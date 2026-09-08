"""训练任务系统测试：互斥 / 终止 / 日志环形缓冲（默认不真跑训练子进程）。

Popen 用构造注入的替身（FakeProcess）替换，日志文件由测试自己写入，任务系统只负责
「起进程 → 轮询 → 增量读日志 → 判定终态」的编排逻辑。kill_process_tree 在测试里一律
替换成记录器：FakeProcess.pid 是假 pid，真终止函数会对它发信号，可能伤及真实进程。
唯一例外是 test_cancel_terminates_real_process_group，它用真实 sleep 子进程验证
start_new_session + 进程组终止确实生效。
"""
import inspect
import os
import subprocess
import sys
import threading
import time

import pytest

from server import paths, tasks
from server.tasks import (
    TERMINAL_STATES,
    TaskConflictError,
    TaskManager,
)

POLL = 0.01  # 测试用短轮询间隔，避免用例变慢


class FakeProcess:
    """subprocess.Popen 替身：前 delay 次 poll() 返回 None（仍在运行），
    之后自然退出为构造给的返回码；finish() 供测试手动推进。"""

    def __init__(self, returncode=0, delay=0):
        self.returncode = None
        self._rc = returncode
        self._delay = delay
        self.pid = 42424

    def poll(self):
        if self.returncode is not None:
            return self.returncode  # 终止后保持终值（finish 与 delay 都要能生效）
        if self._delay > 0:
            self._delay -= 1
            return None
        self.returncode = self._rc
        return self.returncode

    def finish(self):  # 测试手动推进到终态
        self.returncode = self._rc


def wait_until(predicate, timeout=5.0, message="等待条件超时"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL)
    raise AssertionError(message)


def make_manager(monkeypatch, processes, popen_writer=None, **kwargs):
    """构造带 Popen / kill_process_tree 记录器的 TaskManager。

    processes 为每次启动对应的替身（不足时复用最后一个）；popen_writer 为可选的
    「子进程写日志」模拟器（模拟真实子进程向 stdout 重定向文件追加一行）；
    返回 (manager, 启动记录, 终止记录)。
    """
    calls, kills = [], []

    def fake_popen(cmd, **pkwargs):
        if popen_writer is not None:
            popen_writer(cmd)
        proc = processes[min(len(calls), len(processes) - 1)]
        record = dict(cmd=cmd, proc=proc)
        record.update(pkwargs)
        calls.append(record)
        return proc

    def fake_kill(process, process_name="", task_logger=None):
        kills.append((process, process_name))
        process.finish()  # 模拟 SIGTERM 生效
        return True

    monkeypatch.setattr(tasks, "kill_process_tree", fake_kill)
    manager = TaskManager(poll_interval=POLL, popen=fake_popen, **kwargs)
    return manager, calls, kills


@pytest.fixture
def factory(monkeypatch):
    """每个用例独立实例；收尾时终止仍活跃的任务并等待工作线程退出，用例间零残留。"""
    created = {}

    def _factory(processes=None, popen_writer=None, **kwargs):
        manager, calls, kills = make_manager(
            monkeypatch, processes or [FakeProcess(0)], popen_writer=popen_writer, **kwargs
        )
        created["manager"] = manager
        return manager, calls, kills

    yield _factory
    if "manager" in created:
        created["manager"].dispose()


# ---------------------------------------------------------------------------
# 1. 正常生命周期
# ---------------------------------------------------------------------------


def test_create_task_runs_cmds_in_order_and_succeeds(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(0), FakeProcess(0)])
    log_path = tmp_path / "logs" / "mi-test" / "train.log"

    task_id = manager.create_task("fit", ["cmd-a", "cmd-b"], log_path)

    assert isinstance(task_id, str) and task_id
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")
    assert [call["cmd"] for call in calls] == ["cmd-a", "cmd-b"]

    first = calls[0]
    assert first["shell"] is True
    assert first["cwd"] == paths.ROOT
    assert first["stderr"] == subprocess.STDOUT
    assert first["start_new_session"] is True  # POSIX 独立进程组，便于整树终止
    assert first["stdout"].name == str(log_path)
    assert first["stdout"].mode == "wb"  # 默认 truncate=True：每 cmd 启动前截断（对齐 webui）
    assert calls[1]["stdout"] is not first["stdout"]  # 每个 cmd 独立句柄
    assert first["stdout"].closed  # 终态后句柄关闭，不留悬挂 fd

    snapshot = manager.get_task(task_id)
    assert snapshot["error"] is None
    assert snapshot["progress"] is None  # Task 5/6 的编排负责写入
    assert snapshot["cmds"] == ["cmd-a", "cmd-b"]
    assert snapshot["finished_at"] >= snapshot["started_at"] >= snapshot["created_at"]


def test_default_poll_interval_is_one_second():
    assert TaskManager().poll_interval == 1.0


def test_popen_resolved_lazily_so_monkeypatch_works(monkeypatch, tmp_path):
    """未注入 popen 时逐次解析 subprocess.Popen：替换模块属性同样能被任务系统用到。"""
    monkeypatch.setattr(tasks, "kill_process_tree", lambda *args, **kwargs: True)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: FakeProcess(0, delay=1))

    manager = TaskManager(poll_interval=POLL)
    try:
        task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
        wait_until(lambda: manager.get_task(task_id)["state"] == "success")
    finally:
        manager.dispose()


def test_create_task_rejects_empty_cmds(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(0)])
    with pytest.raises(ValueError):
        manager.create_task("fit", [], tmp_path / "train.log")
    assert calls == [] and manager.list_tasks() == []


def test_failure_records_returncode_and_log_tail(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(returncode=3, delay=3)])
    log_path = tmp_path / "extract.log"
    log_path.write_text("准备开始\nboom: 音频损坏\n", encoding="utf8")

    task_id = manager.create_task("extract", ["cmd"], log_path, truncate=False)

    wait_until(lambda: manager.get_task(task_id)["state"] == "failed")
    error = manager.get_task(task_id)["error"]
    assert "返回码" in error and "3" in error
    assert "boom" in error and "准备开始" in error
    assert len(calls) == 1  # 失败后不再启动后续 cmd（pipeline 语义）


def test_failure_message_reports_which_step_failed(factory, tmp_path):
    """extract 双 cmd 共享日志时，前端要能区分是 f0 还是 hubert 挂了。"""
    manager, calls, _ = factory([FakeProcess(0), FakeProcess(returncode=2, delay=2)])

    task_id = manager.create_task(
        "extract", ["f0-cmd", "hubert-cmd"], tmp_path / "extract.log"
    )
    wait_until(lambda: manager.get_task(task_id)["state"] == "failed")

    error = manager.get_task(task_id)["error"]
    assert "第 2 步" in error and "hubert-cmd" in error
    assert "第 1 步" not in error
    assert "返回码" in error and "2" in error
    assert [call["cmd"] for call in calls] == ["f0-cmd", "hubert-cmd"]


def test_log_truncated_per_cmd_by_default(factory, tmp_path):
    """默认 truncate=True：每个 cmd 启动前截断日志（对齐 webui 每阶段截断行为）；
    已读进环形缓冲的旧行不丢（SSE 历史仍完整）。"""
    log_path = tmp_path / "extract.log"

    def writer(cmd):
        with open(log_path, "ab") as handle:
            handle.write(("out:%s\n" % cmd).encode("utf8"))

    manager, calls, _ = factory([FakeProcess(0), FakeProcess(0)], popen_writer=writer)
    log_path.write_text("上一次运行的残留\n", encoding="utf8")

    task_id = manager.create_task("extract", ["cmd-a", "cmd-b"], log_path)
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")

    assert log_path.read_text(encoding="utf8") == "out:cmd-b\n"  # 残留与 cmd-a 输出已被截断
    assert [call["cmd"] for call in calls] == ["cmd-a", "cmd-b"]
    assert list(manager.get_task(task_id)["logs"]) == ["out:cmd-a", "out:cmd-b"]


def test_log_appends_when_truncate_false(factory, tmp_path):
    """truncate=False：二次启动追加不截断（fit 若将来要接 train.log 场景）。"""
    log_path = tmp_path / "train.log"

    def writer(cmd):
        with open(log_path, "ab") as handle:
            handle.write(("out:%s\n" % cmd).encode("utf8"))

    manager, _, _ = factory([FakeProcess(0), FakeProcess(0)], popen_writer=writer)
    log_path.write_text("上一次运行的残留\n", encoding="utf8")

    task_id = manager.create_task("fit", ["cmd-a", "cmd-b"], log_path, truncate=False)
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")

    assert log_path.read_text(encoding="utf8") == "上一次运行的残留\nout:cmd-a\nout:cmd-b\n"


def test_setup_failure_marks_task_failed_without_starting_cmds(factory, tmp_path):
    """Task 5 的接缝：filelist/config 生成等 Python 前置步骤在任务内跑，失败即任务失败而非 500。"""
    manager, calls, _ = factory([FakeProcess(0)])

    def setup():
        raise ValueError("没有可用于训练的有效音频，请先完成数据切分和特征提取")

    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log", setup=setup)
    wait_until(lambda: manager.get_task(task_id)["state"] == "failed")
    assert "没有可用于训练的有效音频" in manager.get_task(task_id)["error"]
    assert calls == []


def test_setup_return_value_appended_to_task_log(factory, tmp_path):
    """setup 可返回提示行（str 或 list）写进任务日志缓冲：编排层用它上报「未使用底模」。
    只写缓冲不落盘——truncate=True 时首个 cmd 启动会截断日志文件，缓冲才是 GET/SSE 的
    任务日志视图。"""
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    log_path = tmp_path / "train.log"

    def setup():
        return "生成器预训练模型不存在，将不使用：assets/pretrained_v2/f0G48k.pth"

    task_id = manager.create_task("fit", ["cmd"], log_path, setup=setup)
    wait_until(
        lambda: manager.get_task(task_id)["logs"]
        == ["生成器预训练模型不存在，将不使用：assets/pretrained_v2/f0G48k.pth"]
    )
    assert manager.get_task(task_id)["log_seq"] == 1  # SSE 游标同步推进，不漏行

    with open(log_path, "a", encoding="utf8") as handle:  # cmd 输出照常进文件
        handle.write("epoch output\n")
        handle.flush()
    wait_until(lambda: len(manager.get_task(task_id)["logs"]) == 2)

    manager.cancel(task_id)  # 收尾，释放互斥


def test_setup_returning_none_writes_nothing(factory, tmp_path):
    manager, _, _ = factory([FakeProcess(0, delay=10**6)])
    task_id = manager.create_task(
        "fit", ["cmd"], tmp_path / "train.log", setup=lambda: None
    )
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")
    time.sleep(0.05)

    assert manager.get_task(task_id)["logs"] == []
    assert manager.get_task(task_id)["log_seq"] == 0

    manager.cancel(task_id)  # 收尾，释放互斥


# ---------------------------------------------------------------------------
# 2. 全局互斥
# ---------------------------------------------------------------------------


def test_global_mutex_blocks_second_task_until_terminal(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    first = manager.create_task("preprocess", ["cmd-1"], tmp_path / "a.log")
    wait_until(lambda: len(calls) == 1)  # 第一个进程已启动

    with pytest.raises(TaskConflictError):
        manager.create_task("extract", ["cmd-2"], tmp_path / "b.log")

    calls[0]["proc"].finish()
    wait_until(lambda: manager.get_task(first)["state"] == "success")

    second = manager.create_task("extract", ["cmd-2"], tmp_path / "b.log")  # 终态后互斥释放
    wait_until(lambda: manager.get_task(second)["state"] == "success")
    assert [call["cmd"] for call in calls] == ["cmd-1", "cmd-2"]


def test_concurrent_create_only_one_wins(factory, tmp_path):
    """互斥标志在并发 create 下必须只放行一个任务。"""
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    started, conflicts, lock = [], [], threading.Lock()

    def worker(index):
        try:
            task_id = manager.create_task("t%d" % index, ["cmd"], tmp_path / "a.log")
            with lock:
                started.append(task_id)
        except TaskConflictError:
            with lock:
                conflicts.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert len(started) == 1
    assert len(conflicts) == 7
    assert len(manager.list_tasks()) == 1
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3. 终止
# ---------------------------------------------------------------------------


def test_cancel_kills_running_process_and_skips_remaining_cmds(factory, tmp_path):
    manager, calls, kills = factory([FakeProcess(returncode=-15, delay=10**6), FakeProcess(0)])

    task_id = manager.create_task("pipeline", ["cmd-1", "cmd-2"], tmp_path / "train.log")
    wait_until(lambda: len(calls) == 1)

    assert manager.cancel(task_id) is True
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")

    assert kills and kills[0][0] is calls[0]["proc"] and kills[0][1] == "pipeline"
    assert len(calls) == 1  # cmd-2 未启动


def test_cancelled_wins_over_nonzero_returncode(factory, tmp_path):
    """人为停止导致的非零返回码不得判成 failed。"""
    manager, _, _ = factory([FakeProcess(returncode=-15, delay=10**6)])
    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
    assert manager.get_task(task_id)["error"] is None


def test_cancel_before_first_cmd_starts_nothing(factory, tmp_path):
    """每个 cmd 起前都要检查停止标志（含首个）。"""
    manager, calls, _ = factory([FakeProcess(0)])
    entered, release = threading.Event(), threading.Event()

    def setup():
        entered.set()
        release.wait(5)

    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log", setup=setup)
    assert entered.wait(5)
    assert manager.cancel(task_id) is True
    release.set()

    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
    assert calls == []


def test_cancel_unknown_or_terminal_task_returns_false(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    assert manager.cancel("no-such-id") is False

    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
    wait_until(lambda: len(calls) == 1)  # 子进程已启动
    calls[0]["proc"].finish()
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")
    assert manager.cancel(task_id) is False


def test_dispose_cancels_running_task(factory, tmp_path):
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")

    manager.dispose()

    assert manager.get_task(task_id)["state"] == "cancelled"
    assert len(calls) == 1  # 未启动后续 cmd


def test_cancel_terminates_real_process_group(monkeypatch, tmp_path):
    """集成验证（唯一真起子进程的用例）：start_new_session + kill_process_tree 确实能整树终止。"""
    real_popen = subprocess.Popen
    procs = []

    def spy_popen(cmd, **kwargs):
        proc = real_popen(cmd, **kwargs)
        procs.append(proc)
        return proc

    manager = TaskManager(poll_interval=POLL, popen=spy_popen)
    try:
        cmd = '"%s" -c "import time; time.sleep(30)"' % sys.executable
        task_id = manager.create_task("sleepy", [cmd], tmp_path / "sleep.log")
        wait_until(lambda: procs and procs[0].poll() is None, message="子进程未启动")

        assert manager.cancel(task_id) is True
        wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
        wait_until(lambda: procs[0].poll() is not None, message="子进程未被终止")
        assert procs[0].returncode != 0
    finally:
        manager.dispose()


# ---------------------------------------------------------------------------
# 4. 日志环形缓冲
# ---------------------------------------------------------------------------


def test_log_buffer_reads_new_lines_incrementally(factory, tmp_path):
    """测试自己写日志文件，任务系统轮询增量读进缓冲（与训练脚本并发 append 兼容）。"""
    manager, _, _ = factory([FakeProcess(0, delay=10**6)])
    log_path = tmp_path / "train.log"
    task_id = manager.create_task("fit", ["cmd"], log_path)
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")

    with open(log_path, "a", encoding="utf8") as handle:
        handle.write("训练轮次：1 [10%]\n")
        handle.flush()
    wait_until(lambda: manager.get_task(task_id)["logs"] == ["训练轮次：1 [10%]"])

    with open(log_path, "a", encoding="utf8") as handle:
        handle.write("loss_disc=1.0, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0\n")
        handle.flush()
    wait_until(lambda: len(manager.get_task(task_id)["logs"]) == 2)

    lines, cursor = manager.read_logs_since(task_id, 0)
    assert list(lines) == [
        "训练轮次：1 [10%]",
        "loss_disc=1.0, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0",
    ]
    assert cursor == 2

    # SSE 增量语义：游标之后只回新行；无新行时游标不变
    lines, cursor = manager.read_logs_since(task_id, cursor)
    assert lines == () and cursor == 2

    manager.cancel(task_id)  # 收尾，释放互斥


def test_log_buffer_is_ring_with_maxlen_1000(factory, tmp_path):
    manager, _, _ = factory([FakeProcess(0, delay=10**6)])
    log_path = tmp_path / "train.log"
    log_path.write_text("".join("line%d\n" % i for i in range(1200)), encoding="utf8")

    task_id = manager.create_task("fit", ["cmd"], log_path, truncate=False)
    wait_until(lambda: manager.get_task(task_id)["log_seq"] == 1200)

    snapshot = manager.get_task(task_id)
    assert len(snapshot["logs"]) == 1000
    assert snapshot["logs"][0] == "line200"  # 旧行被淘汰
    assert snapshot["logs"][-1] == "line1199"

    # 游标已越过缓冲范围：只回缓冲里剩下的部分
    lines, cursor = manager.read_logs_since(task_id, 0)
    assert len(lines) == 1000 and cursor == 1200

    manager.cancel(task_id)  # 收尾，释放互斥


def test_partial_line_is_not_flushed_until_complete(factory, tmp_path):
    """残行（未换行）不进缓冲，等完整行再读——训练日志按行追加，半行会污染解析。"""
    manager, calls, _ = factory([FakeProcess(0, delay=10**6)])
    log_path = tmp_path / "train.log"
    task_id = manager.create_task("fit", ["cmd"], log_path)
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")
    wait_until(lambda: len(calls) == 1, message="子进程未启动")  # 截断打开发生在 Popen 前，先等它

    with open(log_path, "a", encoding="utf8") as handle:
        handle.write("训练轮次：3 [")
        handle.flush()
    time.sleep(0.1)  # 给几个轮询周期
    assert manager.get_task(task_id)["logs"] == []

    with open(log_path, "a", encoding="utf8") as handle:
        handle.write("30%]\n")
        handle.flush()
    wait_until(lambda: manager.get_task(task_id)["logs"] == ["训练轮次：3 [30%]"])

    manager.cancel(task_id)  # 收尾，释放互斥


def test_read_logs_since_unknown_task_returns_none(factory):
    manager, _, _ = factory([FakeProcess(0)])
    assert manager.read_logs_since("no-such-id", 0) is None


# ---------------------------------------------------------------------------
# 5. 快照与进度字段
# ---------------------------------------------------------------------------


def test_get_task_unknown_returns_none(factory):
    manager, _, _ = factory([FakeProcess(0)])
    assert manager.get_task("no-such-id") is None


def test_list_tasks_returns_creation_order_and_isolated_copies(factory, tmp_path):
    manager, _, _ = factory([FakeProcess(0)])
    first = manager.create_task("preprocess", ["cmd-1"], tmp_path / "a.log")
    wait_until(lambda: manager.get_task(first)["state"] == "success")
    second = manager.create_task("extract", ["cmd-2"], tmp_path / "b.log")
    wait_until(lambda: manager.get_task(second)["state"] == "success")

    snapshots = manager.list_tasks()
    assert [item["id"] for item in snapshots] == [first, second]
    assert [item["state"] for item in snapshots] == ["success", "success"]

    snapshots[0]["state"] = "mutated"  # 快照是副本，不影响内部状态
    assert manager.get_task(first)["state"] == "success"


def test_update_progress_clamps_and_reports_unknown_id(factory, tmp_path):
    manager, _, _ = factory([FakeProcess(0, delay=10**6)])
    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")

    assert manager.update_progress(task_id, 0.5) is True
    assert manager.get_task(task_id)["progress"] == 0.5
    manager.update_progress(task_id, 1.5)
    assert manager.get_task(task_id)["progress"] == 1.0
    manager.update_progress(task_id, -3)
    assert manager.get_task(task_id)["progress"] == 0.0
    manager.update_progress(task_id, None)
    assert manager.get_task(task_id)["progress"] is None
    assert manager.update_progress("no-such-id", 0.5) is False

    manager.cancel(task_id)  # 收尾，释放互斥


# ---------------------------------------------------------------------------
# 6. 命令环境与线程生命周期
# ---------------------------------------------------------------------------


def test_popen_env_injects_repo_root_into_pythonpath(monkeypatch):
    """训练脚本以 `python train/xxx.py` 直跑，sys.path[0] 是 train/ 而非仓库根，
    脚本内 `from infer.audio import ...` 这类顶层导入必须靠 PYTHONPATH 注入仓库根。"""
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = tasks._popen_env("cmd", paths.ROOT)

    assert env["PYTHONPATH"] == str(paths.ROOT)
    assert env["PYTHONSAFEPATH"] == "1"  # 不把脚本目录插进 sys.path，防 train.train 遮蔽 train 包


def test_popen_env_prepends_root_to_existing_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/custom/pkg")

    env = tasks._popen_env("cmd", paths.ROOT)

    assert env["PYTHONPATH"] == os.pathsep.join([str(paths.ROOT), "/custom/pkg"])


def test_training_subprocess_imports_repo_packages(monkeypatch, tmp_path):
    """集成（真起子进程，与 test_cancel_terminates_real_process_group 同法）：
    复现 train/xxx.py 的真实 sys.path 形态，验证两层修复，缺一训练子进程都起不来——
    ① PYTHONPATH=仓库根：脚本直跑时 sys.path[0] 是脚本目录（cwd 不参与），顶层导入
       `import server.paths`（与 preprocess.py:21 的 `from infer.audio import ...` 同类）
       只能靠它，否则 ModuleNotFoundError；
    ② PYTHONSAFEPATH=1（python -P）：sys.path[0] 里存在 train.py 时，顶层 `train` 会被
       解析成该文件而非 train/ 命名空间包（真实仓库的 train/train.py 正是如此），
       `from train import helper`（对应 train.py:31 的 `from train import utils`）因此
       ImportError。探针不依赖 torch；修复前两种形态都已手工复现。"""
    probe_dir = tmp_path / "probe_dir"  # sys.path[0]（脚本所在目录）
    probe_dir.mkdir()
    (probe_dir / "train.py").write_text("", encoding="utf8")  # 遮蔽物 = 真实仓库的 train/train.py
    pkg = tmp_path / "probe_pkg"  # 充当「仓库根」：train 是正规包
    (pkg / "train").mkdir(parents=True)
    (pkg / "train" / "__init__.py").write_text("", encoding="utf8")
    (pkg / "train" / "helper.py").write_text("MARKER = 'ok'\n", encoding="utf8")
    probe = probe_dir / "probe.py"
    probe.write_text(
        "import server.paths\nfrom train import helper\nprint(helper.MARKER)\n",
        encoding="utf8",
    )
    monkeypatch.setenv("PYTHONPATH", str(pkg))  # 已有 PYTHONPATH：须被前插而非覆盖

    real_popen = subprocess.Popen
    captured = {}

    def spy_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return real_popen(cmd, **kwargs)

    manager = TaskManager(poll_interval=POLL, popen=spy_popen)
    try:
        task_id = manager.create_task(
            "probe", ['"%s" "%s"' % (sys.executable, probe)], tmp_path / "probe.log"
        )
        wait_until(lambda: manager.get_task(task_id)["state"] == "success")
    finally:
        manager.dispose()

    assert captured["env"]["PYTHONPATH"] == os.pathsep.join([str(paths.ROOT), str(pkg)])
    assert captured["env"]["PYTHONSAFEPATH"] == "1"
    assert manager.get_task(task_id)["logs"] == ["ok"]


def test_train_cmd_disables_cuda_graph_and_env_is_utf8(factory, tmp_path):
    """webui.py:786-790 同源：train/train.py 注入 RVC_CUDA_GRAPH=0；
    全部命令统一注入 PYTHONIOENCODING=utf-8（防 Windows GBK 子进程日志 mojibake）。"""
    manager, calls, _ = factory([FakeProcess(0, delay=1)])
    cmds = [
        '"py" train/train.py -e e -sr 48k',
        '"py" train\\train.py -e e -sr 48k',  # Windows 路径分隔形式同样识别
        '"py" train/preprocess.py "/data/ds"',
    ]
    task_id = manager.create_task("pipeline", cmds, tmp_path / "train.log")
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")

    for index in range(3):
        assert calls[index]["env"]["PYTHONIOENCODING"] == "utf-8", index
    assert calls[0]["env"]["RVC_CUDA_GRAPH"] == "0"
    assert calls[1]["env"]["RVC_CUDA_GRAPH"] == "0"
    assert "RVC_CUDA_GRAPH" not in calls[2]["env"]  # 非训练命令不注入
    assert calls[2]["env"]["PATH"] == os.environ.get("PATH")  # 其余继承父进程环境


def test_worker_thread_exits_after_terminal_state(factory, tmp_path):
    """终态后工作线程必须退出，不留悬挂线程（互斥锁随之释放）。"""
    baseline = threading.active_count()
    manager, _, _ = factory([FakeProcess(0, delay=1)])
    task_id = manager.create_task("fit", ["cmd"], tmp_path / "train.log")
    wait_until(lambda: manager.get_task(task_id)["state"] == "success")
    wait_until(lambda: threading.active_count() == baseline, timeout=5, message="工作线程未退出")


def test_dispose_default_timeout_covers_kill_process_tree_blocking():
    """kill_process_tree 自身可阻塞约 6s（SIGTERM→等待→SIGKILL→wait），默认超时须覆盖它。"""
    assert inspect.signature(TaskManager.dispose).parameters["timeout"].default == 10.0


def test_module_does_not_load_torch_or_configs():
    """延迟导入纪律：import server.tasks 不得连带加载 torch/configs。"""
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import server.tasks;"
        "assert 'torch' not in sys.modules, 'torch 被加载';"
        "assert 'configs.config' not in sys.modules, 'configs 被加载';"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(paths.ROOT), capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_terminal_states_constant():
    assert TERMINAL_STATES == frozenset({"success", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# current_cmd：pipeline 阶段映射的数据源
# ---------------------------------------------------------------------------


def test_current_cmd_follows_running_subcommand(factory, tmp_path):
    """第一个 cmd 阻塞期间 current_cmd 停在 1；取消终态保留最后执行过的序号。"""
    manager, _, _ = factory([FakeProcess(0, delay=10**6), FakeProcess(0)])
    task_id = manager.create_task("multi", ["cmd-a", "cmd-b"], tmp_path / "multi.log")

    wait_until(lambda: manager.get_task(task_id)["current_cmd"] == 1)
    snapshot = manager.get_task(task_id)
    assert snapshot["state"] == "running"
    assert snapshot["cmds"][snapshot["current_cmd"] - 1] == "cmd-a"

    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
    assert manager.get_task(task_id)["current_cmd"] == 1  # 诊断值：出错时停在第几步


def test_current_cmd_keeps_last_index_after_success(factory, tmp_path):
    """成功终态 current_cmd = 最后启动的 cmd 序号（消费方按 state 判成功，序号仅存档）。"""
    manager, _, _ = factory([FakeProcess(0), FakeProcess(0)])
    task_id = manager.create_task("multi", ["cmd-a", "cmd-b"], tmp_path / "multi.log")

    wait_until(lambda: manager.get_task(task_id)["state"] == "success")

    assert manager.get_task(task_id)["current_cmd"] == 2
