"""队列持久化与启动恢复测试（design docs/plans/2026-09-09-training-queue-design.md §4.2）。

TaskStore 用 tmp_path 真文件验证读写往返 / 损坏兜底 / 原子落盘；restore_pending_tasks
用真 TaskManager + 假 Popen 验证恢复入队、id 与提交时间保留、按 kind 重建 setup/meta、
坏记录跳过不阻塞其余恢复。conftest 已把真实单例的 queue_store 断开，这里全部自建实例。
"""
import json

import pytest
from fastapi.testclient import TestClient

from server import paths
from server import tasks as server_tasks
from server.api import datasets as datasets_api  # noqa: F401  separate kind 的注册方
from server.api import training
from server.main import create_app
from server.task_store import TaskStore
from server.tasks import TaskManager


class _FakeProcess:
    """假 Popen 产物：delay 次轮询内保持运行，之后自然成功；finish 模拟 SIGTERM 生效。"""

    def __init__(self, delay=0):
        self.returncode = None
        self._delay = delay
        self.pid = 42424

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._delay > 0:
            self._delay -= 1
            return None
        self.returncode = 0
        return self.returncode

    def finish(self):
        self.returncode = 0


def wait_until(predicate, timeout=5.0, message="等待条件超时"):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


@pytest.fixture(autouse=True)
def isolated_fs(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")
    # 恢复记录里的 log_path 是相对路径字符串（原样来自持久化文件）：
    # 任务启动会按 cwd 打开它，chdir 到 tmp 防止泄漏到仓库根
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """真 TaskManager（假 Popen，进程保持运行）+ tmp 队列文件；training 模块指向它。"""
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProcess(delay=10**6)

    def fake_kill(process, process_name="", task_logger=None):
        process.finish()  # 模拟 SIGTERM 生效（假 pid 不能真发信号）
        return True

    monkeypatch.setattr(server_tasks, "kill_process_tree", fake_kill)
    manager = TaskManager(poll_interval=0.01, popen=fake_popen)
    manager.queue_store = TaskStore(tmp_path / "logs" / "task_queue.json")
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    yield manager, manager.queue_store, calls
    manager.dispose()


def make_record(**overrides):
    """一条持久化的 fit 排队记录（字段与 tasks.py _persist_queue 的快照逐一对齐）。"""
    record = {
        "id": "old-1",
        "created_at": 111.0,
        "name": "fit",
        "cmds": ["cmd-fit"],
        "log_path": "log-fit.log",
        "truncate": True,
        "definition": {
            "kind": "fit",
            "body": {
                "exp_name": "mi-test",
                "sr": "48k",
                "version": "v2",
                "if_f0": True,
                "total_epoch": 20,
                "save_every_epoch": 5,
                "batch_size": 8,
                "save_every_weights": False,
            },
            "batch_note": None,
        },
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# TaskStore：磁盘读写
# ---------------------------------------------------------------------------


def test_store_round_trip(tmp_path):
    store = TaskStore(tmp_path / "logs" / "q.json")
    store.save([{"id": "a"}, {"id": "b"}])
    assert store.load() == [{"id": "a"}, {"id": "b"}]


def test_store_load_missing_file_returns_empty(tmp_path):
    assert TaskStore(tmp_path / "q.json").load() == []


def test_store_save_creates_parent_dirs(tmp_path):
    store = TaskStore(tmp_path / "deep" / "dir" / "q.json")
    store.save([])
    assert (tmp_path / "deep" / "dir" / "q.json").exists()


def test_store_corrupt_file_backed_up_and_emptied(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{not json", encoding="utf8")

    assert TaskStore(path).load() == []

    backups = list(tmp_path.glob("q.json.corrupt-*"))
    assert len(backups) == 1  # 坏文件改名保留，不静默销毁用户数据
    assert backups[0].read_text(encoding="utf8") == "{not json"


def test_store_non_list_payload_treated_as_empty(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('{"id": 1}', encoding="utf8")

    assert TaskStore(path).load() == []


def test_store_save_leaves_no_tmp_leftover(tmp_path):
    store = TaskStore(tmp_path / "q.json")
    store.save([{"id": "a"}])
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# restore_pending_tasks：启动恢复
# ---------------------------------------------------------------------------


def test_restore_requeues_fit_task_preserving_id_and_time(env):
    manager, store, calls = env
    from tests.test_training_api import _fabricate_features

    _fabricate_features(paths.LOGS_DIR / "mi-test")  # setup（filelist/校验）能通过
    store.save([make_record()])

    result = training.restore_pending_tasks()

    assert result == {"restored": 1, "skipped": 0}
    snapshot = manager.get_task("old-1")
    assert snapshot["state"] == "running"  # 空闲服务：恢复即启动
    assert snapshot["created_at"] == 111.0
    wait_until(lambda: len(calls) == 1)  # 子进程已启动
    assert calls == ["cmd-fit"]  # 持久化的最终命令串原样执行
    assert training._TASK_META["old-1"]["exp_name"] == "mi-test"
    assert training._TASK_META["old-1"]["total_epoch"] == 20


def test_restore_preserves_queue_order_when_busy(env):
    manager, store, calls = env
    live = manager.create_task("fit", ["cmd-live"], paths.LOGS_DIR / "live.log")
    wait_until(lambda: len(calls) == 1)

    store.save([make_record(id="old-1"), make_record(id="old-2")])
    result = training.restore_pending_tasks()

    assert result == {"restored": 2, "skipped": 0}
    assert manager.get_task(live)["state"] == "running"  # 不抢占当前任务
    assert manager.get_task("old-1")["state"] == "pending"
    assert manager.get_task("old-1")["queue_position"] == 1
    assert manager.get_task("old-2")["queue_position"] == 2


def test_restore_skips_unknown_kind_and_malformed_records(env):
    manager, store, calls = env
    store.save([
        make_record(id="bad-kind", definition={"kind": "mystery"}),
        {"id": "broken", "created_at": 1.0},  # 缺 name/cmds/log_path
        make_record(),
    ])

    result = training.restore_pending_tasks()

    assert result == {"restored": 1, "skipped": 2}
    assert manager.get_task("old-1") is not None


def test_restore_pipeline_rebuilds_stage_meta(env):
    manager, store, calls = env
    store.save([
        make_record(
            id="pipe-1",
            name="pipeline",
            cmds=["cmd-1", "cmd-2"],
            definition={
                "kind": "pipeline",
                "body": make_record()["definition"]["body"],
                "batch_note": "batch_size 未指定，按设备默认使用 1（无可用显卡）",
                "pipeline_stages": [
                    "preprocess", "preprocess", "extract", "extract",
                    "fit", "fit", "index",
                ],
            },
        )
    ])

    training.restore_pending_tasks()

    meta = training._TASK_META["pipe-1"]
    assert meta["pipeline_stages"][-1] == "index"
    assert meta["total_epoch"] == 20
    assert meta["exp_name"] == "mi-test"


def test_restore_separate_task_registered_by_datasets(env):
    """separate kind 由 datasets 模块注册：恢复无需重建任何东西（cmds 随记录原样执行）。"""
    manager, store, calls = env
    store.save([
        make_record(
            id="sep-1",
            name="separate",
            definition={
                "kind": "separate",
                "input_dir": "/d/alice",
                "output_dir": "/d/alice_vocals",
                "model": "去伴奏",
                "files": [],
            },
        )
    ])

    result = training.restore_pending_tasks()

    assert result == {"restored": 1, "skipped": 0}
    assert manager.get_task("sep-1")["cmds"] == ["cmd-fit"]


def test_restore_without_store_is_noop(env):
    manager, store, calls = env
    manager.queue_store = None

    assert training.restore_pending_tasks() == {"restored": 0, "skipped": 0}


def test_restore_persists_normalized_snapshot(env):
    """恢复路径同样走入队持久化：坏记录被剔除后文件收敛为恢复后的 pending 快照
    （占用当前任务让恢复的任务保持排队，否则它们即刻启动、文件收敛为空）。"""
    manager, store, calls = env
    live = manager.create_task("fit", ["cmd-live"], paths.LOGS_DIR / "live.log")
    wait_until(lambda: len(calls) == 1)

    store.save([
        make_record(id="bad-kind", definition={"kind": "mystery"}),
        make_record(),
    ])

    training.restore_pending_tasks()

    assert store.load() == [make_record()]  # bad-kind 已被剔除


# ---------------------------------------------------------------------------
# lifespan：启动时调用恢复
# ---------------------------------------------------------------------------


def test_lifespan_restores_queue_on_startup(monkeypatch):
    calls = []

    def fake_restore():
        calls.append(1)
        return {"restored": 0, "skipped": 0}

    monkeypatch.setattr(training, "restore_pending_tasks", fake_restore)

    with TestClient(create_app()):
        assert calls == [1]
