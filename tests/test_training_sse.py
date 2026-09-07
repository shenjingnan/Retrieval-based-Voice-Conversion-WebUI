"""任务 SSE（日志/进度/状态流）测试。

用真 TaskManager + 假 Popen（不真跑子进程）：测试自己向任务日志追加行，任务系统按
轮询增量读进环形缓冲，SSE 端点从缓冲取增量——完整覆盖游标协议。假进程不会自然退出，
终态一律由用例显式 finish 推进（流随之以终态 status 收尾，响应体正常读到 EOF）。

TestClient 的 ASGI 传输会缓冲完整响应体且不向应用传播 http.disconnect：前者意味着不能
边读流边与任务交互，流内前置状态（running 的初始 status、分阶段写入的日志）由后台线程
在订阅建立后推进（见 _wait_subscribed）；后者意味着断开清理在生成器层面直接验证
（gen.close() 触发 GeneratorExit → finally 注销订阅）。
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from server import commands, paths
from server.api import training
from server.main import create_app
from server.tasks import TaskManager

POLL = 0.01


class FakeProcess:
    """delay 个轮询周期后以 returncode 退出。"""

    def __init__(self, returncode=0, delay=0):
        self.returncode = None
        self._rc = returncode
        self._delay = delay
        self.pid = 42424

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._delay > 0:
            self._delay -= 1
            return None
        self.returncode = self._rc
        return self.returncode

    def finish(self):  # 测试手动推进到终态
        self.returncode = self._rc

    def wait(self, timeout=None):  # kill_process_tree 终止后会等待
        return self.returncode

    def kill(self):
        self.returncode = self._rc


def wait_until(predicate, timeout=5.0, message="等待条件超时"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL)
    raise AssertionError(message)


@pytest.fixture(autouse=True)
def fast_stream(monkeypatch):
    """注入小轮询/心跳间隔（生产为 1s / 15s）。"""
    monkeypatch.setattr(training, "_SSE_POLL_INTERVAL", POLL)
    monkeypatch.setattr(training, "_SSE_KEEPALIVE_INTERVAL", 0.03)


@pytest.fixture(autouse=True)
def isolated_fs(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")


@pytest.fixture(autouse=True)
def fake_device(monkeypatch):
    monkeypatch.setattr(commands, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(training, "resolve_is_half", lambda: False)


@pytest.fixture
def manager(monkeypatch, tmp_path):
    """真 TaskManager（假 Popen）；返回 (manager, 进程列表) 供测试推进状态。"""
    processes = []

    def fake_popen(cmd, **kwargs):
        proc = FakeProcess(0, delay=10**6)  # 不会自然退出：终态一律由用例显式 finish，避免负载下先于首帧到期
        processes.append(proc)
        return proc

    mgr = TaskManager(poll_interval=POLL, popen=fake_popen)
    monkeypatch.setattr(training, "task_manager", mgr)
    monkeypatch.setattr(training, "_TASK_META", {})
    yield mgr, processes
    mgr.dispose()


@pytest.fixture
def client():
    return TestClient(create_app())


def _create_fit(manager, tmp_path, *, lines=()):
    """建一个 fit 任务（total_epoch=10 供进度换算），可预写日志行。
    返回 (task_id, log_path, processes)；等到假进程已被启动（state=running 早于
    Popen，直接返回会有竞态），测试才能推进它的终态。"""
    mgr, processes = manager
    log_path = tmp_path / "logs" / "mi-test" / "train_task_fit.log"
    task_id = mgr.create_task("fit", ["cmd"], log_path)
    training._TASK_META[task_id] = {"total_epoch": 10}
    wait_until(lambda: mgr.get_task(task_id)["state"] == "running")
    wait_until(lambda: bool(processes), message="假进程未被启动")
    for line in lines:
        with open(log_path, "a", encoding="utf8") as handle:
            handle.write(line + "\n")
            handle.flush()
    if lines:  # 等缓冲收齐再返回：订阅首条 status 的进度换算才确定
        wait_until(
            lambda: mgr.get_task(task_id)["log_seq"] >= len(lines), message="日志未进缓冲"
        )
    return task_id, log_path, processes


def _read_all(client, task_id, cursor=0):
    """读到流自然结束（任务终态收尾），返回原始文本。"""
    with client.stream(
        "GET", f"/api/tasks/{task_id}/events", params={"cursor": cursor}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        return b"".join(response.iter_raw()).decode("utf8")


def _parse_frame(block):
    """单个 SSE 帧（不含结尾空行的行列表）→ (event, data | None)；注释帧 data 为 None。"""
    event = next((l[len("event: "):] for l in block if l.startswith("event: ")), None)
    data = next((l[len("data: "):] for l in block if l.startswith("data: ")), None)
    return (event, json.loads(data) if data is not None else None)


def _iter_frames(lines):
    """SSE 行迭代器 → 帧迭代器（以空行分帧）。"""
    block = []
    for line in lines:
        if line == "":
            if block:
                yield _parse_frame(block)
                block = []
        else:
            block.append(line)
    if block:
        yield _parse_frame(block)


def _frames(text):
    """SSE 文本 → [(event, data | None)]；注释帧（心跳）data 为 None。"""
    return list(_iter_frames(text.split("\n")))


def _append(log_path, line):
    with open(log_path, "a", encoding="utf8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _wait_subscribed(task_id, timeout=5.0):
    """阻塞到服务端 SSE 生成器注册订阅为止。

    TestClient 的 ASGI 传输会缓冲完整响应体（应用结束才返回），因此不能边读流边交互：
    需要在流内出现的前置状态（running 的初始 status、分阶段日志）必须由后台线程在订阅
    建立之后推进，流内的时序才是确定的。"""
    deadline = time.monotonic() + timeout
    while task_id not in training._active_streams and time.monotonic() < deadline:
        time.sleep(0.01)
    assert task_id in training._active_streams, "SSE 订阅未建立"


# ---------------------------------------------------------------------------
# 订阅与事件格式
# ---------------------------------------------------------------------------


def test_stream_starts_with_status_then_replays_logs(manager, client, tmp_path):
    task_id, log_path, processes = _create_fit(
        manager, tmp_path, lines=["INFO:mi-test:训练轮次：5 [40%]"]
    )

    def drive():
        _wait_subscribed(task_id)  # 先保证 running 的初始 status 已入流，再推进终态
        processes[0].finish()

    threading.Thread(target=drive, daemon=True).start()

    with client.stream("GET", f"/api/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        frames = list(_iter_frames(response.iter_lines()))

    assert ("log", {"lines": ["INFO:mi-test:训练轮次：5 [40%]"], "seq": 1}) in frames
    assert ("progress", {"progress": 1.0, "current": None}) in frames  # 终态前推最终进度
    assert frames[0][0] == "status"
    status = frames[0][1]
    assert status["state"] == "running"
    assert status["progress"] == pytest.approx((4 + 0.4) / 10)  # 读时解析换算
    assert "logs" not in status  # 日志由 log 事件承载，status 不重复携带
    assert status["id"] == task_id and status["name"] == "fit"
    assert frames[-1][0] == "status" and frames[-1][1]["state"] == "success"
    assert frames[-1][1]["progress"] == 1.0


def test_stream_pushes_log_events_incrementally(manager, client, tmp_path):
    task_id, log_path, processes = _create_fit(manager, tmp_path)
    staged = [
        "INFO:mi-test:训练轮次：1 [10%]",
        "INFO:mi-test:[120, 0.0001]",
        "INFO:mi-test:loss_disc=1.0, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0",
    ]

    def writer():
        _wait_subscribed(task_id)  # 写入必须发生在订阅之后，才会分批推送
        for line in staged:
            time.sleep(0.05)
            _append(log_path, line)
        processes[0].finish()

    threading.Thread(target=writer, daemon=True).start()

    frames = _frames(_read_all(client, task_id))
    log_frames = [frame for frame in frames if frame[0] == "log"]

    assert len(log_frames) == 3  # 分阶段写入 → 分批推送
    assert [frame[1]["seq"] for frame in log_frames] == [1, 2, 3]
    assert [frame[1]["lines"] for frame in log_frames] == [[line] for line in staged]


def test_cursor_skips_already_delivered_lines(manager, client, tmp_path):
    """M-2 游标协议明确化：cursor 为已消费行数，增量只回其后的行；
    重连一律 cursor=0 重放全量，由前端幂等渲染。"""
    mgr, _ = manager
    task_id, log_path, processes = _create_fit(
        manager, tmp_path, lines=["first line", "second line"]
    )
    wait_until(lambda: mgr.get_task(task_id)["log_seq"] == 2)
    processes[0].finish()
    wait_until(lambda: mgr.get_task(task_id)["state"] == "success")

    resumed = _frames(_read_all(client, task_id, cursor=2))
    assert [frame for frame in resumed if frame[0] == "log"] == []

    replayed = _frames(_read_all(client, task_id, cursor=0))
    log_frames = [frame for frame in replayed if frame[0] == "log"]
    assert log_frames[0][1]["seq"] == 2
    assert log_frames[0][1]["lines"] == ["first line", "second line"]


def test_keepalive_comments_while_idle(manager, client, tmp_path):
    """无新日志、进度不变时按间隔发 `: keepalive` 注释帧，防代理断连。"""
    task_id, _, processes = _create_fit(manager, tmp_path)  # 无日志、progress 不变

    def drive():
        _wait_subscribed(task_id)
        time.sleep(0.3)  # 保持 running 一段时间，让心跳帧有积累
        processes[0].finish()

    threading.Thread(target=drive, daemon=True).start()

    text = _read_all(client, task_id)

    assert ": keepalive" in text
    assert text.count(": keepalive") >= 2


# ---------------------------------------------------------------------------
# 进度来源
# ---------------------------------------------------------------------------


def test_progress_without_meta_passes_raw_value(manager, client, tmp_path):
    """非训练阶段（无 total_epoch 元数据）progress 透传 task.progress 原值；
    订阅首条 status 即携带当前进度，变化时才发 progress 事件。"""
    mgr, processes = manager
    log_path = tmp_path / "logs" / "mi-test" / "preprocess.log"
    task_id = mgr.create_task("preprocess", ["cmd"], log_path)
    wait_until(lambda: bool(processes))
    mgr.update_progress(task_id, 0.42)

    def drive():
        _wait_subscribed(task_id)
        processes[0].finish()

    threading.Thread(target=drive, daemon=True).start()

    frames = _frames(_read_all(client, task_id))

    assert frames[0][0] == "status" and frames[0][1]["progress"] == 0.42
    assert ("progress", {"progress": 1.0, "current": None}) in frames  # 终态变化触发


def test_progress_stays_none_until_first_anchor(manager, client, tmp_path):
    """训练任务尚无 epoch 锚点行时不臆造进度。"""
    task_id, _, processes = _create_fit(manager, tmp_path)

    def drive():
        _wait_subscribed(task_id)
        processes[0].finish()

    threading.Thread(target=drive, daemon=True).start()

    frames = _frames(_read_all(client, task_id))

    assert ("progress", {"progress": 1.0, "current": None}) in frames  # 只有终态那次
    assert ("progress", {"progress": None, "current": None}) not in frames
    assert frames[0][1]["progress"] is None


# ---------------------------------------------------------------------------
# 参数与错误
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cursor", ["abc", "1.5", "-1"])
def test_invalid_cursor_rejected(client, cursor):
    resp = client.get("/api/tasks/whatever/events", params={"cursor": cursor})

    assert resp.status_code == 400
    assert "cursor" in resp.json()["detail"]


def test_unknown_task_returns_404(client):
    resp = client.get("/api/tasks/no-such-id/events")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 断开清理
# ---------------------------------------------------------------------------


def test_generator_close_unregisters_subscription(manager, tmp_path):
    """客户端断开（生成器被 close / GC）必须注销订阅，不留泄漏。"""
    task_id, _, _ = _create_fit(manager, tmp_path)

    generator = training._task_event_stream(task_id, 0)
    first = next(generator)
    assert first.startswith("event: status")
    assert task_id in training._active_streams

    generator.close()  # GeneratorExit → finally

    assert task_id not in training._active_streams
    with pytest.raises(StopIteration):
        next(generator)  # 已关闭


def test_closed_task_stream_does_not_leak_subscription(manager, client, tmp_path):
    """终态收尾路径同样注销订阅。"""
    mgr, _ = manager
    task_id, _, processes = _create_fit(manager, tmp_path)
    processes[0].finish()
    wait_until(lambda: mgr.get_task(task_id)["state"] == "success")

    assert _read_all(client, task_id)

    assert task_id not in training._active_streams


# ---------------------------------------------------------------------------
# 切分/提取阶段的产物计数进度（design §3.5）与终态单条 status
# ---------------------------------------------------------------------------


def test_stage_progress_events_carry_current_file(manager, client, tmp_path):
    """切分阶段：脚本进度行 → SSE progress 事件带数值与当前文件名（design §3.5）。"""
    mgr, _ = manager
    log_path = tmp_path / "logs" / "mi-test" / "preprocess.log"
    task_id = mgr.create_task("preprocess", ["cmd"], log_path)
    _, processes = manager
    wait_until(lambda: bool(processes))

    def writer():
        _wait_subscribed(task_id)  # 进度行须在订阅后写入，progress 事件才可见
        _append(log_path, "[数据切分] 进度：1/2 | a.wav")
        time.sleep(0.05)
        _append(log_path, "[数据切分] 进度：2/2 | b.wav")
        time.sleep(0.15)  # 留出轮询周期让 (1.0, b.wav) 先于终态被推送
        processes[0].finish()

    threading.Thread(target=writer, daemon=True).start()

    frames = _frames(_read_all(client, task_id))

    assert ("progress", {"progress": 0.5, "current": "a.wav"}) in frames
    assert ("progress", {"progress": 1.0, "current": "b.wav"}) in frames


def test_terminal_task_stream_emits_single_status(manager, client, tmp_path):
    """订阅时任务已终态：跳过初始 status，只发终态那一条（避免同一状态推两份）。"""
    mgr, _ = manager
    task_id, _, processes = _create_fit(
        manager, tmp_path, lines=["INFO:mi-test:训练轮次：1 [10%]"]
    )
    processes[0].finish()
    wait_until(lambda: mgr.get_task(task_id)["state"] == "success")

    frames = _frames(_read_all(client, task_id))

    status_frames = [frame for frame in frames if frame[0] == "status"]
    assert len(status_frames) == 1  # 已终态：跳过初始 status，只发终态那条
    assert status_frames[0][1]["state"] == "success"
    assert status_frames[0][1]["progress"] == 1.0  # 进度随终态 status 一次性给出
