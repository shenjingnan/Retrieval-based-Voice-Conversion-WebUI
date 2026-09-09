"""训练历史持久化测试（design docs/plans/2026-09-09-task-centric-training-ui-design.md）。

覆盖三层：TaskHistoryStore 磁盘读写（追加/替换/cap/损坏兜底）、训练记录器
（字段/separate 过滤/幂等/logs_tail）、API 合并查询（列表去重内存优先、详情历史
回退）。conftest 已断开真实单例的 queue_store 并重置历史模块状态，这里自建实例。
"""
import json

import pytest
from fastapi.testclient import TestClient

from server import paths
from server import tasks as server_tasks
from server.api import training
from server.main import create_app
from server.task_store import TaskHistoryStore, TaskStore
from server.tasks import TaskManager


class _FakeProcess:
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
def isolated_env(monkeypatch, tmp_path):
    def _fake_kill(process, process_name="", task_logger=None):
        process.finish()  # 模拟 SIGTERM 生效（假 pid 不能真发信号）
        return True

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(server_tasks, "kill_process_tree", _fake_kill)
    # 训练模块的历史状态逐用例重置（惰性 store / 缓存 / 幂等集合）；
    # raising=False：仅测 Store 层的用例不要求记录器已实现
    monkeypatch.setattr(training, "_history_store", None, raising=False)
    monkeypatch.setattr(training, "_history_cache", None, raising=False)
    monkeypatch.setattr(training, "_recorded_ids", set(), raising=False)


# ---------------------------------------------------------------------------
# TaskHistoryStore：磁盘读写
# ---------------------------------------------------------------------------


def _record(rid, state="success", **overrides):
    rec = {
        "id": rid,
        "name": "pipeline",
        "kind": "pipeline",
        "exp_name": "voice-%s" % rid,
        "state": state,
        "error": None,
        "created_at": 1.0,
        "finished_at": 2.0,
        "params": {"exp_name": "voice-%s" % rid},
        "logs_tail": ["line-1"],
    }
    rec.update(overrides)
    return rec


def test_history_store_round_trip_and_append_order(tmp_path):
    store = TaskHistoryStore(tmp_path / "q.json")
    store.record(_record("a"))
    store.record(_record("b"))
    assert [r["id"] for r in store.load()] == ["a", "b"]  # 追加序


def test_history_store_same_id_replaces_in_place(tmp_path):
    store = TaskHistoryStore(tmp_path / "q.json")
    store.record(_record("a", state="running"))
    store.record(_record("b"))
    store.record(_record("a", state="success"))
    assert [r["id"] for r in store.load()] == ["a", "b"]  # 原位替换，不改变顺序
    assert store.load()[0]["state"] == "success"


def test_history_store_cap_keeps_latest(tmp_path):
    store = TaskHistoryStore(tmp_path / "q.json", cap=3)
    for rid in ("a", "b", "c", "d"):
        store.record(_record(rid))
    assert [r["id"] for r in store.load()] == ["b", "c", "d"]  # 砍头部，保最新


def test_history_store_missing_file_returns_empty(tmp_path):
    assert TaskHistoryStore(tmp_path / "q.json").load() == []


def test_history_store_corrupt_file_backed_up_and_emptied(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{broken", encoding="utf8")

    assert TaskHistoryStore(path).load() == []

    backups = list(tmp_path.glob("q.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf8") == "{broken"


def test_history_store_non_list_payload_returns_empty(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('{"id": 1}', encoding="utf8")
    assert TaskHistoryStore(path).load() == []


def test_history_store_drops_non_dict_or_idless_entries(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([_record("ok"), "junk", {"name": "no-id"}]), encoding="utf8")

    assert [r["id"] for r in TaskHistoryStore(path).load()] == ["ok"]


def test_history_store_no_tmp_leftover(tmp_path):
    store = TaskHistoryStore(tmp_path / "q.json")
    store.record(_record("a"))
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# 记录器：training._record_history
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    """真 TaskManager（假 Popen，进程保持运行）+ tmp 历史文件 + 训练模块记录器。"""
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProcess(delay=10**6)

    manager = TaskManager(poll_interval=0.01, popen=fake_popen)
    manager.queue_store = TaskStore(tmp_path / "logs" / "task_queue.json")
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    manager.on_finish = training._record_history
    yield manager, calls, tmp_path
    manager.dispose()


def _pipeline_definition(exp="voice-x"):
    return {
        "kind": "pipeline",
        "body": {"exp_name": exp, "sr": "40k", "total_epoch": 5},
        "batch_note": None,
        "pipeline_stages": ["preprocess", "preprocess", "extract", "extract", "fit", "fit", "index"],
    }


def test_record_history_writes_complete_record(trainer, tmp_path):
    manager, calls, tmp_path = trainer
    from tests.test_training_api import _fabricate_features

    _fabricate_features(paths.LOGS_DIR / "voice-x")
    manager.create_task(
        "pipeline",
        ["cmd-fit"],
        paths.LOGS_DIR / "voice-x" / "pipeline_task.log",
        definition=_pipeline_definition(),
    )
    task_id = manager.list_tasks()[0]["id"]
    training._TASK_META[task_id] = {
        "exp_name": "voice-x",
        "total_epoch": 5,
        "pipeline_stages": ["preprocess"],
    }
    wait_until(lambda: manager.get_task(task_id)["state"] == "running")
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")

    (record,) = training._get_history_store().load()
    assert record["id"] == task_id
    assert record["exp_name"] == "voice-x"
    assert record["state"] == "cancelled"
    assert record["kind"] == "pipeline"
    assert record["params"] == {"exp_name": "voice-x", "sr": "40k", "total_epoch": 5}
    assert record["pipeline_stages"] == ["preprocess"]
    assert record["cmds"] == ["cmd-fit"]  # 详情解析失败阶段需要命令串
    assert record["progress"] == 0.0  # 取消时无进度锚点，按阶段计算为 0
    assert record["created_at"] > 0 and record["finished_at"] is not None
    assert isinstance(record["logs_tail"], list)


def test_record_history_logs_tail_capped(trainer):
    manager, _, _ = trainer
    manager.create_task(
        "fit",
        ["cmd"],
        paths.LOGS_DIR / "e" / "fit.log",
        definition={"kind": "fit", "body": {"exp_name": "e"}},
    )
    task_id = manager.list_tasks()[0]["id"]
    training._TASK_META[task_id] = {"exp_name": "e"}
    with open(paths.LOGS_DIR / "e" / "fit.log", "a", encoding="utf8") as handle:
        for i in range(400):
            handle.write("line%d\n" % i)
    wait_until(lambda: len(manager.get_task(task_id)["logs"]) >= 350)
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")

    (record,) = training._get_history_store().load()
    assert len(record["logs_tail"]) == training.HISTORY_LOG_TAIL_LINES
    assert record["logs_tail"][-1] == "line399"


def test_record_history_skips_tasks_without_training_meta(trainer):
    """separate 任务没有 _TASK_META 条目：不进训练历史。"""
    manager, _, _ = trainer
    manager.create_task("separate", ["cmd"], paths.LOGS_DIR / "x.separate.log")
    task_id = manager.list_tasks()[0]["id"]
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")

    assert training._get_history_store().load() == []


def test_record_history_idempotent_per_task(trainer):
    manager, _, _ = trainer
    manager.create_task(
        "fit", ["cmd"], paths.LOGS_DIR / "e" / "fit.log",
        definition={"kind": "fit", "body": {"exp_name": "e"}},
    )
    task_id = manager.list_tasks()[0]["id"]
    training._TASK_META[task_id] = {"exp_name": "e"}
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
    training._record_history(manager.get_task(task_id))  # 重复投递

    assert len(training._get_history_store().load()) == 1


def test_record_history_ignores_non_terminal_snapshot(trainer):
    manager, _, _ = trainer
    manager.create_task(
        "fit", ["cmd"], paths.LOGS_DIR / "e" / "fit.log",
        definition={"kind": "fit", "body": {"exp_name": "e"}},
    )
    training._TASK_META[manager.list_tasks()[0]["id"]] = {"exp_name": "e"}
    snapshot = manager.get_task(manager.list_tasks()[0]["id"])
    training._record_history(snapshot)  # running 快照不记录

    assert training._get_history_store().load() == []
    manager.cancel(manager.list_tasks()[0]["id"])


# ---------------------------------------------------------------------------
# API：列表合并 + 详情历史回退
# ---------------------------------------------------------------------------


@pytest.fixture
def history_client(monkeypatch, tmp_path):
    """TestClient + 真单例（queue_store/on_finish 已按需接线）；历史模块态重置。"""
    from tests.test_training_api import StubTaskManager

    store = TaskHistoryStore(tmp_path / "logs" / "task_history.json")
    monkeypatch.setattr(training, "_history_store", store)
    manager = StubTaskManager()
    monkeypatch.setattr(training, "task_manager", manager)
    monkeypatch.setattr(training, "_TASK_META", {})
    client = TestClient(create_app())
    return client, manager, store


def test_tasks_list_merges_history_with_memory_priority(history_client):
    client, manager, store = history_client
    store.record(_record("old-hist", state="failed", error="boom"))
    manager.snapshots["old-hist"] = _stub_snapshot("old-hist", state="running")  # 内存优先
    manager.snapshots["live"] = _stub_snapshot("live", state="pending", queue_position=1)
    training._history_cache = store.load()

    rows = {row["id"]: row for row in client.get("/api/tasks").json()}

    assert rows["old-hist"]["state"] == "running"  # 内存覆盖历史
    assert rows["old-hist"]["history"] is False
    assert rows["live"]["history"] is False
    assert rows["live"]["queue_position"] == 1


def test_tasks_list_history_rows_shape(history_client):
    client, manager, store = history_client
    store.record(_record("h1", state="failed", error="boom", logs_tail=["x", "y"]))
    training._history_cache = store.load()

    (row,) = client.get("/api/tasks").json()

    assert row["id"] == "h1"
    assert row["history"] is True
    assert row["queue_position"] is None
    assert row["exp_name"] == "voice-h1"
    assert row["error"] == "boom"
    assert "logs_tail" not in row  # 列表投影不带日志，详情才带


def test_get_task_falls_back_to_history_with_logs_tail(history_client):
    client, manager, store = history_client
    store.record(_record("h1", logs_tail=["a", "b"]))
    training._history_cache = store.load()

    body = client.get("/api/tasks/h1").json()

    assert body["history"] is True
    assert body["logs_tail"] == ["a", "b"]


def test_get_task_unknown_still_404(history_client):
    client, _, _ = history_client
    assert client.get("/api/tasks/no-such").status_code == 404


def test_memory_terminal_task_not_duplicated_with_history(history_client):
    """本会话内已终态的内存任务：若记录器已把它写进历史（同 id），列表只出现一次。"""
    client, manager, store = history_client
    manager.snapshots["t9"] = _stub_snapshot("t9", state="success")
    store.record(_record("t9", state="success"))
    training._history_cache = store.load()

    rows = client.get("/api/tasks").json()

    assert [r["id"] for r in rows].count("t9") == 1


def _stub_snapshot(task_id, state="running", queue_position=None):
    return {
        "id": task_id,
        "name": "pipeline",
        "state": state,
        "progress": None,
        "error": None,
        "cmds": ["cmd"],
        "current_cmd": None,
        "queue_position": queue_position,
        "log_path": "log",
        "truncate": True,
        "created_at": 1.0,
        "started_at": 2.0,
        "finished_at": None,
        "log_seq": 0,
        "logs": [],
    }


# ---------------------------------------------------------------------------
# 端到端：任务完成 → 历史落盘 → 模拟重启 → 仍可见
# ---------------------------------------------------------------------------


def test_end_to_end_history_survives_restart(trainer, monkeypatch):
    manager, calls, tmp_path = trainer

    def fake_popen_factory():
        def fake_popen(cmd, **kwargs):
            calls.append(cmd)
            return _FakeProcess(delay=10**6)
        return fake_popen

    from tests.test_training_api import _fabricate_features

    _fabricate_features(paths.LOGS_DIR / "voice-r")
    manager.create_task(
        "pipeline",
        ["cmd-fit"],
        paths.LOGS_DIR / "voice-r" / "pipeline_task.log",
        definition=_pipeline_definition("voice-r"),
    )
    task_id = manager.list_tasks()[0]["id"]
    training._TASK_META[task_id] = {"exp_name": "voice-r", "total_epoch": 5}
    wait_until(lambda: len(calls) == 1)
    manager.cancel(task_id)
    wait_until(lambda: manager.get_task(task_id)["state"] == "cancelled")
    assert len(training._get_history_store().load()) == 1

    # 模拟重启：内存任务表清空、镜像与幂等集合重置，store 从磁盘惰性解析
    monkeypatch.setattr(training, "_history_cache", None)
    monkeypatch.setattr(training, "_recorded_ids", set())
    empty_manager = TaskManager(poll_interval=0.01, popen=fake_popen_factory())
    monkeypatch.setattr(training, "task_manager", empty_manager)
    try:
        client = TestClient(create_app())
        rows = client.get("/api/tasks").json()
        assert len(rows) == 1 and rows[0]["history"] is True and rows[0]["exp_name"] == "voice-r"

        detail = client.get("/api/tasks/%s" % task_id).json()
        assert detail["kind"] == "pipeline"
        assert isinstance(detail["logs_tail"], list)
    finally:
        empty_manager.dispose()
