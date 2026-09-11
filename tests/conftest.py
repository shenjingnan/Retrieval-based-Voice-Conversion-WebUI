import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def _isolated_runtime_env(monkeypatch, tmp_path):
    """隔离 infer 代码依赖的运行时环境变量，测试不读写真实 assets/ 与 logs/。"""
    monkeypatch.setenv("weight_root", str(tmp_path / "weights"))
    monkeypatch.setenv("index_root", str(tmp_path / "indices"))
    # 队列持久化单例与历史记录器都指向真实仓库 logs/：测试一律断开（需要持久化
    # 行为的用例自建 TaskManager + TaskStore/TaskHistoryStore(tmp)），恢复入口与
    # 记录器见到 None / 空模块态即跳过
    from server import tasks as server_tasks
    from server.api import training as server_training

    monkeypatch.setattr(server_tasks.task_manager, "queue_store", None, raising=False)
    monkeypatch.setattr(server_tasks.task_manager, "on_finish", None, raising=False)
    monkeypatch.setattr(server_tasks.task_manager, "on_start", None, raising=False)
    # 提交内存守卫读真实系统额度——测试结果不得随宿主机当时的内存状态波动，
    # 一律给健康值；守卫自身的用例在测试内再覆盖为不足值
    monkeypatch.setattr(
        server_tasks, "_available_commit_bytes", lambda: 8 * 1024**3, raising=False
    )
    monkeypatch.setattr(server_training, "_history_store", None, raising=False)
    monkeypatch.setattr(server_training, "_history_cache", None, raising=False)
    monkeypatch.setattr(server_training, "_recorded_ids", set(), raising=False)
