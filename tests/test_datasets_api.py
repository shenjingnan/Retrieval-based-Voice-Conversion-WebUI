"""数据集 API 测试。全部用例把 paths.DATASETS_DIR monkeypatch 到 tmp_path 隔离：
服务跑在 main 工作区、测试跑在 worktree，不共享真实 datasets/ 目录。"""
import pytest
from fastapi.testclient import TestClient

from server import paths
from server.main import create_app


@pytest.fixture(autouse=True)
def _datasets_root(monkeypatch, tmp_path):
    # 属性访问（paths.DATASETS_DIR）而非 from-import，保证 patch 对模块内所有读取生效
    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "datasets")


@pytest.fixture
def client():
    return TestClient(create_app())


def test_list_datasets_missing_root_returns_empty(client):
    """DATASETS_DIR 尚未创建（用户还没上传过任何数据集）→ 空列表而非 500。"""
    assert client.get("/api/datasets").status_code == 200
    assert client.get("/api/datasets").json() == []
