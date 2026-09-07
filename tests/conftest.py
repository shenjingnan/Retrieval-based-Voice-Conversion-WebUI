import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def _isolated_runtime_env(monkeypatch, tmp_path):
    """隔离 infer 代码依赖的运行时环境变量，测试不读写真实 assets/ 与 logs/。"""
    monkeypatch.setenv("weight_root", str(tmp_path / "weights"))
    monkeypatch.setenv("index_root", str(tmp_path / "indices"))
