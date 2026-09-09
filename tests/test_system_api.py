"""系统资源 API 测试：端点用例全部 monkeypatch 采集 helper（不真调 nvidia-smi /
psutil / disk_usage，CI 无 GPU、无 psutil 也能全绿）；nvidia-smi 输出解析是纯函数，
直接喂字符串覆盖多卡 / WDDM [N/A] / 名称含逗号等形态。

替身 fixture 不用 autouse：_read_memory/_read_cpu 的单元用例要调真函数，
autouse 替身会把模块属性覆盖掉。"""
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server.api import system as system_api
from server.main import create_app


@pytest.fixture
def stub_collectors(monkeypatch):
    """采集项替身：即便开发机真有 N 卡也要出确定性断言。"""
    monkeypatch.setattr(system_api, "_read_gpus", lambda: [])
    monkeypatch.setattr(system_api, "_read_cpu", lambda: {"percent": 1.5, "count": 8})
    monkeypatch.setattr(system_api, "_read_memory", lambda: {"used_bytes": 1, "total_bytes": 2})
    monkeypatch.setattr(
        system_api, "_read_disk", lambda: {"used_bytes": 1, "free_bytes": 2, "total_bytes": 3}
    )


# ---------------------------------------------------------------------------
# 端点：GET /api/system/stats
# ---------------------------------------------------------------------------


def test_stats_shape_and_passthrough(stub_collectors):
    resp = TestClient(create_app()).get("/api/system/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"timestamp", "cpu", "memory", "disk", "gpus"}
    assert isinstance(body["timestamp"], float) and body["timestamp"] > 0
    assert body["cpu"] == {"percent": 1.5, "count": 8}
    assert body["memory"] == {"used_bytes": 1, "total_bytes": 2}
    assert body["disk"] == {"used_bytes": 1, "free_bytes": 2, "total_bytes": 3}
    # 无卡是空数组而不是 null/缺键：与「采集项不可用 = null」是两种刻意的降级语义
    assert body["gpus"] == []


def test_gpu_entry_passthrough(stub_collectors, monkeypatch):
    gpu = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "utilization_percent": 12,
        "memory_used_bytes": 2147483648,
        "memory_total_bytes": 24564 * 1024 * 1024,
    }
    monkeypatch.setattr(system_api, "_read_gpus", lambda: [gpu])

    resp = TestClient(create_app()).get("/api/system/stats")

    assert resp.status_code == 200
    assert resp.json()["gpus"] == [gpu]


def test_all_collectors_degraded_still_200(monkeypatch):
    """永不 500 的契约锚点：所有采集项都失败时端点仍 200，各项落到各自的降级值。"""
    monkeypatch.setattr(system_api, "_read_gpus", lambda: [])
    monkeypatch.setattr(system_api, "_read_cpu", lambda: None)
    monkeypatch.setattr(system_api, "_read_memory", lambda: None)
    monkeypatch.setattr(system_api, "_read_disk", lambda: None)

    resp = TestClient(create_app()).get("/api/system/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cpu"] is None
    assert body["memory"] is None
    assert body["disk"] is None
    assert body["gpus"] == []


def test_single_collector_degraded_keeps_others(stub_collectors, monkeypatch):
    monkeypatch.setattr(system_api, "_read_disk", lambda: None)

    resp = TestClient(create_app()).get("/api/system/stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["disk"] is None
    assert body["cpu"] == {"percent": 1.5, "count": 8}
    assert body["memory"] == {"used_bytes": 1, "total_bytes": 2}


# ---------------------------------------------------------------------------
# nvidia-smi 输出解析（纯函数）
# ---------------------------------------------------------------------------


def test_parse_nvidia_smi_multi_gpu():
    text = (
        "0, NVIDIA GeForce RTX 4090, 12, 2048, 24564\n"
        "1, NVIDIA GeForce RTX 3060, 0, 1024, 12288\n"
    )

    gpus = system_api._parse_nvidia_smi(text)

    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "utilization_percent": 12.0,
        "memory_used_bytes": 2048 * 1024 * 1024,
        "memory_total_bytes": 24564 * 1024 * 1024,
    }
    assert gpus[1]["memory_total_bytes"] == 12288 * 1024 * 1024


def test_parse_nvidia_smi_wddm_na_fields():
    """WDDM 驱动常见「util 是 [N/A]/[Not Supported] 但显存正常」：逐字段降级，
    整卡必须保留（丢弃整卡会损失显存信息）。"""
    text = "0, NVIDIA GeForce RTX 4060 Laptop GPU, [Not Supported], 2048, 8188\n"

    gpus = system_api._parse_nvidia_smi(text)

    assert len(gpus) == 1
    assert gpus[0]["utilization_percent"] is None
    assert gpus[0]["memory_used_bytes"] == 2048 * 1024 * 1024
    assert gpus[0]["memory_total_bytes"] == 8188 * 1024 * 1024
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"


def test_parse_nvidia_smi_name_with_comma():
    """名称含逗号（理论风险）：从两端取字段，中间全归 name，数值不错位。"""
    text = "0, GPU A, Model X, 12, 2048, 24564\n"

    gpus = system_api._parse_nvidia_smi(text)

    assert gpus[0]["name"] == "GPU A, Model X"
    assert gpus[0]["utilization_percent"] == 12.0
    assert gpus[0]["memory_used_bytes"] == 2048 * 1024 * 1024


def test_parse_nvidia_smi_garbage():
    assert system_api._parse_nvidia_smi("") == []
    assert system_api._parse_nvidia_smi("hello world\n") == []
    assert system_api._parse_nvidia_smi("0, 1, 2\n") == []  # 字段数不足
    assert system_api._parse_nvidia_smi("x, NVIDIA GPU, 1, 2, 3\n") == []  # index 非数


# ---------------------------------------------------------------------------
# _read_gpus：子进程失败路径全部降级为 []，绝不抛出
# ---------------------------------------------------------------------------


def test_read_gpus_skips_subprocess_when_missing(monkeypatch):
    monkeypatch.setattr(system_api.shutil, "which", lambda name: None)

    def _boom(*args, **kwargs):
        raise AssertionError("nvidia-smi 不在 PATH 时不应启动子进程")

    monkeypatch.setattr(system_api.subprocess, "run", _boom)

    assert system_api._read_gpus() == []


def test_read_gpus_timeout_returns_empty(monkeypatch):
    monkeypatch.setattr(system_api.shutil, "which", lambda name: "/fake/nvidia-smi")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.0)

    monkeypatch.setattr(system_api.subprocess, "run", _timeout)

    assert system_api._read_gpus() == []


def test_read_gpus_nonzero_exit_returns_empty(monkeypatch):
    monkeypatch.setattr(system_api.shutil, "which", lambda name: "/fake/nvidia-smi")
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=1, stdout=b"", stderr=b"boom"
    )
    monkeypatch.setattr(system_api.subprocess, "run", lambda *a, **k: completed)

    assert system_api._read_gpus() == []


# ---------------------------------------------------------------------------
# psutil / 磁盘采集语义
# ---------------------------------------------------------------------------


def test_read_memory_uses_total_minus_available(monkeypatch):
    """锁住「不用 vm.used」的语义：Linux 上 free 被 page cache 吃掉会虚高，
    必须按 total - available 计算（与任务管理器同口径）。"""
    monkeypatch.setattr(
        system_api,
        "_psutil",
        lambda: SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=100, available=40, free=1)),
    )

    assert system_api._read_memory() == {"used_bytes": 60, "total_bytes": 100}


def test_psutil_missing_degrades_cpu_and_memory(monkeypatch):
    """psutil 缺失（部署环境没重装依赖）只降级这两项，不得抛出：patch 模块级
    缓存为「已尝试且失败」态，不依赖 CI 是否装了 psutil。"""
    monkeypatch.setattr(system_api, "_psutil_module", None)
    monkeypatch.setattr(system_api, "_psutil_checked", True)

    assert system_api._read_memory() is None
    assert system_api._read_cpu() is None


def test_read_disk_reports_repo_volume():
    """真实调用（ROOT 恒存在）：三项齐全且 used+free==total 自洽。"""
    stats = system_api._read_disk()

    assert stats is not None
    assert stats["used_bytes"] + stats["free_bytes"] == stats["total_bytes"]
    assert stats["total_bytes"] > 0
