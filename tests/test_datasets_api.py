"""数据集 API 测试。全部用例把 paths.DATASETS_DIR monkeypatch 到 tmp_path 隔离：
服务跑在 main 工作区、测试跑在 worktree，不共享真实 datasets/ 目录。"""
import json
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import paths
from server.api.datasets import _audio_duration, _check_name
from server.main import create_app


@pytest.fixture(autouse=True)
def _datasets_root(monkeypatch, tmp_path):
    # 属性访问（paths.DATASETS_DIR）而非 from-import，保证 patch 对模块内所有读取生效
    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "datasets")


@pytest.fixture
def client():
    return TestClient(create_app())


def _write_wav(path, seconds=1.0, rate=16000):
    """现场生成 wav（1s 正弦），不引入二进制 fixture。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), rate, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# 骨架 / 路由注册
# ---------------------------------------------------------------------------


def test_list_datasets_missing_root_returns_empty(client):
    """DATASETS_DIR 尚未创建（用户还没上传过任何数据集）→ 空列表而非 500。"""
    assert client.get("/api/datasets").status_code == 200
    assert client.get("/api/datasets").json() == []


def test_routes_registered_in_openapi():
    schema = create_app().openapi()
    assert "/api/datasets" in schema["paths"]
    assert "/api/datasets/{name}" in schema["paths"]


# ---------------------------------------------------------------------------
# _check_name：与 training._check_exp_name 同源的非法名矩阵
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../x", ".", "..", "a b", "a$b", 'a"b', ".meta"])
def test_check_name_rejects_forbidden(bad):
    """防穿越（../x、.、..）+ 防 shell 注入（空格、$、引号）+ 拒前导点（.meta 伪装）。"""
    with pytest.raises(HTTPException) as excinfo:
        _check_name(bad)
    assert excinfo.value.status_code == 400


def test_check_name_accepts_plain_names():
    assert _check_name("alice_v2") == "alice_v2"
    assert _check_name("数据集🎉") == "数据集🎉"


@pytest.mark.parametrize(
    "encoded,bad",
    [
        ("%2E", "."),
        ("%2E%2E", ".."),
        ("a%20b", "a b"),
        ("a%24b", "a$b"),
        ("a%22b", 'a"b'),
        (".meta", ".meta"),
    ],
)
def test_detail_invalid_name_returns_400(client, encoded, bad):
    # 字面 ../x 会被 httpx 规范化、%2F 解码后也到不了单段路由（路由层兜住 → 404），
    # 其防穿越逻辑由上面的 _check_name 单元用例覆盖
    resp = client.get("/api/datasets/" + encoded)
    assert resp.status_code == 400
    assert bad in resp.json()["detail"]


def test_detail_allows_cjk_and_emoji_names(client, tmp_path):
    dataset = paths.DATASETS_DIR / "数据集🎉"
    dataset.mkdir(parents=True)
    resp = client.get("/api/datasets/数据集🎉")
    assert resp.status_code == 200
    assert resp.json()["name"] == "数据集🎉"


def test_detail_missing_returns_404(client):
    resp = client.get("/api/datasets/ghost")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 列表
# ---------------------------------------------------------------------------


def test_list_datasets_sorted_with_counts(client):
    root = paths.DATASETS_DIR
    alice = root / "alice"
    _write_wav(alice / "b.wav")
    _write_wav(alice / "a.wav", seconds=2.0)
    (alice / "note.txt").write_text("x", encoding="utf8")
    bob = root / "bob"
    _write_wav(bob / "c.wav")
    (root / "stray.txt").write_text("根目录散文件", encoding="utf8")

    data = client.get("/api/datasets").json()

    assert [d["name"] for d in data] == ["alice", "bob"]  # 只列子目录，按名排序
    assert data[0]["path"] == str(alice)
    assert data[0]["file_count"] == 2
    assert data[0]["other_count"] == 1
    assert data[0]["total_bytes"] == sum(
        p.stat().st_size for p in (alice / "a.wav", alice / "b.wav", alice / "note.txt")
    )
    assert data[0]["total_duration"] == pytest.approx(3.0, abs=0.2)
    assert data[1]["file_count"] == 1
    assert data[1]["other_count"] == 0
    assert data[1]["total_duration"] == pytest.approx(1.0, abs=0.1)


def test_list_skips_meta_hidden_dirs_and_files(client):
    root = paths.DATASETS_DIR
    (root / ".meta").mkdir(parents=True)
    (root / ".meta" / "alice.json").write_text("{}", encoding="utf8")
    (root / ".hidden").mkdir()
    (root / "notadir.txt").write_text("x", encoding="utf8")
    assert client.get("/api/datasets").json() == []


def test_total_duration_aggregates_rest_when_one_probe_fails(client):
    """单个文件时长探测失败记 None、聚合其余，不拖垮整个数据集的统计。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "good.wav")
    (dataset / "broken.wav").write_bytes(b"")  # 0 字节：两路探测都读不出
    summary = client.get("/api/datasets").json()[0]
    assert summary["total_duration"] == pytest.approx(1.0, abs=0.1)
    files = {f["name"]: f["duration"] for f in client.get("/api/datasets/alice").json()["files"]}
    assert files["broken.wav"] is None
    assert files["good.wav"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# 侧车缓存（DATASETS_DIR/.meta/，数据集目录之外）
# ---------------------------------------------------------------------------


def test_sidecar_lives_outside_dataset_dir(client):
    """preprocess 遍历数据集目录不过滤扩展名，侧车放目录内会被当音频再解一次。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")
    client.get("/api/datasets")
    sidecar = paths.DATASETS_DIR / ".meta" / "alice.json"
    assert sidecar.is_file()
    assert list(p.name for p in dataset.iterdir()) == ["a.wav"]


def test_list_duration_cache_hit_skips_reprobe(client, monkeypatch):
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")
    first = client.get("/api/datasets").json()[0]  # 预热侧车

    from server.api import datasets as datasets_api

    calls = []
    real = datasets_api._audio_duration

    def counting(path):
        calls.append(path.name)
        return real(path)

    monkeypatch.setattr(datasets_api, "_audio_duration", counting)
    second = client.get("/api/datasets").json()[0]

    assert calls == []  # {size, mtime_ns} 未变 → 命中缓存，不再探测
    assert second["total_duration"] == first["total_duration"]
    assert second["total_duration"] == pytest.approx(1.0, abs=0.1)


def test_cache_invalidated_when_file_changed_and_rewritten(client, monkeypatch):
    dataset = paths.DATASETS_DIR / "alice"
    audio = _write_wav(dataset / "a.wav")
    client.get("/api/datasets")  # 预热
    _write_wav(audio, seconds=2.0)  # size/mtime 双变，缓存必须失效
    stat = audio.stat()
    os.utime(audio, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    from server.api import datasets as datasets_api

    calls = []

    def fake_probe(path):
        calls.append(path.name)
        return 2.0

    monkeypatch.setattr(datasets_api, "_audio_duration", fake_probe)
    summary = client.get("/api/datasets").json()[0]
    assert calls == ["a.wav"]  # 文件变了 → 重新探测
    assert summary["total_duration"] == pytest.approx(2.0, abs=0.1)

    sidecar = json.loads((paths.DATASETS_DIR / ".meta" / "alice.json").read_text(encoding="utf8"))
    assert sidecar["a.wav"] == {
        "size": audio.stat().st_size,
        "mtime_ns": audio.stat().st_mtime_ns,
        "duration": 2.0,
    }


def test_corrupt_sidecar_treated_as_empty(client):
    """坏 JSON 侧车当空缓存：重新探测并覆写，而不是 500。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")
    meta = paths.DATASETS_DIR / ".meta"
    meta.mkdir(parents=True)
    (meta / "alice.json").write_text("{not json", encoding="utf8")

    data = client.get("/api/datasets").json()
    assert data[0]["total_duration"] == pytest.approx(1.0, abs=0.1)
    rewritten = json.loads((meta / "alice.json").read_text(encoding="utf8"))
    assert rewritten["a.wav"]["duration"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# 详情
# ---------------------------------------------------------------------------


def test_dataset_detail_files_sorted(client):
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "b.wav")
    _write_wav(dataset / "a.wav", seconds=2.0)
    (dataset / "note.txt").write_text("x", encoding="utf8")

    data = client.get("/api/datasets/alice").json()

    assert data["name"] == "alice"
    assert data["path"] == str(dataset)
    assert data["file_count"] == 2
    assert data["other_count"] == 1
    assert [f["name"] for f in data["files"]] == ["a.wav", "b.wav"]
    by_name = {f["name"]: f for f in data["files"]}
    assert by_name["a.wav"]["duration"] == pytest.approx(2.0, abs=0.1)
    assert by_name["a.wav"]["size"] == (dataset / "a.wav").stat().st_size
    assert by_name["b.wav"]["duration"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# _audio_duration：av 主路 / soundfile 兜底 / 全失败 None
# ---------------------------------------------------------------------------


def test_audio_duration_wav_precision(tmp_path):
    path = _write_wav(tmp_path / "a.wav", seconds=1.0)
    assert _audio_duration(path) == pytest.approx(1.0, abs=0.1)


def test_audio_duration_falls_back_to_soundfile(tmp_path, monkeypatch):
    path = _write_wav(tmp_path / "a.wav", seconds=1.0)

    import av

    def broken_open(*args, **kwargs):
        raise RuntimeError("av unavailable")

    monkeypatch.setattr(av, "open", broken_open)
    assert _audio_duration(path) == pytest.approx(1.0, abs=0.1)


def test_audio_duration_returns_none_when_both_probes_fail(tmp_path, monkeypatch):
    path = _write_wav(tmp_path / "a.wav")

    import av
    import soundfile

    def broken(*args, **kwargs):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(av, "open", broken)
    monkeypatch.setattr(soundfile, "info", broken)
    assert _audio_duration(path) is None  # 绝不抛异常


def test_audio_duration_text_file_returns_none(tmp_path):
    path = tmp_path / "fake.wav"
    path.write_text("这不是音频", encoding="utf-8")
    assert _audio_duration(path) is None
