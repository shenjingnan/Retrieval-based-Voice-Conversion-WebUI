"""数据集 API 测试。全部用例把 paths.DATASETS_DIR monkeypatch 到 tmp_path 隔离：
服务跑在 main 工作区、测试跑在 worktree，不共享真实 datasets/ 目录。"""
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import paths
from server.api.datasets import (
    _audio_duration,
    _check_name,
    _safe_filename,
    _unique_path,
)
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


def _wav_bytes(seconds=1.0, rate=16000):
    """内存中的 wav 字节（上传 body 用）。BytesIO 无文件名，须显式给 format。"""
    buf = io.BytesIO()
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    sf.write(
        buf,
        (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"),
        rate,
        subtype="PCM_16",
        format="WAV",
    )
    return buf.getvalue()


def _upload(client, name, files):
    """POST multipart：files 为 [(filename, bytes)]。"""
    return client.post(
        f"/api/datasets/{name}/files",
        files=[("files", (fname, data)) for fname, data in files],
    )


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
# 上传：落盘 / 重名序号 / skip / 大小软限制
# ---------------------------------------------------------------------------


def test_safe_filename_normalizes_path_and_control_chars():
    """basename 化（/ 与 \\ 都当分隔符）+ C0 控制字符替换为 _；常规字符原样保留。"""
    assert _safe_filename("C:\\Users\\me\\a.wav") == "a.wav"
    assert _safe_filename("/tmp/x/b.wav") == "b.wav"
    assert _safe_filename("a\x01\x02b.wav") == "a__b.wav"
    assert _safe_filename("a\r\nb.wav") == "a__b.wav"
    assert _safe_filename("中文🎉 c.wav") == "中文🎉 c.wav"
    assert _safe_filename("") == ""
    assert _safe_filename("\\\\") == ""  # 纯分隔符 → basename 为空 → 调用方 skip


def test_unique_path_avoids_disk_and_batch_collisions(tmp_path):
    """重名递增 _1/_2；磁盘已有文件（含恰好叫 a_1 的）与同批已分配名都视为占用。"""
    (tmp_path / "a.wav").write_bytes(b"first")
    taken: set = set()
    assert _unique_path(tmp_path, "a.wav", taken).name == "a_1.wav"
    (tmp_path / "a_1.wav").write_bytes(b"preexisting")
    assert _unique_path(tmp_path, "a.wav", taken).name == "a_2.wav"
    assert _unique_path(tmp_path, "a.wav", taken).name == "a_3.wav"
    assert (tmp_path / "a.wav").read_bytes() == b"first"  # 永不覆盖


def test_upload_single_file_creates_dataset(client):
    payload = _wav_bytes()
    resp = _upload(client, "alice", [("a.wav", payload)])

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == "alice"
    assert body["created"] is True
    assert body["added"] == ["a.wav"]
    assert body["skipped"] == []
    assert body["path"] == str(paths.DATASETS_DIR / "alice")
    assert body["file_count"] == 1
    assert body["total_duration"] > 0
    on_disk = paths.DATASETS_DIR / "alice" / "a.wav"
    assert on_disk.read_bytes() == payload  # 字节一致（流式落盘不改动内容）


def test_upload_batch_multiple_files(client):
    resp = _upload(
        client,
        "alice",
        [("b.wav", _wav_bytes()), ("a.wav", _wav_bytes(seconds=2.0))],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert sorted(body["added"]) == ["a.wav", "b.wav"]
    assert body["file_count"] == 2
    assert body["total_duration"] == pytest.approx(3.0, abs=0.2)
    assert (paths.DATASETS_DIR / "alice" / "a.wav").is_file()


def test_upload_duplicate_name_appends_sequence(client):
    first, second = _wav_bytes(), _wav_bytes(seconds=2.0)
    assert _upload(client, "alice", [("a.wav", first)]).status_code == 200

    resp = _upload(client, "alice", [("a.wav", second)])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 200
    assert resp.json()["added"] == ["a_1.wav"]
    assert (dataset / "a.wav").read_bytes() == first  # 原文件字节不变（永不覆盖）
    assert (dataset / "a_1.wav").read_bytes() == second


def test_upload_same_batch_duplicates_take_distinct_names(client):
    resp = _upload(
        client, "alice", [("a.wav", _wav_bytes())] * 3
    )

    assert resp.json()["added"] == ["a.wav", "a_1.wav", "a_2.wav"]
    assert resp.json()["file_count"] == 3


def test_upload_same_stem_different_suffix_coexist(client):
    resp = _upload(client, "alice", [("a.wav", _wav_bytes()), ("a.mp3", _wav_bytes(seconds=2.0))])

    assert sorted(resp.json()["added"]) == ["a.mp3", "a.wav"]


def test_upload_preserves_unicode_and_space_filenames(client):
    resp = _upload(client, "alice", [("中文 歌.wav", _wav_bytes()), ("🎉.wav", _wav_bytes())])

    assert set(resp.json()["added"]) == {"🎉.wav", "中文 歌.wav"}
    assert (paths.DATASETS_DIR / "alice" / "中文 歌.wav").is_file()


def test_upload_non_audio_skipped_with_reason(client):
    resp = _upload(client, "alice", [("a.wav", _wav_bytes()), ("notes.txt", b"hello")])

    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == ["a.wav"]
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["name"] == "notes.txt"
    assert ".txt" in body["skipped"][0]["reason"]  # reason 注明格式
    assert not (paths.DATASETS_DIR / "alice" / "notes.txt").exists()


def test_upload_empty_file_skipped(client):
    resp = _upload(client, "alice", [("a.wav", b""), ("b.wav", _wav_bytes())])

    body = resp.json()
    assert body["added"] == ["b.wav"]
    assert [s["name"] for s in body["skipped"]] == ["a.wav"]
    assert not (paths.DATASETS_DIR / "alice" / "a.wav").exists()  # 空文件不落盘


def test_upload_all_skipped_on_new_dataset_returns_400_without_dir(client):
    resp = _upload(client, "fresh", [("notes.txt", b"x"), ("junk.exe", b"MZ")])

    assert resp.status_code == 400
    assert not (paths.DATASETS_DIR / "fresh").exists()  # 不留空目录
    assert client.get("/api/datasets").json() == []


def test_upload_all_skipped_on_existing_dataset_keeps_dir(client):
    """对照：目录本就存在时全 skip 不是错误（追加零个文件），200 如实上报。"""
    (paths.DATASETS_DIR / "alice").mkdir(parents=True)
    resp = _upload(client, "alice", [("notes.txt", b"x")])

    assert resp.status_code == 200
    assert resp.json()["added"] == []
    assert resp.json()["file_count"] == 0


def test_upload_missing_files_field_returns_422(client):
    resp = client.post("/api/datasets/alice/files", data={})

    assert resp.status_code == 422  # FastAPI 对必填 File 字段缺省的默认行为


def test_upload_invalid_dataset_name_returns_400(client):
    resp = client.post(
        "/api/datasets/%2E%2E/files", files=[("files", ("a.wav", _wav_bytes()))]
    )

    assert resp.status_code == 400


def test_upload_oversize_file_aborts_with_413_no_partial(client, monkeypatch):
    from server.api import datasets as datasets_api

    monkeypatch.setattr(datasets_api, "MAX_FILE_BYTES", 16)
    resp = _upload(client, "alice", [("a.wav", _wav_bytes())])

    assert resp.status_code == 413
    dataset = paths.DATASETS_DIR / "alice"
    assert not (dataset / "a.wav").exists()  # 半成品已删
    # 目录是本次新建且零落成 → 一并清掉（与全 skip 400 的「不留空目录」同一纪律）
    assert not dataset.exists()
    assert client.get("/api/datasets").json() == []


def test_upload_batch_limit_keeps_completed_files(client, monkeypatch):
    """累计超限：已完成的文件保留计入 added（不回滚），中止的文件不留半成品。"""
    from server.api import datasets as datasets_api

    a, b, c = _wav_bytes(), _wav_bytes(), _wav_bytes()
    monkeypatch.setattr(datasets_api, "MAX_BATCH_BYTES", len(a) + 100)

    resp = _upload(client, "alice", [("a.wav", a), ("b.wav", b), ("c.wav", c)])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 413
    assert (dataset / "a.wav").read_bytes() == a  # 已落盘的完整保留
    assert not (dataset / "b.wav").exists()  # 写盘中止的半成品已删
    assert not (dataset / "c.wav").exists()  # 中止后不再处理后续文件


def test_upload_large_file_spools_to_disk(client):
    """>1MB 的文件走 SpooledTemporaryFile 滚盘路径（不整体进内存），内容仍逐字节一致。"""
    payload = _wav_bytes(seconds=60.0)
    assert len(payload) > 1024 * 1024

    resp = _upload(client, "alice", [("long.wav", payload)])

    assert resp.status_code == 200
    assert (paths.DATASETS_DIR / "alice" / "long.wav").read_bytes() == payload
    assert resp.json()["total_duration"] == pytest.approx(60.0, abs=1.0)


def test_upload_append_existing_dataset_counts_accumulate(client):
    assert _upload(client, "alice", [("a.wav", _wav_bytes())]).status_code == 200

    resp = _upload(client, "alice", [("b.wav", _wav_bytes(seconds=2.0))])
    body = resp.json()

    assert body["created"] is False
    assert body["file_count"] == 2  # 追加后是数据集总数
    assert body["total_duration"] == pytest.approx(3.0, abs=0.2)


def test_upload_writes_sidecar_with_duration(client):
    resp = _upload(client, "alice", [("a.wav", _wav_bytes(seconds=2.0))])
    assert resp.status_code == 200

    sidecar = json.loads(
        (paths.DATASETS_DIR / ".meta" / "alice.json").read_text(encoding="utf8")
    )
    on_disk = paths.DATASETS_DIR / "alice" / "a.wav"
    assert sidecar["a.wav"]["duration"] == pytest.approx(2.0, abs=0.1)
    assert sidecar["a.wav"]["size"] == on_disk.stat().st_size
    assert sidecar["a.wav"]["mtime_ns"] == on_disk.stat().st_mtime_ns


def test_upload_io_failure_returns_500_with_progress(client, monkeypatch):
    """落盘 IO 失败 → 500，detail 注明已成功写入的数量；已落盘文件保留。"""
    real_open = Path.open

    def flaky_open(self, mode="r", *args, **kwargs):
        if self.name == "bad.wav" and "w" in mode:
            raise OSError("disk full")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    good = _wav_bytes()

    resp = _upload(client, "alice", [("good.wav", good), ("bad.wav", _wav_bytes())])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 500
    assert "1" in resp.json()["detail"]  # 已成功 N 个
    assert (dataset / "good.wav").read_bytes() == good
    assert not (dataset / "bad.wav").exists()  # 失败的半成品不留


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
