"""数据集 API 测试。全部用例把 paths.DATASETS_DIR monkeypatch 到 tmp_path 隔离：
服务跑在 main 工作区、测试跑在 worktree，不共享真实 datasets/ 目录。"""
import errno
import io
import json
import os
import sys
import threading
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
    assert "/api/datasets/{name}/files" in schema["paths"]
    assert "post" in schema["paths"]["/api/datasets/{name}/files"]
    assert "delete" in schema["paths"]["/api/datasets/{name}"]


# ---------------------------------------------------------------------------
# _check_name：与 training._check_exp_name 同源的非法名矩阵
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../x", ".", "..", "a b", "a$b", 'a"b', ".meta", "a\0b"])
def test_check_name_rejects_forbidden(bad):
    """防穿越（../x、.、..）+ 防 shell 注入（空格、$、引号）+ 拒前导点（.meta 伪装）
    + 拒 NUL（漏到 mkdir/unlink 层是 ValueError → 500，必须在 400 拦下）。"""
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


def test_list_tolerates_dataset_deleted_mid_scan(client, monkeypatch):
    """并发删除的竞态（扫描到一半目录消失）：跳过该项，不 500 整次列表。"""
    root = paths.DATASETS_DIR
    _write_wav(root / "alice" / "a.wav")
    _write_wav(root / "bob" / "b.wav")

    from server.api import datasets as datasets_api

    real_scan = datasets_api._scan_dataset

    def flaky_scan(directory):
        if directory.name == "alice":
            raise FileNotFoundError(directory)  # 模拟 listdir/stat 时目录已被并发删掉
        return real_scan(directory)

    monkeypatch.setattr(datasets_api, "_scan_dataset", flaky_scan)
    data = client.get("/api/datasets").json()

    assert [d["name"] for d in data] == ["bob"]


def test_detail_tolerates_dataset_deleted_mid_scan(client, monkeypatch):
    """404 检查之后、扫描过程中被并发删除 → 如实按不存在处理，不 500。"""
    _write_wav(paths.DATASETS_DIR / "alice" / "a.wav")

    from server.api import datasets as datasets_api

    def gone_scan(directory):
        raise FileNotFoundError(directory)

    monkeypatch.setattr(datasets_api, "_scan_dataset", gone_scan)
    assert client.get("/api/datasets/alice").status_code == 404


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


def test_upload_dot_prefixed_filename_skipped(client):
    """点开头文件（AppleDouble 的 ._a.wav 是真实场景）落盘也进不了列表（_iter_audio
    的隐藏文件口径），preprocess 还会把它当音频解一次——直接 skip，不自相矛盾。"""
    resp = _upload(
        client, "alice", [("._hidden.wav", _wav_bytes()), ("a.wav", _wav_bytes())]
    )

    body = resp.json()
    assert body["added"] == ["a.wav"]
    assert [s["name"] for s in body["skipped"]] == ["._hidden.wav"]
    assert not (paths.DATASETS_DIR / "alice" / "._hidden.wav").exists()


def test_upload_rejects_oversize_declared_body(client):
    """Content-Length 预检：body 由框架在 handler 之前整体 spool 进 $TMPDIR，超大声明
    直接 413，不进落盘流程（连目录都不建）。"""
    from server.api.datasets import MAX_BATCH_BYTES

    resp = client.post(
        "/api/datasets/alice/files",
        files=[("files", ("a.wav", _wav_bytes()))],
        headers={"content-length": str(MAX_BATCH_BYTES + 1024 * 1024 + 1)},
    )

    assert resp.status_code == 413
    assert not (paths.DATASETS_DIR / "alice").exists()


def test_upload_race_between_check_and_create_gets_own_file(client, monkeypatch):
    """exists() 检查与真正 open 之间存在窗口（线程池并发上传），落盘用 O_EXCL 独占
    创建把「永不覆盖」交给文件系统裁决：窗口内路径被占 → 换名，绝不覆盖。stub 在
    _unique_path 返回后、独占创建前把目标路径占掉，确定性复现这个窗口。"""
    from server.api import datasets as datasets_api

    dataset = paths.DATASETS_DIR / "alice"
    real_unique = datasets_api._unique_path
    injected = {"done": False}

    def unique_then_rival(directory, filename, taken):
        target = real_unique(directory, filename, taken)
        if not injected["done"]:
            injected["done"] = True
            target.write_bytes(b"rival got here first")  # 竞态对手抢先落位
        return target

    monkeypatch.setattr(datasets_api, "_unique_path", unique_then_rival)
    payload = _wav_bytes(seconds=2.0)

    resp = _upload(client, "alice", [("same.wav", payload)])

    assert resp.status_code == 200
    assert resp.json()["added"] == ["same_1.wav"]  # 换名，不覆盖对手
    assert (dataset / "same.wav").read_bytes() == b"rival got here first"
    assert (dataset / "same_1.wav").read_bytes() == payload


def test_upload_concurrent_same_name_never_overwrites():
    """真并发：两线程各传同名文件，两个文件都在且字节完好（各归各的路径）。"""
    payload_a, payload_b = _wav_bytes(seconds=8.0), _wav_bytes(seconds=16.0)
    barrier = threading.Barrier(2, timeout=60)
    results = {}

    def run(tag, payload):
        with TestClient(create_app()) as per_thread_client:
            barrier.wait()  # 两侧同时进入请求，最大化撞窗概率
            results[tag] = _upload(per_thread_client, "alice", [("same.wav", payload)])

    threads = [
        threading.Thread(target=run, args=("a", payload_a)),
        threading.Thread(target=run, args=("b", payload_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert sorted(r.status_code for r in results.values()) == [200, 200]
    dataset = paths.DATASETS_DIR / "alice"
    names = sorted(p.name for p in dataset.iterdir())
    assert names == ["same.wav", "same_1.wav"]
    sizes = sorted((dataset / name).stat().st_size for name in names)
    assert sizes == sorted([len(payload_a), len(payload_b)])  # 两份字节都完好


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
    """落盘 IO 失败 → 500，detail 注明已成功写入的数量；已落盘文件保留。

    落盘句柄由 O_EXCL 独占创建产生、经 os.fdopen 接管，打桩点在 fdopen（第 2 个文件
    即失败）；fd 先 close 再抛，模拟打开即失败，顺带覆盖 except 里的防 fd 泄漏分支。
    """
    real_fdopen = os.fdopen
    seen = {"w": 0}

    def flaky_fdopen(fd, mode="r", *args, **kwargs):
        if "w" in mode:
            seen["w"] += 1
            if seen["w"] == 2:  # 第 2 个文件（bad.wav）
                os.close(fd)
                raise OSError("disk full")
        return real_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", flaky_fdopen)
    good = _wav_bytes()

    resp = _upload(client, "alice", [("good.wav", good), ("bad.wav", _wav_bytes())])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 500
    assert "已成功写入 1 个文件" in resp.json()["detail"]
    assert (dataset / "good.wav").read_bytes() == good
    assert not (dataset / "bad.wav").exists()  # 失败的半成品不留


def _patch_open_failure(monkeypatch, fail_on, code=errno.ENOSPC):
    """在 O_EXCL 独占创建处注入 open 失败（fail_on 为第几个文件，1 起）。

    只对落进数据集根的路径生效，避免误伤框架自身的临时文件创建。
    """
    real_open = os.open
    seen = {"n": 0}

    def flaky_open(path, flags, *args, **kwargs):
        if str(path).startswith(str(paths.DATASETS_DIR)):
            seen["n"] += 1
            if seen["n"] == fail_on:
                raise OSError(code, os.strerror(code))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", flaky_open)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EACCES])
def test_upload_open_failure_cleans_empty_new_dir(client, monkeypatch, code):
    """os.open 非 EEXIST 失败（ENOSPC/EACCES）→ 与写盘失败同一收尾：500 + 空目录清理。

    独占创建在写盘 try 块之内：EEXIST 之外的 OSError 不允许绕过
    _discard_empty_new_dir 直接 500，否则本次新建的数据集目录会以空目录形态留在列表。
    """
    _patch_open_failure(monkeypatch, fail_on=1, code=code)

    resp = _upload(client, "alice", [("a.wav", _wav_bytes())])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 500
    assert resp.json()["detail"] == "写入 a.wav 失败（已成功写入 0 个文件，已落盘文件保留）"
    assert not dataset.exists()  # 本次新建且零落成 → 不留空目录
    assert client.get("/api/datasets").json() == []


def test_upload_open_failure_keeps_completed_files(client, monkeypatch):
    """第 2 个文件 open 失败：已落盘的第 1 个保留、失败项不留空壳（与写盘失败对齐）。"""
    _patch_open_failure(monkeypatch, fail_on=2)
    good = _wav_bytes()

    resp = _upload(client, "alice", [("good.wav", good), ("bad.wav", _wav_bytes())])
    dataset = paths.DATASETS_DIR / "alice"

    assert resp.status_code == 500
    assert "已成功写入 1 个文件" in resp.json()["detail"]
    assert (dataset / "good.wav").read_bytes() == good
    assert not (dataset / "bad.wav").exists()


# ---------------------------------------------------------------------------
# 删除：尽力删 + 训练任务互斥
# ---------------------------------------------------------------------------


def test_delete_dataset_removes_dir_and_sidecar(client):
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")
    client.get("/api/datasets")  # 顺带生成侧车
    assert (paths.DATASETS_DIR / ".meta" / "alice.json").is_file()

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "failed_files": []}
    assert not dataset.exists()
    assert not (paths.DATASETS_DIR / ".meta" / "alice.json").exists()  # 侧车一并删除


def test_delete_missing_dataset_returns_404(client):
    assert client.delete("/api/datasets/ghost").status_code == 404


@pytest.mark.parametrize("encoded", ["%2E%2E", "%2E"])
def test_delete_traversal_encoded_names_rejected(client, encoded):
    """编码后的 `..` / `.` 到不了目录层（_check_name 400），与 GET 详情同源防线。"""
    assert client.delete("/api/datasets/" + encoded).status_code == 400


def _cmd_referencing(directory) -> str:
    """模拟 preprocess/pipeline 的 cmd：数据集目录以双引号内嵌（护栏按此匹配）。"""
    return '"%s" train/preprocess.py "%s"' % (sys.executable, directory)


def test_delete_blocked_while_task_references_dataset(client, monkeypatch):
    """删除互斥是引用级的：非终态任务的 cmds 引用了该数据集目录才拒删
    （排队中的 preprocess/pipeline 还等着读它）。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    def fake_list_tasks():
        return [
            {
                "id": "t1",
                "name": "pipeline",
                "state": "running",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "alice")],
            }
        ]

    monkeypatch.setattr(datasets_api.task_manager, "list_tasks", fake_list_tasks)

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 409
    assert "pipeline" in resp.json()["detail"]  # 409 文案带占用任务名
    assert dataset.exists()  # 互斥拦截：目录原样保留


def test_delete_blocked_while_queued_task_references_dataset(client, monkeypatch):
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    def fake_list_tasks():
        return [
            {
                "id": "t2",
                "name": "pipeline",
                "state": "pending",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "alice")],
            }
        ]

    monkeypatch.setattr(datasets_api.task_manager, "list_tasks", fake_list_tasks)

    assert client.delete("/api/datasets/alice").status_code == 409


def test_delete_allowed_when_active_task_references_other_dataset(client, monkeypatch):
    """排队开放后的核心场景：任务 A 运行中时准备任务 B 的数据——运行/排队任务
    未引用的数据集可以随意删除。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    def fake_list_tasks():
        return [
            {
                "id": "t1",
                "name": "pipeline",
                "state": "running",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "bob")],  # 另一个数据集
            }
        ]

    monkeypatch.setattr(datasets_api.task_manager, "list_tasks", fake_list_tasks)

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 200
    assert not dataset.exists()


def test_delete_allowed_when_active_task_has_no_dataset_cmd(client, monkeypatch):
    """fit/index 类任务只引用 logs/{exp} 产物，不构成对原始数据集的互斥。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    def fake_list_tasks():
        return [{"id": "t1", "name": "fit", "state": "running", "cmds": ['"py" train/train.py -e e']}]

    monkeypatch.setattr(datasets_api.task_manager, "list_tasks", fake_list_tasks)

    assert client.delete("/api/datasets/alice").status_code == 200


def test_delete_path_prefix_does_not_match_sibling_dataset(client, monkeypatch):
    """alice 与 alice2 是相邻目录：双引号包夹匹配，前缀命中不得误伤兄弟数据集。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    def fake_list_tasks():
        return [
            {
                "id": "t1",
                "name": "pipeline",
                "state": "running",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "alice2")],
            }
        ]

    monkeypatch.setattr(datasets_api.task_manager, "list_tasks", fake_list_tasks)

    assert client.delete("/api/datasets/alice").status_code == 200


def test_delete_allowed_when_tasks_are_terminal(client, monkeypatch):
    """保守判定只拦非终态：历史终态任务（success/failed）不构成互斥。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")

    from server.api import datasets as datasets_api

    monkeypatch.setattr(
        datasets_api.task_manager,
        "list_tasks",
        lambda: [
            {"id": "t1", "name": "old", "state": "success"},
            {"id": "t2", "name": "new", "state": "failed"},
        ],
    )

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 200
    assert not dataset.exists()


def test_delete_partial_failure_reports_failed_files(client, monkeypatch):
    """逐项删除失败：尽力删其余，失败项如实上报（200，不回滚也不 500）。"""
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "locked.wav")
    _write_wav(dataset / "free.wav")

    real_unlink = os.unlink

    def stubborn_unlink(path, *args, **kwargs):
        if Path(path).name == "locked.wav":
            raise OSError("Resource busy")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", stubborn_unlink)

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_files"] == ["locked.wav"]
    assert body["deleted"] is False  # 目录仍有残留，如实标未删净
    assert (dataset / "locked.wav").exists()
    assert not (dataset / "free.wav").exists()  # 其余文件照删


@pytest.mark.skipif(os.name == "nt", reason="POSIX 权限位语义（Windows 的 chmod 只置只读位）")
def test_delete_total_failure_returns_500(client):
    """目录整体无法推进（顶层不可进入）且无逐项失败可报 → 500。

    chmod 000 下 rmtree 的顶层 open 失败走回调且 path 即目录自身——与实现的
    「目录自身失败不进 failed_files，交整体判定兜底」分支对应，无需打桩。
    """
    dataset = paths.DATASETS_DIR / "alice"
    _write_wav(dataset / "a.wav")
    dataset.chmod(0o000)
    try:
        resp = client.delete("/api/datasets/alice")
    finally:
        dataset.chmod(0o700)  # 恢复权限，避免 pytest 清理 tmp_path 时连带失败

    assert resp.status_code == 500
    assert dataset.exists()


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


def _encode_compressed(path, codec, seconds=3.0, rate=44100):
    """用 av 现场编码压缩音频（AAC→m4a / libmp3lame→mp3），不引入二进制 fixture。

    wav 用哪条探测路都绿，暴露不了主路的真实行为——soundfile 读不了 AAC（libsndfile
    不支持），m4a 用例是「PyAV 主路真正生效」的唯一证明；这类用例缺失正是上一版
    frames/average_rate 写法失效却全绿的原因。
    """
    import av

    samples = int(rate * seconds)
    tone = (0.3 * np.sin(2 * np.pi * 440 * (np.arange(samples) / rate))).astype("float32")
    step = 1024 if codec == "aac" else 1152  # AAC/MP3 每帧采样数
    with av.open(str(path), "w") as out:  # 容器格式按后缀推断
        stream = out.add_stream(codec, rate=rate)
        for off in range(0, samples, step):
            chunk = tone[off : off + step]
            frame = av.AudioFrame.from_ndarray(
                np.stack([chunk, chunk]), format="fltp", layout="stereo"
            )
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


def _encoder_available(codec: str) -> bool:
    """PyAV 的非 wheel 构建可能缺 AAC/lame 编码器：缺则跳过对应用例（探测主路本身
    由其余用例与兜底路径覆盖）。收集期求值，av 顶层导入无 torch 依赖，安全。"""
    import av

    try:
        av.codec.Codec(codec, "w")
        return True
    except Exception:  # noqa: BLE001 未知编码器在 av 里以 ValueError/UnknownCodecError 表达
        return False


@pytest.mark.skipif(not _encoder_available("aac"), reason="PyAV 构建无 AAC 编码器")
def test_audio_duration_m4a_aac_from_primary_path(tmp_path):
    """AAC（m4a）：libsndfile 读不了，时长必须由 PyAV 主路给出（实测误差 ~0.02s）。"""
    path = _encode_compressed(tmp_path / "a.m4a", "aac")
    assert _audio_duration(path) == pytest.approx(3.0, abs=0.2)


@pytest.mark.skipif(not _encoder_available("libmp3lame"), reason="PyAV 构建无 lame 编码器")
def test_audio_duration_mp3_compressed(tmp_path):
    path = _encode_compressed(tmp_path / "a.mp3", "libmp3lame")
    assert _audio_duration(path) == pytest.approx(3.0, abs=0.3)


@pytest.mark.skipif(not _encoder_available("libmp3lame"), reason="PyAV 构建无 lame 编码器")
def test_audio_duration_truncated_mp3_never_raises(tmp_path):
    """截断的压缩文件（元数据/帧流不完整）：探测失败走兜底，绝不向调用方抛异常。"""
    full = _encode_compressed(tmp_path / "full.mp3", "libmp3lame")
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(full.read_bytes()[: full.stat().st_size // 3])

    result = _audio_duration(broken)

    assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# 人声分离：端点校验 / 任务创建 / 衍生标记（设计 docs/plans/2026-09-07-…-design.md）
# ---------------------------------------------------------------------------


def _make_dataset(name, files=("a.wav",)):
    for file_name in files:
        _write_wav(paths.DATASETS_DIR / name / file_name)


def test_separate_creates_task_with_default_model(client, monkeypatch):
    """默认模型去伴奏；命令 / 日志 / 衍生标记三处落点全部可核。"""
    _make_dataset("alice")

    from server.api import datasets as datasets_api

    seen = {}

    def fake_create(name, cmds, log_path, truncate=True, setup=None, **kwargs):
        seen.update(name=name, cmds=list(cmds), log_path=Path(log_path))
        return "task-1"

    monkeypatch.setattr(datasets_api.task_manager, "create_task", fake_create)

    resp = client.post("/api/datasets/alice/separate", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-1"
    assert body["output_dataset"] == "alice_vocals"
    assert body["output_path"] == str(paths.DATASETS_DIR / "alice_vocals")
    # 任务命令：runner 入口 + 输入/输出目录 + 默认模型
    assert seen["name"] == "separate"
    cmd = seen["cmds"][0]
    assert "-m tools.vocal_dataset" in cmd
    assert '"%s"' % (paths.DATASETS_DIR / "alice") in cmd
    assert '"%s"' % (paths.DATASETS_DIR / "alice_vocals") in cmd
    assert '--model "去伴奏"' in cmd
    # 任务日志在 .meta（数据集目录之外）
    assert seen["log_path"].parent == paths.DATASETS_DIR / ".meta"
    assert seen["log_path"].name == "alice.separate.log"
    # 建任务即写衍生标记（失败重跑同样有血缘可展示）
    marker = json.loads(
        (paths.DATASETS_DIR / ".meta" / "alice_vocals.derived.json").read_text(encoding="utf8")
    )
    assert marker == {"source": "alice", "model": "去伴奏"}


def test_separate_explicit_model_passthrough(client, monkeypatch):
    _make_dataset("alice")

    from server.api import datasets as datasets_api

    seen = {}
    monkeypatch.setattr(
        datasets_api.task_manager,
        "create_task",
        lambda name, cmds, log_path, truncate=True, setup=None, **kwargs: seen.update(
            cmds=list(cmds)
        ) or "task-1",
    )

    resp = client.post("/api/datasets/alice/separate", json={"model": "去混响"})

    assert resp.status_code == 200
    assert '--model "去混响"' in seen["cmds"][0]
    marker = json.loads(
        (paths.DATASETS_DIR / ".meta" / "alice_vocals.derived.json").read_text(encoding="utf8")
    )
    assert marker["model"] == "去混响"


def test_separate_unknown_model_returns_400(client):
    _make_dataset("alice")

    resp = client.post("/api/datasets/alice/separate", json={"model": "不存在的模型"})

    assert resp.status_code == 400
    assert "去伴奏" in resp.json()["detail"]  # 可选列表进 detail


def test_separate_missing_dataset_returns_404(client):
    assert client.post("/api/datasets/ghost/separate", json={}).status_code == 404


def test_separate_invalid_name_returns_400(client):
    assert client.post("/api/datasets/%2E%2E/separate", json={}).status_code == 400


def test_separate_empty_dataset_returns_400(client):
    (paths.DATASETS_DIR / "hollow").mkdir(parents=True)

    resp = client.post("/api/datasets/hollow/separate", json={})

    assert resp.status_code == 400
    assert "没有可分离的音频文件" in resp.json()["detail"]


def test_separate_queues_while_training_active(client, monkeypatch):
    """与训练共用串行队列：训练任务非终态时提交分离 → 200 且任务排队（不再 409
    拒绝），响应带 queued / queue_position 供前端提示。"""
    _make_dataset("alice")

    from server.api import datasets as datasets_api

    monkeypatch.setattr(
        datasets_api.task_manager,
        "create_task",
        lambda name, cmds, log_path, truncate=True, setup=None, **kwargs: "task-9",
    )
    monkeypatch.setattr(
        datasets_api.task_manager,
        "get_task",
        lambda task_id: {"id": task_id, "state": "pending", "queue_position": 1},
    )

    resp = client.post("/api/datasets/alice/separate", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task-9"
    assert body["queued"] is True
    assert body["queue_position"] == 1
    assert body["output_dataset"] == "alice_vocals"


def test_separate_runs_immediately_when_idle(client, monkeypatch):
    _make_dataset("alice")

    from server.api import datasets as datasets_api

    monkeypatch.setattr(
        datasets_api.task_manager,
        "create_task",
        lambda name, cmds, log_path, truncate=True, setup=None, **kwargs: "task-9",
    )
    monkeypatch.setattr(
        datasets_api.task_manager,
        "get_task",
        lambda task_id: {"id": task_id, "state": "running", "queue_position": None},
    )

    body = client.post("/api/datasets/alice/separate", json={}).json()

    assert body["queued"] is False
    assert body["queue_position"] is None


def test_list_and_detail_report_derived_from(client):
    _make_dataset("alice")
    _make_dataset("alice_vocals", files=("a_vocals.wav",))

    from server.api import datasets as datasets_api

    datasets_api._write_derived_marker("alice_vocals", "alice", "去伴奏")

    listed = {d["name"]: d for d in client.get("/api/datasets").json()}
    assert listed["alice"]["derived_from"] is None
    assert listed["alice_vocals"]["derived_from"] == "alice"

    detail = client.get("/api/datasets/alice_vocals").json()
    assert detail["derived_from"] == "alice"


def test_corrupt_derived_marker_reads_as_none(client):
    """坏标记与坏时长侧车同一纪律：不可信即视为无标记，绝不 500。"""
    _make_dataset("alice")
    meta = paths.DATASETS_DIR / ".meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "alice.derived.json").write_text("{not-json", encoding="utf8")

    data = client.get("/api/datasets").json()

    assert data[0]["derived_from"] is None


def test_delete_removes_marker_and_separate_log_of_deleted_name(client):
    """清理范围是「以被删数据集自己的名字命名的」.meta 档案（时长侧车 / 衍生标记 /
    分离日志）。衍生数据集 alice_vocals 是独立数据集：删源数据集 alice 不动它的
    标记（血缘是历史事实），删它自己时才清自己的标记。"""
    _make_dataset("alice")
    _make_dataset("alice_vocals", files=("a_vocals.wav",))
    meta = paths.DATASETS_DIR / ".meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "alice.separate.log").write_text("log", encoding="utf8")

    from server.api import datasets as datasets_api

    datasets_api._write_derived_marker("alice_vocals", "alice", "去伴奏")

    resp = client.delete("/api/datasets/alice")

    assert resp.status_code == 200
    assert not (meta / "alice.separate.log").exists()  # 源的分离日志随源删除
    assert (meta / "alice_vocals.derived.json").exists()  # 衍生标记不受牵连

    resp = client.delete("/api/datasets/alice_vocals")

    assert resp.status_code == 200
    assert not (meta / "alice_vocals.derived.json").exists()  # 衍生删除时清自己的标记


# ---------------------------------------------------------------------------
# 单文件删除：就地纠错 / 互斥 / 空目录收敛
# ---------------------------------------------------------------------------


def test_delete_file_removes_file_and_sidecar_entry(client):
    _make_dataset("alice", files=("a.wav", "b.wav"))
    client.get("/api/datasets")  # 生成侧车

    resp = client.delete("/api/datasets/alice/files/a.wav")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
    assert not (paths.DATASETS_DIR / "alice" / "a.wav").exists()
    assert (paths.DATASETS_DIR / "alice" / "b.wav").exists()  # 其余文件不受牵连
    sidecar = json.loads((paths.DATASETS_DIR / ".meta" / "alice.json").read_text(encoding="utf8"))
    assert "a.wav" not in sidecar  # 时长缓存条目一并清
    assert "b.wav" in sidecar


def test_delete_last_file_removes_empty_dataset_and_meta(client):
    """删到空 → 目录连同 .meta 档案一起移除，列表不留 0 文件空壳（与上传
    「不留空目录」同一纪律）。"""
    _make_dataset("alice", files=("a.wav",))

    assert client.delete("/api/datasets/alice/files/a.wav").status_code == 200

    assert not (paths.DATASETS_DIR / "alice").exists()
    assert client.get("/api/datasets").json() == []


def test_delete_file_missing_file_or_dataset_returns_404(client):
    _make_dataset("alice", files=("a.wav",))

    assert client.delete("/api/datasets/alice/files/ghost.wav").status_code == 404
    assert client.delete("/api/datasets/ghost/files/a.wav").status_code == 404


@pytest.mark.parametrize("bad", ["%2E%2E", ".hidden.wav"])
def test_delete_file_invalid_names_rejected(client, bad):
    """点开头文件不进列表（_iter_audio 口径），也不该能被删；`..` 防穿越。"""
    _make_dataset("alice", files=("a.wav",))

    assert client.delete(f"/api/datasets/alice/files/{bad}").status_code == 400


def test_delete_file_blocked_while_task_references_dataset(client, monkeypatch):
    _make_dataset("alice", files=("a.wav",))

    from server.api import datasets as datasets_api

    monkeypatch.setattr(
        datasets_api.task_manager,
        "list_tasks",
        lambda: [
            {
                "id": "t1",
                "name": "preprocess",
                "state": "running",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "alice")],
            }
        ],
    )

    resp = client.delete("/api/datasets/alice/files/a.wav")

    assert resp.status_code == 409
    assert (paths.DATASETS_DIR / "alice" / "a.wav").exists()  # 互斥拦截：文件原样保留


def test_delete_file_allowed_when_task_references_other_dataset(client, monkeypatch):
    _make_dataset("alice", files=("a.wav",))

    from server.api import datasets as datasets_api

    monkeypatch.setattr(
        datasets_api.task_manager,
        "list_tasks",
        lambda: [
            {
                "id": "t1",
                "name": "preprocess",
                "state": "running",
                "cmds": [_cmd_referencing(paths.DATASETS_DIR / "bob")],
            }
        ],
    )

    resp = client.delete("/api/datasets/alice/files/a.wav")

    assert resp.status_code == 200
    assert not (paths.DATASETS_DIR / "alice" / "a.wav").exists()


def test_delete_file_with_unicode_and_space_name(client):
    """上传允许空格/中文名（_safe_filename 保留），删除同口径放行。"""
    _make_dataset("alice")
    _write_wav(paths.DATASETS_DIR / "alice" / "中文 歌.wav")

    resp = client.delete("/api/datasets/alice/files/%E4%B8%AD%E6%96%87%20%E6%AD%8C.wav")

    assert resp.status_code == 200
    assert not (paths.DATASETS_DIR / "alice" / "中文 歌.wav").exists()


def test_separate_with_file_subset_builds_file_args(client, monkeypatch):
    """逐文件分离：命令带 --file；字符校验 + 存在性校验 + 去重。"""
    _make_dataset("alice", files=("a.wav", "b.wav"))

    from server.api import datasets as datasets_api

    seen = {}
    monkeypatch.setattr(
        datasets_api.task_manager,
        "create_task",
        lambda name, cmds, log_path, truncate=True, setup=None, **kwargs: seen.update(
            cmds=list(cmds)
        )
        or "task-1",
    )

    resp = client.post(
        "/api/datasets/alice/separate", json={"files": ["b.wav", "b.wav"]}
    )

    assert resp.status_code == 200
    assert '--file "b.wav"' in seen["cmds"][0]  # 去重后只出现一次
    assert "--file" not in seen["cmds"][0].split("--model")[1].split("--file")[0]


def test_separate_file_subset_rejects_unknown_and_unsafe_names(client):
    _make_dataset("alice", files=("a.wav",))

    unknown = client.post(
        "/api/datasets/alice/separate", json={"files": ["ghost.wav"]}
    )
    assert unknown.status_code == 400
    assert "没有该音频文件" in unknown.json()["detail"]

    # $ 反引号 引号 反斜杠：_safe_filename 放行它们，但它们会进 shell 命令，必须在此拦下
    for bad in ["a$(x).wav", "a`x`.wav", 'a"b.wav', "a\\b.wav"]:
        resp = client.post("/api/datasets/alice/separate", json={"files": [bad]})
        assert resp.status_code == 400, bad
        assert "文件名非法" in resp.json()["detail"]


def test_file_content_serves_audio_with_media_type(client):
    """试听端点：字节原样返回 + 按后缀给媒体类型。"""
    _make_dataset("alice", files=("a.wav",))
    payload = (paths.DATASETS_DIR / "alice" / "a.wav").read_bytes()

    resp = client.get("/api/datasets/alice/files/a.wav/content")

    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"].startswith("audio/wav")
    # 允许内联播放（不设 attachment）
    assert "attachment" not in resp.headers.get("content-disposition", "")


def test_file_content_rejects_invalid_and_missing(client):
    _make_dataset("alice", files=("a.wav",))

    assert client.get("/api/datasets/alice/files/.hidden/content").status_code == 400
    assert client.get("/api/datasets/alice/files/ghost.wav/content").status_code == 404
    assert client.get("/api/datasets/ghost/files/a.wav/content").status_code == 404


def test_file_content_supports_range_requests(client):
    """<audio> 拖动进度条依赖 Range 分段；Starlette FileResponse 原生支持。"""
    _make_dataset("alice", files=("a.wav",))
    full = (paths.DATASETS_DIR / "alice" / "a.wav").read_bytes()

    resp = client.get(
        "/api/datasets/alice/files/a.wav/content", headers={"range": "bytes=0-9"}
    )

    assert resp.status_code == 206
    assert resp.content == full[:10]
