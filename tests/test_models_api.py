import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from server.api.models import _index_matches
from server.main import create_app


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_models_scan_and_pair(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice_v2.pth")
    _touch(weights / "alice_v2_e20_s100.pth")  # 带 epoch/step 后缀的变体
    added = _touch(indices / "added_IVFxxx_Flat_nprobe_1_alice_v2.index")
    trained = _touch(indices / "trained_IVFxxx_alice_v2.index")  # trained 必须被忽略
    # 把 trained 的 mtime 推到未来：一旦过滤失效，它必然按"最新索引"胜出，断言必失败
    future = trained.stat().st_mtime + 1000
    os.utime(trained, (future, future))
    _touch(weights / "bob.pth")  # 无索引
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    client = TestClient(create_app())
    resp = client.get("/api/models")

    assert resp.status_code == 200
    items = resp.json()
    by_name = {m["name"]: m for m in items}
    assert by_name["alice_v2.pth"]["index"] == str(added)
    assert by_name["alice_v2_e20_s100.pth"]["index"] == str(added)
    assert by_name["bob.pth"]["index"] is None


def test_models_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "none")
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "none")
    client = TestClient(create_app())
    assert client.get("/api/models").json() == []


def test_models_ignore_dangling_symlink_index(monkeypatch, tmp_path):
    """悬空的索引软链（logs/{exp} 被删）必须被忽略而不是 500：
    训练产出的索引是链到 logs/{exp}/ 的软链，glob 仍能列出它，但 stat/读取
    都 FileNotFoundError（E2E 实测把 GET /api/models 打成 500）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    weights.mkdir(parents=True)
    indices.mkdir(parents=True)
    _touch(weights / "alice_v2.pth")
    real = _touch(tmp_path / "logs" / "alice_v2" / "added_x_alice_v2.index")
    linked = indices / "alice_v2_added_IVF_x.index"
    linked.symlink_to(real)
    dangling = indices / "ghost_added_IVF_x.index"
    dangling.symlink_to(tmp_path / "logs" / "ghost" / "added_x_ghost.index")  # 目标不存在
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    client = TestClient(create_app())
    resp = client.get("/api/models")

    assert resp.status_code == 200
    assert resp.json() == [{"name": "alice_v2.pth", "path": str(weights / "alice_v2.pth"),
                            "index": str(linked)}]


def test_models_pair_edge_cases(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "Alice_V2.pth")  # 大小写不同也应配对
    _touch(weights / "_e20_s100.pth")  # 退化文件名剥不出实验名，无法配对
    _touch(weights / "bob.pth")
    _touch(weights / "bobby.pth")  # 相似名不得串用索引
    older = _touch(indices / "added_IVF64_Flat_nprobe_1_alice_v2.index")
    newer = _touch(indices / "Alice_V2_added_IVF96_Flat_nprobe_1_Alice_V2.index")
    _touch(indices / "added_IVF128_Flat_nprobe_1_bobby.index")
    _touch(indices / "added_IVF128_Flat_nprobe_1_alice_v2_spkid3.index")  # spkid 需跳过
    older_time = older.stat().st_mtime - 1000
    os.utime(older, (older_time, older_time))
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    by_name = {
        m["name"]: m for m in TestClient(create_app()).get("/api/models").json()
    }

    # 多个候选时取 mtime 最新的索引
    assert by_name["Alice_V2.pth"]["index"] == str(newer)
    assert by_name["_e20_s100.pth"]["index"] is None
    assert by_name["bob.pth"]["index"] is None  # 不误用 bobby 的索引
    assert by_name["bobby.pth"]["index"].endswith("_bobby.index")


@pytest.mark.parametrize(
    "stem,exp,ok",
    [
        ("added_IVF800_Flat_nprobe_1_alice_v2", "alice_v2", True),
        ("alice_v2_added_IVF800_Flat_nprobe_1", "alice_v2", True),
        ("alice_v2", "alice_v2", True),
        ("added_IVF800_Flat_nprobe_1_alice_v2_v2", "alice_v2", True),
        ("added_IVF128_Flat_nprobe_1_bobby", "bob", False),
        ("added_IVF128_Flat_nprobe_1_bob", "bobby", False),
    ],
)
def test_index_matches(stem, exp, ok):
    assert _index_matches(stem, exp) is ok


# ---------------------------------------------------------------------------
# DELETE /api/models/{name}
# ---------------------------------------------------------------------------


def test_delete_model_with_paired_indices(monkeypatch, tmp_path):
    """正常删除：.pth 移除，且该实验名下全部候选索引联动删除（含外链副本形态）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice_v2_e20_s100.pth")
    added = _touch(indices / "added_IVFxxx_Flat_nprobe_1_alice_v2.index")
    linked = _touch(indices / "alice_v2_added_IVFyyy_Flat_nprobe_1.index")
    other = _touch(indices / "added_IVFzzz_bob.index")  # 别人的索引必须保留
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/alice_v2_e20_s100.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_model"] == "alice_v2_e20_s100.pth"
    assert sorted(body["deleted_indices"]) == sorted([added.name, linked.name])
    assert not (weights / "alice_v2_e20_s100.pth").exists()
    assert not added.exists() and not linked.exists()
    assert other.exists()


def test_delete_model_without_index(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "bob.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/bob.pth")

    assert resp.status_code == 200
    assert resp.json()["deleted_indices"] == []
    assert not (weights / "bob.pth").exists()


def test_delete_model_degenerate_stem_keeps_indices(monkeypatch, tmp_path):
    """退化文件名（_e20_s100.pth）剥不出实验名 → 联动为空，索引一个都不动。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "_e20_s100.pth")
    kept = _touch(indices / "added_IVF1_Flat_nprobe_1_bob.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/_e20_s100.pth")

    assert resp.status_code == 200
    assert resp.json() == {"deleted_model": "_e20_s100.pth", "deleted_indices": [], "failed_indices": []}
    assert kept.exists()


def test_delete_model_shared_index_documented_ambiguity(monkeypatch, tmp_path):
    """【文档化决策】配对规则的多对一歧义：added_..._alice_v2.index 同时命中
    alice.pth（"_alice_" 子串）与 alice_v2.pth（"_alice_v2" 结尾）。删除任一方都会把
    该索引删掉，另一方（仍存在）失去索引——确认文案已向用户声明该影响范围。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    _touch(weights / "alice_v2.pth")
    shared = _touch(indices / "added_IVF1_Flat_nprobe_1_alice_v2.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    client = TestClient(create_app())

    # 前置：GET 配对下两份模型确实共享同一索引
    paired = {m["name"]: m["index"] for m in client.get("/api/models").json()}
    assert paired["alice.pth"] == paired["alice_v2.pth"] == str(shared)

    resp = client.delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert resp.json()["deleted_indices"] == [shared.name]
    assert (weights / "alice_v2.pth").exists()  # 模型还在，但失去索引：
    assert not shared.exists()
    remaining = {m["name"]: m["index"] for m in client.get("/api/models").json()}
    assert remaining["alice_v2.pth"] is None


class _UnlinkFails:
    """unlink 必抛的索引替身：模拟权限/占用等删除失败（M-2 非原子场景）。
    stem 与 Path.stem 同语义（不含后缀），否则配对过滤不会命中它。"""

    def __init__(self, name: str):
        self.name = name
        self.stem = name.removesuffix(".index")

    def unlink(self):
        raise PermissionError(13, "denied")


def test_delete_model_reports_failed_indices(monkeypatch, tmp_path):
    """索引删除失败不回滚、不 500：如实上报 failed_indices，其余照常删除。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    good = _touch(indices / "added_IVF1_alice.index")
    bad = _UnlinkFails("added_IVF2_alice.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    monkeypatch.setattr("server.api.models._list_indices", lambda: [good, bad])

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_indices"] == [good.name]
    assert body["failed_indices"] == [bad.name]
    assert not good.exists() and (weights / "alice.pth").exists() is False


def test_delete_model_missing(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete("/api/models/carol.pth")

    assert resp.status_code == 404
    assert (weights / "alice.pth").exists()  # 别的模型不受影响


@pytest.mark.parametrize("name", ["notes.txt", "readme", "alice.pth.bak"])
def test_delete_model_rejects_non_pth(monkeypatch, tmp_path, name):
    """只删权重文件（与 GET 扫描的 *.pth 一致），防误删 weights 下的杂物。"""
    weights = tmp_path / "weights"
    _touch(weights / name)
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete(f"/api/models/{name}")

    assert resp.status_code == 404
    assert (weights / name).exists()


@pytest.mark.parametrize("bad", ["%2E%2E", "%2E"])
def test_delete_model_rejects_traversal(monkeypatch, tmp_path, bad):
    """防穿越：非 basename 一律 404，绝不触达 weights 目录之外的文件。

    用百分号编码直达 handler——客户端会把裸 `..` / `.` 规范化掉（`/a/../b` → `/b`），
    测不到 handler 自身的校验。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    outside = _touch(tmp_path / "outside.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete(f"/api/models/{bad}")

    assert resp.status_code == 404
    assert outside.exists() and (weights / "alice.pth").exists()


@pytest.mark.parametrize("bad", ["../alice.pth", "sub/alice.pth"])
def test_delete_model_multi_segment_never_reaches_handler(monkeypatch, tmp_path, bad):
    """带路径分隔符的形态根本匹配不上 `/models/{name}` 单段路由（防御在路由层）：
    无论如何都不能有模型被删。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete(f"/api/models/{bad}")

    assert resp.status_code in (404, 405)
    assert (weights / "alice.pth").exists()


def test_delete_model_keeps_similar_experiment_index(monkeypatch, tmp_path):
    """联动范围精确性：删除 bob 不得牵连 bobby 的索引（裸子串会误配）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "bob.pth")
    bobby_index = _touch(indices / "added_IVF128_Flat_nprobe_1_bobby.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/bob.pth")

    assert resp.status_code == 200
    assert resp.json()["deleted_indices"] == []
    assert bobby_index.exists()


def test_delete_model_ignores_spkid_index(monkeypatch, tmp_path):
    """*_spkidN 索引只含单说话人向量（P1 配对规则即跳过），删除时不联动。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    spkid = _touch(indices / "added_IVF128_Flat_nprobe_1_alice_spkid3.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert resp.json()["deleted_indices"] == []
    assert spkid.exists()


# ---------------------------------------------------------------------------
# GET /api/models/{name}/download（zip 打包下载：pth + 配对索引）
# ---------------------------------------------------------------------------


def test_download_model_bundles_pth_and_index(monkeypatch, tmp_path):
    """正常下载：zip 内含实验名目录下的 pth 与配对索引，字节与源文件一致。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    model = weights / "alice_v2_e20_s100.pth"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"pth-bytes")
    index = _touch(indices / "added_IVFxxx_Flat_nprobe_1_alice_v2.index")
    index.write_bytes(b"index-bytes")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).get("/api/models/alice_v2_e20_s100.pth/download")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert resp.headers["content-disposition"] == 'attachment; filename="alice_v2_e20_s100.zip"'
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == [
        "alice_v2/alice_v2_e20_s100.pth",
        "alice_v2/added_IVFxxx_Flat_nprobe_1_alice_v2.index",
    ]
    assert zf.read("alice_v2/alice_v2_e20_s100.pth") == b"pth-bytes"
    assert zf.read("alice_v2/added_IVFxxx_Flat_nprobe_1_alice_v2.index") == b"index-bytes"


def test_download_model_without_index_only_pth(monkeypatch, tmp_path):
    """缺索引的模型照常可下载，包内只有 pth，不被拦截。"""
    weights = tmp_path / "weights"
    _touch(weights / "bob.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get("/api/models/bob.pth/download")

    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.namelist() == ["bob/bob.pth"]
    assert zf.read("bob/bob.pth") == b"x"


def test_download_model_degenerate_stem_falls_back_to_stem_dir(monkeypatch, tmp_path):
    """退化文件名（_e20_s100.pth）剥不出实验名 → zip 目录退回模型 stem，不能是空目录名。"""
    weights = tmp_path / "weights"
    _touch(weights / "_e20_s100.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get("/api/models/_e20_s100.pth/download")

    assert resp.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(resp.content)).namelist() == ["_e20_s100/_e20_s100.pth"]


def test_download_model_multi_chunk_stream_stays_intact(monkeypatch, tmp_path):
    """缓冲中途多次 flush 后 zip 仍必须完整：zipfile 在可 seek 的缓冲上写完数据会
    回头 seek 改写 local header，已被 flush 出去的字节就再也改不到了（真实 55MB
    模型端到端实测包损坏，单测里 1 字节假文件从不触发中途 flush 所以漏过）。
    把块大小打到极小强制多次 flush，包必须仍过 zipfile 的 CRC 完整性校验。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    payload = bytes(range(256)) * 40  # 10KB，远超下方 64 字节的块大小
    model = weights / "alice.pth"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    index = _touch(indices / "added_IVF1_Flat_nprobe_1_alice.index")
    index.write_bytes(payload[::-1])
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    monkeypatch.setattr("server.api.models._ZIP_CHUNK", 64)

    resp = TestClient(create_app()).get("/api/models/alice.pth/download")

    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert zf.testzip() is None
    assert zf.read("alice/alice.pth") == payload
    assert zf.read("alice/added_IVF1_Flat_nprobe_1_alice.index") == payload[::-1]


def test_download_model_chinese_name_uses_rfc5987(monkeypatch, tmp_path):
    """非 ASCII 模型名：filename= 放不下，必须走 filename*=utf-8''（RFC 5987），
    否则非 latin-1 字节会把响应头打爆。"""
    weights = tmp_path / "weights"
    _touch(weights / "小明_v2.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get("/api/models/小明_v2.pth/download")

    assert resp.status_code == 200
    assert (
        resp.headers["content-disposition"]
        == "attachment; filename*=utf-8''%E5%B0%8F%E6%98%8E_v2.zip"
    )
    assert zipfile.ZipFile(io.BytesIO(resp.content)).namelist() == ["小明_v2/小明_v2.pth"]


def test_download_model_missing(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get("/api/models/carol.pth/download")

    assert resp.status_code == 404


@pytest.mark.parametrize("name", ["notes.txt", "readme", "alice.pth.bak"])
def test_download_model_rejects_non_pth(monkeypatch, tmp_path, name):
    """与 DELETE 同口径：只打包权重文件，weights 下的杂物不给下。"""
    weights = tmp_path / "weights"
    _touch(weights / name)
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get(f"/api/models/{name}/download")

    assert resp.status_code == 404


@pytest.mark.parametrize("bad", ["%2E%2E", "%2E"])
def test_download_model_rejects_traversal(monkeypatch, tmp_path, bad):
    """防穿越：非 basename 一律 404（百分号编码直达 handler，同 DELETE 的理由）。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    outside = _touch(tmp_path / "outside.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get(f"/api/models/{bad}/download")

    assert resp.status_code == 404
    assert outside.exists() and (weights / "alice.pth").exists()


@pytest.mark.parametrize("bad", ["../alice.pth", "sub/alice.pth"])
def test_download_model_multi_segment_never_reaches_handler(monkeypatch, tmp_path, bad):
    """带路径分隔符的形态匹配不上单段路由（防御在路由层），不能有任何文件被读走。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).get(f"/api/models/{bad}/download")

    assert resp.status_code in (404, 405)
    assert (weights / "alice.pth").exists()
