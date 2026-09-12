import io
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import paths
from server.api import models as models_module
from server.api import training as training_module
from server.api.models import _index_matches
from server.main import create_app


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


@pytest.fixture(autouse=True)
def isolated_logs(monkeypatch, tmp_path):
    """删除接口会 rmtree logs/{exp}：把 LOGS_DIR 指到 tmp，测试绝不触碰真实 logs/。
    只 patch LOGS_DIR 不 patch ROOT——models 的其余路径都不依赖 ROOT，缩小爆炸半径。"""
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")


def _make_logs(tmp_path, exp):
    """造一个最小但形态齐全的 logs/{exp} 训练产物目录（特征/权重/索引真身）。"""
    logs_exp = tmp_path / "logs" / exp
    _touch(logs_exp / "0_gt_wavs" / "a.wav")
    _touch(logs_exp / "G_50.pth")
    return logs_exp


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
    mtime = (weights / "alice_v2.pth").stat().st_mtime
    assert resp.json() == [{"name": "alice_v2.pth", "path": str(weights / "alice_v2.pth"),
                            "index": str(linked), "mtime": int(mtime)}]


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
    assert body["deleted_models"] == ["alice_v2_e20_s100.pth"]
    assert body["failed_models"] == []
    assert sorted(body["deleted_indices"]) == sorted([added.name, linked.name])
    # 组内只有这一个成员，logs/{exp} 不存在：附带清理为"无目标"，不算失败
    assert body["logs"] == {"target": None, "removed": False, "failed_files": []}
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
    body = resp.json()
    assert body["deleted_indices"] == []
    assert body["deleted_models"] == ["bob.pth"]
    assert not (weights / "bob.pth").exists()


def test_delete_model_degenerate_stem_keeps_indices(monkeypatch, tmp_path):
    """退化文件名（_e20_s100.pth）剥不出实验名 → 联动为空，索引一个都不动，
    也不碰 logs（保持旧行为：只删被点名的文件）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "_e20_s100.pth")
    kept = _touch(indices / "added_IVF1_Flat_nprobe_1_bob.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/_e20_s100.pth")

    assert resp.status_code == 200
    assert resp.json() == {
        "deleted_model": "_e20_s100.pth",
        "deleted_models": ["_e20_s100.pth"],
        "failed_models": [],
        "deleted_indices": [],
        "failed_indices": [],
        "logs": {"target": None, "removed": False, "failed_files": []},
    }
    assert kept.exists()


def test_delete_model_shared_index_documented_ambiguity(monkeypatch, tmp_path):
    """【文档化决策】配对规则的多对一歧义：added_..._alice_v2.index 同时命中
    alice.pth（"_alice_" 子串）与 alice_v2.pth（"_alice_v2" 结尾）。删除任一方都会把
    该索引删掉，另一方（仍存在）失去索引——确认文案已向用户声明该影响范围。
    组口径下该歧义依旧存在：alice 与 alice_v2 是两个不同的组。"""
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
    """联动范围精确性：删除 bob 不得牵连 bobby 的索引（裸子串会误配），
    也不得动 bobby 的权重与 logs（组判定带同样边界）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "bob.pth")
    bobby = _touch(weights / "bobby.pth")
    bobby_index = _touch(indices / "added_IVF128_Flat_nprobe_1_bobby.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/bob.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_indices"] == []
    assert body["deleted_models"] == ["bob.pth"]
    assert bobby_index.exists() and bobby.exists()


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
# DELETE /api/models/{name}：彻底删除（整组权重 + 索引 + logs/{exp}）
# ---------------------------------------------------------------------------


def test_delete_group_removes_all_members_indices_and_logs(monkeypatch, tmp_path):
    """删组主路径：最终产物 + 全部中间轮次 + 配对索引 + logs/{exp} 全链路删净，
    别人的权重不受牵连。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    _touch(weights / "alice_e10_s5000.pth")
    _touch(weights / "alice_e20_s100.pth")
    kept = _touch(weights / "bob.pth")
    added = _touch(indices / "added_IVF1_Flat_nprobe_1_alice.index")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert sorted(body["deleted_models"]) == [
        "alice.pth",
        "alice_e10_s5000.pth",
        "alice_e20_s100.pth",
    ]
    assert body["failed_models"] == []
    assert body["deleted_indices"] == [added.name]
    assert body["logs"] == {"target": str(logs_exp), "removed": True, "failed_files": []}
    assert not added.exists()
    assert not logs_exp.exists()
    assert kept.exists()


def test_delete_group_case_insensitive_members_and_logs(monkeypatch, tmp_path):
    """组成员与 logs 目录都按大小写不敏感判定：请求大写文件名也要删掉小写
    实验名的组员与 logs 目录（Linux 大小写敏感 FS 上这里是 lower 探测在起作用）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "Alice_e20_s100.pth")
    _touch(weights / "alice.pth")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete("/api/models/Alice_e20_s100.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert sorted(body["deleted_models"]) == ["Alice_e20_s100.pth", "alice.pth"]
    assert body["logs"]["removed"] is True
    # 目标目录名随平台探测路径不同（macOS 大小写不敏感 FS 会以别名路径命中），
    # 只断言实验名命中，不锁全路径
    assert Path(body["logs"]["target"]).name.lower() == "alice"
    assert not logs_exp.exists()


@pytest.mark.parametrize("logs_setup", ["missing_root", "missing_exp"])
def test_delete_group_tolerates_missing_logs_dir(monkeypatch, tmp_path, logs_setup):
    """logs 根目录不存在 / 存在但无该实验子目录：都视为"无产物可清"，正常 200。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    if logs_setup == "missing_exp":
        _touch(tmp_path / "logs" / "bob" / "G_50.pth")  # logs 根存在，但没有 alice
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert resp.json()["logs"] == {"target": None, "removed": False, "failed_files": []}
    assert not (weights / "alice.pth").exists()


def test_delete_group_from_intermediate_representative(monkeypatch, tmp_path):
    """noFinal 组（训练中断只剩中间产物）：从代表项（中间轮次）发起删除同样删整组。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice_e20_s100.pth")  # 没有 alice.pth
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete("/api/models/alice_e20_s100.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_models"] == ["alice_e20_s100.pth"]
    assert body["logs"]["removed"] is True
    assert not logs_exp.exists()


class _StubTaskManager:
    """list_tasks 返回预置快照的最小替身（_ensure_exp_idle 只用这三个键）。"""

    def __init__(self, snapshots):
        self._snapshots = snapshots

    def list_tasks(self):
        return self._snapshots


@pytest.mark.parametrize("meta_exp,weight_name", [
    ("alice", "alice.pth"),  # 精确命中
    ("alice", "Alice_e20_s100.pth"),  # lower 候选命中：文件名大小写与任务不一致
    ("Alice", "Alice.pth"),  # 原始 case 候选命中
])
def test_delete_rejected_while_experiment_training(
    monkeypatch, tmp_path, meta_exp, weight_name
):
    """在训守卫：实验存在非终态任务时 409，权重/索引/logs 原封不动。
    守卫在 _ensure_exp_idle 内部解析 training 命名空间的 task_manager/_TASK_META，
    故 patch 的是 training 模块而不是 models。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / weight_name)
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    monkeypatch.setattr(
        training_module, "task_manager", _StubTaskManager(
            [{"id": "t1", "state": "running", "name": "train alice"}]
        )
    )
    monkeypatch.setattr(training_module, "_TASK_META", {"t1": {"exp_name": meta_exp}})

    resp = TestClient(create_app()).delete(f"/api/models/{weight_name}")

    assert resp.status_code == 409
    assert (weights / weight_name).exists()  # 一个字节都没动
    assert logs_exp.exists()


def test_delete_allowed_with_terminal_task(monkeypatch, tmp_path):
    """终态任务不拦（重新提交同一实验是合法的重跑路径，守卫只挡非终态）。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")
    monkeypatch.setattr(
        training_module, "task_manager", _StubTaskManager(
            [{"id": "t1", "state": "success", "name": "train alice"}]
        )
    )
    monkeypatch.setattr(training_module, "_TASK_META", {"t1": {"exp_name": "alice"}})

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert resp.json()["logs"]["removed"] is True
    assert not logs_exp.exists()


def test_delete_group_reports_failed_models(monkeypatch, tmp_path):
    """组内某个成员 unlink 失败：不回滚、不 500，failed_models 如实上报，
    其余成员与 logs 照常删除。"""
    weights = tmp_path / "weights"
    stuck = weights / "alice_e10_s5000.pth"
    real_delete = models_module._delete_file

    def flaky(path):
        if path.name == stuck.name:
            raise PermissionError(13, "denied")
        return real_delete(path)

    _touch(weights / "alice.pth")
    _touch(stuck)
    _touch(weights / "alice_e20_s100.pth")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")
    monkeypatch.setattr(models_module, "_delete_file", flaky)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_models"] == [stuck.name]
    assert stuck.exists()  # 删不掉的就是还在
    assert sorted(body["deleted_models"]) == ["alice.pth", "alice_e20_s100.pth"]
    assert body["logs"]["removed"] is True
    assert not logs_exp.exists()


def test_delete_named_model_failure_is_500_and_leaves_group(monkeypatch, tmp_path):
    """点名权重自身 unlink 失败 → 500：删除未发生，整组（权重/索引/logs）原样。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    named = weights / "alice.pth"
    real_delete = models_module._delete_file

    def flaky(path):
        if path.name == named.name:
            raise PermissionError(13, "denied")
        return real_delete(path)

    _touch(named)
    _touch(weights / "alice_e20_s100.pth")
    added = _touch(indices / "added_IVF1_Flat_nprobe_1_alice.index")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    monkeypatch.setattr(models_module, "_delete_file", flaky)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 500
    assert named.exists() and (weights / "alice_e20_s100.pth").exists()
    assert added.exists() and logs_exp.exists()


def test_delete_group_reports_partial_logs_failure(monkeypatch, tmp_path):
    """logs 树部分文件删不掉：200 + logs.removed=False + failed_files 上报，
    权重与索引不回滚（它们才是本接口的产品，logs 只是附带清理）。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    added = _touch(indices / "added_IVF1_Flat_nprobe_1_alice.index")
    logs_exp = _make_logs(tmp_path, "alice")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    def half_broken_rmtree(directory, failed_files):
        failed_files.append("G_50.pth")  # 模拟残留：目录本身不动

    monkeypatch.setattr(models_module, "_rmtree_best_effort", half_broken_rmtree)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    body = resp.json()
    assert body["logs"] == {"target": str(logs_exp), "removed": False, "failed_files": ["G_50.pth"]}
    assert body["deleted_models"] == ["alice.pth"]
    assert not added.exists()


def test_delete_group_sweeps_residual_links_into_logs(monkeypatch, tmp_path):
    """删 logs 后清扫仍指向它的残余软链（spkid / 悬空链，不在配对候选里），
    指向别的实验的链接不受牵连。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice.pth")
    indices.mkdir(parents=True, exist_ok=True)
    real_index = _touch(tmp_path / "logs" / "alice" / "added_alice.index")
    spkid = indices / "added_IVF1_alice_spkid3.index"
    spkid.symlink_to(real_index)
    dangling = indices / "alice_added_IVF2.index"
    dangling.symlink_to(tmp_path / "logs" / "alice" / "ghost.index")  # 目标本就不存在
    foreign_real = _touch(tmp_path / "logs" / "bob" / "added_bob.index")
    foreign = indices / "added_IVF1_bob.index"
    foreign.symlink_to(foreign_real)
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert resp.json()["logs"]["removed"] is True
    assert not spkid.exists()  # 指向被删 logs 的残余链接被清扫
    assert not dangling.exists()
    assert foreign.exists()  # 别人的链接不受牵连
    assert not (tmp_path / "logs" / "alice").exists()


def test_delete_group_keeps_other_experiment_logs(monkeypatch, tmp_path):
    """logs 清理的精确性：删 alice 组不得动 logs/bobby（与索引不牵连对偶）。"""
    weights = tmp_path / "weights"
    _touch(weights / "alice.pth")
    alice_logs = _make_logs(tmp_path, "alice")
    bobby_logs = _make_logs(tmp_path, "bobby")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = TestClient(create_app()).delete("/api/models/alice.pth")

    assert resp.status_code == 200
    assert not alice_logs.exists()
    assert bobby_logs.exists()
    assert (bobby_logs / "G_50.pth").exists()


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


# ---------------------------------------------------------------------------
# POST /api/models/upload（外部音色模型导入）
# ---------------------------------------------------------------------------

# pth 的 torch zip 容器魔数 + 伪内容：上传接口只做轻校验，不需要真实 checkpoint
_PTH_BYTES = b"PK\x03\x04not-a-real-checkpoint"
_IDX_BYTES = b"fake-faiss-index-bytes"


def _upload(client, model=None, index=None):
    """按字段名组装 multipart 上传；字段值为 (filename, bytes) 元组，None 表示不传。"""
    files = []
    if model is not None:
        files.append(("model", model))
    if index is not None:
        files.append(("index", index))
    return client.post("/api/models/upload", files=files)


def test_upload_model_with_index_pairs_and_lists(monkeypatch, tmp_path):
    """pth + index 双文件上传成功：两文件各自落盘，响应与 GET /api/models 条目
    同构且索引已配对，列表接口能看到同一模型。"""
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    client = TestClient(create_app())

    resp = _upload(
        client,
        model=("alice.pth", _PTH_BYTES),
        index=("added_IVFxxx_alice.index", _IDX_BYTES),
    )

    assert resp.status_code == 200
    assert (weights / "alice.pth").read_bytes() == _PTH_BYTES
    assert (indices / "added_IVFxxx_alice.index").read_bytes() == _IDX_BYTES
    listed = client.get("/api/models").json()
    assert resp.json() in listed  # 响应条目与列表条目同构且一致
    assert listed[0]["index"] == str(indices / "added_IVFxxx_alice.index")


def test_upload_model_without_index(monkeypatch, tmp_path):
    """只传 pth 是合法操作：响应与列表里 index 均为 None。"""
    weights = tmp_path / "weights"
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")
    client = TestClient(create_app())

    resp = _upload(client, model=("bob.pth", _PTH_BYTES))

    assert resp.status_code == 200
    assert resp.json()["index"] is None
    assert (weights / "bob.pth").exists()


def test_upload_duplicate_model_name_is_409(monkeypatch, tmp_path):
    """与 weights 下已有模型重名 → 409，原文件保持原样，也不写任何半成品。"""
    weights = tmp_path / "weights"
    original = _touch(weights / "alice.pth")
    original.write_bytes(_PTH_BYTES)
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = _upload(
        TestClient(create_app()), model=("alice.pth", _PTH_BYTES), index=("a.index", _IDX_BYTES)
    )

    assert resp.status_code == 409
    assert original.read_bytes() == _PTH_BYTES  # 原内容未被覆盖
    assert list(weights.iterdir()) == [original]  # 无 .part 残留
    assert not (tmp_path / "indices" / "a.index").exists()  # 拒绝时索引也不落盘


def test_upload_duplicate_index_name_is_409(monkeypatch, tmp_path):
    """与 indices 下已有索引重名 → 409（两目录独立查重），模型也不落盘。"""
    indices = tmp_path / "indices"
    original = _touch(indices / "alice.index")
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    resp = _upload(
        TestClient(create_app()), model=("alice.pth", _PTH_BYTES), index=("alice.index", _IDX_BYTES)
    )

    assert resp.status_code == 409
    assert not (tmp_path / "weights" / "alice.pth").exists()


@pytest.mark.parametrize(
    ("model", "index"),
    [
        ("../evil.pth", ("ok.index", _IDX_BYTES)),  # 模型名穿越
        ("ok.pth", ("../evil.index", _IDX_BYTES)),  # 索引名穿越
    ],
)
def test_upload_rejects_traversal(monkeypatch, tmp_path, model, index):
    """非 basename 一律 400，绝不落盘到目录之外。"""
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = _upload(TestClient(create_app()), model=model, index=index)

    assert resp.status_code == 400
    assert not (tmp_path / "evil.pth").exists()
    assert not (tmp_path / "evil.index").exists()
    assert not (tmp_path / "weights" / "ok.pth").exists()
    assert not (tmp_path / "indices" / "ok.index").exists()


@pytest.mark.parametrize(
    ("model", "index"),
    [
        ("notes.txt", ("ok.index", _IDX_BYTES)),  # 模型后缀非法
        ("ok.pth", ("notes.idx", _IDX_BYTES)),  # 索引后缀非法：整体拒绝，模型也不落盘
    ],
)
def test_upload_rejects_wrong_suffix(monkeypatch, tmp_path, model, index):
    """后缀白名单：模型必须 .pth、索引必须 .index；任一非法整体拒绝（400），
    校验发生在任何落盘之前。"""
    weights = tmp_path / "weights"
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    model_bytes = _PTH_BYTES if model.endswith(".pth") else b"whatever"
    resp = _upload(
        TestClient(create_app()), model=(model, model_bytes), index=index
    )

    assert resp.status_code == 400
    assert not (weights / "ok.pth").exists()


def test_upload_rejects_uppercase_suffix(monkeypatch, tmp_path):
    """后缀必须小写：_scan 的 glob 与配对口径大小写敏感，大写后缀会造出列表
    看不见的文件，必须显式拒绝而不是静默改写。"""
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = _upload(TestClient(create_app()), model=("ALICE.PTH", _PTH_BYTES))

    assert resp.status_code == 400


def test_upload_rejects_oversized(monkeypatch, tmp_path):
    """超过大小上限 → 413，无任何落盘残留（含 .part 半成品）。"""
    weights = tmp_path / "weights"
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")
    monkeypatch.setattr("server.api.models.MAX_UPLOAD_BYTES", 16)

    resp = _upload(
        TestClient(create_app()), model=("big.pth", b"PK\x03\x04" + b"x" * 64)
    )

    assert resp.status_code == 413
    assert list(weights.iterdir() if weights.exists() else []) == []


def test_upload_rejects_non_zip_model_content(monkeypatch, tmp_path):
    """pth 魔数校验：非 zip 容器（缺 PK\x03\x04）→ 400，不落盘。"""
    weights = tmp_path / "weights"
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = _upload(TestClient(create_app()), model=("fake.pth", b"definitely not a zip"))

    assert resp.status_code == 400
    assert not (weights / "fake.pth").exists()
    assert list(weights.iterdir() if weights.exists() else []) == []


def test_upload_missing_model_field(monkeypatch, tmp_path):
    """model 是必选 multipart 字段：缺省由 FastAPI 校验层拒绝（422）。"""
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "indices")

    resp = _upload(TestClient(create_app()), index=("a.index", _IDX_BYTES))

    assert resp.status_code == 422
