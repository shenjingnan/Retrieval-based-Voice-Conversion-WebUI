import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import paths
from server.api import infer as infer_api
from server.main import create_app


class CountingVC:
    """可计数的 FakeVC：记录 get_vc 的总调用数与成功加载的模型名。"""

    def __init__(self, fail_first=None):
        self.loaded = None
        self.calls = 0
        self.load_count = 0
        self.fail_first = fail_first

    def get_vc(self, sid):
        self.calls += 1
        if self.fail_first is not None:
            err, self.fail_first = self.fail_first, None
            raise err
        self.loaded = sid
        self.load_count += 1

    def vc_single(self, sid, path, f0_up_key, f0_method, file_index,
                  index_rate, resample_sr, rms_mix_rate, protect):
        return "状态：成功", (32000, np.zeros(3200, dtype=np.int16))


class FailPipelineVC(CountingVC):
    def vc_single(self, *a, **k):
        return "【单次推理】\n状态：失败\nboom", (None, None)


@pytest.fixture(autouse=True)
def _fresh_vc_cache(monkeypatch):
    """清空模块级缓存，避免用例间串扰。"""
    monkeypatch.setattr(infer_api, "_vc_cache", {})


def _post(client, model, **overrides):
    data = {"model": model, "transpose": "0", "f0_method": "rmvpe",
            "index_rate": "0.75", "resample_sr": "0",
            "rms_mix_rate": "0.25", "protect": "0.33", "index_path": ""}
    data.update(overrides)
    return client.post(
        "/api/infer",
        files={"audio": ("in.wav", b"x", "audio/wav")},
        data=data,
    )


def _fake_weights_dir(monkeypatch, tmp_path, name="m.pth"):
    """WEIGHTS_DIR 指向 tmp 并造假模型文件，让前置存在性检查通过。"""
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path)
    (tmp_path / name).write_bytes(b"fake")


def test_infer_ok(monkeypatch, tmp_path):
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = CountingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert vc.loaded == "m.pth"


def test_infer_model_not_found(monkeypatch, tmp_path):
    # 造 m.pth 让前置检查通过，缺失由 VC.get_vc 内部抛出，走兜底网
    _fake_weights_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        infer_api, "_build_vc",
        lambda: CountingVC(fail_first=FileNotFoundError("m.pth 下载失败")),
    )

    client = TestClient(create_app())
    resp = _post(client, "m.pth")

    assert resp.status_code == 404
    assert "m.pth" in resp.json()["detail"]


def test_infer_pipeline_failure(monkeypatch, tmp_path):
    _fake_weights_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: FailPipelineVC())

    client = TestClient(create_app())
    resp = _post(client, "m.pth")

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_infer_path_traversal_rejected(monkeypatch, tmp_path):
    _fake_weights_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    client = TestClient(create_app())
    resp = _post(client, "../evil.pth")

    assert resp.status_code == 404


class RecordingVC(CountingVC):
    """记录 vc_single 收到的 file_index，用于断言解析结果。"""

    def __init__(self):
        super().__init__()
        self.seen_index = "unset"

    def vc_single(self, sid, path, f0_up_key, f0_method, file_index,
                  index_rate, resample_sr, rms_mix_rate, protect):
        self.seen_index = file_index
        return super().vc_single(
            sid, path, f0_up_key, f0_method, file_index,
            index_rate, resample_sr, rms_mix_rate, protect,
        )


def _fake_indices_dir(monkeypatch, tmp_path, name="alice.index"):
    """INDICES_DIR 指向 tmp 并造假索引文件，返回其绝对路径。"""
    indices = tmp_path / "indices"
    indices.mkdir()
    index = indices / name
    index.write_bytes(b"fake")
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)
    return index


def test_infer_index_traversal_rejected(monkeypatch, tmp_path):
    """index_path 含路径分隔符且不在 INDICES_DIR 内 → 404，不透传给 faiss。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    _fake_indices_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path="../evil.index")

    assert resp.status_code == 404
    assert vc.seen_index == "unset"  # 未触达 vc_single


def test_infer_index_absolute_outside_rejected(monkeypatch, tmp_path):
    """绝对路径但不在 INDICES_DIR 下 → 404（防探测任意路径）。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    _fake_indices_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    outside = tmp_path / "elsewhere" / "evil.index"
    outside.parent.mkdir()
    outside.write_bytes(b"fake")

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path=str(outside))

    assert resp.status_code == 404
    assert vc.seen_index == "unset"


def test_infer_index_relative_name_resolved_to_indices_dir(monkeypatch, tmp_path):
    """纯文件名形态：解析为 INDICES_DIR 下的绝对路径再交给 vc_single。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    index = _fake_indices_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path="alice.index")

    assert resp.status_code == 200
    assert vc.seen_index == str(index)


def test_infer_index_absolute_path_from_models_accepted(monkeypatch, tmp_path):
    """/api/models 返回的是绝对路径，前端原样回传也必须可用。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    index = _fake_indices_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path=str(index))

    assert resp.status_code == 200
    assert vc.seen_index == str(index)


def test_infer_index_symlinked_dir_and_file_accepted(monkeypatch, tmp_path):
    """真实仓库布局：assets/indices 是目录软链，index 又是链到 logs/{exp}/ 的软链。

    GET /api/models 返回的是「软链目录下的软链文件」路径，前端原样回传后
    resolve() 会穿到 logs 下，按文件自身归属判定会误报 404（E2E 实测）——
    归属判定必须只 resolve 父目录。
    """
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    real_indices = tmp_path / "real_indices"
    real_indices.mkdir()
    real_logs_index = tmp_path / "logs" / "exp"
    real_logs_index.mkdir(parents=True)
    target = real_logs_index / "added_x.index"
    target.write_bytes(b"fake")

    linked_dir = tmp_path / "assets" / "indices"
    linked_dir.parent.mkdir()
    linked_dir.symlink_to(real_indices)
    index = linked_dir / "exp_added_IVF_x.index"
    index.symlink_to(target)
    monkeypatch.setattr("server.paths.INDICES_DIR", linked_dir)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path=str(index))

    assert resp.status_code == 200
    assert vc.seen_index == str(index)


def test_infer_index_symlinked_subdir_escape_rejected(monkeypatch, tmp_path):
    """INDICES_DIR 下的子目录若是软链，父目录 resolve 后落到目录之外 → 404：
    归属判定的口径是「传入路径须落在 INDICES_DIR 内」，网络请求方不能借
    目录软链探测/读取任意路径（文件软链外指属已知取舍：训练产物 index 本身
    就是链到 logs/{exp}/ 的软链，按软链目标判定会把主流程误判成 404）。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.index").write_bytes(b"fake")

    indices_dir = tmp_path / "assets" / "indices"
    indices_dir.parent.mkdir()
    indices_dir.mkdir()
    escape = indices_dir / "escape"
    escape.symlink_to(outside)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices_dir)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path=str(escape / "secret.index"))

    assert resp.status_code == 404
    assert vc.seen_index == "unset"


def test_infer_index_missing_file_rejected(monkeypatch, tmp_path):
    """名字合法但文件缺失 → 404，而非把缺失路径丢给 faiss 后炸堆栈。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    _fake_indices_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path="ghost.index")

    assert resp.status_code == 404
    assert vc.seen_index == "unset"


def test_infer_index_empty_means_no_index(monkeypatch, tmp_path):
    """空串保持"不使用索引"语义，传给 vc_single 的是 None。"""
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = RecordingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth", index_path="")

    assert resp.status_code == 200
    assert vc.seen_index is None


def test_get_vc_cached_reuses_and_skips_reload(monkeypatch):
    made = []

    def factory():
        vc = CountingVC()
        made.append(vc)
        return vc

    monkeypatch.setattr(infer_api, "_build_vc", factory)

    a1 = infer_api.get_vc_cached("a.pth")
    a2 = infer_api.get_vc_cached("a.pth")
    assert a1 is a2 is made[0]
    assert made[0].load_count == 1  # 同模型第二次命中缓存，跳过重载

    b = infer_api.get_vc_cached("b.pth")
    assert b is made[1]
    assert made[1].load_count == 1

    a3 = infer_api.get_vc_cached("a.pth")
    assert a3 is made[0]
    assert made[0].load_count == 1  # 切回 a.pth 不再加载


def test_get_vc_cached_fills_missing_rmvpe_root(monkeypatch):
    """rmvpe_root 缺失时由缓存层兜底设置，避免 pipeline.get_f0 直取环境变量时 KeyError。"""
    monkeypatch.delenv("rmvpe_root", raising=False)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    infer_api.get_vc_cached("a.pth")

    assert os.environ["rmvpe_root"] == str(paths.ASSETS / "rmvpe")


def test_get_vc_cached_keeps_existing_rmvpe_root(monkeypatch):
    """外部已显式设置 rmvpe_root 时不得覆盖（setdefault 语义）。"""
    monkeypatch.setenv("rmvpe_root", "/custom/rmvpe")
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    infer_api.get_vc_cached("a.pth")

    assert os.environ["rmvpe_root"] == "/custom/rmvpe"


def test_get_vc_cached_fills_missing_index_roots(monkeypatch):
    """index_root / outside_index_root 缺失时由缓存层兜底设置：
    VC.get_vc → get_index_path_from_model 遍历这两个目录配对默认索引，
    roots[0]（outside_index_root）为 None 会在 os.path.abspath 处 TypeError
    （webui 靠模块顶层 setdefault，create_app() 直起的服务进程没人设它）。"""
    monkeypatch.delenv("index_root", raising=False)
    monkeypatch.delenv("outside_index_root", raising=False)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    infer_api.get_vc_cached("a.pth")

    assert os.environ["index_root"] == str(paths.INDICES_DIR)
    assert os.environ["outside_index_root"] == str(paths.INDICES_DIR)


def test_get_vc_cached_keeps_existing_index_roots(monkeypatch):
    """外部已显式设置 index_root / outside_index_root 时不得覆盖（setdefault 语义）。"""
    monkeypatch.setenv("index_root", "/custom/indices")
    monkeypatch.setenv("outside_index_root", "/custom/outside")
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    infer_api.get_vc_cached("a.pth")

    assert os.environ["index_root"] == "/custom/indices"
    assert os.environ["outside_index_root"] == "/custom/outside"


def test_infer_same_model_reuses_weights(monkeypatch, tmp_path):
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = CountingVC()
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    assert _post(client, "m.pth").status_code == 200
    assert _post(client, "m.pth").status_code == 200
    assert vc.load_count == 1
    assert vc.loaded == "m.pth"


def test_infer_load_failure_retryable(monkeypatch, tmp_path):
    _fake_weights_dir(monkeypatch, tmp_path)
    vc = CountingVC(fail_first=FileNotFoundError("权重缺失"))
    monkeypatch.setattr(infer_api, "_build_vc", lambda: vc)

    client = TestClient(create_app())
    resp = _post(client, "m.pth")
    assert resp.status_code == 404
    assert "权重缺失" in resp.json()["detail"]

    vc.fail_first = None  # 修复后同模型重试，缓存里的失败 entry 允许重试
    resp = _post(client, "m.pth")
    assert resp.status_code == 200
    assert vc.calls == 2
    assert vc.load_count == 1


# ---------------------------------------------------------------------------
# LRU 上限与释放（长会话内存积累的治理，见 server/api/infer.py 模块注释）
# ---------------------------------------------------------------------------


def test_vc_cache_lru_evicts_oldest(monkeypatch):
    """超过 _VC_CACHE_MAX 时按 LRU 逐出最旧条目：上限 2 下 a→b→回a→c 应逐出 b。"""
    made = {}

    def factory():
        vc = CountingVC()
        made[f"vc{len(made)}"] = vc
        return vc

    monkeypatch.setattr(infer_api, "_build_vc", factory)
    monkeypatch.setattr(infer_api, "_VC_CACHE_MAX", 2)

    a = infer_api.get_vc_cached("a.pth")
    b = infer_api.get_vc_cached("b.pth")
    infer_api.get_vc_cached("a.pth")  # 命中即触活：LRU 序变为 b（旧）、a（新）

    c = infer_api.get_vc_cached("c.pth")  # 超限 → 逐出最旧的 b
    assert len(infer_api._vc_cache) == 2
    assert "b.pth" not in infer_api._vc_cache
    assert {"a.pth", "c.pth"} == set(infer_api._vc_cache)
    assert infer_api.get_vc_cached("a.pth") is a  # a 未被误逐出
    assert c is not b


def test_vc_cache_disposes_evicted_entry(monkeypatch):
    """逐出时对被逐出的 VC 做图缓存清理（_dispose_vc），即便 torch 栈未加载也不炸。"""
    disposed = []
    monkeypatch.setattr(infer_api, "_VC_CACHE_MAX", 1)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())
    monkeypatch.setattr(infer_api, "_dispose_vc", lambda vc: disposed.append(vc))

    infer_api.get_vc_cached("a.pth")
    infer_api.get_vc_cached("b.pth")

    assert len(disposed) == 1  # a.pth 的 VC 被逐出时清理
    assert set(infer_api._vc_cache) == {"b.pth"}


def test_release_all_vc_clears_cache_and_reports(monkeypatch):
    made = [CountingVC(), CountingVC()]
    it = iter(made)
    monkeypatch.setattr(infer_api, "_build_vc", lambda: next(it))

    infer_api.get_vc_cached("a.pth")
    infer_api.get_vc_cached("b.pth")
    note = infer_api.release_all_vc()

    assert infer_api._vc_cache == {}
    assert note is not None and "2" in note  # 摘要行带释放个数，进任务日志


def test_release_all_vc_empty_returns_none(monkeypatch):
    monkeypatch.setattr(infer_api, "_build_vc", lambda: CountingVC())

    assert infer_api.release_all_vc() is None  # 无可释放时不往任务日志塞噪音行
