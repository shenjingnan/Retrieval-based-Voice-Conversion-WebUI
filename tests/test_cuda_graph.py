"""tools/cuda_graph 策略用例：帧数上限、启用判定、捕获 OOM 逐出重试，以及
server/main.py 默认禁图与 webui.py:17-18 的同源约束。torch 一律函数内延迟导入
（pytest 收集期纪律，见 server/api/infer.py._reclaim_torch_memory）。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_max_capture_frames_env(monkeypatch):
    from tools import cuda_graph as cg

    monkeypatch.delenv(cg.MAX_FRAMES_ENV, raising=False)
    assert cg.max_capture_frames() == cg.DEFAULT_MAX_CAPTURE_FRAMES
    monkeypatch.setenv(cg.MAX_FRAMES_ENV, "500")
    assert cg.max_capture_frames() == 500
    monkeypatch.setenv(cg.MAX_FRAMES_ENV, "abc")
    assert cg.max_capture_frames() == cg.DEFAULT_MAX_CAPTURE_FRAMES


def test_applies_respects_frame_limit(monkeypatch):
    import torch

    from tools import cuda_graph as cg

    monkeypatch.setattr(cg, "cuda_graph_enabled", lambda device: True)
    short = torch.zeros(1, 3000, 768)
    too_long = torch.zeros(1, 3001, 768)

    monkeypatch.setenv(cg.MAX_FRAMES_ENV, "3000")
    assert cg.cuda_graph_applies("cuda", short) is True
    assert cg.cuda_graph_applies("cuda", too_long) is False

    monkeypatch.setenv(cg.MAX_FRAMES_ENV, "0")  # 0 = 不设上限
    assert cg.cuda_graph_applies("cuda", too_long) is True


def test_applies_requires_enabled(monkeypatch):
    import torch

    from tools import cuda_graph as cg

    monkeypatch.setattr(cg, "cuda_graph_enabled", lambda device: False)
    assert cg.cuda_graph_applies("cuda", torch.zeros(1, 4, 3)) is False


def _fake_call_cls(oom_times, log):
    """替身 _CapturedCall：前 oom_times 次构造抛 OOM，replay 走原函数、不碰 CUDA 流。"""

    class _FakeCall:
        capture_ms = 1.0

        def __init__(self, function, inputs):
            self.function = function
            self.inputs = inputs
            log.append("capture")
            if log.count("capture") <= oom_times:
                import torch

                raise torch.OutOfMemoryError("fake OOM")

        def replay(self, inputs):
            log.append("replay")
            return self.function(*inputs)

    return _FakeCall


def _oom_recorder(monkeypatch):
    """替换 torch.cuda.empty_cache 并记录调用次数。"""
    import torch

    calls = []

    def _record():
        calls.append(1)

    monkeypatch.setattr(torch.cuda, "empty_cache", _record)
    return calls


def test_capture_oom_evicts_pools_then_retries(monkeypatch):
    import torch

    from tools import cuda_graph as cg

    log = []
    emptied = _oom_recorder(monkeypatch)
    monkeypatch.setattr(cg, "_CapturedCall", _fake_call_cls(oom_times=1, log=log))
    monkeypatch.setenv(cg.MAX_CACHE_ENV, "2")

    cache = cg._GraphCache()
    inputs = (torch.zeros(1, 4, 3),)
    output = cache.run(("ns",), lambda t: "graph-result", inputs)

    assert output == "graph-result"
    assert log == ["capture", "capture", "replay"]  # 第一次 OOM，逐出后重试成功
    assert emptied == [1]
    assert len(cache.entries) == 1
    # 第二次同形状命中缓存，不再捕获
    cache.run(("ns",), lambda t: "graph-result", inputs)
    assert log == ["capture", "capture", "replay", "replay"]


def test_capture_double_oom_blacklists_and_falls_back(monkeypatch):
    import torch

    from tools import cuda_graph as cg

    log = []
    _oom_recorder(monkeypatch)
    monkeypatch.setattr(cg, "_CapturedCall", _fake_call_cls(oom_times=2, log=log))
    monkeypatch.setenv(cg.MAX_CACHE_ENV, "2")

    cache = cg._GraphCache()
    inputs = (torch.zeros(1, 4, 3),)
    output = cache.run(("ns",), lambda t: "eager-result", inputs)

    assert output == "eager-result"
    assert log == ["capture", "capture"]  # 两次捕获均 OOM，逐出后仍失败
    assert len(cache.failures) == 1
    assert cache.fallback_count == 1
    assert not cache.entries
    # 拉黑后同形状直接 eager，不再尝试捕获
    cache.run(("ns",), lambda t: "eager-result", inputs)
    assert log == ["capture", "capture"]
    assert cache.fallback_count == 2


def test_capture_non_oom_error_blacklists_without_retry(monkeypatch):
    from tools import cuda_graph as cg

    class _BrokenCall:
        capture_ms = 1.0

        def __init__(self, function, inputs):
            raise RuntimeError("capture unsupported")

    emptied = _oom_recorder(monkeypatch)
    monkeypatch.setattr(cg, "_CapturedCall", _BrokenCall)

    cache = cg._GraphCache()
    import torch

    output = cache.run(("ns",), lambda t: "eager-result", (torch.zeros(1, 4, 3),))

    assert output == "eager-result"
    assert emptied == []  # 非 OOM 失败不逐出池
    assert len(cache.failures) == 1
    assert cache.fallback_count == 1


def test_server_build_vc_defaults_cuda_graph_off():
    """server/api/infer.py._build_vc 与 webui.py:17-18 同源：默认 eager，
    RVC_OFFLINE_CUDA_GRAPH=1 才开图。纯源码断言——真实 _build_vc 会拉起 torch 并
    探测设备，且单测进程禁写 os.environ（模块级写回会污染 test_tasks 对非训练
    命令子进程环境的断言，全量跑序下实测）。"""
    text = (REPO_ROOT / "server" / "api" / "infer.py").read_text(encoding="utf-8")
    assert 'os.environ.get("RVC_OFFLINE_CUDA_GRAPH", "0") == "1"' in text
    assert 'os.environ["RVC_CUDA_GRAPH"]' in text
