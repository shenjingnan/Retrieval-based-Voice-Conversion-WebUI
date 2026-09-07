"""训练命令拼装快照测试：与 webui.py 命令模板逐字对齐（cwd=仓库根）。"""
import subprocess
import sys
import threading
import time
from types import ModuleType

from server import commands, paths
from server.commands import (
    SR_DICT,
    build_extract_f0_cmd,
    build_extract_hubert_cmd,
    build_fit_cmd,
    build_index_cmd,
    build_preprocess_cmd,
    get_pretrained_paths,
)

PY = sys.executable
ROOT = str(paths.ROOT)


def test_sr_dict():
    assert SR_DICT == {"32k": 32000, "40k": 40000, "48k": 48000}


def test_build_preprocess_cmd_snapshot():
    # webui.py:867-877（单说话人无 manifest 段）；sr 已由 SR_DICT 转为 int
    cmd = build_preprocess_cmd("/data/ds", 40000, 8, "mi-test", False, 3.7)
    assert cmd == (
        f'"{PY}" train/preprocess.py "/data/ds" 40000 8 "{ROOT}/logs/mi-test" False 3.7'
    )


def test_build_preprocess_cmd_noparallel_true():
    cmd = build_preprocess_cmd("d", 32000, 4, "e", True, 3.0)
    assert cmd.endswith('" True 3.0')


def test_build_extract_f0_cmd_snapshot():
    # webui.py:973-979（无 rmvpe GPU 卡时的 CPU 分支）
    cmd = build_extract_f0_cmd("mi-test", 8, "rmvpe")
    assert cmd == f'"{PY}" train/dataset/extract_f0.py cpu "{ROOT}/logs/mi-test" 8 rmvpe'


def test_build_extract_hubert_cmd_snapshot(monkeypatch):
    # webui.py:1028-1036（无 GPU 分支，7 参数形态）；device 延迟解析，测试注入避免加载 torch
    monkeypatch.setattr("server.commands._resolve_device", lambda: "cpu")
    cmd = build_extract_hubert_cmd("mi-test", "v2", False)
    assert cmd == (
        f'"{PY}" train/dataset/extract_hubert_feature.py cpu 1 0 "{ROOT}/logs/mi-test" v2 False'
    )


def test_build_extract_hubert_cmd_device_injected(monkeypatch):
    monkeypatch.setattr("server.commands._resolve_device", lambda: "mps")
    cmd = build_extract_hubert_cmd("mi-test", "v1", True)
    assert "extract_hubert_feature.py mps 1 0 " in cmd
    assert cmd.endswith('logs/mi-test" v1 True')


def test_resolve_device_memoized(monkeypatch):
    """device 缓存生效：注入假 configs.config（不加载 torch），Config 只被构造一次。"""
    calls = []

    class FakeConfig:
        device = "mps"

        def __init__(self):
            calls.append(1)

    fake_module = ModuleType("configs.config")
    fake_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "configs.config", fake_module)
    monkeypatch.setattr(commands, "_device", None)  # 结束后由 monkeypatch 还原

    assert commands._resolve_device() == "mps"
    assert commands._resolve_device() == "mps"
    assert len(calls) == 1


def test_config_probe_ignores_foreign_argv(monkeypatch):
    """uvicorn/pytest 的附加命令行参数不能让 Config 的 argparse 退出（否则首次拼
    extract / pipeline 命令的请求直接 SystemExit）。"""
    seen = {}

    class FakeConfig:
        device = "cpu"
        is_half = False

        def __init__(self):
            seen["argv"] = list(sys.argv)

    fake_module = ModuleType("configs.config")
    fake_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "configs.config", fake_module)
    monkeypatch.setattr(commands, "_device", None)
    monkeypatch.setattr(commands, "_is_half", None)
    monkeypatch.setattr(sys, "argv", ["pytest", "-q", "--port", "7861"])

    assert commands.resolve_is_half() is False
    assert commands._resolve_device() == "cpu"
    assert seen["argv"] == ["pytest"]  # 附加参数被屏蔽
    assert sys.argv == ["pytest", "-q", "--port", "7861"]  # 构造后原样还原


def test_build_fit_cmd_with_pretrained_snapshot():
    cmd = build_fit_cmd(
        "mi-test", "48k", True, 8, 20, 5, True,
        "assets/pretrained_v2/f0G48k.pth", "assets/pretrained_v2/f0D48k.pth",
        version="v2",
    )
    assert cmd == (
        f'"{PY}" train/train.py -e "mi-test" -sr 48k -f0 1 -bs 8 -te 20 -se 5'
        " -pg assets/pretrained_v2/f0G48k.pth"
        " -pd assets/pretrained_v2/f0D48k.pth"
        " -l 0 -c 0 -sw 1 -v v2"
    )


def test_build_fit_cmd_without_pretrained():
    cmd = build_fit_cmd("e", "48k", False, 8, 20, 5, False, "", "", version="v2")
    assert cmd == (
        f'"{PY}" train/train.py -e "e" -sr 48k -f0 0 -bs 8 -te 20 -se 5'
        " -l 0 -c 0 -sw 0 -v v2"
    )


def test_build_fit_cmd_no_gpus_flag():
    # P2 单进程编排：不拼 -g（webui 的多卡分支不在移植范围）
    cmd = build_fit_cmd("e", "40k", True, 4, 10, 5, True, "", "", version="v1")
    assert " -g " not in cmd and not cmd.endswith(" -g")
    assert cmd.endswith(" -sw 1 -v v1")


def test_build_index_cmd_snapshot():
    # webui.py:1460-1470；outside_index_root 恒为 assets/indices（webui.py:25）
    cmd = build_index_cmd("mi-test", "v2", 8)
    assert cmd == f'"{PY}" train/train_index.py "mi-test" v2 "assets/indices" 8 single'


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_get_pretrained_paths_exist(monkeypatch, tmp_path):
    _touch(tmp_path / "assets/pretrained_v2/f0G48k.pth")
    _touch(tmp_path / "assets/pretrained_v2/f0D48k.pth")
    monkeypatch.setattr(paths, "ROOT", tmp_path)  # 按 ROOT 解析，不受进程 cwd 影响
    g, d = get_pretrained_paths("48k", True, "v2")
    assert g == "assets/pretrained_v2/f0G48k.pth"
    assert d == "assets/pretrained_v2/f0D48k.pth"


def test_get_pretrained_paths_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    assert get_pretrained_paths("48k", True, "v2") == ("", "")


def test_get_pretrained_paths_partial(monkeypatch, tmp_path):
    # 只有 G 存在：D 返回空串，命令里不带 -pd（webui get_pretrained_models 语义）
    _touch(tmp_path / "assets/pretrained_v2/G48k.pth")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    g, d = get_pretrained_paths("48k", False, "v2")
    assert g == "assets/pretrained_v2/G48k.pth"
    assert d == ""


def test_get_pretrained_paths_v1_32k_forces_40k(monkeypatch, tmp_path):
    # webui.py:1120-1123 change_version19：v1 + 32k 底模一律取 40k
    _touch(tmp_path / "assets/pretrained/f0G40k.pth")
    _touch(tmp_path / "assets/pretrained/f0D40k.pth")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    g, d = get_pretrained_paths("32k", True, "v1")
    assert (g, d) == ("assets/pretrained/f0G40k.pth", "assets/pretrained/f0D40k.pth")


def test_get_pretrained_paths_v1_uses_plain_dir(monkeypatch, tmp_path):
    _touch(tmp_path / "assets/pretrained/f0G48k.pth")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    g, _ = get_pretrained_paths("48k", True, "v1")
    assert g == "assets/pretrained/f0G48k.pth"


def test_modules_do_not_load_torch_or_configs():
    """延迟导入纪律：pytest 收集期 import 本模块不得加载 torch/configs。"""
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import server.commands;"
        "assert 'torch' not in sys.modules, 'torch 被加载';"
        "assert 'configs.config' not in sys.modules, 'configs 被加载';"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(paths.ROOT), capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_config_probe_is_thread_safe(monkeypatch):
    """并发首调：屏蔽→构造→恢复必须全程互斥。无锁时各线程互相把被截断的 argv 当原样
    恢复，sys.argv 被永久截断（uvicorn 启动方式下后续请求会连锁失败）。

    场景循环 10 轮：4 个线程能否同时悬在屏蔽窗口里取决于调度，单轮存在「恰好串行化、
    无覆盖」的漏检可能；10 轮内无锁实现必然复现污染，有锁实现只是多花几秒等待超时。"""
    original = ["pytest", "--port", "7861"]

    for round_index in range(10):
        entered = []

        class FakeConfig:
            device = "cpu"
            is_half = False

            def __init__(self):
                entered.append(list(sys.argv))
                # 等到 4 个线程都进入构造再放行：无锁实现会同时悬在各自的屏蔽窗口里，
                # 恢复顺序必然互相覆盖；有锁实现则串行进入，各自等到超时即可
                deadline = time.monotonic() + 0.05
                while len(entered) < 4 and time.monotonic() < deadline:
                    time.sleep(0.005)

        fake_module = ModuleType("configs.config")
        fake_module.Config = FakeConfig
        monkeypatch.setitem(sys.modules, "configs.config", fake_module)
        monkeypatch.setattr(sys, "argv", list(original))

        barrier = threading.Barrier(4)
        errors = []

        def worker():
            try:
                barrier.wait(timeout=5)
                commands._load_config()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)

        assert errors == [], round_index
        assert len(entered) == 4, round_index
        assert all(argv == ["pytest"] for argv in entered), (round_index, entered)
        assert sys.argv == original, (round_index, sys.argv)  # 未被截断/污染


# ---------------------------------------------------------------------------
# batch_size 设备默认（webui.py:175-183 滑条预填逻辑）
# ---------------------------------------------------------------------------


def _fake_gpu_config(monkeypatch, *, memory):
    """注入假 configs.config 的模块级 GPU 探测结果。
    GPU_MEMORY 的值带 +0.4 容差（config.py：total_memory / 1024**3 + 0.4），
    8 GiB 卡在该表里约是 8.4。"""
    fake_module = ModuleType("configs.config")
    fake_module.IS_GPU = bool(memory)
    fake_module.GPU_INDEX = set(memory)
    fake_module.GPU_MEMORY = memory
    monkeypatch.setitem(sys.modules, "configs.config", fake_module)


def test_resolve_default_batch_size_gpu_8gb(monkeypatch):
    _fake_gpu_config(monkeypatch, memory={0: 8.4})
    assert commands.resolve_default_batch_size() == 4


def test_resolve_default_batch_size_multi_gpu_takes_min(monkeypatch):
    # webui：min(GPU_MEMORY[i] for i in sorted(GPU_INDEX))——按最小的那张卡算
    _fake_gpu_config(monkeypatch, memory={1: 24.4, 0: 8.4})
    assert commands.resolve_default_batch_size() == 4


def test_resolve_default_batch_size_no_gpu(monkeypatch):
    _fake_gpu_config(monkeypatch, memory={})
    assert commands.resolve_default_batch_size() == 1


def test_resolve_default_batch_size_tiny_gpu_floor_at_one(monkeypatch):
    # max(1, ...)：小显存（1 GiB → 1 // 2 = 0）不得给出 0（webui 同款下限）
    _fake_gpu_config(monkeypatch, memory={0: 1.4})
    assert commands.resolve_default_batch_size() == 1


def test_default_batch_size_note_gpu(monkeypatch):
    _fake_gpu_config(monkeypatch, memory={0: 12.4, 1: 8.4})
    assert commands.default_batch_size_note(4) == (
        "batch_size 未指定，按设备默认使用 4（可用显卡最小显存 8.4 GB ÷ 2）"
    )


def test_default_batch_size_note_no_gpu(monkeypatch):
    _fake_gpu_config(monkeypatch, memory={})
    assert commands.default_batch_size_note(1) == (
        "batch_size 未指定，按设备默认使用 1（无可用显卡）"
    )
