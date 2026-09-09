"""整机系统资源快照（只读遥测）：GPU 走 nvidia-smi 子进程，内存/CPU 走 psutil，
磁盘走 shutil.disk_usage。一期只做整机视角，不做训练子进程 per-process 视角
（将来可用 nvidia-smi --query-compute-apps 按任务 PID 过滤，见 server.tasks 的进程句柄）。

依赖纪律：
- 绝不 import torch（server 侧延迟导入纪律，见 server.api.training 模块 docstring）。
- psutil 延迟导入（_psutil）：未安装只让内存/CPU 两项显示「不可用」，不得让
  create_app() 失败——部署环境（Windows runtime\\ 与 .venv）可能没重装依赖。
- 各采集项独立 try/except 并降级为 null / 空列表：本端点被前端 3s 轮询
  （frontend/src/hooks/useSystemStats.ts），任何一项失败都不应把整页打成错误，
  因此 handler 层不设 try/except（也就没有 500）。
"""
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any

from fastapi import APIRouter

from server import paths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system")

# nvidia-smi 典型耗时 50-300ms，Windows WDDM/多卡冷启动可到 ~1s；取 2s 覆盖冷启动，
# 同时把最坏情况钉在一个轮询周期（3s）以内——子进程挂死（驱动异常）时必须靠超时兜底，
# 否则每次轮询占死一个线程池线程（FastAPI 同步 def 在 anyio 线程池执行）。
_NVIDIA_SMI_TIMEOUT = 2.0
_NVIDIA_SMI_ARGS = (
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
# nounits 输出形如 "0, NVIDIA GeForce RTX 4090, 12, 2048, 24564"（每卡一行）。
# 数值字段允许的形态白名单；[N/A] / [Not Supported] / N/A 一律按「该字段缺失」处理
_NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_MIB = 1024 * 1024

# psutil 延迟加载的模块级缓存：None 表示「加载过且失败」，与「还没试过」用
# _psutil_checked 区分（psutil 缺失是稳定事实，不能每帧重试 import）
_psutil_module = None
_psutil_checked = False


def _psutil():
    """延迟加载 psutil，返回模块或 None。加载成功即做一次 cpu_percent(None) 预热：
    psutil 的非阻塞采样是「相对上一次调用」的差值，不预热的话首个有效样本会是
    进程启动以来的均值（几乎恒等于一个很小的数）。预热后首个请求帧读到 ~0，
    第 2 帧（3s 后）起就是正常的 3s 窗口值，前端无需特判。"""
    global _psutil_module, _psutil_checked
    if not _psutil_checked:
        _psutil_checked = True
        try:
            import psutil
        except Exception as exc:  # ImportError，以及极端环境下的加载期 OSError
            logger.warning("psutil 不可用，内存/CPU 将显示为「不可用」：%s", exc)
        else:
            psutil.cpu_percent(interval=None)  # 预热调用，丢弃返回值
            _psutil_module = psutil
    return _psutil_module


def _to_number(raw: str) -> float | None:
    text = raw.strip()
    # WDDM 驱动不支持的字段输出 "[N/A]" / "[Not Supported]"（nounits 也拦不住）；
    # 个别驱动输出裸 "N/A"。统一按「该字段缺失」处理，绝不弃掉整卡
    if not _NUMBER_RE.match(text):
        return None
    return float(text)


def _parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """纯函数（单测直接喂字符串，不碰子进程）。从两端切分以容忍产品名内出现逗号：
    parts[0]=index，parts[-3:]=util/used/total，中间全归 name。"""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:  # 空行 / 残缺行
            continue
        index = _to_number(parts[0])
        name = ", ".join(parts[1:-3])
        if index is None or not name:
            continue
        used = _to_number(parts[-2])
        total = _to_number(parts[-1])
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "utilization_percent": _to_number(parts[-3]),
                # 单位契约是 bytes（与 client.ts 既有 total_bytes 口径一致），MiB 只在
                # 这里换算成整数 bytes，GiB 化留给前端展示层
                "memory_used_bytes": int(used * _MIB) if used is not None else None,
                "memory_total_bytes": int(total * _MIB) if total is not None else None,
            }
        )
    return gpus


_last_gpu_note: str | None = None


def _note_gpu_failure(reason: str) -> None:
    """3s 轮询下同一原因会刷屏（20 条/分钟），只在原因变化时记一次 WARNING。"""
    global _last_gpu_note
    if reason != _last_gpu_note:
        _last_gpu_note = reason
        logger.warning("GPU 资源不可用（前端将显示「不可用」）：%s", reason)


def _read_gpus() -> list[dict[str, Any]]:
    try:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            # macOS / 未装驱动的 Windows 下的常态，不算异常，不记日志
            return []
        completed = subprocess.run(
            [executable, *_NVIDIA_SMI_ARGS],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_NVIDIA_SMI_TIMEOUT,
            check=False,
            # tools/pymss_webui.py 同款：Windows 下防止每次轮询弹出控制台窗口
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        _note_gpu_failure(f"nvidia-smi 超时（>{_NVIDIA_SMI_TIMEOUT}s）")
        return []
    except OSError as exc:
        _note_gpu_failure(f"nvidia-smi 无法启动：{exc}")
        return []
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _note_gpu_failure(f"nvidia-smi 返回码 {completed.returncode}：{detail}")
        return []
    gpus = _parse_nvidia_smi(completed.stdout.decode("utf-8", errors="replace"))
    if not gpus:
        _note_gpu_failure("nvidia-smi 输出无法解析")
    return gpus


def _read_memory() -> dict[str, int] | None:
    psutil = _psutil()
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
    except Exception as exc:  # psutil 在受限容器/权限下会抛 AccessDenied 等
        logger.warning("内存采集失败：%s", exc)
        return None
    # 用 total - available 而非 vm.used：Linux 上 free 被 page cache 吃掉，
    # vm.used 会虚高到 ~95%；available 与 psutil.percent / 任务管理器同口径
    return {"used_bytes": vm.total - vm.available, "total_bytes": vm.total}


def _read_cpu() -> dict[str, Any] | None:
    psutil = _psutil()
    if psutil is None:
        return None
    try:
        return {
            "percent": round(psutil.cpu_percent(interval=None), 1),
            "count": psutil.cpu_count(logical=True) or os.cpu_count() or 0,
        }
    except Exception as exc:
        logger.warning("CPU 采集失败：%s", exc)
        return None


def _read_disk() -> dict[str, int] | None:
    try:
        # 模块属性访问（paths.ROOT）而非 from-import，维持测试 monkeypatch 纪律
        # （见 server/api/datasets.py 模块 docstring）；ROOT 是仓库根目录且恒存在，
        # logs/ 与 datasets/ 都在其下，比单取某个子目录稳
        usage = shutil.disk_usage(paths.ROOT)
    except OSError as exc:
        logger.warning("磁盘采集失败：%s", exc)
        return None
    return {
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }


@router.get("/stats")
def system_stats():
    """整机资源快照。永不 500：各采集项独立降级（cpu/memory/disk → null，gpus → []）。"""
    return {
        "timestamp": time.time(),
        "cpu": _read_cpu(),
        "memory": _read_memory(),
        "disk": _read_disk(),
        "gpus": _read_gpus(),
    }
