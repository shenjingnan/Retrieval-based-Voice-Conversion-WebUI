"""推理编排：缓存 VC 实例，透传参数给 infer.vc.modules.VC.vc_single。"""
import gc
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException
from fastapi.responses import FileResponse

from server import paths

router = APIRouter(prefix="/api")

_vc_lock = threading.Lock()
# 缓存上限（LRU）：每个 entry 驻留一整套 VC（HuBERT + 生成器 + checkpoint + Pipeline，
# 合计 GB 级），而 torch 的 CPU/GPU 缓存分配器只增不减——无上限时一次「逐个试听多个
# 模型」的会话就能把服务进程吃掉几十 GB 提交内存，挤死同进程编排的训练子进程
# （Windows 上表现为「页面文件太小，无法完成操作」）。取 2：单用户「对比两三个音色」
# 的常见来回切换不受影响（tests/test_infer_api.py 的 a→b→a 用例依赖 ≥2）。
_VC_CACHE_MAX = 2
_vc_cache: dict = {}


def _dispose_vc(vc) -> None:
    """释放单个 VC 携带的显存清理链路（CUDA 图缓存挂在其模型属性上，见
    tools/cuda_graph.run_cuda_graph 的 owner 形参；GC 不会自动归还图私有内存池，
    必须显式 clear）。只做图缓存清理与引用声明，不原地改写模型属性——可能存在
    并发进行中的 vc_single（推理在 _vc_lock 外执行），原地置 None 会打断它；
    缓存条目移除后，VC 对象随引用消失由 GC 回收。"""
    graph_mod = sys.modules.get("tools.cuda_graph")
    if graph_mod is None:
        return  # torch 栈从未加载（测试替身 / 纯编排进程）：无图可清
    for value in list(vars(vc).values()):
        graph_mod.clear_cuda_graph_cache(value)


def _reclaim_torch_memory() -> None:
    """逐出后把 torch 缓存分配器持有的空闲块还给系统。sys.modules 门卫：torch 未
    加载（测试）时是 no-op——本模块顶层不得引入 torch（pytest 收集期纪律）。"""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return
    gc.collect()  # 先让被逐出的 VC 真正变成不可达，empty_cache 才有块可还
    torch.cuda.empty_cache()


def release_all_vc() -> str | None:
    """清空全部推理缓存并归还显存（任务启动前的内存让路，server/tasks 的 on_start
    钩子调用）。串行队列保证此刻没有训练在跑；正在进行的单次推理持有 VC 的局部
    引用，不受清空影响，跑完自然释放。返回写进任务日志的摘要行，无可释放时 None。"""
    with _vc_lock:
        if not _vc_cache:
            return None
        entries = list(_vc_cache.values())
        _vc_cache.clear()
    for entry in entries:
        _dispose_vc(entry.vc)
    _reclaim_torch_memory()
    return "已释放推理模型缓存：%d 个（含 HuBERT/生成器/CUDA 图），为子进程腾出内存" % len(entries)


class _CacheEntry:
    __slots__ = ("vc", "loaded")

    def __init__(self, vc):
        self.vc = vc
        self.loaded = None


def _build_vc():
    """构造 VC 实例（测试 patch 点：隔离 torch 导入）。"""
    # 延迟导入：configs.config 在 import 期就探测计算设备，避免 pytest 收集期加载 torch
    from configs.config import Config
    from infer.vc.modules import VC

    return VC(Config())


def get_vc_cached(name: str):
    """缓存 VC 实例并记录已加载的权重名，命中则跳过 get_vc。

    LRU 语义：命中即移到最新端；超过 _VC_CACHE_MAX 时从最旧端逐出（显式清理
    CUDA 图缓存后丢弃引用，见 _dispose_vc / _reclaim_torch_memory）。"""
    with _vc_lock:
        entry = _vc_cache.pop(name, None)  # pop+回插：dict 序即 LRU 序（最新在尾）
        if entry is None:
            # 幂等兜底：绕过 main() 的入口若未设 weight_root，VC.get_vc 会拼出 "None/xxx.pth" 的误导性路径
            os.environ.setdefault("weight_root", str(paths.WEIGHTS_DIR))
            # rmvpe_root 同理：pipeline.get_f0 用 os.environ["rmvpe_root"] 直取，缺了直接 KeyError
            os.environ.setdefault("rmvpe_root", str(paths.ASSETS / "rmvpe"))
            # index_root / outside_index_root 同理：VC.get_vc → get_index_path_from_model
            # 遍历这两个目录做默认索引配对，外面只看 webui.py:25/27 的 setdefault；
            # create_app() 直起（如 E2E 的 uvicorn 工厂调用）时进程里两者皆缺，
            # roots[0] 为 None 会让 os.path.abspath(roots[0]) 抛 TypeError（E2E 实测）
            os.environ.setdefault("index_root", str(paths.INDICES_DIR))
            os.environ.setdefault("outside_index_root", str(paths.INDICES_DIR))
            entry = _CacheEntry(_build_vc())
        if entry.loaded != name:
            entry.vc.get_vc(name)  # 抛异常时 loaded 保持 None，下次自动重试，不污染缓存
            entry.loaded = name
        _vc_cache[name] = entry
        evicted = False
        while len(_vc_cache) > max(_VC_CACHE_MAX, 1):
            oldest = next(iter(_vc_cache))  # 刚插入的 name 恒在末尾，不会被选中
            _dispose_vc(_vc_cache.pop(oldest).vc)
            evicted = True
    if evicted:
        _reclaim_torch_memory()  # 锁外做 gc.collect + empty_cache（可能耗时几十 ms）
    return entry.vc


def _resolve_index_path(index_path: str) -> str:
    """index_path 与 model 同等防御：只接受 INDICES_DIR 下的文件。

    兼容两种合法形态：
    - 纯文件名（相对 INDICES_DIR 解析）
    - GET /api/models 返回的绝对路径（前端原样回传，须落在 INDICES_DIR 内）
    空串保持"不使用索引"的语义。非法来源或文件缺失一律 404，
    避免把任意路径透传给 faiss 后泄漏内部堆栈。

    归属判定只 resolve 父目录、不 resolve 文件本身：仓库布局里 assets/indices
    常是软链，而训练产出的 index 又是链到 logs/{exp}/ 的软链（train_index.py），
    Path.resolve() 会把整条链解析到底、落到 logs 下，is_relative_to(INDICES_DIR)
    因此误判越界——GET /api/models 返回的路径原样回传就会被 404（E2E 实测）。
    父目录 resolve 到 INDICES_DIR 内 + is_file()（跟随文件软链）已足够封住
    「借 INDICES_DIR 之外的真实路径探测任意文件」的口子。
    """
    if not index_path:
        return ""
    candidate = Path(index_path)
    if candidate.name == index_path:
        candidate = paths.INDICES_DIR / index_path
    if not candidate.parent.resolve().is_relative_to(paths.INDICES_DIR.resolve()):
        raise HTTPException(404, f"索引不存在: {index_path}")
    if not candidate.is_file():
        raise HTTPException(404, f"索引不存在: {index_path}")
    return str(candidate)


@router.post("/infer")
def infer(
    audio: bytes = File(...),
    model: str = Form(...),
    transpose: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
    index_path: str = Form(""),
):
    # basename 校验防路径穿越（如 "../x.pth"），通过后再查存在性
    if Path(model).name != model or not (paths.WEIGHTS_DIR / model).is_file():
        raise HTTPException(404, f"模型不存在: {model}")

    resolved_index = _resolve_index_path(index_path)

    # src/dst 落在系统临时目录，带可辨识前缀 + uuid 防并发冲突；单用户规模不设后台清理
    src = Path(tempfile.gettempdir()) / f"rvc_in_{uuid.uuid4().hex}.wav"
    src.write_bytes(audio)
    dst = Path(tempfile.gettempdir()) / f"rvc_out_{uuid.uuid4().hex}.wav"
    try:
        vc = get_vc_cached(model)
        # 推理在锁外执行，hubert 懒加载并发首请求会重复加载但不致损坏
        info, result = vc.vc_single(
            0, str(src), transpose, f0_method, resolved_index or None,
            index_rate, resample_sr, rms_mix_rate, protect,
        )
        if result is None or result[1] is None:
            raise HTTPException(500, f"推理失败: {info}")
        sr, audio_opt = result
        sf.write(dst, audio_opt, sr)
        return FileResponse(dst, media_type="audio/wav", filename="converted.wav")
    except FileNotFoundError as e:
        # 兜底网而非主校验路径：模型存在性已前置检查，这里兜 get_vc / vc_single 内部抛出的缺失
        raise HTTPException(404, str(e))
