"""训练子进程命令拼装（移植 webui.py 的命令模板，逐字对齐）。

只做字符串拼装与底模路径探测，不 import configs/torch（pytest 收集期不得加载 torch）。
所有命令均在 cwd=仓库根 下执行，因此模板中的相对路径（train/...、assets/...）与
get_pretrained_paths 返回的相对底模路径都按仓库根解析（webui 用 now_dir=os.getcwd()，同源）。

模板权威来源（单说话人分支）：
- webui.py:867-877   train/preprocess.py（无 multispeaker manifest 段）
- webui.py:973-979   train/dataset/extract_f0.py 的 CPU 分支
- webui.py:1028-1036 train/dataset/extract_hubert_feature.py 的无 GPU 分支（7 参数形态靠 argc 区分）
- webui.py:1308-1346 train/train.py 的无 -g 分支
- webui.py:1460-1470 train/train_index.py
"""
import logging
import sys
import threading
from typing import Literal

from server import paths

logger = logging.getLogger(__name__)

# webui.py:279-283 的 sr_dict：preprocess.py 内部 int(sys.argv[2])
SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}

# 用户可见的采样率档位（接口层用），与 SR_DICT 的键一致
SampleRate = Literal["32k", "40k", "48k"]

# webui.py:25 os.environ.setdefault("outside_index_root", "assets/indices")，
# train_index.py 会把索引链接到该目录，与 P1 /api/models 的扫描目录闭环；
# 由 paths.INDICES_DIR 派生，避免与 paths.py 双源漂移
INDEX_ROOT = paths.INDICES_DIR.relative_to(paths.ROOT).as_posix()

_device: str | None = None
_is_half: bool | None = None
#: 保护「屏蔽 sys.argv → 构造 Config → 恢复」全程（并发首请求下不加锁会互相覆盖恢复值）
_config_lock = threading.Lock()


def _load_config():
    """构造 configs.config.Config（单例，首次才加载 torch 并探测设备）。

    Config.arg_parse() 内部用 argparse.parse_args() 解析进程命令行——webui 自己是入口
    进程没问题，但本服务可能由 `uvicorn server.main:app --port 7861` 等方式拉起，附加
    参数会让 parse_args 直接 SystemExit，首个拼 extract/pipeline 命令的请求被打挂。
    因此构造期间把 sys.argv 屏蔽到只剩程序名，且全程持锁：并发首请求若不加锁，一个
    线程会在另一线程的屏蔽窗口外构造（看到未屏蔽 argv），或把被截断的 argv 当原样
    恢复，sys.argv 从此被永久截断。"""
    from configs.config import Config  # 延迟导入：见模块 docstring

    with _config_lock:
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            return Config()
        finally:
            sys.argv = argv


def _resolve_device() -> str:
    """惰性取 configs.config.Config().device，首次调用才触发 torch 加载。"""
    global _device
    if _device is None:
        _device = str(_load_config().device)
    return _device


def resolve_is_half() -> bool:
    """webui 的 config.is_half 同源（extract_hubert_feature.py 的最后一个参数）。
    Config 是单例，与 _resolve_device 共享同一次 torch 探测开销；公开为函数是因为
    编排层（server/api/training.py）拼 extract 命令时也要取它。"""
    global _is_half
    if _is_half is None:
        _is_half = bool(_load_config().is_half)
    return _is_half


def build_preprocess_cmd(dataset_dir, sr: int, n_p, exp, noparallel, per) -> str:
    """sr 传 int（调用方先经 SR_DICT 转换，webui.py:853 同源），与 build_fit_cmd 的
    SampleRate 档位串不同轨；noparallel 传 bool（%s → True/False）；
    per 用 %.1f 对齐 webui 模板（preprocess.py 里 float(sys.argv[6])）。"""
    return '"%s" train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        sys.executable,
        dataset_dir,
        sr,
        n_p,
        paths.ROOT,
        exp,
        noparallel,
        per,
    )


def build_extract_f0_cmd(exp, n_p, f0_method) -> str:
    """CPU 分支（webui.py:973-979）；rmvpe 多卡多进程分支不在 P2 范围。"""
    return '"%s" train/dataset/extract_f0.py cpu "%s/logs/%s" %s %s' % (
        sys.executable,
        paths.ROOT,
        exp,
        n_p,
        f0_method,
    )


def build_extract_hubert_cmd(exp, version, is_half) -> str:
    """无 GPU 分支，7 参数形态（extract_hubert_feature.py 靠 argc 区分）；
    is_half 传 bool → "True"/"False"，脚本内 .lower() == "true"。"""
    return '"%s" train/dataset/extract_hubert_feature.py %s 1 0 "%s/logs/%s" %s %s' % (
        sys.executable,
        _resolve_device(),
        paths.ROOT,
        exp,
        version,
        is_half,
    )


def build_fit_cmd(
    exp, sr: SampleRate, if_f0, batch_size, total_epoch, save_every, save_every_weights,
    pretrain_G, pretrain_D, *, version,
) -> str:
    """webui.py:1329-1346 无 -g 分支；-l/-c 固定 0（P2 不做 latest/cache 选项）。
    -pg/-pd 仅在非空串时拼入（与 webui 的条件 %s 等价，仅去掉它留下的双空格）。
    version 必传（对应 webui 的 version19 下拉框值，API 层来自请求体）。"""
    cmd = '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s' % (
        sys.executable,
        exp,
        sr,
        1 if if_f0 else 0,
        batch_size,
        total_epoch,
        save_every,
    )
    if pretrain_G:
        cmd += " -pg %s" % pretrain_G
    if pretrain_D:
        cmd += " -pd %s" % pretrain_D
    cmd += " -l 0 -c 0 -sw %s -v %s" % (1 if save_every_weights else 0, version)
    return cmd


def build_index_cmd(exp, version, n_cpu) -> str:
    """末位传 single：显式单说话人模式（webui 非多说话人训练时传 single/auto）。"""
    return '"%s" train/train_index.py "%s" %s "%s" %s single' % (
        sys.executable,
        exp,
        version,
        INDEX_ROOT,
        n_cpu,
    )


def build_fitprep_cmd(exp, sr: SampleRate, version, if_f0) -> str:
    """pipeline 专用：把 fit 的 Python 前置（filelist/config/产物校验/底模提示）作为
    一个子进程 cmd 插在 extract 之后、train.py 之前。

    为什么是子进程而不是任务级 setup 钩子：pipeline 是单任务多 cmd，setup 钩子只在
    第一个 cmd 之前跑一次，无法满足「filelist 必须在 preprocess/extract 完成后生成」
    的时序；`python -m server.api.training` 依赖 cwd=仓库根 解析包（tasks.py 的 Popen
    正是 cwd=仓库根），代价是一次无 torch 的解释器启动（每个 pipeline 一次）。"""
    return '"%s" -m server.api.training fitprep "%s" %s %s %s' % (
        sys.executable,
        exp,
        sr,
        version,
        1 if if_f0 else 0,
    )


def build_precheck_cmd(exp) -> str:
    """pipeline 专用：preprocess 完成后立刻校验切分产物（分步流程里由 extract 的
    setup 钩子做同样的检查），让空数据集在进入耗时的特征提取前就失败。"""
    return '"%s" -m server.api.training precheck "%s"' % (sys.executable, exp)


def pretrained_rel_paths(sr: SampleRate, if_f0, version):
    """底模的相对路径串（不查存在性）：`assets/pretrained{path_str}/{f0}{G|D}{sr}.pth`。
    供 get_pretrained_paths 与编排层的「未使用底模」提示共用，避免两处拼法漂移。"""
    path_str = "" if version == "v1" else "_v2"
    if version == "v1" and sr == "32k":  # webui.py:1120-1123：v1 无 32k 底模，一律取 40k
        sr = "40k"
    f0_str = "f0" if if_f0 else ""
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr),
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr),
    )


def get_pretrained_paths(sr: SampleRate, if_f0, version):
    """移植 webui.py:1083-1146 get_pretrained_models 与 :1118-1146 的 path_str/f0_str/sr 组合。
    存在性检查按仓库根（paths.ROOT）解析，不随进程 cwd 漂移（否则从其他目录起服务会静默退化成
    从零训练）；返回值仍为相对仓库根的路径串（命令在 cwd=仓库根 的子进程里执行）。
    任一缺失返回空串，调用方不拼 -pg/-pd，并须把缺失信息写进任务日志。"""
    generator_rel, discriminator_rel = pretrained_rel_paths(sr, if_f0, version)
    has_g = (paths.ROOT / generator_rel).exists()
    has_d = (paths.ROOT / discriminator_rel).exists()
    if not has_g:
        logger.warning("生成器预训练模型不存在，将不使用：%s", generator_rel)
    if not has_d:
        logger.warning("判别器预训练模型不存在，将不使用：%s", discriminator_rel)
    return (generator_rel if has_g else "", discriminator_rel if has_d else "")
