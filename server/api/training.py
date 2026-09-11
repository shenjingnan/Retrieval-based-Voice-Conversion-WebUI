"""训练编排：filelist / config.json 生成（纯函数）+ 4 步与一键训练路由 + 任务查询。

移植来源：
- webui.py:1169-1307 run_train_model 的 filelist + config 生成段
- webui.py:692-720   validate_preprocess_outputs / validate_feature_outputs
- webui.py:1118-1123 change_version19 的 v1+32k → 40k 归一化

延迟导入纪律：本模块不 import configs/torch——config 模板直接读 configs/v*/{sr}.json
文件（webui 经 config.json_config[config_path] 读的是同一批文件），i18n 文案取 zh 默认值；
device/is_half 经 server.commands 的惰性解析函数取（首次调用才加载 torch）。

同时是 pipeline 的 fit 前置子进程入口：`python -m server.api.training fitprep ...`
（cwd=仓库根，见 server.commands.build_fitprep_cmd）。
"""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from random import shuffle  # 与 webui.py:76 同源，测试可注入替身

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server import paths
from server.commands import (  # noqa: F401  SampleRate 仅用于类型注解
    SR_DICT,
    SampleRate,
    build_extract_f0_cmd,
    build_extract_hubert_cmd,
    build_fit_cmd,
    build_fitprep_cmd,
    build_index_cmd,
    build_precheck_cmd,
    build_preprocess_cmd,
    default_batch_size_note,
    get_pretrained_paths,
    pretrained_rel_paths,
    resolve_default_batch_size,
    resolve_is_half,
)
from server.progress import parse_stage_progress_line, parse_train_line
from server.task_store import TaskHistoryStore
from server.tasks import (
    PENDING,
    RESTORE_FACTORIES,
    SUCCESS,
    TERMINAL_STATES,
    task_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# webui 的 config.preprocess_per / config.noparallel 同源默认值（P2 不暴露调参入口；
# webui 在 CPU/小显存机器上会把 per 降到 3.0，这里按计划规格固定默认 3.7）
PREPROCESS_PER = 3.7
NOPARALLEL = False

# 异常分类规则：webui 抛 RuntimeError 的照抄（validate_preprocess_outputs /
# validate_feature_outputs）；唯一新增分类是 generate_filelist 的 ValueError
# （数据未就绪，属用户可修复状态），编排层两类都要接住并转成任务失败。
NO_VALID_AUDIO_MSG = "没有可用于训练的有效音频，请先完成数据切分和特征提取"

# webui.py:1207-1208：v1 用 256 维特征，v2 用 768 维
FEATURE_DIM = {"v1": 256, "v2": 768}

# ---------------------------------------------------------------------------
# 请求模型与参数校验
# ---------------------------------------------------------------------------


class PreprocessBody(BaseModel):
    exp_name: str
    dataset_dir: str
    sr: str = "40k"  # 对齐 webui 默认（40k 底模最稳，webui.py:2436-2445）
    n_p: int | None = None  # 缺省取 os.cpu_count()（webui 的 config.n_cpu 同源）


class ExtractBody(BaseModel):
    exp_name: str
    f0_method: str = "rmvpe"
    version: str = "v2"
    if_f0: bool = True


class FitBody(BaseModel):
    exp_name: str
    sr: str
    version: str
    if_f0: bool
    total_epoch: int
    save_every_epoch: int
    # None（未指定）→ 按设备自适应解析（webui 滑条预填逻辑：最小显存GB÷2，无卡为 1）
    batch_size: int | None = None
    save_every_weights: bool = False


class IndexBody(BaseModel):
    exp_name: str
    version: str = "v2"


class PipelineBody(FitBody):
    dataset_dir: str
    f0_method: str = "rmvpe"
    # False（默认）＝新提交：实验目录已存在则 409（自选实验名防冲突）。True 仅由
    # 任务历史「重新提交」携带：失败任务的目录必然已存在，重提交是显式续训意图
    allow_existing: bool = False


# 命令串按双引号包裹路径，这些字符会破坏参数边界（注入面），实验名额外拒绝空白字符；
# $ 与反引号会触发命令替换（实验名出现在 -e "..." 与日志路径里，一并拒绝）。
# server.api.datasets._check_name 复用同一张表（数据集名与实验名同规则），改动须两侧同步
_EXP_FORBIDDEN = ' \t\r\n"\\$`'
_F0_METHODS = frozenset({"pm", "rmvpe"})
_VERSIONS = frozenset({"v1", "v2"})
_SR_KEYS = frozenset(SR_DICT)


def _dump_body(body: BaseModel) -> dict:
    """请求体 → 可序列化 dict，兼容 pydantic v1/v2。model_dump() 是 v2 API，服务
    .venv 里是 pydantic 1.10（老 webui 的 gradio 3.14 连带钉死 fastapi 0.99），v1 下
    直接调 model_dump 会 AttributeError → 500（开发环境跑测试用的是 v2，故测试未拦截）。
    v2 在位时优先走 v2 路径，环境升级后无需改这里。"""
    dumper = getattr(body, "model_dump", None)
    return dumper() if dumper is not None else body.dict()


def _check_exp_name(value: str) -> str:
    """非空、无路径分量（防穿越）、无引号/反斜杠/空白（防 shell 注入）。
    "." 与 ".." 单独拒绝：Python 3.13 起 Path("..").name 返回 ".."，不再能靠 name 兜住。
    NUL 单独拒绝：漏到任务编排层（mkdir/日志路径）是 ValueError → 500，必须在 400
    拦下（与 server.api.datasets._check_name 同步补的判定，两侧同步）。"""
    if (
        not value
        or "\0" in value
        or value in (".", "..")
        or Path(value).name != value
        or any(ch in _EXP_FORBIDDEN for ch in value)
    ):
        raise HTTPException(
            400,
            "实验名非法：%r（不得为空、不得含路径分隔符、引号、反斜杠或空白字符）" % value,
        )
    return value


def _check_dataset_dir(value: str) -> str:
    """命令串按双引号包裹路径，这些字符会破坏参数边界或触发命令替换：
    引号/换行截断参数，$ 与反引号触发命令替换，结尾反斜杠吞掉闭引号。"""
    if (
        not value
        or any(ch in value for ch in '"`$\r\n')
        or value.endswith("\\")
    ):
        raise HTTPException(400, "数据集路径非法：%r（不得含引号、$、反引号、换行或以反斜杠结尾）" % value)
    if not Path(value).is_dir():
        raise HTTPException(400, "数据集目录不存在：%s" % value)
    return value


def _check_sr(value: str) -> SampleRate:
    if value not in _SR_KEYS:
        raise HTTPException(400, "采样率仅支持 32k / 40k / 48k：%r" % value)
    return value


def _check_version(value: str) -> str:
    if value not in _VERSIONS:
        raise HTTPException(400, "版本仅支持 v1 / v2：%r" % value)
    return value


def _check_f0_method(value: str) -> str:
    if value not in _F0_METHODS:
        raise HTTPException(400, "音高提取算法仅支持 pm / rmvpe：%r" % value)
    return value


def _check_epochs(body: FitBody) -> None:
    for name in ("total_epoch", "save_every_epoch", "batch_size"):
        value = getattr(body, name)
        if value is not None and value < 1:  # batch_size 允许 None（设备默认）
            raise HTTPException(400, "%s 必须为正整数：%s" % (name, value))


def _resolve_batch_size(batch_size: int | None) -> tuple[int, str | None]:
    """编排层的 batch_size 归一：显式值原样透传；None → 设备自适应值 + 任务日志提示行
    （返回 (值, 提示行)，提示行仅在走默认时非 None，由 setup 钩子写进任务日志）。"""
    if batch_size is not None:
        return batch_size, None
    resolved = resolve_default_batch_size()
    return resolved, default_batch_size_note(resolved)


def _normalize_sr(sr: str, version: str) -> str:
    """硬性要求①（webui.py:1118-1123 change_version19）：v1 没有 32k 档，一律归一化为
    40k——train.py 的 -sr、config 模板选择、filelist 的 mute 文件名都依赖该一致性。
    在所有校验之后、任何使用之前调用。"""
    return "40k" if (version == "v1" and sr == "32k") else sr


def _resolve_n_p(n_p) -> int:
    if n_p is None:
        return os.cpu_count()
    if n_p < 1:
        raise HTTPException(400, "n_p 必须为正整数：%s" % n_p)
    return n_p


def _task_log(exp_name: str, filename: str) -> Path:
    """任务专属日志文件。fit 用 train_task_fit.log 而非 train.log（硬性要求④）：
    train/train.py 的 get_logger 会向 logs/{exp}/train.log 另挂 FileHandler，与 stdout
    重定向双写会让解析锚点行出现两份，故任务日志一律走专属文件，train.log 只留给
    训练脚本自身的 FileHandler。"""
    return paths.LOGS_DIR / exp_name / filename


# 任务元数据（task_id → 附加信息）：进度换算需要 total_epoch，而 tasks.py 的任务表只存
# 命令串。单用户规模下任务数有限，随进程生命周期存活即可
_TASK_META: dict = {}



def _feature_dir_name(version: str) -> str:
    return "3_feature%s" % FEATURE_DIM[version]


def _stems(directory: Path) -> set:
    """webui.py:1184-1195：name.split(".")[0] 取 stem，不按后缀过滤——
    x.wav 与 x.wav.npy 归为同一 stem（双扩展文件的既有行为，照抄）。
    目录缺失返回空集：跳过 preprocess/extract 直接 fit 时给出「数据未就绪」的可操作
    提示，而不是裸的 FileNotFoundError errno。"""
    if not directory.is_dir():
        return set()
    return {name.split(".")[0] for name in os.listdir(directory)}


def _ensure_mute_assets(sr: SampleRate, version: str) -> None:
    """补齐 mute 行引用的全局静音资产（logs/mute/ 下 1 个静音 wav + 3 个零值 npy）。

    filelist 末尾的 2 条 mute 行教会模型「静音不发声」，train/data_utils 会
    getsize 那个 wav 并 np.load 三个 npy——缺任何一个，训练子进程在加载数据集时
    就会 FileNotFoundError 崩溃，而流水线仍会继续跑索引，造成「看起来成功、实际
    没训」的假象。旧版 webui 同样只引用不生成（上游靠历史目录兜底），这是移植
    缺口，这里补上。幂等：资产齐备时跳过。"""
    import wave

    import numpy as np

    mute_dir = paths.LOGS_DIR / "mute"
    wav_path = mute_dir / "0_gt_wavs" / ("mute%s.wav" % sr)
    feature_path = mute_dir / ("3_feature%s" % FEATURE_DIM[version]) / "mute.npy"
    f0_path = mute_dir / "2a_f0" / "mute.wav.npy"
    f0nsf_path = mute_dir / "2b-f0nsf" / "mute.wav.npy"
    if (
        wav_path.is_file()
        and wav_path.stat().st_size > 44  # 至少容得下 wav 头（空壳视为缺失重建）
        and feature_path.is_file()
        and f0_path.is_file()
        and f0nsf_path.is_file()
    ):
        return

    # 静音 wav：0.4 秒 16bit 单声道，stdlib wave 直写（服务进程保持 torch-free）
    sample_rate = int(sr[:-1]) * 1000  # "40k" → 40000
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * 0.4))

    # 特征/音高占位 npy：全零 50 帧（get_audio_text_pair 会把各模态裁到对齐长度）
    frames = 50
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, np.zeros((frames, FEATURE_DIM[version]), dtype="float32"))
    f0_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(f0_path, np.zeros(frames, dtype="int64"))
    f0nsf_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(f0nsf_path, np.zeros(frames, dtype="float32"))


def generate_filelist(exp_dir: Path, sr: SampleRate, version: str, if_f0: bool) -> None:
    """四目录 stem 交集 → filelist.txt（末尾追加 2 条 mute 行，shuffle 后写入）。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    _ensure_mute_assets(sr, version)  # mute 行引用的资产必须先落地（见函数 docstring）
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / _feature_dir_name(version)
    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            _stems(gt_wavs_dir) & _stems(feature_dir) & _stems(f0_dir) & _stems(f0nsf_dir)
        )
    else:
        names = _stems(gt_wavs_dir) & _stems(feature_dir)
    if not names:
        raise ValueError(NO_VALID_AUDIO_MSG)

    # 目录段做反斜杠转义（webui.py:1217 等 .replace("\\", "\\\\")）：POSIX 上是恒等替换，
    # Windows 上 filelist 以 | 分隔、训练侧按 \ 解析，必须保留
    gt = str(gt_wavs_dir).replace("\\", "\\\\")
    fea = str(feature_dir).replace("\\", "\\\\")
    opt = []
    for name in sorted(names):
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|0"
                % (
                    gt, name,
                    fea, name,
                    str(f0_dir).replace("\\", "\\\\"), name,
                    str(f0nsf_dir).replace("\\", "\\\\"), name,
                )
            )
        else:
            opt.append("%s/%s.wav|%s/%s.npy|0" % (gt, name, fea, name))

    # mute 行不转义（webui.py:1251-1271 同样直接用 now_dir 拼接）；单说话人 sid 固定 0
    fea_dim = FEATURE_DIM[version]
    root = str(paths.ROOT)
    for _ in range(2):
        if if_f0:
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy"
                "|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|0"
                % (root, sr, root, fea_dim, root, root)
            )
        else:
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|0"
                % (root, sr, root, fea_dim)
            )

    shuffle(opt)
    with open(exp_dir / "filelist.txt", "w", encoding="utf8") as f:
        f.write("\n".join(opt))
    logger.debug("训练文件列表写入完成：%s", exp_dir / "filelist.txt")


def write_config(exp_dir: Path, sr: SampleRate, version: str) -> None:
    """webui.py:1282-1307：v1 或 40k 用 v1 模板（v2 无 40k 模板），其余用 v2；
    已有 config.json 则沿用（续训语义），单说话人一律去掉 speaker_info。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    if version == "v1" or sr == "40k":
        config_path = "v1/%s.json" % sr
    else:
        config_path = "v2/%s.json" % sr
    config_save_path = exp_dir / "config.json"
    if config_save_path.exists():
        config_data = json.loads(config_save_path.read_text(encoding="utf8"))
    else:
        template_path = paths.CONFIGS_DIR / config_path
        config_data = json.loads(template_path.read_text(encoding="utf8"))
    config_data.pop("speaker_info", None)  # 单说话人：webui.py:1300
    with open(config_save_path, "w", encoding="utf8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")


def _artifact_names(directory: Path, suffix: str) -> set:
    """webui.py:685-692 artifact_names：按后缀过滤后取 stem；目录不存在返回空集。"""
    if not directory.is_dir():
        return set()
    return {
        name.split(".")[0]
        for name in os.listdir(directory)
        if name.lower().endswith(suffix)
    }


def validate_preprocess_outputs(exp_dir: Path) -> None:
    """webui.py:694-705：0_gt_wavs 与 1_16k_wavs 的 wav stem 交集非空。"""
    gt_names = _artifact_names(exp_dir / "0_gt_wavs", ".wav")
    wav16_names = _artifact_names(exp_dir / "1_16k_wavs", ".wav")
    if not gt_names:
        raise RuntimeError("数据切分没有生成有效训练音频，请检查训练集和数据切分日志")
    if not wav16_names:
        raise RuntimeError("数据切分没有生成16k音频，已停止后续特征提取和训练")
    if not gt_names & wav16_names:
        raise RuntimeError("数据切分输出文件不匹配，已停止后续特征提取和训练")


def validate_feature_outputs(exp_dir: Path, version: str, if_f0: bool) -> set:
    """webui.py:707-720：HuBERT 特征（f0 时再校验 2a_f0 / 2b-f0nsf）。
    返回最终匹配的 stem 集合（webui 同名函数返回值，供编排层复用）。"""
    wav16_names = _artifact_names(exp_dir / "1_16k_wavs", ".wav")
    feature_names = _artifact_names(exp_dir / _feature_dir_name(version), ".npy")
    matched = wav16_names & feature_names
    if not feature_names or not matched:
        raise RuntimeError("HuBERT特征提取没有生成有效结果，已停止训练")
    if if_f0:
        f0_names = _artifact_names(exp_dir / "2a_f0", ".npy")
        f0nsf_names = _artifact_names(exp_dir / "2b-f0nsf", ".npy")
        matched &= f0_names & f0nsf_names
        if not f0_names or not f0nsf_names or not matched:
            raise RuntimeError("F0提取没有生成有效结果，已停止训练")
    return matched


# ---------------------------------------------------------------------------
# fit 前置（fit 端点的 setup 钩子与 pipeline 的 fitprep 子进程共用）
# ---------------------------------------------------------------------------


def _pretrained_notices(sr: SampleRate, if_f0: bool, version: str,
                        pretrain_g: str, pretrain_d: str) -> list:
    """底模缺失提示行（webui.py:1091-1100 的 warning 文案）。硬性要求②：
    缺失必须可见——不拼 -pg/-pd 属正常分支，但静默从零训练会让人误以为在续训。"""
    expected_g, expected_d = pretrained_rel_paths(sr, if_f0, version)
    lines = []
    if not pretrain_g:
        lines.append("生成器预训练模型不存在，将不使用：%s" % expected_g)
    if not pretrain_d:
        lines.append("判别器预训练模型不存在，将不使用：%s" % expected_d)
    return lines


def prepare_fit(exp_dir: Path, sr: SampleRate, version: str, if_f0: bool) -> list:
    """fit 的全部 Python 前置：底模选择 → filelist → config.json → 产物校验。

    返回须写进任务日志的提示行。ValueError（数据未就绪）与 RuntimeError（产物校验
    失败）向上抛，由调用方任务化：
    - fit 端点：setup 钩子在任务工作线程内执行，tasks.py 捕获后落 failed 终态
    - pipeline：fitprep 子进程非零退出，tasks.py 以日志尾部转 failed 终态
    底模存在性在此处与命令拼装处各查一次（纯文件系统检查，开销可忽略），保证
    fitprep 的提示与 train.py 的 -pg/-pd 永远来自同一函数。"""
    pretrain_g, pretrain_d = get_pretrained_paths(sr, if_f0, version)
    lines = _pretrained_notices(sr, if_f0, version, pretrain_g, pretrain_d)
    generate_filelist(exp_dir, sr, version, if_f0)
    write_config(exp_dir, sr, version)
    validate_feature_outputs(exp_dir, version, if_f0)
    return lines


FITPREP_USAGE = (
    "用法：python -m server.api.training precheck <实验名>\n"
    "      python -m server.api.training fitprep <实验名> <sr> <version> <if_f0: 0|1>"
)


def fitprep_main(argv: list) -> int:
    """pipeline 子进程 cmd 的入口（cwd=仓库根，见 build_precheck_cmd / build_fitprep_cmd）：

    - `precheck <实验名>`：校验切分产物（对应分步流程 extract 的 setup 钩子）
    - `fitprep <实验名> <sr> <version> <0|1>`：fit 的 Python 前置（底模提示 → filelist
      → config.json → 产物校验）

    参数由 API 层校验后拼进命令串，这里不再重复校验。退出码约定（tasks.py 按非零判
    失败，stdout/stderr 已被重定向进任务日志）：0 成功；1 前置失败（原因打到 stderr，
    进任务日志尾部）；2 用法错误。"""
    if len(argv) == 2 and argv[0] == "precheck":
        exp_dir = paths.LOGS_DIR / argv[1]
        try:
            validate_preprocess_outputs(exp_dir)
        except Exception as exc:  # noqa: BLE001 子进程边界：失败以非零退出上报
            print("数据切分校验未通过：%s" % exc, file=sys.stderr)
            return 1
        print("数据切分产物校验通过", flush=True)
        return 0

    if len(argv) != 5 or argv[0] != "fitprep" or argv[4] not in ("0", "1"):
        print(FITPREP_USAGE, file=sys.stderr)
        return 2
    _, exp_name, sr, version, if_f0_flag = argv
    exp_dir = paths.LOGS_DIR / exp_name
    try:
        lines = prepare_fit(exp_dir, sr, version, if_f0_flag == "1")
    except Exception as exc:  # noqa: BLE001 子进程边界：任何前置失败都以非零退出上报
        traceback.print_exc()
        print("训练前置生成失败：%s" % exc, file=sys.stderr)
        return 1
    for line in lines:
        print(line, flush=True)
    print("训练前置生成完成：%s" % (exp_dir / "filelist.txt"), flush=True)
    return 0



# ---------------------------------------------------------------------------
# 路由：4 步 + 一键
# ---------------------------------------------------------------------------


def _ensure_exp_idle(exp_name: str) -> None:
    """同名实验互斥（队列化后唯一保留的 409，见
    docs/plans/2026-09-09-training-queue-design.md §2）：非终态任务（运行中或排队中）
    里已有相同 exp_name 时拒绝——两个任务共写 logs/{exp} 会互相覆盖权重与索引。
    终态任务不受限（重新提交同一实验是合法的重跑/续训路径）。"""
    for snapshot in task_manager.list_tasks():
        if snapshot["state"] in TERMINAL_STATES:
            continue
        meta = _TASK_META.get(snapshot["id"]) or {}
        if meta.get("exp_name") == exp_name:
            phase = "运行中" if snapshot["state"] == "running" else "排队中"
            raise HTTPException(
                409,
                "实验 %s 已有任务在%s（%s），请等待完成或先停止"
                % (exp_name, phase, snapshot["name"]),
            )


def _ensure_exp_dir_free(exp_name: str, allow_existing: bool) -> None:
    """自选实验名防冲突（只挂 pipeline 新提交）：logs/{exp} 目录已存在即 409——
    防止误覆盖已有产物或无意识续进旧检查点。续训是显式动作（allow_existing=True，
    任务历史「重新提交」专用），不走这条静默路径。与 _ensure_exp_idle 互补：
    那里管「排队/运行中的活动任务」，这里管「磁盘上已有产物目录」。"""
    if allow_existing:
        return
    if (paths.LOGS_DIR / exp_name).exists():
        raise HTTPException(
            409,
            "实验名 %s 已存在，请换一个名字；要基于已有产物续训，请在任务历史中"
            "使用「重新提交」" % exp_name,
        )


def _create(name: str, cmds: list, log_path: Path, *, setup=None, meta: dict | None = None,
            definition: dict | None = None):
    """登记任务并返回 {task_id, queued, queue_position}。

    已有任务运行中时不再拒绝：任务自动排队（design §4.1），响应里的 queued 与
    queue_position 供前端展示「排在第几位」。meta 记入任务元数据表（进度换算锚点
    + 同名互斥 + 队列面板展示）；definition 为可序列化任务定义，随 pending 快照
    持久化、重启后按 kind 重建（见 restore_pending_tasks）。"""
    task_id = task_manager.create_task(
        name, cmds, log_path, setup=setup, definition=definition
    )
    if meta:
        _TASK_META[task_id] = meta
    snapshot = task_manager.get_task(task_id) or {}
    return {
        "task_id": task_id,
        "queued": snapshot.get("state") == PENDING,
        "queue_position": snapshot.get("queue_position"),
    }


@router.post("/train/preprocess")
def start_preprocess(body: PreprocessBody):
    """数据切分。注意本接口没有 version 字段，v1+32k → 40k 的归一化发生在 fit /
    pipeline（webui change_version19 语义）；分步模式下的采样率一致性由前端保证。"""
    exp_name = _check_exp_name(body.exp_name)
    dataset_dir = _check_dataset_dir(body.dataset_dir)
    sr = _check_sr(body.sr)
    n_p = _resolve_n_p(body.n_p)
    _ensure_exp_idle(exp_name)
    cmd = build_preprocess_cmd(dataset_dir, SR_DICT[sr], n_p, exp_name, NOPARALLEL, PREPROCESS_PER)
    return _create(
        "preprocess",
        [cmd],
        _task_log(exp_name, "preprocess.log"),
        meta={"exp_name": exp_name},
        definition={"kind": "preprocess", "body": _dump_body(body)},
    )


@router.post("/train/extract")
def start_extract(body: ExtractBody):
    exp_name = _check_exp_name(body.exp_name)
    f0_method = _check_f0_method(body.f0_method)
    version = _check_version(body.version)
    exp_dir = paths.LOGS_DIR / exp_name
    cmds = []
    if body.if_f0:  # webui 同序：先 f0 后 HuBERT，两段共享同一日志文件
        cmds.append(build_extract_f0_cmd(exp_name, os.cpu_count(), f0_method))
    cmds.append(build_extract_hubert_cmd(exp_name, version, resolve_is_half()))
    _ensure_exp_idle(exp_name)
    return _create(
        "extract",
        cmds,
        _task_log(exp_name, "extract_f0_feature.log"),
        setup=_preprocess_check_setup(exp_dir),
        meta={"exp_name": exp_name},
        definition={"kind": "extract", "body": _dump_body(body)},
    )


def _preprocess_check_setup(exp_dir: Path):
    """extract 的 setup 钩子：启动子进程前校验切分产物（失败 → 任务 failed 而非 500）。"""
    def setup():
        validate_preprocess_outputs(exp_dir)

    return setup


@router.post("/train/fit")
def start_fit(body: FitBody):
    exp_name = _check_exp_name(body.exp_name)
    version = _check_version(body.version)
    sr = _normalize_sr(_check_sr(body.sr), version)
    _check_epochs(body)
    batch_size, batch_note = _resolve_batch_size(body.batch_size)
    exp_dir = paths.LOGS_DIR / exp_name
    _ensure_exp_idle(exp_name)
    return _create(
        "fit",
        [_fit_cmd(body, sr, version, batch_size)],
        _task_log(exp_name, "train_task_fit.log"),
        setup=_fit_setup(exp_dir, sr, version, body.if_f0, batch_note),
        meta={"total_epoch": body.total_epoch, "exp_name": exp_name},
        definition={"kind": "fit", "body": _dump_body(body), "batch_note": batch_note},
    )


def _fit_cmd(body: FitBody, sr: str, version: str, batch_size: int) -> str:
    pretrain_g, pretrain_d = get_pretrained_paths(sr, body.if_f0, version)
    return build_fit_cmd(
        body.exp_name,
        sr,
        body.if_f0,
        batch_size,
        body.total_epoch,
        body.save_every_epoch,
        body.save_every_weights,
        pretrain_g,
        pretrain_d,
        version=version,
    )


def _fit_setup(exp_dir: Path, sr: str, version: str, if_f0: bool, batch_note: str | None = None):
    """fit 的 setup 钩子：返回值（batch_size 默认值说明 + 底模缺失提示行）由 tasks.py
    写进任务日志缓冲。"""
    def setup():
        return ([batch_note] if batch_note else []) + prepare_fit(exp_dir, sr, version, if_f0)

    return setup


@router.get("/train/defaults")
def train_defaults():
    """训练表单的设备自适应默认值（webui 滑条预填值同源）。轻量只读端点，供前端
    把解析值预填进输入框；只回 batch_size，不承载其它配置。"""
    return {"batch_size": resolve_default_batch_size()}


@router.get("/train/exp-name/exists")
def exp_name_exists(name: str):
    """实验名占用检查（前端输入框防抖查询用）。非法名与提交接口同一张校验表
    （_check_exp_name，文案一致，前端可直接透出）；合法名返回 logs/{name} 目录
    是否已存在。纯只读：不创建目录、不登记任务。"""
    _check_exp_name(name)
    return {"exists": (paths.LOGS_DIR / name).exists()}


@router.post("/train/index")
def start_index(body: IndexBody):
    exp_name = _check_exp_name(body.exp_name)
    version = _check_version(body.version)
    _ensure_exp_idle(exp_name)
    cmd = build_index_cmd(exp_name, version, os.cpu_count())
    return _create(
        "index",
        [cmd],
        _task_log(exp_name, "train_index.log"),
        meta={"exp_name": exp_name},
        definition={"kind": "index", "body": _dump_body(body)},
    )


@router.post("/train/pipeline")
def start_pipeline(body: PipelineBody):
    """一键全流程：单任务多 cmd（任一 cmd 失败/取消后 tasks.py 不再启动后续）。

    precheck / fitprep 两个 Python 前置以子进程 cmd 的形式插在对应子进程之后
    （见 build_precheck_cmd / build_fitprep_cmd），时序因此与分步流程完全一致；
    任务级 setup 钩子做不到这一点——它只在首个 cmd 之前跑一次，会让新实验在
    preprocess 还没执行时就被产物校验判死。因此这里的 setup 只承担与时序无关的
    batch_size 默认值提示（不做任何产物校验）。
    """
    exp_name = _check_exp_name(body.exp_name)
    dataset_dir = _check_dataset_dir(body.dataset_dir)
    version = _check_version(body.version)
    f0_method = _check_f0_method(body.f0_method)
    sr = _normalize_sr(_check_sr(body.sr), version)
    _check_epochs(body)
    batch_size, batch_note = _resolve_batch_size(body.batch_size)
    n_p = os.cpu_count()
    exp_dir = paths.LOGS_DIR / exp_name
    _ensure_exp_dir_free(exp_name, body.allow_existing)

    cmds = [build_preprocess_cmd(dataset_dir, SR_DICT[sr], n_p, exp_name, NOPARALLEL, PREPROCESS_PER)]
    cmds.append(build_precheck_cmd(exp_name))
    # 每条 cmd 的阶段归属表（与 cmds 等长）：_progress_state 据此把 current_cmd
    # 映射到加权进度的阶段（前端步骤条的阶段映射用各自的命令串标记，互不依赖）
    stages = ["preprocess", "preprocess"]
    if body.if_f0:
        cmds.append(build_extract_f0_cmd(exp_name, n_p, f0_method))
        stages.append("extract")
    cmds.append(build_extract_hubert_cmd(exp_name, version, resolve_is_half()))
    stages.append("extract")
    cmds.append(build_fitprep_cmd(exp_name, sr, version, body.if_f0))
    stages.append("fit")
    cmds.append(_fit_cmd(body, sr, version, batch_size))
    stages.append("fit")
    cmds.append(build_index_cmd(exp_name, version, os.cpu_count()))
    stages.append("index")
    _ensure_exp_idle(exp_name)
    return _create(
        "pipeline",
        cmds,
        _task_log(exp_name, "pipeline_task.log"),
        setup=_pipeline_notice_setup(batch_note),
        meta={
            "total_epoch": body.total_epoch,
            "pipeline_stages": stages,
            "exp_name": exp_name,
        },
        definition={
            "kind": "pipeline",
            "body": _dump_body(body),
            "batch_note": batch_note,
            "pipeline_stages": stages,
        },
    )


def _pipeline_notice_setup(batch_note: str | None):
    """pipeline 的 setup 钩子：只回 batch_size 说明行（可能为 None → 无输出），
    与 fit 的产物校验前置严格分离——见 start_pipeline 的时序说明。"""
    def setup():
        return [batch_note] if batch_note else []

    return setup


# ---------------------------------------------------------------------------
# 路由：任务查询 / 取消
# ---------------------------------------------------------------------------


def _progress_state(snapshot: dict) -> dict:
    """读时即时计算的进度（无解析线程、无额外 I/O：直接用快照里的日志缓冲）。

    返回 {"progress": float | None, "current": str | None}：

    - success → 1.0（index 等秒级阶段没有过程行，在这里收敛为完成）
    - pipeline（_TASK_META 登记 pipeline_stages）→ 全流程加权进度（见
      _pipeline_progress_state），单一刻度单调递增，阶段切换不再归零
    - fit（登记 total_epoch）且缓冲里有 epoch 行 → 按训练进度换算，
      训练段没有「当前文件」语义
    - 其余（preprocess/extract/index）→ 用脚本自身的进度行
      （design §3.5）换算 done/total，并带上当前文件名
    - 都没有锚点 → 透传 task.progress 原值
    """
    if snapshot["state"] == SUCCESS:
        return {"progress": 1.0, "current": None}
    meta = _TASK_META.get(snapshot["id"]) or {}
    stages = meta.get("pipeline_stages")
    if stages:
        return _pipeline_progress_state(snapshot, meta, stages)
    total_epoch = meta.get("total_epoch")
    if total_epoch is not None:
        for line in reversed(snapshot["logs"]):
            parsed = parse_train_line(line)
            if parsed and "epoch" in parsed:
                progress = ((parsed["epoch"] - 1) + parsed["pct"] / 100.0) / total_epoch
                return {"progress": max(0.0, min(1.0, progress)), "current": None}
    for line in reversed(snapshot["logs"]):
        parsed = parse_stage_progress_line(line)
        if parsed and parsed["total"] > 0:
            progress = parsed["done"] / parsed["total"]
            return {"progress": max(0.0, min(1.0, progress)), "current": parsed["current"]}
    return {"progress": snapshot["progress"], "current": None}


# pipeline 全流程进度的阶段权重（展示启发值，总和 1）：训练占绝对大头，切分/提取
# 在 CPU 上也可能吃时间故给 extract 15%；索引秒级。只用于 pipeline，分步任务保持
# 「进度 = 本阶段 done/total」的原始语义
_PIPELINE_STAGE_WEIGHTS = {"preprocess": 0.05, "extract": 0.15, "fit": 0.75, "index": 0.05}


def _pipeline_progress_state(snapshot: dict, meta: dict, stages: list) -> dict:
    """pipeline 的全流程加权进度：已完成阶段的权重 + 当前阶段内进度 × 阶段权重。

    - 阶段归属来自建任务时随 cmds 一起登记的 stages 表（与 cmds 等长），配合快照
      的 current_cmd（1-based）定位当前阶段——阶段切换不再让 bar 归零重爬
    - 单调钳位：F0 提取是多进程分批跑的，每批的 done/total 从低处重新计数，不钳位
      的话 bar 会在批间抖动。读写竞态最坏丢一次 max，下一次读即恢复，无锁必要
    - current 沿用最近的阶段进度行文件名（训练段无文件语义，为 None）
    """
    current_cmd = snapshot.get("current_cmd")
    if current_cmd is None:
        return {"progress": 0.0, "current": None}
    idx = min(current_cmd, len(stages)) - 1
    stage = stages[idx]
    # 回溯到当前阶段的首条 cmd（同阶段可占多条 cmd，如 fitprep + train.py 都属
    # fit）：base 只累加该阶段之前已完成阶段的权重，否则当前阶段的权重会被自己
    # 吃进 base（fit 段直接顶到 95%）
    stage_start = idx
    while stage_start > 0 and stages[stage_start - 1] == stage:
        stage_start -= 1
    base = sum(_PIPELINE_STAGE_WEIGHTS[s] for s in dict.fromkeys(stages[:stage_start]))
    fraction = 0.0
    current = None
    if stage == "fit":
        total_epoch = meta.get("total_epoch")
        if total_epoch:
            for line in reversed(snapshot["logs"]):
                parsed = parse_train_line(line)
                if parsed and "epoch" in parsed:
                    fraction = ((parsed["epoch"] - 1) + parsed["pct"] / 100.0) / total_epoch
                    break
    else:
        for line in reversed(snapshot["logs"]):
            parsed = parse_stage_progress_line(line)
            if parsed and parsed["total"] > 0:
                fraction = parsed["done"] / parsed["total"]
                current = parsed["current"]
                break
    progress = base + max(0.0, min(1.0, fraction)) * _PIPELINE_STAGE_WEIGHTS[stage]
    last = meta.get("last_pipeline_progress")
    if last is not None:
        progress = max(progress, last)
    meta["last_pipeline_progress"] = progress
    return {"progress": max(0.0, min(1.0, progress)), "current": current}


def _progress_of(snapshot: dict):
    """进度数值版（GET /api/tasks 的紧凑投影等只需要数字的场景）。"""
    return _progress_state(snapshot)["progress"]


def _status_payload(snapshot: dict) -> dict:
    """SSE 的 status 事件体：完整快照去掉 logs 与 definition——日志由 log 事件承载
    （避免每条 status 重复携带最多 1000 行），definition 是内部持久化字段
    （params 投影已携带等价信息）；progress/current 换算为读时计算值。"""
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in ("logs", "definition")
    }
    payload.update(_progress_state(snapshot))
    return payload


@router.get("/tasks")
def list_tasks():
    """任务列表（前端队列面板数据源）：内存任务 + 磁盘历史合并（按 id 去重，
    内存优先）。历史行是终态快照投影，不含 logs_tail（展开时经 GET /tasks/{id}
    一次性拉取），queue_position 恒为 None。"""
    rows = []
    memory_ids = set()
    for snapshot in task_manager.list_tasks():
        memory_ids.add(snapshot["id"])
        rows.append(_task_row(snapshot, history=False))
    for record in _history_rows():
        if record.get("id") in memory_ids:
            continue
        rows.append(_task_row(record, history=True))
    return rows


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    """任务详情。内存任务：完整快照 + 读时进度；否则回退历史记录（含 logs_tail，
    供前端展开历史队列项时一次性拉取日志与 loss 数据）；都不存在 → 404。"""
    snapshot = task_manager.get_task(task_id)
    if snapshot is not None:
        snapshot.pop("definition", None)  # 内部字段不外露（params 投影已携带等价信息）
        snapshot.update(_progress_state(snapshot))
        snapshot["history"] = False
        return snapshot
    for record in _history_rows():
        if record.get("id") == task_id:
            return {**record, "queue_position": None, "history": True}
    raise HTTPException(404, "任务不存在：%s" % task_id)


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    """触发终止（进程组 SIGTERM→SIGKILL）。已终态时 cancel 是幂等 no-op，仍返回 202
    并带上当前状态——前端「停止」按钮与任务自然结束竞态时不需要区分。排队中的任务
    即时出队落 cancelled。"""
    snapshot = task_manager.get_task(task_id)
    if snapshot is None:
        raise HTTPException(404, "任务不存在：%s" % task_id)
    task_manager.cancel(task_id)
    return JSONResponse(
        status_code=202,
        content={"id": task_id, "state": (task_manager.get_task(task_id) or snapshot)["state"]},
    )


@router.delete("/tasks")
def clear_tasks():
    """「停止并清空队列」（design §4.1）：终止当前任务（进程组终止由其工作线程执行）
    并取消全部排队任务。幂等：空队列时是 no-op。"""
    return task_manager.clear_queue()


# ---------------------------------------------------------------------------
# SSE：任务日志 / 进度 / 状态流
# ---------------------------------------------------------------------------

# 轮询与心跳间隔（模块常量：测试注入小间隔）；间隔读自模块全局，monkeypatch 即生效
_SSE_POLL_INTERVAL = 1.0
_SSE_KEEPALIVE_INTERVAL = 15.0

# 当前活跃的 SSE 订阅（task_id 级）：仅用于排障与测试断言清理，不参与推送
_active_streams: set = set()


def _sse_event(name: str, payload: dict) -> str:
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, ensure_ascii=False))


@router.get("/tasks/{task_id}/events")
def stream_task_events(task_id: str, cursor: str = "0"):
    """SSE 订阅：先推一条 status（当前态 + 进度），随后按游标增量推 log、按变化推
    progress，终态补发最终 status 后关闭流。

    游标协议（M-2 遗留的明确化）：cursor 是已消费的累计行数，0 表示从缓冲起点重放；
    断线重连一律带 cursor=0 全量重放，由前端幂等渲染（环形缓冲淘汰的旧行会自动收敛到
    缓冲内剩余部分，见 TaskManager.read_logs_since）。
    """
    try:
        cursor_value = int(cursor)
    except ValueError:
        raise HTTPException(400, "cursor 必须为整数：%r" % cursor)
    if cursor_value < 0:
        raise HTTPException(400, "cursor 不能为负数：%s" % cursor_value)
    if task_manager.get_task(task_id) is None:
        raise HTTPException(404, "任务不存在：%s" % task_id)
    return StreamingResponse(
        _task_event_stream(task_id, cursor_value),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _task_event_stream(task_id: str, cursor: int):
    """事件生成器。同步生成器由 Starlette 放进线程池迭代，poll 期间的 sleep 不阻塞
    事件循环；无论正常收尾还是客户端断开（GeneratorExit）都走 finally 注销订阅。"""
    snapshot = task_manager.get_task(task_id)
    if snapshot is None:  # 订阅后任务表被清空（仅服务重启可能），直接收流
        return
    poll = _SSE_POLL_INTERVAL
    keepalive_interval = _SSE_KEEPALIVE_INTERVAL
    _active_streams.add(task_id)
    try:
        initial = _status_payload(snapshot)
        if snapshot["state"] not in TERMINAL_STATES:  # 已终态只发下面那条终态 status
            yield _sse_event("status", initial)
        last_progress = {key: initial[key] for key in ("progress", "current")}
        last_current_cmd = snapshot.get("current_cmd")
        last_sent = time.monotonic()
        while True:
            fetched = task_manager.read_logs_since(task_id, cursor)
            snapshot = task_manager.get_task(task_id)
            if fetched is None or snapshot is None:
                return
            lines, cursor = fetched
            if lines:
                yield _sse_event("log", {"lines": list(lines), "seq": cursor})
                last_sent = time.monotonic()

            # current_cmd 变化（pipeline 切换子命令）必须补发 status：中途只有
            # log/progress 事件在飞，而它们不携带 cmds/current_cmd——前端步骤条的
            # 阶段映射靠这个切换信号推进，漏发会让步骤条冻结在上一阶段。
            # 同步重置 last_progress：status 载荷里已含本次进度，避免下轮重复推送
            if snapshot.get("current_cmd") != last_current_cmd:
                last_current_cmd = snapshot.get("current_cmd")
                progress_now = _progress_state(snapshot)
                last_progress = {
                    key: progress_now[key] for key in ("progress", "current")
                }
                yield _sse_event("status", _status_payload(snapshot))
                last_sent = time.monotonic()

            progress_state = _progress_state(snapshot)
            if progress_state != last_progress:  # 进度按变化推送：静默期才有心跳的意义
                last_progress = progress_state
                yield _sse_event("progress", progress_state)
                last_sent = time.monotonic()
            if snapshot["state"] in TERMINAL_STATES:
                yield _sse_event("status", _status_payload(snapshot))
                return

            time.sleep(poll)
            if time.monotonic() - last_sent >= keepalive_interval:
                yield ": keepalive\n\n"
                last_sent = time.monotonic()
    finally:
        _active_streams.discard(task_id)


# ---------------------------------------------------------------------------
# 启动恢复（main.py lifespan startup 调用，design docs/plans/2026-09-09-training-
# queue-design.md §4.2）：把持久化的排队任务按原顺序重新入队
# ---------------------------------------------------------------------------


def _restore_training_task(definition: dict):
    """按 kind 重建任务记录里不可序列化的部分，返回 (setup, meta)。

    cmds/log_path 等已随记录持久化（提交时定格了 batch_size 自适应、device/is_half、
    底模存在性探测的结果），因此重建是纯 Python 文件系统操作，不触发 torch 加载。
    setup 闭包在任务真正开始执行时才跑，校验的是执行时刻的产物状态。"""
    kind = definition.get("kind")
    body = definition.get("body") or {}
    exp_name = body.get("exp_name")
    if kind == "preprocess":
        return None, {"exp_name": exp_name}
    if kind == "extract":
        return _preprocess_check_setup(paths.LOGS_DIR / exp_name), {"exp_name": exp_name}
    if kind == "fit":
        return (
            _fit_setup(
                paths.LOGS_DIR / exp_name,
                body["sr"],
                body["version"],
                body["if_f0"],
                definition.get("batch_note"),
            ),
            {"total_epoch": body["total_epoch"], "exp_name": exp_name},
        )
    if kind == "index":
        return None, {"exp_name": exp_name}
    if kind == "pipeline":
        return (
            _pipeline_notice_setup(definition.get("batch_note")),
            {
                "total_epoch": body["total_epoch"],
                "exp_name": exp_name,
                "pipeline_stages": definition["pipeline_stages"],
            },
        )
    return None


for _kind in ("preprocess", "extract", "fit", "index", "pipeline"):
    RESTORE_FACTORIES[_kind] = _restore_training_task


def restore_pending_tasks() -> dict:
    """读取持久化的排队任务并按文件顺序重新入队（保留原 id 与提交时间），返回
    {"restored", "skipped"}。首条任务在空闲服务上即刻启动；已有任务运行中则全部
    排队。缺字段 / kind 未注册的记录跳过并告警（恢复完成后由入队路径把文件收敛
    为恢复后的快照，坏记录自然剔除）。queue_store 为 None（测试隔离）时是 no-op。"""
    store = task_manager.queue_store
    if store is None:
        return {"restored": 0, "skipped": 0}
    records = store.load()
    restored = 0
    skipped = 0
    for record in records:
        definition = record.get("definition") or {}
        if not isinstance(definition, dict) or definition.get("kind") not in RESTORE_FACTORIES:
            logger.warning("恢复队列：未知任务定义，跳过记录 %s", record.get("id"))
            skipped += 1
            continue
        try:
            setup, meta = RESTORE_FACTORIES[definition["kind"]](definition)
            task_manager.create_task(
                record["name"],
                record["cmds"],
                Path(record["log_path"]),
                truncate=bool(record.get("truncate", True)),
                setup=setup,
                definition=definition,
                task_id=record["id"],
                created_at=float(record["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("恢复队列：跳过损坏记录 %s：%s", record.get("id"), exc)
            skipped += 1
            continue
        if meta:
            _TASK_META[record["id"]] = meta
        restored += 1
    if records:
        logger.info("队列恢复完成：%d 个排队任务已恢复，%d 条记录跳过", restored, skipped)
    return {"restored": restored, "skipped": skipped}


# ---------------------------------------------------------------------------
# 训练历史（design docs/plans/2026-09-09-task-centric-training-ui-design.md §3）：
# 任务落终态时记录快照落盘，跨重启可回看（loss 尾部趋势 / 日志 / 参数）。
# 磁盘上的任务日志不完整（pipeline 多 cmd 共用文件且逐 cmd 截断），内存环形缓冲
# 是唯一完整来源，因此历史必须在终态时随快照落盘。
# ---------------------------------------------------------------------------

#: 日志尾部快照行数：≈15-30 epoch 的 loss 尾部趋势 + 失败原因，单条记录 ~36KB
HISTORY_LOG_TAIL_LINES = 300
#: 历史记录上限（新进旧出）
HISTORY_CAP = 50

# 模块级历史状态。store 走惰性解析（路径随 paths.ROOT 的测试 monkeypatch 跟随，
# 不得在 import 时固化）；cache 是磁盘内容的内存镜像（GET /api/tasks 3s 轮询直接
# 读它，避免反复解析大 JSON）；recorded_ids 保证同一任务只记一次。
_history_store: TaskHistoryStore | None = None
_history_cache: list | None = None
_recorded_ids: set = set()


def _get_history_store() -> TaskHistoryStore:
    global _history_store
    if _history_store is None:
        _history_store = TaskHistoryStore(paths.history_file(), cap=HISTORY_CAP)
    return _history_store


def _history_rows() -> list:
    """历史记录（旧→新序），首次访问从磁盘惰性加载。"""
    global _history_cache
    if _history_cache is None:
        try:
            _history_cache = _get_history_store().load()
        except OSError:
            logger.exception("训练历史读取失败，按空历史处理")
            _history_cache = []
    return _history_cache


def _record_history(snapshot: dict) -> None:
    """on_finish 回调（server/tasks.py）：把终态任务写入历史。
    非终态快照、重复投递、无训练元数据的任务（separate 等）一律跳过；
    写盘失败降级为日志（历史故障不得影响任务与队列）。"""
    global _history_cache
    if snapshot.get("state") not in TERMINAL_STATES:
        return
    task_id = snapshot.get("id")
    if not task_id or task_id in _recorded_ids:
        return
    meta = _TASK_META.get(task_id)
    if not meta:  # separate 等无训练语义的任务
        return
    definition = snapshot.get("definition") or {}
    record = {
        "id": task_id,
        "name": snapshot.get("name"),
        "kind": definition.get("kind"),
        "exp_name": meta.get("exp_name"),
        "state": snapshot.get("state"),
        "error": snapshot.get("error"),
        "created_at": snapshot.get("created_at"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
        "current_cmd": snapshot.get("current_cmd"),
        "progress": _progress_state(snapshot)["progress"],
        "cmds": snapshot.get("cmds") or [],
        "params": definition.get("body") or {},
        "pipeline_stages": meta.get("pipeline_stages"),
        "log_path": snapshot.get("log_path"),
        "logs_tail": list(snapshot.get("logs") or [])[-HISTORY_LOG_TAIL_LINES:],
    }
    try:
        _get_history_store().record(record)
    except OSError:
        logger.exception("训练历史写入失败（不影响任务）：%s", task_id)
        return
    _recorded_ids.add(task_id)
    if _history_cache is None:
        _history_cache = _get_history_store().load()  # 含本条，且已按 cap 截断
    else:
        _history_cache = [row for row in _history_cache if row.get("id") != task_id]
        _history_cache.append(record)
        if len(_history_cache) > HISTORY_CAP:
            _history_cache = _history_cache[len(_history_cache) - HISTORY_CAP:]


task_manager.on_finish = _record_history  # 进程级单例挂记录器（测试实例各自接线）


def _task_row(snapshot: dict, *, history: bool) -> dict:
    """GET /api/tasks 的行投影（内存与历史两类行字段形状完全一致，前端一套渲染）。"""
    meta = _TASK_META.get(snapshot.get("id")) or {}
    definition = snapshot.get("definition") or {}
    return {
        "id": snapshot.get("id"),
        "name": snapshot.get("name"),
        "state": snapshot.get("state"),
        "progress": _progress_state(snapshot)["progress"] if not history else snapshot.get("progress"),
        "error": snapshot.get("error"),
        "queue_position": snapshot.get("queue_position"),
        "exp_name": meta.get("exp_name") or snapshot.get("exp_name"),
        "kind": definition.get("kind") or snapshot.get("kind"),
        "params": definition.get("body") or snapshot.get("params") or {},
        "pipeline_stages": meta.get("pipeline_stages") or snapshot.get("pipeline_stages"),
        "created_at": snapshot.get("created_at"),
        "finished_at": snapshot.get("finished_at"),
        "history": history,
    }


if __name__ == "__main__":  # pipeline 的 fit 前置子进程（见 build_fitprep_cmd）
    sys.exit(fitprep_main(sys.argv[1:]))
