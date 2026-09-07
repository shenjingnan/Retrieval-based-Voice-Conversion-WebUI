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
from server.tasks import SUCCESS, TERMINAL_STATES, TaskConflictError, task_manager

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


# 命令串按双引号包裹路径，这些字符会破坏参数边界（注入面），实验名额外拒绝空白字符；
# $ 与反引号会触发命令替换（实验名出现在 -e "..." 与日志路径里，一并拒绝）
_EXP_FORBIDDEN = ' \t\r\n"\\$`'
_F0_METHODS = frozenset({"pm", "rmvpe"})
_VERSIONS = frozenset({"v1", "v2"})
_SR_KEYS = frozenset(SR_DICT)


def _check_exp_name(value: str) -> str:
    """非空、无路径分量（防穿越）、无引号/反斜杠/空白（防 shell 注入）。
    "." 与 ".." 单独拒绝：Python 3.13 起 Path("..").name 返回 ".."，不再能靠 name 兜住。"""
    if (
        not value
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


def generate_filelist(exp_dir: Path, sr: SampleRate, version: str, if_f0: bool) -> None:
    """四目录 stem 交集 → filelist.txt（末尾追加 2 条 mute 行，shuffle 后写入）。"""
    exp_dir.mkdir(parents=True, exist_ok=True)
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


def _create(name: str, cmds: list, log_path: Path, *, setup=None, meta: dict | None = None):
    """登记任务并把 TaskConflictError 映射成 409；meta 记入任务元数据表。"""
    try:
        task_id = task_manager.create_task(name, cmds, log_path, setup=setup)
    except TaskConflictError as exc:
        raise HTTPException(409, str(exc))
    if meta:
        _TASK_META[task_id] = meta
    return {"task_id": task_id}


@router.post("/train/preprocess")
def start_preprocess(body: PreprocessBody):
    """数据切分。注意本接口没有 version 字段，v1+32k → 40k 的归一化发生在 fit /
    pipeline（webui change_version19 语义）；分步模式下的采样率一致性由前端保证。"""
    exp_name = _check_exp_name(body.exp_name)
    dataset_dir = _check_dataset_dir(body.dataset_dir)
    sr = _check_sr(body.sr)
    n_p = _resolve_n_p(body.n_p)
    cmd = build_preprocess_cmd(dataset_dir, SR_DICT[sr], n_p, exp_name, NOPARALLEL, PREPROCESS_PER)
    return _create("preprocess", [cmd], _task_log(exp_name, "preprocess.log"))


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
    return _create(
        "extract",
        cmds,
        _task_log(exp_name, "extract_f0_feature.log"),
        setup=_preprocess_check_setup(exp_dir),
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
    return _create(
        "fit",
        [_fit_cmd(body, sr, version, batch_size)],
        _task_log(exp_name, "train_task_fit.log"),
        setup=_fit_setup(exp_dir, sr, version, body.if_f0, batch_note),
        meta={"total_epoch": body.total_epoch},
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


@router.post("/train/index")
def start_index(body: IndexBody):
    exp_name = _check_exp_name(body.exp_name)
    version = _check_version(body.version)
    cmd = build_index_cmd(exp_name, version, os.cpu_count())
    return _create("index", [cmd], _task_log(exp_name, "train_index.log"))


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

    cmds = [build_preprocess_cmd(dataset_dir, SR_DICT[sr], n_p, exp_name, NOPARALLEL, PREPROCESS_PER)]
    cmds.append(build_precheck_cmd(exp_name))
    if body.if_f0:
        cmds.append(build_extract_f0_cmd(exp_name, n_p, f0_method))
    cmds.append(build_extract_hubert_cmd(exp_name, version, resolve_is_half()))
    cmds.append(build_fitprep_cmd(exp_name, sr, version, body.if_f0))
    cmds.append(_fit_cmd(body, sr, version, batch_size))
    cmds.append(build_index_cmd(exp_name, version, os.cpu_count()))
    return _create(
        "pipeline",
        cmds,
        _task_log(exp_name, "pipeline_task.log"),
        setup=_pipeline_notice_setup(batch_note),
        meta={"total_epoch": body.total_epoch},
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
    - fit/pipeline（_TASK_META 登记 total_epoch）且缓冲里有 epoch 行 → 按训练进度换算，
      训练段没有「当前文件」语义
    - 其余（preprocess/extract/index，或 pipeline 的切分/提取段）→ 用脚本自身的进度行
      （design §3.5）换算 done/total，并带上当前文件名
    - 都没有锚点 → 透传 task.progress 原值
    """
    if snapshot["state"] == SUCCESS:
        return {"progress": 1.0, "current": None}
    meta = _TASK_META.get(snapshot["id"]) or {}
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


def _progress_of(snapshot: dict):
    """进度数值版（GET /api/tasks 的紧凑投影等只需要数字的场景）。"""
    return _progress_state(snapshot)["progress"]


def _status_payload(snapshot: dict) -> dict:
    """SSE 的 status 事件体：完整快照去掉 logs——日志由 log 事件承载，避免每条 status
    重复携带最多 1000 行；progress/current 换算为读时计算值。"""
    payload = {key: value for key, value in snapshot.items() if key != "logs"}
    payload.update(_progress_state(snapshot))
    return payload


@router.get("/tasks")
def list_tasks():
    return [
        {
            "id": snapshot["id"],
            "name": snapshot["name"],
            "state": snapshot["state"],
            "progress": _progress_of(snapshot),
            "error": snapshot["error"],
        }
        for snapshot in task_manager.list_tasks()
    ]


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    snapshot = task_manager.get_task(task_id)
    if snapshot is None:
        raise HTTPException(404, "任务不存在：%s" % task_id)
    snapshot.update(_progress_state(snapshot))
    return snapshot


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    """触发终止（进程组 SIGTERM→SIGKILL）。已终态时 cancel 是幂等 no-op，仍返回 202
    并带上当前状态——前端「停止」按钮与任务自然结束竞态时不需要区分。"""
    snapshot = task_manager.get_task(task_id)
    if snapshot is None:
        raise HTTPException(404, "任务不存在：%s" % task_id)
    task_manager.cancel(task_id)
    return JSONResponse(
        status_code=202,
        content={"id": task_id, "state": (task_manager.get_task(task_id) or snapshot)["state"]},
    )


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


if __name__ == "__main__":  # pipeline 的 fit 前置子进程（见 build_fitprep_cmd）
    sys.exit(fitprep_main(sys.argv[1:]))
