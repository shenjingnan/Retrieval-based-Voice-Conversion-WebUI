"""数据集人声分离 runner（server 任务的子进程入口，设计见
docs/plans/2026-09-07-dataset-vocal-separation-design.md）。

输入 datasets/{name}/，把每个音频的目标 stem（人声）落盘到 datasets/{name}_vocals/，
伴奏残余丢弃。分离核心复用旧版 webui 的批量栈（tools.pymss_webui.MSSTBatchSeparator：
模型只加载一次、线程池落盘、NaN 校验），但 torch 栈全部 lazy import——本模块顶层只
依赖标准库（tools/pymss 的 __init__ 会级联加载 torch，故不放包内），pytest 可安全
导入并单测纯逻辑部分（fake batch / fake spec 注入见 tests/test_vocal_dataset.py）。

stdout 协议（tasks.py 重定向进任务日志，server/progress.parse_stage_progress_line
按「进度：n/m」锚点解析，因此进度行骨架与 train/preprocess.py 的计数行一致）：

    [人声分离] 进度：3/12 | song.wav        进度行（done 含失败与跳过，同 extract_f0 口径）
    [人声分离] 跳过（产物已存在）：song.wav  幂等跳过行

失败文件打印「文件名 -> 失败」+ traceback 后继续下一个（单文件损坏不拖垮整批）。
退出码：0 正常（含全部跳过 / 有失败但有产出）；1 无产出且有失败 / 模型缺失；
2 用法错误（目录不存在 / 无音频 / 未知模型）。
"""
import argparse
import sys
import traceback
from pathlib import Path

# 与 server.api.datasets.AUDIO_SUFFIXES 同源（后端上传校验与本地收集必须同一口径，改动须两侧同步）
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})

# 训练集统一 wav；输出格式不作为参数暴露（YAGNI，见设计 §4.1）
OUTPUT_FORMAT = "wav"

# 与 server.api.datasets.SEPARATION_MODELS 的默认值同源：MODEL_SPECS 中去伴奏条目的 label
DEFAULT_MODEL = "去伴奏"

USAGE = "用法：python -m tools.vocal_dataset <输入数据集目录> <输出数据集目录> [--model <模型名>]"


def collect_inputs(input_dir: Path) -> list:
    """待分离音频列表（按名排序）。口径与 server.api.datasets._iter_audio 一致：
    点开头文件（侧车/系统杂物）、子目录、非音频后缀一律排除。"""
    if not input_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in AUDIO_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def expected_output(output_dir: Path, input_path: Path, desired_suffix: str) -> Path:
    """目标 stem 产物路径：{stem}_{desired_suffix}.wav，与 MSSTBatchSeparator._save_output
    的命名同源——幂等跳过的判断必须与真实落盘名逐字一致，漂移会导致重复分离或误跳过。"""
    return output_dir / ("%s_%s.%s" % (input_path.stem, desired_suffix, OUTPUT_FORMAT))


def progress_line(done: int, total: int, name: str) -> str:
    return "[人声分离] 进度：%d/%d | %s" % (done, total, name)


def run_separation(input_dir, output_dir, model, *, resolve_spec, batch_factory, emit=print, only_files=None) -> int:
    """批量分离主体。resolve_spec/batch_factory/emit 注入以便单测；生产接线见 main。

    - resolve_spec(model) → 含 desired_suffix 的模型描述（未知模型抛 ValueError）
    - batch_factory(spec, output_dir) → 上下文管理器，separate_file(path) 落盘目标 stem
      （模型文件缺失抛 FileNotFoundError）
    - emit(line) 逐行输出协议行
    - only_files：只处理这些文件名（逐文件分离入口）；None = 目录内全部音频。
      名单里匹配不到的逐行说明（可见性），全部匹配不到才算用法错误
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # 目录互相覆盖是 CLI 直跑才可能出现的误用（API 侧衍生名恒不同名），先拦下
    if input_dir.exists() and output_dir.exists() and input_dir.resolve() == output_dir.resolve():
        emit("输入与输出目录相同，会覆盖源数据集：%s" % input_dir)
        emit(USAGE)
        return 2

    inputs = collect_inputs(input_dir)
    if only_files:
        wanted = set(only_files)
        for name in dict.fromkeys(only_files):
            if name not in {p.name for p in inputs}:
                emit("输入目录中没有该音频文件：%s" % name)
        inputs = [p for p in inputs if p.name in wanted]
    if not inputs:
        emit("输入目录没有可处理的音频文件：%s" % input_dir)
        emit(USAGE)
        return 2

    try:
        spec = resolve_spec(model)
    except ValueError as exc:
        emit("未知分离模型：%s" % exc)
        emit(USAGE)
        return 2

    try:
        batch = batch_factory(spec, str(output_dir))
    except FileNotFoundError as exc:
        # 权重 ckpt 不在 git 内（assets/pymss_weights 只入库 yaml 配置），首次使用最常见的失败
        emit("模型文件不存在：%s（请将模型与配置下载后放入 assets/pymss_weights）" % exc)
        return 1

    total = len(inputs)
    produced = skipped = failed = 0
    with batch:
        for index, path in enumerate(inputs, 1):
            target = expected_output(output_dir, path, spec.desired_suffix)
            if target.exists():
                skipped += 1
                emit("[人声分离] 跳过（产物已存在）：%s" % path.name)
            else:
                try:
                    batch.separate_file(str(path))
                    produced += 1
                except Exception:  # noqa: BLE001 单文件失败不中断整批，明细进任务日志
                    failed += 1
                    emit("%s -> 失败" % path.name)
                    emit(traceback.format_exc())
            emit(progress_line(index, total, path.name))
    emit(
        "[人声分离] 完成：成功 %d / 跳过 %d / 失败 %d（输出目录：%s）"
        % (produced, skipped, failed, output_dir)
    )
    if produced == 0 and failed > 0:
        return 1
    return 0


def _resolve_spec(model):
    """label → ModelSpec（tools.pymss_webui.MSSTBatchSeparator 所需形态）。
    lazy import：tools.pymss_webui 顶层构造 configs.Config（加载 torch），只允许在
    真正执行分离的子进程里触发。import 前先屏蔽本进程命令行参数——configs.Config
    的 arg_parse 用裸 argparse 解析 sys.argv，见到本 runner 的位置参数会直接
    SystemExit(2)（旧版 pymss worker 子进程在 import 前同样先 sys.argv[:]=sys.argv[:1]；
    server/commands._load_config 是同一手法的服务侧）。此刻参数已解析完，不再需要。"""
    sys.argv = sys.argv[:1]
    from tools.pymss_webui import resolve_model

    return resolve_model(model)


def _batch_factory(spec, output_dir):
    """生产 batch：secondary_root=None → 只落盘目标 stem，伴奏残余不编码不落盘
    （见 tools.pymss_webui.MSSTBatchSeparator 对该分支的支持）。"""
    from tools.pymss_webui import MSSTBatchSeparator

    return MSSTBatchSeparator(spec, OUTPUT_FORMAT, output_dir, None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.vocal_dataset",
        description="把数据集目录中的音频批量分离出人声 stem（RVC 训练集预处理）。",
    )
    parser.add_argument("input_dir", help="输入数据集目录")
    parser.add_argument("output_dir", help="输出数据集目录（衍生数据集）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="分离模型（tools.pymss_webui.MODEL_SPECS 的 label）")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        default=[],
        help="只处理指定文件名（可重复）；缺省处理目录内全部音频",
    )
    args = parser.parse_args(argv)
    return run_separation(
        args.input_dir,
        args.output_dir,
        args.model,
        resolve_spec=_resolve_spec,
        batch_factory=_batch_factory,
        only_files=args.files or None,
    )


if __name__ == "__main__":
    sys.exit(main())
