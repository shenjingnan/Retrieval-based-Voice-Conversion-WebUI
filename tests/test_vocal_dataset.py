"""tools/vocal_dataset.py 纯逻辑单测。

fake spec / batch / emit 全部注入（模块顶层不 import torch，本文件可被 pytest 安全
收集）。进度行协议与 server.progress.parse_stage_progress_line 的锚点互通，顺带做
交叉断言——runner 打印的行必须能被服务端进度解析器读懂，这是两边唯一的契约点。
"""
from pathlib import Path

import pytest

from server.progress import parse_stage_progress_line
from tools import vocal_dataset as vd


class FakeSpec:
    """resolve_spec 的替身：只带 runner 实际读取的 desired_suffix 字段。"""

    def __init__(self, desired_suffix="vocals"):
        self.desired_suffix = desired_suffix


class FakeBatch:
    """batch_factory 的替身：separate_file 落盘与 MSSTBatchSeparator 同名的目标 stem
    文件（{stem}_{suffix}.wav），可指定按文件名失败。"""

    def __init__(self, spec, output_dir, fail_stems=()):
        self.spec = spec
        self.output_dir = Path(output_dir)
        # 真实 batch 在构造时建目录（MSSTBatchSeparator ctor 的 os.makedirs）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fail_stems = set(fail_stems)
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False

    def separate_file(self, path):
        self.calls.append(path)
        stem = Path(path).stem
        if stem in self.fail_stems:
            raise RuntimeError("boom: %s" % stem)
        (self.output_dir / ("%s_%s.wav" % (stem, self.spec.desired_suffix))).touch()


def _factory(spec, output_dir, fail_stems=()):
    return FakeBatch(spec, output_dir, fail_stems)


def run(tmp_path, inputs=(), output_name="ds_vocals", model="去伴奏", **kwargs):
    """驱动 run_separation 的便捷封装：返回 (退出码, 输出行, batch 或 None)。"""
    input_dir = tmp_path / "ds"
    input_dir.mkdir(exist_ok=True)
    for name in inputs:
        (input_dir / name).write_bytes(b"x")
    output_dir = tmp_path / output_name
    lines = []
    holder = {}
    code = vd.run_separation(
        input_dir,
        output_dir,
        model,
        resolve_spec=lambda _model: kwargs.get("spec", FakeSpec()),
        batch_factory=lambda spec, out: holder.setdefault(
            "batch", _factory(spec, out, kwargs.get("fail_stems", ()))
        ),
        emit=lines.append,
        only_files=kwargs.get("only_files"),
    )
    return code, lines, holder.get("batch")


# ---------------------------------------------------------------------------
# 输入收集与命名
# ---------------------------------------------------------------------------


def test_collect_inputs_filters_suffixes_dotfiles_and_dirs(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    for name in ("a.wav", "b.MP3", "c.FlAc", "d.ogg", "e.m4a"):
        (root / name).write_bytes(b"x")
    (root / "score.txt").write_bytes(b"x")  # 非音频
    (root / ".hidden.wav").write_bytes(b"x")  # 点开头
    (root / "sub").mkdir()
    (root / "sub" / "nested.wav").write_bytes(b"x")  # 子目录不递归

    assert [p.name for p in vd.collect_inputs(root)] == [
        "a.wav",
        "b.MP3",
        "c.FlAc",
        "d.ogg",
        "e.m4a",
    ]


def test_collect_inputs_missing_dir_returns_empty(tmp_path):
    assert vd.collect_inputs(tmp_path / "ghost") == []


def test_expected_output_naming_matches_batch_save(tmp_path):
    # {stem}_{suffix}.wav：幂等跳过判断与 MSSTBatchSeparator._save_output 的落盘名逐字同源
    assert vd.expected_output(tmp_path, tmp_path / "song.wav", "vocals") == (
        tmp_path / "song_vocals.wav"
    )
    assert vd.expected_output(tmp_path, tmp_path / "song.MP3", "noreverb") == (
        tmp_path / "song_noreverb.wav"
    )


def test_progress_line_parseable_by_server_progress():
    # 契约点：runner 的进度行必须被 server/progress.parse_stage_progress_line 解析
    parsed = parse_stage_progress_line(vd.progress_line(3, 12, "a.wav"))
    assert parsed == {"done": 3, "total": 12, "current": "a.wav"}


# ---------------------------------------------------------------------------
# run_separation：正常流 / 幂等 / 失败
# ---------------------------------------------------------------------------


def test_happy_path_separates_all_and_exits_zero(tmp_path):
    code, lines, batch = run(tmp_path, inputs=("a.wav", "b.flac"))

    assert code == 0
    assert [Path(c).name for c in batch.calls] == ["a.wav", "b.flac"]
    assert vd.progress_line(1, 2, "a.wav") in lines
    assert vd.progress_line(2, 2, "b.flac") in lines
    assert any("完成：成功 2 / 跳过 0 / 失败 0" in line for line in lines)
    assert batch.closed  # with 语义：收尾必须关闭（真实实现里释放模型显存）


def test_existing_outputs_are_skipped_idempotently(tmp_path):
    # 预置一份产物：该文件被跳过且不触发 separate_file（断点续跑不重复计算）
    out = tmp_path / "ds_vocals"
    out.mkdir()
    (out / "a_vocals.wav").touch()

    code, lines, batch = run(tmp_path, inputs=("a.wav", "b.wav"))

    assert code == 0
    assert batch.calls == [str(tmp_path / "ds" / "b.wav")]
    assert "[人声分离] 跳过（产物已存在）：a.wav" in lines
    assert any("成功 1 / 跳过 1 / 失败 0" in line for line in lines)


def test_all_skipped_still_exits_zero(tmp_path):
    out = tmp_path / "ds_vocals"
    out.mkdir()
    for stem in ("a", "b"):
        (out / ("%s_vocals.wav" % stem)).touch()

    code, lines, batch = run(tmp_path, inputs=("a.wav", "b.wav"))

    assert code == 0
    assert batch.calls == []
    assert any("成功 0 / 跳过 2 / 失败 0" in line for line in lines)


def test_single_file_failure_continues_batch(tmp_path):
    # 有产出 → 退出码 0（单文件损坏不拖垮整批）；traceback 与失败文件名进日志
    code, lines, batch = run(
        tmp_path, inputs=("bad.wav", "good.wav"), fail_stems=("bad",)
    )

    assert code == 0
    assert len(batch.calls) == 2  # 失败后继续处理下一个
    assert any(line.startswith("bad.wav -> 失败") for line in lines)
    assert any("Traceback" in line or "boom: bad" in line for line in lines)
    assert any("成功 1 / 跳过 0 / 失败 1" in line for line in lines)


def test_no_output_with_failures_exits_one(tmp_path):
    code, lines, _batch = run(
        tmp_path, inputs=("a.wav", "b.wav"), fail_stems=("a", "b")
    )

    assert code == 1
    assert any("成功 0 / 跳过 0 / 失败 2" in line for line in lines)


# ---------------------------------------------------------------------------
# run_separation：用法错误与环境错误
# ---------------------------------------------------------------------------


def test_empty_dataset_returns_usage_error(tmp_path):
    code, lines, batch = run(tmp_path, inputs=())

    assert code == 2
    assert batch is None
    assert any("没有可处理的音频文件" in line for line in lines)
    assert vd.USAGE in lines


def test_input_dir_missing_returns_usage_error(tmp_path):
    lines = []
    code = vd.run_separation(
        tmp_path / "ghost",
        tmp_path / "out",
        "去伴奏",
        resolve_spec=lambda _m: FakeSpec(),
        batch_factory=lambda spec, out: FakeBatch(spec, out),
        emit=lines.append,
    )

    assert code == 2
    assert vd.USAGE in lines


def test_same_input_and_output_dir_rejected(tmp_path):
    same = tmp_path / "ds"
    same.mkdir()
    (same / "a.wav").write_bytes(b"x")
    lines = []
    code = vd.run_separation(
        same,
        same,
        "去伴奏",
        resolve_spec=lambda _m: FakeSpec(),
        batch_factory=lambda spec, out: FakeBatch(spec, out),
        emit=lines.append,
    )

    assert code == 2
    assert any("输入与输出目录相同" in line for line in lines)


def test_unknown_model_returns_usage_error(tmp_path):
    def broken_resolver(_model):
        raise ValueError("Unknown separation model: 不存在")

    input_dir = tmp_path / "ds"
    input_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"x")
    lines = []
    code = vd.run_separation(
        input_dir,
        tmp_path / "out",
        "不存在",
        resolve_spec=broken_resolver,
        batch_factory=lambda spec, out: FakeBatch(spec, out),
        emit=lines.append,
    )

    assert code == 2
    assert any("未知分离模型" in line for line in lines)
    assert vd.USAGE in lines


def test_missing_model_files_returns_one(tmp_path):
    # assets/pymss_weights 缺 ckpt 是首次使用最常见的失败：明确文案 + 任务失败退出码
    input_dir = tmp_path / "ds"
    input_dir.mkdir()
    (input_dir / "a.wav").write_bytes(b"x")

    def missing_model_factory(spec, out):
        raise FileNotFoundError(f"{out}/model.ckpt")

    lines = []
    code = vd.run_separation(
        input_dir,
        tmp_path / "out",
        "去伴奏",
        resolve_spec=lambda _m: FakeSpec(),
        batch_factory=missing_model_factory,
        emit=lines.append,
    )

    assert code == 1
    assert any("模型文件不存在" in line for line in lines)
    assert any("assets/pymss_weights" in line for line in lines)


def test_custom_desired_suffix_used_for_skip_and_naming(tmp_path):
    # 提主旋律模型的 desired_suffix 是 main_vocal：跳过判断必须跟随后缀而非写死 vocals
    out = tmp_path / "ds_vocals"
    out.mkdir()
    (out / "a_main_vocal.wav").touch()

    code, lines, batch = run(
        tmp_path, inputs=("a.wav",), spec=FakeSpec(desired_suffix="main_vocal")
    )

    assert code == 0
    assert batch.calls == []
    assert "[人声分离] 跳过（产物已存在）：a.wav" in lines


# ---------------------------------------------------------------------------
# main（argparse 装配层）
# ---------------------------------------------------------------------------


def test_main_defaults_to_builtin_model(tmp_path, monkeypatch):
    # --model 省略 → DEFAULT_MODEL 传给 run_separation；run_separation 本身打桩
    seen = {}

    def fake_run(input_dir, output_dir, model, **kwargs):
        seen["model"] = model
        return 0

    monkeypatch.setattr(vd, "run_separation", fake_run)
    assert vd.main([str(tmp_path / "ds"), str(tmp_path / "out")]) == 0
    assert seen["model"] == vd.DEFAULT_MODEL == "去伴奏"


# ---------------------------------------------------------------------------
# 逐文件分离（only_files 子集）
# ---------------------------------------------------------------------------


def test_only_files_processes_subset(tmp_path):
    """逐文件分离：只处理名单内文件，进度分母随之缩小。"""
    code, lines, batch = run(
        tmp_path, inputs=("a.wav", "b.wav", "c.wav"), only_files=["b.wav"]
    )

    assert code == 0
    assert [Path(c).name for c in batch.calls] == ["b.wav"]
    assert vd.progress_line(1, 1, "b.wav") in lines
    assert any("成功 1 / 跳过 0 / 失败 0" in line for line in lines)


def test_only_files_reports_missing_names_but_continues(tmp_path):
    """名单里匹配不到的逐行说明（可见性），有命中的照常处理。"""
    code, lines, batch = run(
        tmp_path, inputs=("a.wav",), only_files=["a.wav", "ghost.wav"]
    )

    assert code == 0
    assert [Path(c).name for c in batch.calls] == ["a.wav"]
    assert any("输入目录中没有该音频文件：ghost.wav" in line for line in lines)


def test_only_files_all_missing_returns_usage_error(tmp_path):
    code, lines, batch = run(tmp_path, inputs=("a.wav",), only_files=["ghost.wav"])

    assert code == 2
    assert batch is None
    assert any("没有可处理的音频文件" in line for line in lines)
