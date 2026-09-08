"""filelist / config.json 生成与产物校验（移植 webui.py:694-720、1169-1307）。"""
import json
import subprocess
import sys

import pytest

from server import paths
from server.api.training import (
    generate_filelist,
    validate_feature_outputs,
    validate_preprocess_outputs,
    write_config,
)

ROOT = str(paths.ROOT)


@pytest.fixture(autouse=True)
def _logs_dir(monkeypatch, tmp_path):
    """generate_filelist 会向 LOGS_DIR/mute 写静音资产：钉到 tmp，不污染真实仓库。"""
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")


def _touch(path, name):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(b"x")


def _make_exp(tmp_path, *, names=("a", "b"), extra_gt=(), version="v2", f0=True):
    """造假产物目录树；names 里的 stem 四目录齐全，extra_gt 只进 0_gt_wavs。"""
    exp = tmp_path / "logs" / "mi-test"
    fea_name = "3_feature256" if version == "v1" else "3_feature768"
    dirs = {
        "gt": exp / "0_gt_wavs",
        "fea": exp / fea_name,
        "f0": exp / "2a_f0",
        "f0nsf": exp / "2b-f0nsf",
    }
    for stem in names:
        _touch(dirs["gt"], f"{stem}.wav")
        _touch(dirs["fea"], f"{stem}.npy")
        if f0:
            _touch(dirs["f0"], f"{stem}.wav.npy")
            _touch(dirs["f0nsf"], f"{stem}.wav.npy")
    for stem in extra_gt:
        _touch(dirs["gt"], f"{stem}.wav")
    return exp


def test_generate_filelist_f0(monkeypatch, tmp_path):
    exp = _make_exp(tmp_path, names=("a", "b"), extra_gt=("only_gt",), version="v2")
    monkeypatch.setattr("server.api.training.shuffle", lambda x: None)  # 固定顺序做快照

    generate_filelist(exp, "48k", "v2", True)

    lines = (exp / "filelist.txt").read_text(encoding="utf8").split("\n")
    assert len(lines) == 4, lines  # 2 条数据行 + 2 条 mute 行
    gt, fea, f0, f0nsf = (
        str(exp / "0_gt_wavs").replace("\\", "\\\\"),
        str(exp / "3_feature768").replace("\\", "\\\\"),
        str(exp / "2a_f0").replace("\\", "\\\\"),
        str(exp / "2b-f0nsf").replace("\\", "\\\\"),
    )
    assert lines[0] == f"{gt}/a.wav|{fea}/a.npy|{f0}/a.wav.npy|{f0nsf}/a.wav.npy|0"
    assert lines[1] == f"{gt}/b.wav|{fea}/b.npy|{f0}/b.wav.npy|{f0nsf}/b.wav.npy|0"
    # mute 行：仓库根 + mute{sr}.wav + feature 维度按 version；无结尾换行
    assert lines[2] == (
        f"{ROOT}/logs/mute/0_gt_wavs/mute48k.wav|{ROOT}/logs/mute/3_feature768/mute.npy"
        f"|{ROOT}/logs/mute/2a_f0/mute.wav.npy|{ROOT}/logs/mute/2b-f0nsf/mute.wav.npy|0"
    )
    assert lines[3] == lines[2]
    assert not (exp / "filelist.txt").read_text(encoding="utf8").endswith("\n")


def test_generate_filelist_no_f0(monkeypatch, tmp_path):
    exp = _make_exp(tmp_path, names=("a",), version="v2", f0=False)
    monkeypatch.setattr("server.api.training.shuffle", lambda x: None)

    generate_filelist(exp, "40k", "v2", False)

    lines = (exp / "filelist.txt").read_text(encoding="utf8").split("\n")
    assert len(lines) == 3
    assert lines[0] == (
        f"{exp}/0_gt_wavs/a.wav|{exp}/3_feature768/a.npy|0"
    )
    assert lines[1] == (
        f"{ROOT}/logs/mute/0_gt_wavs/mute40k.wav|{ROOT}/logs/mute/3_feature768/mute.npy|0"
    )


def test_generate_filelist_v1_uses_feature256(monkeypatch, tmp_path):
    exp = _make_exp(tmp_path, names=("a",), version="v1")
    monkeypatch.setattr("server.api.training.shuffle", lambda x: None)

    generate_filelist(exp, "48k", "v1", True)

    lines = (exp / "filelist.txt").read_text(encoding="utf8").split("\n")
    assert "3_feature256" in lines[0]
    assert f"{ROOT}/logs/mute/3_feature256/mute.npy" in lines[2]


def test_generate_filelist_stem_intersection_ignores_double_extension(tmp_path):
    # stem 取 name.split(".")[0]：x.wav 与 x.wav.npy 视为同一 stem（webui 既有行为）
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "0_gt_wavs", "x.wav")
    _touch(exp / "3_feature768", "x.wav.npy")
    _touch(exp / "2a_f0", "x.wav.npy")
    _touch(exp / "2b-f0nsf", "x.wav.npy")

    generate_filelist(exp, "48k", "v2", True)

    assert len((exp / "filelist.txt").read_text(encoding="utf8").split("\n")) == 3


def test_generate_filelist_shuffle_called(monkeypatch, tmp_path):
    exp = _make_exp(tmp_path, names=("a", "b", "c"))
    seen = []
    monkeypatch.setattr("server.api.training.shuffle", lambda opt: seen.append(list(opt)))

    generate_filelist(exp, "48k", "v2", True)

    assert len(seen) == 1 and len(seen[0]) == 5  # 3 数据行 + 2 mute 行一起被打乱


def test_generate_filelist_empty_intersection_raises(tmp_path):
    exp = _make_exp(tmp_path, names=("a",), version="v2")
    # 特征目录被清空 → 交集为空
    for f in (exp / "3_feature768").iterdir():
        f.unlink()

    with pytest.raises(ValueError) as e:
        generate_filelist(exp, "48k", "v2", True)
    assert str(e.value) == "没有可用于训练的有效音频，请先完成数据切分和特征提取"


def test_generate_filelist_missing_dirs_give_actionable_message(tmp_path):
    """跳过 preprocess/extract 直接 fit：目录缺失按空集处理，报「数据未就绪」而非
    裸的 FileNotFoundError errno。"""
    exp = tmp_path / "logs" / "fresh-exp"
    exp.mkdir(parents=True)

    with pytest.raises(ValueError) as e:
        generate_filelist(exp, "48k", "v2", True)
    assert "没有可用于训练的有效音频" in str(e.value)


def test_write_config_v2_48k_uses_v2_template(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    exp.mkdir(parents=True)

    write_config(exp, "48k", "v2")

    got = json.loads((exp / "config.json").read_text(encoding="utf8"))
    template = json.loads((paths.ROOT / "configs" / "v2" / "48k.json").read_text(encoding="utf8"))
    assert got == template
    assert got["model"]["spk_embed_dim"] == 109
    assert "speaker_info" not in got


def test_write_config_v2_40k_falls_back_to_v1_template(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    exp.mkdir(parents=True)

    write_config(exp, "40k", "v2")

    template = json.loads((paths.ROOT / "configs" / "v1" / "40k.json").read_text(encoding="utf8"))
    assert json.loads((exp / "config.json").read_text(encoding="utf8")) == template


@pytest.mark.parametrize("sr,version", [("48k", "v1"), ("32k", "v1"), ("40k", "v1")])
def test_write_config_v1_uses_v1_template(tmp_path, sr, version):
    exp = tmp_path / "logs" / "mi-test"
    exp.mkdir(parents=True)

    write_config(exp, sr, version)

    template = json.loads((paths.ROOT / "configs" / "v1" / f"{sr}.json").read_text(encoding="utf8"))
    assert json.loads((exp / "config.json").read_text(encoding="utf8")) == template


def test_write_config_existing_kept(tmp_path):
    # 续训语义：已有 config.json 原样沿用（仅 pop speaker_info），不回填模板
    exp = tmp_path / "logs" / "mi-test"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text(
        json.dumps({"model": {"spk_embed_dim": 109, "keep_me": 1}, "speaker_info": [{"id": 0, "name": "a"}]}),
        encoding="utf8",
    )

    write_config(exp, "48k", "v2")

    got = json.loads((exp / "config.json").read_text(encoding="utf8"))
    assert got["model"]["keep_me"] == 1
    assert "speaker_info" not in got
    assert got["model"]["spk_embed_dim"] == 109


def test_write_config_sorted_indent_trailing_newline(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    exp.mkdir(parents=True)

    write_config(exp, "48k", "v2")

    raw = (exp / "config.json").read_text(encoding="utf8")
    assert raw.endswith("\n")
    assert raw == json.dumps(json.loads(raw), ensure_ascii=False, indent=4, sort_keys=True) + "\n"


def test_validate_preprocess_outputs_ok(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "0_gt_wavs", "a.wav")
    _touch(exp / "0_gt_wavs", "b.wav")
    _touch(exp / "1_16k_wavs", "a.wav")
    _touch(exp / "1_16k_wavs", "c.wav")  # 只要求交集非空，多余文件允许

    validate_preprocess_outputs(exp)  # 不抛即通过


def test_validate_preprocess_outputs_no_gt(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")

    with pytest.raises(RuntimeError) as e:
        validate_preprocess_outputs(exp)
    assert "数据切分没有生成有效训练音频" in str(e.value)


def test_validate_preprocess_outputs_no_16k(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "0_gt_wavs", "a.wav")

    with pytest.raises(RuntimeError) as e:
        validate_preprocess_outputs(exp)
    assert "数据切分没有生成16k音频" in str(e.value)


def test_validate_preprocess_outputs_mismatch(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "0_gt_wavs", "a.wav")
    _touch(exp / "1_16k_wavs", "z.wav")

    with pytest.raises(RuntimeError) as e:
        validate_preprocess_outputs(exp)
    assert "数据切分输出文件不匹配" in str(e.value)


def test_validate_feature_outputs_ok_with_f0(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")
    _touch(exp / "3_feature768", "a.npy")
    _touch(exp / "2a_f0", "a.wav.npy")
    _touch(exp / "2b-f0nsf", "a.wav.npy")

    matched = validate_feature_outputs(exp, "v2", True)
    assert matched == {"a"}


def test_validate_feature_outputs_ok_without_f0(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")
    _touch(exp / "3_feature256", "a.npy")  # v1 → 256

    assert validate_feature_outputs(exp, "v1", False) == {"a"}


def test_validate_feature_outputs_missing_f0(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")
    _touch(exp / "3_feature768", "a.npy")
    _touch(exp / "2a_f0", "a.wav.npy")
    # 2b-f0nsf 缺失

    with pytest.raises(RuntimeError) as e:
        validate_feature_outputs(exp, "v2", True)
    assert "F0提取没有生成有效结果" in str(e.value)


def test_validate_feature_outputs_no_feature(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")

    with pytest.raises(RuntimeError) as e:
        validate_feature_outputs(exp, "v2", False)
    assert "HuBERT特征提取没有生成有效结果" in str(e.value)


def test_validate_feature_outputs_mismatch(tmp_path):
    exp = tmp_path / "logs" / "mi-test"
    _touch(exp / "1_16k_wavs", "a.wav")
    _touch(exp / "3_feature768", "z.npy")

    with pytest.raises(RuntimeError):
        validate_feature_outputs(exp, "v2", False)


def test_training_module_does_not_load_torch():
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import server.api.training;"
        "assert 'torch' not in sys.modules and 'configs.config' not in sys.modules;"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(paths.ROOT), capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# mute 静音资产：缺失会让训练子进程加载数据集时崩溃（且流水线仍继续跑索引）
# ---------------------------------------------------------------------------


def test_generate_filelist_creates_mute_assets(monkeypatch, tmp_path):
    exp = _make_exp(tmp_path, names=("a",))
    generate_filelist(exp, "40k", "v2", True)

    import wave

    import numpy as np

    mute_dir = paths.LOGS_DIR / "mute"
    wav = mute_dir / "0_gt_wavs" / "mute40k.wav"
    assert wav.is_file() and wav.stat().st_size > 44
    with wave.open(str(wav)) as handle:
        assert handle.getframerate() == 40000
        assert handle.getnchannels() == 1
    fea = np.load(mute_dir / "3_feature768" / "mute.npy")
    assert fea.shape == (50, 768) and fea.dtype == np.float32
    assert np.load(mute_dir / "2a_f0" / "mute.wav.npy").dtype == np.int64
    assert np.load(mute_dir / "2b-f0nsf" / "mute.wav.npy").dtype == np.float32


def test_generate_filelist_mute_assets_idempotent_and_per_version(monkeypatch, tmp_path):
    """幂等：重写 filelist 不重新生成；v1 用 256 维特征（与 v2 的 npy 分开）。"""
    exp = _make_exp(tmp_path, names=("a",))
    generate_filelist(exp, "48k", "v2", True)
    wav = paths.LOGS_DIR / "mute" / "0_gt_wavs" / "mute48k.wav"
    first_mtime = wav.stat().st_mtime_ns

    generate_filelist(exp, "48k", "v2", True)
    assert wav.stat().st_mtime_ns == first_mtime

    exp_v1 = _make_exp(tmp_path, names=("a",), version="v1")
    generate_filelist(exp_v1, "48k", "v1", True)
    assert (paths.LOGS_DIR / "mute" / "3_feature256" / "mute.npy").is_file()
