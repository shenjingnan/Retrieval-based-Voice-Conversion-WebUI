"""训练进度解析测试：train.log 行解析 / 产物计数 / 日志增量读取。

锚点格式取自 train/train.py:530-546 的真实输出（fixture 用真实格式片段）。
"""
import subprocess
import sys
from pathlib import Path

import pytest

from server import paths
from server.progress import (
    TailReader,
    count_dir_progress,
    parse_stage_progress_line,
    parse_train_line,
)

FIXTURE = Path(__file__).parent / "fixtures" / "train_log_sample.log"

LOSS_LINE = "loss_disc=12.345, loss_gen=23.456, loss_fm=1.234,loss_mel=45.678, loss_kl=0.912"
LOSS_DICT = {
    "loss_disc": 12.345,
    "loss_gen": 23.456,
    "loss_fm": 1.234,
    "loss_mel": 45.678,
    "loss_kl": 0.912,
}


# ---------------------------------------------------------------------------
# 1. parse_train_line
# ---------------------------------------------------------------------------


def test_parse_epoch_line_chinese():
    assert parse_train_line("训练轮次：20 [34%]") == {"epoch": 20, "pct": 34.0}


def test_parse_epoch_line_english_short():
    assert parse_train_line("Epoch: 20 [34%]") == {"epoch": 20, "pct": 34.0}


def test_parse_epoch_line_english_locale():
    # i18n/locale/en_US.json:239 的真实译文（train/utils.py 的 basicConfig 带 logging 前缀）
    assert parse_train_line("INFO:train.train:Training epoch: 20 [34%]") == {
        "epoch": 20,
        "pct": 34.0,
    }


def test_parse_epoch_line_keeps_decimal_pct():
    assert parse_train_line("训练轮次：3 [7.5%]") == {"epoch": 3, "pct": 7.5}


def test_parse_step_lr_line():
    assert parse_train_line("[1234, 0.0001]") == {"step": 1234, "lr": 0.0001}


def test_parse_step_lr_line_scientific_notation():
    # logger.info([global_step, lr]) 的 str(list) 输出，lr 常见 1e-05 形态
    assert parse_train_line("[1235, 9.9e-05]") == {"step": 1235, "lr": 9.9e-05}


def test_parse_step_lr_line_with_logging_prefix():
    # 真实形态：train/train.py:138 用 utils.get_logger(hps.model_dir) 把 logger 名绑成实验名
    # （train/utils.py:448 logging.getLogger(os.path.basename(model_dir))），stdout 重定向后的行是
    # "INFO:{实验名}:[step, lr]"；前缀必须与 logger 名字符集无关（连字符实验名也要能解析）
    assert parse_train_line("INFO:mi-test:[1234, 0.0001]") == {"step": 1234, "lr": 0.0001}


@pytest.mark.parametrize(
    "line, expected",
    [
        ("INFO:mi-test:[1234, 0.0001]", {"step": 1234, "lr": 0.0001}),  # 连字符实验名
        ("INFO:my_exp_v2:[99, 9.9e-05]", {"step": 99, "lr": 9.9e-05}),  # 下划线
        ("INFO:train.train:[7, 0.1]", {"step": 7, "lr": 0.1}),  # 模块路径形态的 logger 名
        ("WARNING:mi-test:[7, 0.1]", {"step": 7, "lr": 0.1}),  # 级别不限于 INFO
        ("[7, 0.1]", {"step": 7, "lr": 0.1}),  # 无前缀（fixture 形态）
        ("[7, 0.1]  ", {"step": 7, "lr": 0.1}),  # 尾部空白
        ("INFO:mi-test:INFO:[1, 0.1]", {"step": 1, "lr": 0.1}),  # 畸形双重前缀：仍按尾部锚点解析
    ],
)
def test_parse_step_lr_line_prefix_is_logger_name_agnostic(line, expected):
    assert parse_train_line(line) == expected


def test_parse_step_lr_line_negative_step_accepted():
    # global_step 实际不会为负，负号仅作防御；行为有定义即可
    assert parse_train_line("INFO:mi-test:[-1, 0.1]") == {"step": -1, "lr": 0.1}


@pytest.mark.parametrize(
    "line",
    [
        "INFO:mi-test:[1234, 0.0001] trailing text",  # 锚点不在行尾
        "INFO:mi-test:[1234]",  # 缺 lr
        "INFO:mi-test:[1234, ]",  # lr 为空
        "INFO:mi-test:[1234,abc]",  # lr 非数字
        "2026-09-06 12:00:00,000\tmi-test\tINFO\t[1234, 0.0001]\textra",  # asctime 行加尾巴
    ],
)
def test_parse_step_lr_line_lookalikes_return_none(line):
    assert parse_train_line(line) is None


def test_parse_step_lr_line_inside_asctime_formatted_line():
    """train.log 的 FileHandler 行（train/utils.py:451 的 asctime\\t实验名\\tLEVEL\\t行）尾部也是
    [step, lr]，尾锚定能解析出数值；但 Task 5 不得把任务日志指向 train.log——FileHandler 与
    stdout 重定向双写会让锚点行双份（见 docs/plans/2026-09-06-p2-training-plan.md Task 5 第 9 条）。"""
    line = "2026-09-06 12:00:00,000\tmi-test\tINFO\t[1234, 0.0001]"
    assert parse_train_line(line) == {"step": 1234, "lr": 0.0001}


def test_parse_loss_line():
    assert parse_train_line(LOSS_LINE) == LOSS_DICT


def test_parse_loss_line_negative_value():
    # GAN 判别器损失可为负，正则放行负号
    line = "loss_disc=-0.125, loss_gen=23.456, loss_fm=1.234,loss_mel=45.678, loss_kl=0.912"
    assert parse_train_line(line)["loss_disc"] == -0.125


def test_parse_loss_line_requires_exact_layout():
    # loss_fm 后的逗号无空格是 train/train.py:545 的实际输出，空格变体不匹配
    spaced = "loss_disc=1.0, loss_gen=2.0, loss_fm=3.0, loss_mel=4.0, loss_kl=5.0"
    assert parse_train_line(spaced) is None
    assert parse_train_line("loss_disc=1.0, loss_gen=2.0") is None


@pytest.mark.parametrize(
    "line",
    ["", "   ", "\n", "INFO:train.train:Saving model and optimizer state at epoch 20", "0%|"],
)
def test_parse_unrecognised_line_returns_none(line):
    assert parse_train_line(line) is None


def test_fixture_covers_all_anchors():
    lines = FIXTURE.read_text(encoding="utf8").splitlines()
    parsed = [parse_train_line(line) for line in lines]
    assert parsed == [
        {"epoch": 20, "pct": 34.0},
        {"step": 1234, "lr": 0.0001},
        LOSS_DICT,
        {"epoch": 20, "pct": 34.0},
        None,
    ]


@pytest.mark.parametrize(
    "line",
    [
        "训练轮次：3 [7.5.5%]",  # pct 畸形：多段小数
        "训练轮次：3 [%]",  # pct 缺失
        "loss_disc=1.2.3, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0",  # 畸形小数
        "INFO:mi-test:[1, 0.1.2]",  # lr 畸形：多段小数
        "INFO:mi-test:[1, --0.1]",  # lr 畸形：双负号
        "INFO:mi-test:[1, 0.1e]",  # lr 畸形：指数无尾数
        "INFO:mi-test:[1.5, 0.1]",  # step 必须是整数
    ],
)
def test_parse_malformed_numbers_return_none(line):
    """畸形数值必须返回 None 而不是让 float() 抛 ValueError——下游在工作线程/SSE 里调用，
    异常会把健康训练误判 failed 或中断日志流。"""
    assert parse_train_line(line) is None


def test_parse_float_forms_still_accepted():
    """收紧后的正则仍要放行真实日志会出现的全部数值形态。"""
    assert parse_train_line("[1, 1e-05]") == {"step": 1, "lr": 1e-05}
    assert parse_train_line("[1, .5]") == {"step": 1, "lr": 0.5}
    assert parse_train_line("[1, 1.]") == {"step": 1, "lr": 1.0}
    assert parse_train_line("[1, -0.25]") == {"step": 1, "lr": -0.25}
    assert parse_train_line("[1, 2e5]") == {"step": 1, "lr": 200000.0}
    assert parse_train_line("训练轮次：3 [0%]") == {"epoch": 3, "pct": 0.0}
    line = "loss_disc=-0.5, loss_gen=2, loss_fm=3.25,loss_mel=4.0, loss_kl=5.125"
    assert parse_train_line(line) == {
        "loss_disc": -0.5,
        "loss_gen": 2.0,
        "loss_fm": 3.25,
        "loss_mel": 4.0,
        "loss_kl": 5.125,
    }


# ---------------------------------------------------------------------------
# 2. count_dir_progress
# ---------------------------------------------------------------------------


def test_count_dir_progress_ratio(tmp_path):
    target = tmp_path / "1_16k_wavs"
    target.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        (target / name).write_bytes(b"x")
    assert count_dir_progress(target, 4) == 0.75


def test_count_dir_progress_empty_dir_is_zero(tmp_path):
    target = tmp_path / "2a_f0"
    target.mkdir()
    assert count_dir_progress(target, 10) == 0.0


def test_count_dir_progress_clamps_to_one(tmp_path):
    target = tmp_path / "1_16k_wavs"
    target.mkdir()
    for name in ("a.wav", "b.wav", "c.wav", "d.wav", "e.wav", "f.wav"):
        (target / name).write_bytes(b"x")
    assert count_dir_progress(target, 4) == 1.0


def test_count_dir_progress_ignores_subdirectories(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "a.npy").write_bytes(b"x")
    (target / "sub").mkdir()
    assert count_dir_progress(target, 1) == 1.0


def test_count_dir_progress_invalid_total_returns_none(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    assert count_dir_progress(target, 0) is None
    assert count_dir_progress(target, -3) is None


def test_count_dir_progress_missing_dir_returns_none(tmp_path):
    assert count_dir_progress(tmp_path / "nope", 10) is None


# ---------------------------------------------------------------------------
# 3. TailReader
# ---------------------------------------------------------------------------


def test_tail_reader_missing_file_returns_empty(tmp_path):
    reader = TailReader(tmp_path / "not_yet.log")
    assert reader.read_new_lines() == []


def test_tail_reader_reads_new_lines_only(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("第一行\n第二行\n", encoding="utf8")
    reader = TailReader(log)

    assert reader.read_new_lines() == ["第一行", "第二行"]
    assert reader.read_new_lines() == []  # 无新内容

    with open(log, "a", encoding="utf8") as handle:
        handle.write("第三行\n")
    assert reader.read_new_lines() == ["第三行"]


def test_tail_reader_keeps_partial_line(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("", encoding="utf8")
    reader = TailReader(log)

    with open(log, "a", encoding="utf8") as handle:
        handle.write("训练轮次：3 [")
    assert reader.read_new_lines() == []  # 残行不进结果

    with open(log, "a", encoding="utf8") as handle:
        handle.write("30%]\n")
    assert reader.read_new_lines() == ["训练轮次：3 [30%]"]


def test_tail_reader_resets_offset_on_truncate(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("旧日志第一行\n旧日志第二行\n", encoding="utf8")
    reader = TailReader(log)
    assert reader.read_new_lines() == ["旧日志第一行", "旧日志第二行"]

    # 编排层在启动前截断日志（对齐 webui），读取器须从头重读而不是报错/丢内容
    with open(log, "w", encoding="utf8") as handle:
        handle.write("新的一行\n")
    assert reader.read_new_lines() == ["新的一行"]


def test_tail_reader_truncated_below_previous_offset(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("a\n" * 50, encoding="utf8")
    reader = TailReader(log)
    assert len(reader.read_new_lines()) == 50

    log.write_text("b\n", encoding="utf8")  # 截断到远小于上次偏移
    assert reader.read_new_lines() == ["b"]


def test_tail_reader_preserves_multibyte_content(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("", encoding="utf8")
    reader = TailReader(log)
    with open(log, "a", encoding="utf8") as handle:
        handle.write("loss_disc=1.0, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0\n")
    assert parse_train_line(reader.read_new_lines()[0])["loss_disc"] == 1.0


def test_module_does_not_load_torch_or_configs():
    """延迟导入纪律：import server.progress 不得连带加载 torch/configs。"""
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import server.progress;"
        "assert 'torch' not in sys.modules, 'torch 被加载';"
        "assert 'configs.config' not in sys.modules, 'configs 被加载';"
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(paths.ROOT), capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# 2. 阶段进度行（design §3.5：切分/提取的进度与当前文件名）
# ---------------------------------------------------------------------------


def test_parse_stage_progress_slicing_chinese():
    line = "[数据切分] 进度：12/34 | a.wav"
    assert parse_stage_progress_line(line) == {"done": 12, "total": 34, "current": "a.wav"}


def test_parse_stage_progress_f0_chinese():
    line = "[F0提取] 进度：3/9 | 成功：2 | 跳过：1 | b.flac"
    assert parse_stage_progress_line(line) == {"done": 3, "total": 9, "current": "b.flac"}


def test_parse_stage_progress_hubert_ignores_shape_field():
    """HuBERT 行末字段是特征形状元组，当前文件名取其前一个含扩展名的字段。"""
    line = "[HuBERT特征] 进度：7/10 | 成功：7 | 失败：0 | c.wav.npy | (1, 768)"
    assert parse_stage_progress_line(line) == {
        "done": 7, "total": 10, "current": "c.wav.npy",
    }


@pytest.mark.parametrize(
    "line",
    [
        "[Data slicing] Progress: 12/34 | a.wav",
        "[F0 extraction] Progress: 3/9 | Success: 2 | Skipped: 1 | b.flac",
        "[HuBERT features] Progress: 7/10 | Success: 7 | Failed: 0 | c.wav.npy | (1, 768)",
    ],
)
def test_parse_stage_progress_english_anchors(line):
    # i18n/locale/en_US.json:20/27/36 的真实译文
    assert parse_stage_progress_line(line)["done"] > 0


def test_parse_stage_progress_with_logging_prefix():
    assert parse_stage_progress_line("INFO:mi-test:[数据切分] 进度：1/2 | a.wav") == {
        "done": 1, "total": 2, "current": "a.wav",
    }


@pytest.mark.parametrize(
    "line",
    [
        "[数据切分] 子任务完成 | 成功：12 | 失败：0",  # 完成行无 done/total
        "[数据切分] 待处理：34 | 进程数：8",
        "loss_disc=1.0, loss_gen=2.0, loss_fm=3.0,loss_mel=4.0, loss_kl=5.0",
        "训练轮次：20 [34%]",
        "",
    ],
)
def test_parse_stage_progress_non_progress_lines_return_none(line):
    assert parse_stage_progress_line(line) is None


def test_parse_stage_progress_without_file_field():
    """[索引训练] 写入进度：5/9 无 | 字段：进度可解析，当前文件名为 None。"""
    assert parse_stage_progress_line("[索引训练] 写入进度：5/9") == {
        "done": 5, "total": 9, "current": None,
    }
