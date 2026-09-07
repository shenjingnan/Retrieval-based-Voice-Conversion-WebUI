"""训练进度解析：train.log 行解析 + 产物目录计数 + 日志增量读取。

锚点来自 train/train.py:530-546 的真实输出。训练循环的 logger 是实验名 logger
（train/train.py:138 `utils.get_logger(hps.model_dir)` → train/utils.py:448
`logging.getLogger(os.path.basename(model_dir))`），经 train/utils.py:17 的 basicConfig
StreamHandler 打到 stdout，再由编排层重定向进任务日志，因此真实行形如
`INFO:{实验名}:[step, lr]`——前缀字符集不可假设，匹配一律以消息体为锚：

    训练轮次：20 [34%]     /  Training epoch: 20 [34%]   epoch 行（i18n 双语锚点）
    [1234, 0.0001]                                        logger.info([step, lr])（train/train.py:543）
    loss_disc=12.345, loss_gen=23.456, loss_fm=1.234,loss_mel=45.678, loss_kl=0.912
                             （loss_fm 后的逗号无空格，照抄 train/train.py:545 的 f-string）

注意 get_logger 还会向 `logs/{exp}/train.log` 另挂 FileHandler（train/utils.py:451，
asctime\t实验名\tLEVEL\t行），与 stdout 重定向双写会让锚点行出现两份——因此任务日志
必须用专属文件（见 docs/plans/2026-09-06-p2-training-plan.md Task 5 第 9 条）。

progress 由编排层消费：切分/提取阶段用 count_dir_progress 数产物（分母=1_16k_wavs 文件数，
设计 §3.5），训练阶段用 parse_train_line + TailReader 增量解析任务日志。

延迟导入纪律：本模块顶层不 import torch/configs。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# epoch 行：中文锚点 + i18n 译文锚点（en_US.json:239 "Training epoch: ..."）与简写兜底。
# 数值用严格小数（不匹配 1.2.3 这类畸形串），保证 float() 永不抛 ValueError
EPOCH_PATTERN = re.compile(
    r"(?:训练轮次：|Training epoch: |Epoch: )(\d+) \[(-?\d+(?:\.\d+)?)%\]"
)
# [step, lr] 行：行尾锚定 + search。前缀是 logging 的 "LEVEL:logger 名:"（train/utils.py:451
# 之外的默认 basicConfig 格式），而 logger 名来自 train/train.py:138 的
# utils.get_logger(hps.model_dir) → logging.getLogger(os.path.basename(model_dir))，
# 即实验名本身（连字符/下划线/点都可能出现），因此锚点只认行尾的 [step, lr]，
# 不对前缀字符集做任何假设；畸形前缀（如 INFO:mi-test:INFO:[1, 0.1]）同样按尾部锚点解析。
# 全仓库仅 train/train.py:543 一处输出该形态，无误判来源。
# lr 形态：可选负号 + 整数/小数（含 .5、1. ）+ 可选科学计数（1e-05）
STEP_PATTERN = re.compile(r"\[(-?\d+), (-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\]\s*$")
# loss 行：五项顺序与分隔照抄 train/train.py:545，负值放行（判别器损失可为负），
# 数值同样是严格小数
LOSS_PATTERN = re.compile(
    r"loss_disc=(-?\d+(?:\.\d+)?), loss_gen=(-?\d+(?:\.\d+)?), loss_fm=(-?\d+(?:\.\d+)?),"
    r"loss_mel=(-?\d+(?:\.\d+)?), loss_kl=(-?\d+(?:\.\d+)?)"
)
LOSS_KEYS = ("loss_disc", "loss_gen", "loss_fm", "loss_mel", "loss_kl")

# 阶段进度行（design §3.5：切分/提取的进度与当前文件名）。锚点是脚本自身的计数行，
# 中英双语共用同一骨架（i18n/locale/en_US.json:20/27/36 的译文）：
#   [数据切分] 进度：12/34 | a.wav                                   train/preprocess.py:112
#   [F0提取] 进度：3/9 | 成功：2 | 跳过：1 | b.flac                   train/dataset/extract_f0.py:160
#   [HuBERT特征] 进度：7/10 | 成功：7 | 失败：0 | c.wav.npy | (1, 768) train/dataset/extract_hubert_feature.py:142
#   [索引训练] 写入进度：5/9                                        train/train_index.py:227
# done/total 锚定「进度：n/m」（英文 "Progress: n/m"），前缀不做任何假设；
# 当前文件名取尾部 | 字段里最后一个含 "." 的段——HuBERT 行最后一段是特征形状元组，须跳过
STAGE_PROGRESS_PATTERN = re.compile(r"(?:进度：|Progress: )(\d+)/(\d+)")


def parse_train_line(line: str):
    """解析一行训练日志；不认识的行（空行/普通日志行）返回 None。

    返回三种形态之一：
      {"epoch": int, "pct": float}
      {"step": int, "lr": float}
      {"loss_disc": float, "loss_gen": float, "loss_fm": float,
       "loss_mel": float, "loss_kl": float}

    step 锚点放最后：它是行尾锚定的宽松形态，让更具体的 loss/epoch 锚点优先。
    """
    if not line:
        return None
    loss = LOSS_PATTERN.search(line)
    if loss:
        return dict(zip(LOSS_KEYS, (float(value) for value in loss.groups())))
    epoch = EPOCH_PATTERN.search(line)
    if epoch:
        return {"epoch": int(epoch.group(1)), "pct": float(epoch.group(2))}
    step = STEP_PATTERN.search(line)
    if step:
        return {"step": int(step.group(1)), "lr": float(step.group(2))}
    return None


def parse_stage_progress_line(line: str):
    """解析切分/提取阶段的进度行；不是进度行返回 None。

    返回 {"done": int, "total": int, "current": str | None}，current 为该行携带的
    当前文件名（无 | 字段或字段都不含扩展名时为 None）。
    """
    if not line:
        return None
    match = STAGE_PROGRESS_PATTERN.search(line)
    if not match:
        return None
    tail = line[match.end():]
    current = None
    for part in reversed(tail.split("|")):
        part = part.strip()
        if "." in part:
            current = part
            break
    return {"done": int(match.group(1)), "total": int(match.group(2)), "current": current}


def count_dir_progress(product_dir, total):
    """产物进度 = 目录内文件数 / total，钳位 [0,1]；分母无效或目录不存在返回 None。

    只数文件不数子目录（产物目录是扁平的 .wav/.npy）；is_file 跟随软链，
    因此 train_index 链接进来的产物也计入。
    """
    if total is None or total <= 0:
        return None
    directory = Path(product_dir)
    if not directory.is_dir():
        return None
    count = sum(1 for entry in os.scandir(directory) if entry.is_file())
    return max(0.0, min(1.0, count / total))


class TailReader:
    """日志增量读取：按字节偏移只消费完整行。

    - 文件尚不存在（首个进程还没写入）→ 返回空列表，不抛错
    - 文件被截断（size 变小，编排层启动前截断日志）→ 复位偏移从头重读
    - 末尾半行（进程正在写）→ 留到下次凑成完整行再返回，避免污染解析
    """

    def __init__(self, path):
        self._path = Path(path)
        self._offset = 0

    def read_new_lines(self):
        try:
            size = self._path.stat().st_size
        except OSError:
            return []
        if size < self._offset:  # 被截断：从头重读
            self._offset = 0
        try:
            with open(self._path, "rb") as handle:  # 二进制读：子进程输出不经编码转换
                handle.seek(self._offset)
                raw = handle.read()
        except OSError:
            return []
        if not raw:
            return []
        end = self._offset + len(raw)
        if not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n") + 1
            raw = raw[:cut]
            end = self._offset + cut
        if not raw:
            return []
        self._offset = end
        return raw.decode("utf-8", errors="replace").splitlines()
