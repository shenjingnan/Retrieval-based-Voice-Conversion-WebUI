"""训练数据集管理：列出 / 详情 / 上传 / 删除。

目录布局：paths.DATASETS_DIR 下每个一级子目录是一个数据集，训练侧拿到的是该子目录的
绝对路径（作为 dataset_dir 传给 preprocess）。时长探测用 PyAV + soundfile 兜底，
结果缓存进 DATASETS_DIR/.meta/ 侧车（见 META_DIRNAME 注释）。
"""
from fastapi import APIRouter

from server import paths

router = APIRouter(prefix="/api/datasets")

# 时长探测与上传校验的音频口径。preprocess 遍历目录不过滤扩展名（train/preprocess.py
# 的 load_audio 走 ffmpeg），这里只列常见音频后缀，其余文件计入 other_count
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
# 侧车缓存目录：必须位于各数据集目录之外（放数据集目录内会被 preprocess 当音频再解一次）
META_DIRNAME = ".meta"


@router.get("")
def list_datasets():
    return []
