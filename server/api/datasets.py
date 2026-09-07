"""训练数据集管理：列出 / 详情 / 上传 / 删除。

目录布局：paths.DATASETS_DIR 下每个一级子目录是一个数据集，训练侧拿到的是该子目录的
绝对路径（作为 dataset_dir 传给 preprocess）。时长探测用 PyAV + soundfile 兜底，
结果缓存进 DATASETS_DIR/.meta/ 侧车（见 META_DIRNAME 注释）。

注意 monkeypatch 纪律：paths 以模块属性访问（paths.DATASETS_DIR），不得 from-import，
否则测试把根目录指到 tmp_path 的隔离手段失效。
"""
import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from server import paths
# 与 training._check_exp_name 同源的非法字符表（勿在本模块复制一份，改一处必须同源）
from server.api.training import _EXP_FORBIDDEN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets")

# 时长探测与上传校验的音频口径。preprocess 遍历目录不过滤扩展名（train/preprocess.py
# 的 load_audio 走 ffmpeg），这里只列常见音频后缀，其余文件计入 other_count
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
# 侧车缓存目录：必须位于各数据集目录之外（放数据集目录内会被 preprocess 当音频再解一次）
META_DIRNAME = ".meta"


# ---------------------------------------------------------------------------
# 校验与目录扫描
# ---------------------------------------------------------------------------


def _check_name(value: str) -> str:
    """数据集名沿用实验名同源规则（training._check_exp_name 的语义：防路径穿越 + 防
    shell 注入，非法字符表即 _EXP_FORBIDDEN），另拒前导点：.meta 侧车目录与隐藏目录
    不能被当成数据集访问（否则列表/删除会误伤缓存）。"""
    if (
        not value
        or value.startswith(".")
        or Path(value).name != value
        or any(ch in _EXP_FORBIDDEN for ch in value)
    ):
        raise HTTPException(
            400,
            "数据集名非法：%r（不得为空、不得以点开头、不得含路径分隔符、引号、反斜杠或空白字符）"
            % value,
        )
    return value


def _dataset_dir(name: str) -> Path:
    """校验后的数据集目录（不保证存在）。"""
    return paths.DATASETS_DIR / _check_name(name)


def _iter_audio(directory: Path) -> tuple[list, list]:
    """(音频文件, 其他文件)，均按名排序；点开头文件一律跳过（.meta 侧车、系统杂物）。"""
    audios: list = []
    others: list = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        path = directory / name
        if not path.is_file():  # 数据集不嵌套子目录；is_file 同时排除悬空软链
            continue
        (audios if path.suffix.lower() in AUDIO_SUFFIXES else others).append(path)
    return audios, others


# ---------------------------------------------------------------------------
# 时长探测
# ---------------------------------------------------------------------------


def _audio_duration(path: Path) -> float | None:
    """音频时长（秒）。PyAV 取音频流 frames/average_rate——container.duration 在 mp3 上
    偏大 5-8%，不可用；PyAV 读不出（格式不支持 / 损坏 / 未装 av）走 soundfile 的
    sf.info 兜底；两路都失败返回 None。绝不向调用方抛异常。"""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            frames = stream.frames
            rate = stream.average_rate
            if frames and rate:
                return float(frames) / float(rate)
    except Exception:  # noqa: BLE001 探测失败属预期分支，交给 soundfile 兜底
        logger.debug("PyAV 时长探测失败：%s", path, exc_info=True)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return info.frames / info.samplerate
    except Exception:  # noqa: BLE001 兜底也失败 → 时长未知
        logger.debug("soundfile 时长探测失败：%s", path, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# 侧车缓存（DATASETS_DIR/.meta/{name}.json：文件名 → {size, mtime_ns, duration}）
# ---------------------------------------------------------------------------


def _cache_load(name: str) -> dict:
    """坏 JSON / 缺失一律当空缓存（下次全量重探测并覆写）。"""
    try:
        data = json.loads(
            (paths.DATASETS_DIR / META_DIRNAME / f"{name}.json").read_text(encoding="utf8")
        )
    except Exception:  # noqa: BLE001 OSError 与 JSONDecodeError 同类：缓存不可信即视为空
        return {}
    return data if isinstance(data, dict) else {}


def _cache_store(name: str, cache: dict) -> None:
    """原子写（tmp + os.replace）；写失败静默——缓存只影响探测耗时，不影响正确性。"""
    meta_dir = paths.DATASETS_DIR / META_DIRNAME
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = meta_dir / f"{name}.json.tmp"
        tmp_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf8")
        os.replace(tmp_path, meta_dir / f"{name}.json")
    except OSError:
        logger.debug("数据集侧车缓存写入失败：%s", name, exc_info=True)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def _scan_dataset(directory: Path) -> tuple[dict, list]:
    """扫描一个数据集目录 → (概览, 文件明细)。

    时长任一文件探测失败记 None（聚合时跳过，见 total_duration 注释）；缓存键为
    文件名 + size + mtime_ns，外部改动文件后必然失效重探。
    """
    audios, others = _iter_audio(directory)
    cache = _cache_load(directory.name)
    files: list = []
    total_bytes = 0
    total_duration = 0.0
    probed_any = False
    dirty = False
    for path in audios:
        stat = path.stat()
        total_bytes += stat.st_size
        entry = cache.get(path.name)
        if (
            isinstance(entry, dict)
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            duration = entry.get("duration")
        else:
            duration = _audio_duration(path)
            cache[path.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "duration": duration,
            }
            dirty = True
        if duration is not None:
            total_duration += duration
            probed_any = True
        files.append({"name": path.name, "size": stat.st_size, "duration": duration})
    for path in others:
        total_bytes += path.stat().st_size
    if dirty:
        _cache_store(directory.name, cache)
    summary = {
        "name": directory.name,
        "path": str(directory),
        "file_count": len(audios),
        "other_count": len(others),
        # 时长探测全失败（含空数据集）→ None，前端显示"未知"而非误导性的 0 秒
        "total_bytes": total_bytes,
        "total_duration": total_duration if probed_any else None,
    }
    return summary, files


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.get("")
def list_datasets():
    """列出数据集（一级子目录，按名排序）；根目录不存在（还没上传过）→ 空列表。"""
    if not paths.DATASETS_DIR.is_dir():
        return []
    entries = sorted(
        (
            entry
            for entry in paths.DATASETS_DIR.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        ),
        key=lambda entry: entry.name,
    )
    return [_scan_dataset(entry)[0] for entry in entries]


@router.get("/{name}")
def dataset_detail(name: str):
    """数据集详情：概览字段 + files 明细（音频文件，按名排序，含 size 与时长）。"""
    directory = _dataset_dir(name)
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    summary, files = _scan_dataset(directory)
    summary["files"] = files
    return summary
