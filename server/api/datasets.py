"""训练数据集管理：列出 / 详情 / 上传 / 删除。

目录布局：paths.DATASETS_DIR 下每个一级子目录是一个数据集，训练侧拿到的是该子目录的
绝对路径（作为 dataset_dir 传给 preprocess）。时长探测用 PyAV + soundfile 兜底，
结果缓存进 DATASETS_DIR/.meta/ 侧车（见 META_DIRNAME 注释）。

注意 monkeypatch 纪律：paths 以模块属性访问（paths.DATASETS_DIR），不得 from-import，
否则测试把根目录指到 tmp_path 的隔离手段失效。
"""
import contextlib
import json
import logging
import os
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server import paths
from server.commands import build_separate_cmd
# 与 training._check_exp_name 同源的非法字符表（勿在本模块复制一份，改一处必须同源）
from server.api.training import _EXP_FORBIDDEN
# 删除互斥的判定依据：任务状态机常量与进程级任务表单例（server.tasks 顶层不加载
# torch，与 server.main 的导入深度一致，pytest 收集期安全）
from server.tasks import TERMINAL_STATES, TaskConflictError, task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets")

# 时长探测与上传校验的音频口径。preprocess 遍历目录不过滤扩展名（train/preprocess.py
# 的 load_audio 走 ffmpeg），这里只列常见音频后缀，其余文件计入 other_count
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
# 侧车缓存目录：必须位于各数据集目录之外（放数据集目录内会被 preprocess 当音频再解一次）
META_DIRNAME = ".meta"
# 流式落盘的 read 粒度：内存占用与拷贝次数的折中（FastAPI 的 UploadFile spool 阈值
# 也是 1MB，超过即滚到磁盘临时文件，配合本粒度全程只有一块数据在内存里）
CHUNK_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# 人声分离（tools/vocal_dataset.py，设计见 docs/plans/2026-09-07-dataset-vocal-separation-design.md）
# ---------------------------------------------------------------------------

# 分离模型 label 白名单：与 tools.pymss_webui.MODEL_SPECS 的 label 逐字同源（勿在本
# 模块重排或改名，改动须两处同步）。服务进程不得 import 该模块做校验——它顶层构造
# configs.Config 会加载 torch，违背 server 侧延迟导入纪律，故静态维护；runner 侧
# tools.pymss_webui.resolve_model 仍会二次校验兜底。顺序即前端下拉顺序。
SEPARATION_MODELS = ("去混响", "去混响（激进）", "去伴奏", "去伴奏（激进）", "提主旋律")
# 默认模型与 tools.vocal_dataset.DEFAULT_MODEL 同源（去伴奏：训练集最常见的净化需求）
DEFAULT_SEPARATION_MODEL = "去伴奏"
# 衍生数据集固定后缀：分离产物恒为 {name}_vocals（去混响产物也在其中，语义为该
# 数据集的「净化版」；真实来源与模型记录在衍生标记里）
DERIVED_SUFFIX = "_vocals"


# ---------------------------------------------------------------------------
# 校验与目录扫描
# ---------------------------------------------------------------------------


def _check_name(value: str) -> str:
    """数据集名沿用实验名同源规则（training._check_exp_name 的语义：防路径穿越 + 防
    shell 注入，非法字符表即 _EXP_FORBIDDEN），另拒前导点：.meta 侧车目录与隐藏目录
    不能被当成数据集访问（否则列表/删除会误伤缓存）。NUL 单独拒绝：漏到 mkdir/unlink
    层是 ValueError → 500，必须先在 400 拦下（training 侧已加同一判定，两侧同步）。"""
    if (
        not value
        or "\0" in value
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


class SeparateBody(BaseModel):
    """POST /{name}/separate 请求体。model 为 SEPARATION_MODELS 的 label；
    files 省略 = 分离整目录，提供 = 只分离这些文件（逐文件分离入口）。"""

    model: str = DEFAULT_SEPARATION_MODEL
    files: list[str] | None = None


# 文件名会以双引号形式进入分离子进程的 shell 命令，这些字符会破坏参数边界或触发
# 命令替换（_safe_filename 只清洗控制字符，$ 反引号 引号 反斜杠在上传时可能存活）
_SEPARATE_FILE_FORBIDDEN = '"\\$`'


def _derived_marker_path(name: str) -> Path:
    """衍生标记：.meta/{name}.derived.json（.meta 在各数据集目录之外，与时长侧车同理——
    放数据集目录内会被 preprocess 当音频再解一次）。"""
    return paths.DATASETS_DIR / META_DIRNAME / ("%s.derived.json" % name)


def _write_derived_marker(name: str, source: str, model: str) -> None:
    """写衍生标记（建任务时写入）。尽力而为：写失败只影响列表的 derived_from 展示，
    不影响分离产物本身。"""
    try:
        meta_dir = paths.DATASETS_DIR / META_DIRNAME
        meta_dir.mkdir(parents=True, exist_ok=True)
        _derived_marker_path(name).write_text(
            json.dumps({"source": source, "model": model}, ensure_ascii=False),
            encoding="utf8",
        )
    except OSError:
        logger.debug("衍生标记写入失败：%s", name, exc_info=True)


def _read_derived_source(name: str) -> str | None:
    """衍生标记里的源数据集名；无标记 / 坏 JSON → None（语义同 _cache_load：缓存不可信
    即视为无标记）。"""
    try:
        data = json.loads(_derived_marker_path(name).read_text(encoding="utf8"))
    except Exception:  # noqa: BLE001 OSError 与 JSONDecodeError 同类处理
        return None
    source = data.get("source") if isinstance(data, dict) else None
    return source if isinstance(source, str) and source else None


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------


class _QuotaExceeded(Exception):
    """上传超出大小软限制（单文件或批次累计）；API 层转 413。limit 供 detail 展示。"""

    def __init__(self, message: str, limit: int):
        super().__init__(message)
        self.limit = limit


def _safe_filename(name: str) -> str:
    """multipart 的 filename → 可安全落盘的单段文件名。

    客户端可能把完整路径塞进 filename（Windows 常见）：`/` 与反斜杠都当路径分隔符
    取 basename。C0 控制字符（0x00-0x1f，含回车换行）替换为 `_` 而非删除——删除会让
    控制字符版本与删除后的名字撞名；中文/emoji/空格等常规字符原样保留。basename 后
    为空（空串或纯分隔符）→ 返回空串，调用方按 skip 处理。
    """
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join("_" if ord(ch) < 0x20 else ch for ch in base)


def _unique_path(directory: Path, filename: str, taken: set) -> Path:
    """目录内不重名的落盘候选路径：重名 → stem_1.ext 递增。

    taken 是本批次已分配的名字——同批两份 a.wav 落盘前谁也不 exists()，必须靠集合
    防互撞；磁盘已有文件（无论是不是音频）也一律视为占用。这里只负责「挑一个大概率
    可用的名字」；真正的「永不覆盖」由调用方的 O_EXCL 独占创建裁决——exists() 检查
    与 open 之间存在窗口（sync handler 在线程池，两个并发上传可同时通过检查）。
    """
    candidate = directory / filename
    if candidate.name in taken or candidate.exists():
        stem, suffix = candidate.stem, candidate.suffix
        n = 0
        while candidate.name in taken or candidate.exists():
            n += 1
            candidate = directory / f"{stem}_{n}{suffix}"
    taken.add(candidate.name)
    return candidate


def _save_stream(upload: UploadFile, target: Path, fd: int, file_limit: int, batch_budget: int) -> int:
    """把上传文件的 spool 流式写到已独占创建的 fd，返回写入字节数。

    「超限中止」针对落盘到数据集目录的字节：请求 body 已由框架在 handler 之前整体
    spool 进 $TMPDIR（声明的体积由 handler 入口的 Content-Length 预检挡下，谎报的
    小体积大 body 由这里兜底）。读一块写一块（CHUNK_BYTES 粒度），写盘中即时核对
    两条限额：file_limit 为单文件上限，batch_budget 为本批剩余额度（累计上限 − 已
    落盘字节数）。任何异常路径都先删掉半成品再上抛：磁盘上只允许出现完整文件。
    """
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := upload.file.read(CHUNK_BYTES):
                written += len(chunk)
                if written > file_limit:
                    raise _QuotaExceeded("单个文件超过大小上限", file_limit)
                if written > batch_budget:
                    raise _QuotaExceeded("超过单次上传的累计大小上限", batch_budget)
                out.write(chunk)
    except BaseException:  # noqa: BLE001 半成品清理必须覆盖 413 与 IO 错误两条路径
        # 两个清理动作各自 suppress：正常退出路径 fd 已随 with 关闭，close 必抛 EBADF，
        # 与 unlink 混在同一个 suppress 里会把半成品清理整个跳过
        with contextlib.suppress(OSError):
            os.close(fd)  # fdopen 失败时防 fd 泄漏；正常路径 EBADF 被抑制
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)
        raise
    return written


def _probe_spool_size(upload: UploadFile) -> int:
    """spool 里的字节数（FastAPI 在 handler 之前已把整个 part 收进 spool，seek/tell
    即真实大小）。0 字节文件据此提前 skip，不落盘。"""
    upload.file.seek(0, os.SEEK_END)
    size = upload.file.tell()
    upload.file.seek(0)
    return size


# ---------------------------------------------------------------------------
# 时长探测
# ---------------------------------------------------------------------------


def _audio_duration(path: Path) -> float | None:
    """音频时长（秒）。PyAV 主路取音频流 duration×time_base——不取 container.duration
    （mp3 容器时长偏大 5-8%），也不取 frames/average_rate（frames 在 wav/mp3 等格式上
    恒为 0，AudioStream 更没有 average_rate 属性——旧写法是永不生效的死代码，av 15.1.0
    实测）；m4a（AAC）libsndfile 读不了、只能靠这条主路给出时长。PyAV 读不出（格式
    不支持 / 损坏 / 未装 av）走 soundfile 的 sf.info 兜底；两路都失败返回 None。绝不向
    调用方抛异常。"""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base:
                return float(stream.duration * stream.time_base)
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
    """原子写（tmp + os.replace）；写失败静默——缓存只影响探测耗时，不影响正确性。
    tmp 名掺 pid + 线程 id：sync handler 在线程池，两个请求同时写同一数据集的侧车时
    不能共用一个 tmp 名（后写者会把先写者尚未 replace 的半成品抢走发布）。"""
    meta_dir = paths.DATASETS_DIR / META_DIRNAME
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = meta_dir / f"{name}.json.{os.getpid()}.{threading.get_ident()}.tmp"
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
    summaries = []
    for entry in entries:
        try:
            summary, _ = _scan_dataset(entry)
        except OSError:
            # 并发删除的竞态（扫描到一半目录消失）只影响该数据集，跳过即可，
            # 不让整个列表 500
            continue
        # 衍生标记在并发删除的窗口里可能已随之消失，读不到按普通数据集展示
        summary["derived_from"] = _read_derived_source(entry.name)
        summaries.append(summary)
    return summaries


@router.get("/{name}")
def dataset_detail(name: str):
    """数据集详情：概览字段 + files 明细（音频文件，按名排序，含 size 与时长）。"""
    directory = _dataset_dir(name)
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    try:
        summary, files = _scan_dataset(directory)
    except OSError:
        # 404 检查之后、扫描过程中被并发删除：如实按不存在处理
        raise HTTPException(404, "数据集不存在：%s" % name)
    summary["files"] = files
    summary["derived_from"] = _read_derived_source(name)
    return summary


@router.post("/{name}/files")
def upload_dataset_files(
    name: str, request: Request, files: list[UploadFile] = File(...)
):
    """上传音频（multipart `files`，浏览器可多选/拖拽）：目录不存在则建、存在则追加。

    必须用 list[UploadFile]（SpooledTemporaryFile 流式，>1MB 滚到磁盘临时文件），不可
    换成 `bytes = File(...)` 的全量内存模式。逐文件：名字可安全化 → 非点开头 → 后缀是
    音频 → 非 0 字节 → 独占创建不重名路径 → 流式落盘 → 探测时长写侧车。大小限制是软
    限制：写盘中计数、即时中止、半成品删除；已完成的文件保留计入 added（不回滚），
    所以 413 之后磁盘上可能已有本次的部分文件。全 skip 且目录是本次新建 → 400 并删掉
    刚建的空目录。
    """
    directory = _dataset_dir(name)  # 非法名 → 400
    if not files:
        raise HTTPException(400, "未选择任何文件")
    # Content-Length 预检：body 由框架在 handler 之前整体 spool 进 $TMPDIR，超大 body
    # 会先写满临时目录并阻塞接收——软限制防不住 body 本体。该头由客户端声明、可以谎报，
    # 只做便宜的预检（缺失或非数字则跳过，交由写盘中计数兜底）；余量容掉 multipart
    # 边界的编码开销。
    declared = request.headers.get("content-length")
    if (
        declared
        and declared.isdigit()
        and int(declared) > MAX_BATCH_BYTES + CHUNK_BYTES
    ):
        raise HTTPException(
            413, "上传体积超过单次累计上限（%d MB）" % (MAX_BATCH_BYTES // (1024 * 1024))
        )
    created = not directory.is_dir()
    if created:
        try:
            # exist_ok：并发请求刚好先建了同一目录时按追加处理，不打 500
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("数据集目录创建失败：%s", directory)
            raise HTTPException(500, "数据集目录创建失败：%s" % name)

    added: list[str] = []
    skipped: list[dict] = []
    saved: list[Path] = []
    batch_written = 0
    taken: set = set()

    def _discard_empty_new_dir() -> None:
        """中止路径（413/500）的收尾：目录是本次新建且一个文件都没落成 → 删掉，
        不让列表里出现空数据集（与全 skip 的 400 同一纪律）。"""
        if created and not added:
            with contextlib.suppress(OSError):
                directory.rmdir()

    for upload in files:
        original = upload.filename or ""
        safe = _safe_filename(original)
        if not safe:
            skipped.append({"name": original, "reason": "文件名为空"})
            continue
        if safe.startswith("."):
            # 与 _iter_audio 的隐藏文件口径一致：点开头文件即使落盘也进不了列表/详情，
            # preprocess 还会把它当音频解一次（macOS 复制出的 AppleDouble ._a.wav 是
            # 真实场景）——与其自相矛盾不如直接 skip
            skipped.append(
                {"name": safe, "reason": "点开头的文件不会出现在数据集列表中，已跳过"}
            )
            continue
        suffix = Path(safe).suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            supported = " ".join(sorted(AUDIO_SUFFIXES))
            skipped.append(
                {
                    "name": safe,
                    "reason": "不支持的音频格式：%s（支持 %s）" % (suffix or "无后缀", supported),
                }
            )
            continue
        if _probe_spool_size(upload) == 0:
            skipped.append({"name": safe, "reason": "空文件（0 字节）"})
            continue

        try:
            while True:
                # O_EXCL 独占创建：exists() 检查与 open 之间的窗口（线程池并发上传同名
                # 文件）由文件系统裁决，被占即换名重试，绝不覆盖。循环放在下面的
                # except OSError 收尾范围内：os.open 除 EEXIST 重试外还可能直接失败
                # （ENOSPC/EACCES…），必须与写盘失败走同一条 500 + 空目录清理路径
                target = _unique_path(directory, safe, taken)
                try:
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    break
                except FileExistsError:
                    taken.add(target.name)
            written = _save_stream(
                upload, target, fd, MAX_FILE_BYTES, MAX_BATCH_BYTES - batch_written
            )
        except _QuotaExceeded as exc:
            _discard_empty_new_dir()
            raise HTTPException(
                413, "%s：%s（上限 %d MB）" % (exc, safe, exc.limit // (1024 * 1024))
            )
        except OSError:
            logger.exception("上传落盘失败：%s", target)
            _discard_empty_new_dir()
            raise HTTPException(
                500, "写入 %s 失败（已成功写入 %d 个文件，已落盘文件保留）" % (safe, len(added))
            )
        added.append(target.name)
        saved.append(target)
        batch_written += written

    if not added and created:
        # 全 skip 且目录是本次新建 → 400，不留空目录
        with contextlib.suppress(OSError):
            directory.rmdir()
        detail = "; ".join(f"{s['name']}（{s['reason']}）" for s in skipped)
        raise HTTPException(400, "没有可用的音频文件，全部被跳过：%s" % detail)

    # 时长探测写侧车（键与 _scan_dataset 一致）；随后的统计扫描会命中刚写的缓存，
    # 不会重复探测。413/500 中止路径不走这里——没探测的文件由下次 GET 补侧车。
    if saved:
        cache = _cache_load(name)
        for path in saved:
            stat = path.stat()
            cache[path.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "duration": _audio_duration(path),
            }
        _cache_store(name, cache)

    summary, _ = _scan_dataset(directory)
    return {
        "dataset": name,
        "path": str(directory),
        "created": created,
        "added": added,
        "skipped": skipped,
        "file_count": summary["file_count"],
        "total_duration": summary["total_duration"],
    }


@router.post("/{name}/separate")
def separate_dataset(name: str, body: SeparateBody):
    """发起人声分离任务：datasets/{name}/ → 衍生数据集 {name}_vocals/（仅人声 stem，
    伴奏残余丢弃；模型可选，默认去伴奏）。files 省略分离整目录，提供时只分离这些
    文件（前端逐文件分离入口；产物落到同一衍生目录，按文件幂等天然衔接后续的全量
    分离——已分离过的文件自动跳过）。

    与训练共用 task_manager 全局互斥（409）——分离与训练都是重负载任务，Mac 统一
    内存/显存不允许并发；这也让 delete_dataset 的「任务进行中 409 拒删」自动覆盖
    分离中的源与衍生数据集。产物按文件幂等（已存在的目标 stem 跳过），任务失败或
    中断后重新发起即续跑，不需要额外参数。衍生标记在建任务时写入（而非成功终态）：
    列表的血缘徽章对失败重跑同样有意义，且与目录由 runner 启动即创建的时序一致。
    """
    directory = _dataset_dir(name)  # 非法名 → 400
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    if body.model not in SEPARATION_MODELS:
        raise HTTPException(
            400,
            "未知分离模型：%r（可选：%s）" % (body.model, " / ".join(SEPARATION_MODELS)),
        )
    audios, _ = _iter_audio(directory)
    if not audios:
        raise HTTPException(400, "数据集没有可分离的音频文件：%s" % name)
    audio_names = {path.name for path in audios}
    files: list[str] | None = None
    if body.files:
        for filename in body.files:
            if (
                not filename
                or any(ch in _SEPARATE_FILE_FORBIDDEN for ch in filename)
                or any(ord(ch) < 0x20 for ch in filename)
            ):
                raise HTTPException(400, "文件名非法：%r" % filename)
            if filename not in audio_names:
                raise HTTPException(400, "数据集中没有该音频文件：%s" % filename)
        files = list(dict.fromkeys(body.files))  # 去重保序
    output_name = name + DERIVED_SUFFIX
    output_dir = paths.DATASETS_DIR / output_name
    cmd = build_separate_cmd(str(directory), str(output_dir), body.model, files)
    # 任务日志放 .meta（数据集目录之外，与衍生标记同理）；truncate 默认 True：
    # 每次发起都是全新日志，与训练各阶段的截断语义一致
    log_path = paths.DATASETS_DIR / META_DIRNAME / ("%s.separate.log" % name)
    try:
        task_id = task_manager.create_task("separate", [cmd], log_path)
    except TaskConflictError as exc:
        raise HTTPException(409, str(exc))
    _write_derived_marker(output_name, name, body.model)
    return {
        "task_id": task_id,
        "output_dataset": output_name,
        "output_path": str(output_dir),
    }


@router.delete("/{name}/files/{filename}")
def delete_dataset_file(name: str, filename: str):
    """删除数据集内的单个音频文件（上传错了/多传了的就地纠错）。

    互斥口径与整目录删除一致：任何训练类任务非终态时 409 拒删——preprocess 正在
    遍历该目录时不能动里面的文件，宁可误拦（用户可停任务后重试）。删除最后一个
    文件后目录为空则连目录一起移除（与上传「不留空目录」同一纪律），目录消失时
    .meta 下的侧车/标记/日志一并清理，不让列表留下 0 文件的空壳。
    """
    directory = _dataset_dir(name)  # 非法名 → 400
    # filename 校验：路由参数本身不含路径分隔符（单段），这里拒点开头与 NUL——
    # 口径同 _iter_audio（点开头文件不进列表，也就不该能被删）；文件名允许空格与
    # 中文（_safe_filename 保留它们），不经 shell，因此不套 _EXP_FORBIDDEN
    if (
        not filename
        or filename.startswith(".")
        or filename in (".", "..")
        or "\0" in filename
    ):
        raise HTTPException(400, "文件名非法：%r" % filename)
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    active = [
        task for task in task_manager.list_tasks() if task["state"] not in TERMINAL_STATES
    ]
    if active:
        raise HTTPException(
            409,
            "训练任务进行中（%s），不能删除文件；请先停止或等待任务完成"
            % active[0]["name"],
        )
    target = directory / filename
    if not target.is_file():
        raise HTTPException(404, "文件不存在：%s" % filename)
    try:
        target.unlink()
    except OSError:
        logger.exception("数据集文件删除失败：%s", target)
        raise HTTPException(500, "删除文件失败：%s" % filename)

    # 时长侧车的对应条目一并清（尽力而为；残留条目也不会被读到——扫描按现存文件取缓存）
    cache = _cache_load(name)
    if filename in cache:
        cache.pop(filename)
        _cache_store(name, cache)

    # 最后一个文件删掉 → 目录空了就移除（iterdir 含点开头杂物，有杂物则留着不动）
    if not any(directory.iterdir()):
        removed = False
        with contextlib.suppress(OSError):
            directory.rmdir()
            removed = True
        if removed:
            for meta_name in (f"{name}.json", f"{name}.derived.json", f"{name}.separate.log"):
                with contextlib.suppress(OSError):
                    (paths.DATASETS_DIR / META_DIRNAME / meta_name).unlink(missing_ok=True)

    return {"deleted": True}


@router.get("/{name}/files/{filename}/content")
def get_dataset_file_content(name: str, filename: str):
    """音频文件内容（试听播放）。路径校验与单文件删除同口径（拒点开头/NUL，允许
    空格与中文）；只读操作，不做任务互斥。走 FileResponse 以获得 Range 分段支持
    （浏览器 <audio> 拖动进度条依赖它）。"""
    directory = _dataset_dir(name)  # 非法名 → 400
    if (
        not filename
        or filename.startswith(".")
        or filename in (".", "..")
        or "\0" in filename
    ):
        raise HTTPException(400, "文件名非法：%r" % filename)
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    target = directory / filename
    if not target.is_file():
        raise HTTPException(404, "文件不存在：%s" % filename)
    media_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(target, media_type=media_type)


def _rmtree_best_effort(directory: Path, failed_files: list) -> None:
    """尽力删除目录树：逐项失败不中断，残留文件名收进 failed_files。

    目录自身的失败（顶层扫描被拒 / 最终 rmdir 因残留非空）不进 failed_files——那不
    是文件级信息，由调用方以「目录是否还在」做整体判定。onexc 是 3.12+ 参数，
    onerror 是等价旧签名（回调末参 excinfo 换成 exc，这里用不到），按版本回退。
    """
    def on_error(_func, path, exc) -> None:
        if Path(path) != directory:
            failed_files.append(Path(path).name)
        logger.warning("删除数据集目录时无法移除 %s：%r", path, exc)

    try:
        shutil.rmtree(directory, onexc=on_error)
    except TypeError:  # Python < 3.12 没有 onexc
        shutil.rmtree(directory, onerror=lambda func, path, excinfo: on_error(func, path, excinfo))


@router.delete("/{name}")
def delete_dataset(name: str):
    """删除数据集目录与侧车。训练任务非终态时 409 拒删——不解析 cmds 是否引用该
    数据集，宁可误拦（用户可停任务后重试），也不冒删掉正在被读的目录的风险。

    删除是尽力而为：单个文件删不掉（占用/权限）不回滚也不 500，残留项进
    failed_files 如实上报（models.delete_model 同纪律）；目录整体无法推进且无
    逐项失败可报时才 500。
    """
    directory = _dataset_dir(name)  # 非法名 → 400
    if not directory.is_dir():
        raise HTTPException(404, "数据集不存在：%s" % name)
    active = [
        task for task in task_manager.list_tasks() if task["state"] not in TERMINAL_STATES
    ]
    if active:
        raise HTTPException(
            409,
            "训练任务进行中（%s），不能删除数据集 %s；请先停止或等待任务完成"
            % (active[0]["name"], name),
        )

    failed_files: list = []
    try:
        _rmtree_best_effort(directory, failed_files)
    except OSError:
        logger.exception("数据集目录删除失败：%s", directory)
        raise HTTPException(500, "数据集目录删除失败：%s" % name)

    if directory.exists():
        if failed_files:
            # 部分失败：删掉的已删掉、残留项如实上报，不回滚
            return {"deleted": False, "failed_files": failed_files}
        raise HTTPException(500, "数据集目录无法移除：%s" % name)

    # 侧车一起删（只是缓存/标记/任务日志，失败静默——残留文件不会被任何路径读到）；
    # 衍生标记与分离日志同在 .meta，源数据集删除后一并清掉，不让列表留下指向
    # 已删除目录的血缘徽章
    with contextlib.suppress(OSError):
        (paths.DATASETS_DIR / META_DIRNAME / f"{name}.json").unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        _derived_marker_path(name).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        (paths.DATASETS_DIR / META_DIRNAME / f"{name}.separate.log").unlink(missing_ok=True)

    return {"deleted": True, "failed_files": []}
