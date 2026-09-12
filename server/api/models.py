"""扫描 assets/weights 与 assets/indices，返回模型及其索引配对；
删除模型 = 彻底删除：整组权重 + 配对索引 + logs/{exp} 训练产物；
上传模型 = 外部音色模型导入：校验后写入 weights/indices，入库即生效。"""
import contextlib
import logging
import os
import re
import zipfile
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from server import paths
# 跨模块私有 import 有先例（datasets.py 对 training._EXP_FORBIDDEN）；main.py 的
# create_app 本就全量加载两个 router，无环、无额外 import 重量。测试 patch 的是
# training 命名空间里的 task_manager/_TASK_META——_ensure_exp_idle 在调用期从
# training 模块 globals 解析它们，因此借用方（本模块）不需要任何 patch。
from server.api.datasets import _rmtree_best_effort
from server.api.training import _ensure_exp_idle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_EPOCH_SUFFIX = re.compile(r"_e\d+_s\d+$", re.IGNORECASE)
_SPKID_SUFFIX = re.compile(r"_spkid\d+$", re.IGNORECASE)

# zip 流的写出缓冲：每次攒满这么多字节才往响应里推一块，内存占用恒定
_ZIP_CHUNK = 1024 * 1024


def experiment_name(model_stem: str) -> str:
    """alice_v2_e20_s100 -> alice_v2，与 infer/vc/utils.py 规则一致。"""
    return _EPOCH_SUFFIX.sub("", model_stem)


def _index_matches(index_stem: str, exp: str) -> bool:
    # 索引名两种形态：训练产物 added_IVF*_{exp} 与外链副本 {exp}_added_IVF*，
    # 故同时认 _exp 边界与 exp_added_ 前缀；裸子串会把 bob 误配到 bobby。
    # P1 简化：不做 exact_model_match / 多级优先级评分。
    return (
        index_stem == exp
        or index_stem.startswith(f"{exp}_added_")
        or f"_{exp}_" in index_stem
        or index_stem.endswith(f"_{exp}")
    )


def _pick_index(exp: str, indices):
    if not exp:  # 退化文件名（如 _e20_s100.pth）剥不出实验名，无法配对
        return None
    candidates = _paired_indices(exp, indices)
    if not candidates:
        return None
    # 多个候选取最新构建的索引，与 infer/vc/utils.py 的 mtime 偏好一致
    return min(candidates, key=lambda i: (-i.stat().st_mtime, i.name.lower()))


def _paired_indices(exp: str, indices) -> list:
    """会配对到该实验的全部索引（_pick_index 的候选全集）。
    *_spkidN 索引只含单个说话人的向量，P1 无说话人上下文，与 infer/vc/utils.py 一致地跳过。"""
    if not exp:
        return []
    return [
        i
        for i in indices
        if not _SPKID_SUFFIX.search(i.stem) and _index_matches(i.stem.lower(), exp)
    ]


def _list_indices() -> list:
    """assets/indices 下的候选索引；trained 档是训练中间产物，与 _scan 一并排除。

    is_file() 跟随软链：训练产出的索引是链到 logs/{exp}/ 的软链（train_index.py），
    实验目录被删后留下悬空软链——glob 仍能列出它，但 stat/读取都会 FileNotFoundError，
    会把整个 GET /api/models 打成 500（E2E 实测）。is_file() 为 False 的条目直接排除，
    与 /api/infer 的索引接受口径（同样 is_file）保持一致。
    """
    if not paths.INDICES_DIR.is_dir():
        return []
    return [
        p
        for p in paths.INDICES_DIR.glob("*.index")
        if "trained" not in p.name.lower() and p.is_file()
    ]


def _scan():
    indices = _list_indices()
    models = []
    if paths.WEIGHTS_DIR.is_dir():
        for p in sorted(paths.WEIGHTS_DIR.glob("*.pth")):
            exp = experiment_name(p.stem).lower()
            index = _pick_index(exp, indices)
            models.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "index": str(index) if index else None,
                    # 文件修改时间（epoch 秒）：前端模型页按它把最近训练的组排前面
                    "mtime": int(p.stat().st_mtime),
                }
            )
    return models


@router.get("/models")
def list_models():
    return _scan()


def _delete_file(path: Path) -> None:
    """unlink 的薄封装：仅为测试提供打桩点（组内某个成员删除失败的可测注入）。"""
    path.unlink()


def _locate_logs_dir(exp: str, exp_raw: str) -> Path | None:
    """定位 logs/{exp}，找不到返回 None（视为已清理，不是错误）。

    logs 目录名保留用户提交实验名时的原始大小写（EXP_NAME_RE 允许大写，Linux
    大小写敏感），而 weights 文件名与索引配对走 lower 口径——故先按文件名原始
    大小写探测，再按 lower 探测，最后遍历 LOGS_DIR 做大小写不敏感兜底。
    安全兜底：候选必须仍是 basename（Path(x).name != x 或 "." / ".." 一律跳过，
    绝不 rmtree LOGS_DIR 之外的路径）；logs/mute 是全局共享静音资产（训练侧
    preprocess 的 0 秒静音样本源），永不删除。
    """
    if not paths.LOGS_DIR.is_dir():
        return None
    for cand in dict.fromkeys([exp_raw, exp]):  # 去重保序：原始 case 优先
        if not cand or cand in (".", "..") or Path(cand).name != cand:
            continue
        target = paths.LOGS_DIR / cand
        if target.is_dir() and target.name.lower() != "mute":
            return target
    for entry in paths.LOGS_DIR.iterdir():
        if entry.name.lower() == exp and entry.is_dir() and entry.name.lower() != "mute":
            return entry
    return None


@router.delete("/models/{name}")
def delete_model(name: str):
    """彻底删除一个模型组：组内全部权重 + 配对索引 + logs/{exp} 训练产物。

    「组」= weights 下 experiment_name(stem).lower() 相同的全部 *.pth（最终产物
    与全部中间轮次），与索引配对同口径（case-insensitive）。前端只在组卡片上提供
    删除按钮；noFinal 组（训练中断只剩中间产物）由代表项（最大轮次）发起，同样
    删整组。

    索引联动范围取配对规则的候选全集（_index_matches 命中且非 spkid），而不是只删
    GET /api/models 当前选中的那一个——否则同实验的旧索引会残留成孤儿；相似实验名
    （bob / bobby）因匹配规则带边界不会被牵连。防穿越与 infer.py 同手法：非 basename
    一律 404，绝不触达 weights 目录之外的路径（"." / ".." 单独拒绝：Python 3.13 起
    Path("..").name 返回 ".."，name 比较兜不住；带路径分隔符的形态到不了本 handler，
    路由层只匹配单段）。仅接受 .pth 后缀（与 GET 扫描的 *.pth 一致），weights 下的
    杂物文件（说明文档、备份残片等）不由本接口删除。

    logs/{exp} 是训练产物的体积大头（checkpoint、预处理特征、索引真实体），随组
    一并删除——删除后无法再用这些产物补训索引。目录名保留用户提交实验名时的原始
    大小写而 weights/索引配对走 lower 口径，故按原始 case → lower → 遍历兜底定位
    （_locate_logs_dir）；logs/mute 是全局共享静音资产，永不删除。删除后再清扫
    assets/indices 里仍指向该目录的残余软链（*_spkidN 等不在配对候选里的链接），
    否则它们会永久悬空堆积（GET 靠 is_file() 过滤不会 500，但会越积越多）。

    在训保护：该实验（原始 case 与 lower 两个候选都查——任务元数据里的 exp_name
    是提交时的原始大小写，weights 文件名大小写可能与之不一致）存在非终态任务时
    409，绝不截断正在写 logs 的训练。

    退化文件名（如 _e20_s100.pth）剥不出实验名（exp == ""）：退回旧行为，只删被
    点名的文件，不联动索引、不碰 logs。

    失败纪律（与 datasets.delete_dataset 有意分歧）：datasets 里目录就是产品，整树
    删不动给 500；本接口的产品是权重与索引，logs 只是附带清理。点名权重自身 unlink
    失败 → 500（删除未发生）；组内其余权重、索引、logs 树的逐项失败不回滚也不
    500，分别收进 failed_models / failed_indices / logs.failed_files 如实上报。
    """
    if (
        name in (".", "..")
        or Path(name).name != name
        or Path(name).suffix != ".pth"
        or not (paths.WEIGHTS_DIR / name).is_file()
    ):
        raise HTTPException(404, f"模型不存在: {name}")
    model_path = paths.WEIGHTS_DIR / name
    exp_raw = experiment_name(model_path.stem)  # 原始大小写：logs 目录名候选
    exp = exp_raw.lower()

    deleted_models: list = []
    failed_models: list = []
    deleted_indices: list = []
    failed_indices: list = []
    logs_result = {"target": None, "removed": False, "failed_files": []}

    if exp:  # 退化 stem（exp == ""）走不到这里：只删点名文件
        # 在训守卫放最前：任何删除动作都不该发生在活跃任务存在时
        for cand in dict.fromkeys([exp_raw, exp]):
            _ensure_exp_idle(cand)  # 命中即抛 HTTPException(409)

        # 删任何东西之前先完成枚举与定位，失败语义不依赖中途状态
        members = [
            p
            for p in paths.WEIGHTS_DIR.glob("*.pth")
            if p.is_file() and experiment_name(p.stem).lower() == exp
        ]
        logs_dir = _locate_logs_dir(exp, exp_raw)
        paired = _paired_indices(exp, _list_indices())

        # 点名者先行：它删不掉就 500，整组保持原样
        try:
            _delete_file(model_path)
        except OSError:
            logger.exception("删除模型失败：%s", model_path)
            raise HTTPException(500, f"模型删除失败: {name}") from None
        deleted_models.append(name)
        for member in members:
            # is_file 兜底两类同名异写：macOS/Windows 大小写不敏感 FS 上"alice.pth"
            # 与磁盘上的 "Alice.pth" 是同一文件（点名删除已把它删掉，再按组员删会
            # FileNotFoundError），也顺带防扫描与删除之间的竞态。
            if member == model_path or not member.is_file():
                continue
            try:
                _delete_file(member)
            except OSError:
                logger.exception("删除组内权重失败：%s", member)
                failed_models.append(member.name)
            else:
                deleted_models.append(member.name)

        for index_path in paired:
            try:
                index_path.unlink()
            except OSError:
                logger.exception("删除索引失败：%s", index_path)
                failed_indices.append(index_path.name)
            else:
                deleted_indices.append(index_path.name)

        if logs_dir is not None:
            failed_files: list = []
            try:
                _rmtree_best_effort(logs_dir, failed_files)
            except OSError:
                logger.exception("训练产物目录删除失败：%s", logs_dir)
            logs_result["target"] = str(logs_dir)
            logs_result["failed_files"] = failed_files
            logs_result["removed"] = not logs_dir.exists()

            # 清扫仍指向该 logs 目录的残余软链（spkid / 悬空链不在配对候选里）。
            # 只处理 symlink：Windows 硬链在 indices 下是独立实体，rmtree 不影响它。
            if paths.INDICES_DIR.is_dir():
                for link in paths.INDICES_DIR.glob("*.index"):
                    with contextlib.suppress(OSError):
                        if link.is_symlink() and link.resolve().is_relative_to(logs_dir):
                            link.unlink(missing_ok=True)

    else:
        try:
            _delete_file(model_path)
        except OSError:
            logger.exception("删除模型失败：%s", model_path)
            raise HTTPException(500, f"模型删除失败: {name}") from None
        deleted_models.append(name)

    return {
        "deleted_model": name,  # 兼容保留：被点名者
        "deleted_models": deleted_models,
        "failed_models": failed_models,
        "deleted_indices": deleted_indices,
        "failed_indices": failed_indices,
        "logs": logs_result,
    }


def _zip_stream(entries: list[tuple[str, Path]]) -> Iterator[bytes]:
    """把 entries 逐块流式打成 zip。ZIP_STORED 不压缩：pth/index 是二进制张量，
    压缩率极低，白烧 CPU 还拖慢下载；攒一块吐一块让内存占用与文件大小无关，
    不落临时文件也就没有清理问题。force_zip64 兜住超大文件（>4GB）的极端情况。

    落给 zipfile 的是一个只有 write/tell、没有 seek 的 sink——这是关键：可 seek
    的缓冲会让 zipfile 写完数据后回头 seek 改写 local header，而被中途 flush 出去
    的字节再也改不到（真实大模型端到端实测包损坏）；缺 seek 则逼它走
    data-descriptor 顺序写模式，全程只追加，天然适配流式响应。"""
    out: deque[bytes] = deque()
    written = 0

    class _StreamSink:
        def write(self, data: bytes) -> int:
            nonlocal written
            out.append(data)
            written += len(data)
            return len(data)

        def tell(self) -> int:
            return written

        def flush(self) -> None:
            pass
        # 故意不提供 seek：见 docstring

    with zipfile.ZipFile(_StreamSink(), "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, src in entries:
            with zf.open(arcname, "w", force_zip64=True) as dst, src.open("rb") as f:
                while chunk := f.read(_ZIP_CHUNK):
                    dst.write(chunk)
                    while out:
                        yield out.popleft()
    while out:
        yield out.popleft()


def _content_disposition(download_name: str) -> str:
    """Content-Disposition：ASCII 名走 filename=，否则 filename*=utf-8''（RFC 5987），
    与 Starlette FileResponse 同一套分支——非 latin-1 字节进 filename= 会打爆响应头。"""
    quoted = quote(download_name)
    if quoted == download_name:
        return f'attachment; filename="{download_name}"'
    return f"attachment; filename*=utf-8''{quoted}"


@router.get("/models/{name}/download")
def download_model(name: str):
    """一键打包下载：pth + 配对索引打成一个 zip 流式返回。

    zip 是 Windows/macOS/Linux 三端原生可解压的格式，客户端无需任何处理；
    包内一层实验名目录，解压后不散落当前文件夹。配对索引与列表页展示的
    是同一个结果（_pick_index），看到的即所得；缺索引的模型包内只有 pth，
    下载不被拦截。入口校验与 delete_model 同手法：非 basename / 非重量级
    .pth / 不存在一律 404，绝不触达 weights 目录之外的路径。
    """
    if (
        name in (".", "..")
        or Path(name).name != name
        or Path(name).suffix != ".pth"
        or not (paths.WEIGHTS_DIR / name).is_file()
    ):
        raise HTTPException(404, f"模型不存在: {name}")
    model_path = paths.WEIGHTS_DIR / name
    stem = model_path.stem
    exp = experiment_name(stem).lower()
    index_path = _pick_index(exp, _list_indices())
    # zip 内目录名：剥得出实验名用实验名，退化文件名退回模型 stem（不能是空目录名）
    arc_dir = exp or stem
    entries = [(f"{arc_dir}/{model_path.name}", model_path)]
    if index_path is not None:
        entries.append((f"{arc_dir}/{index_path.name}", index_path))
    return StreamingResponse(
        _zip_stream(entries),
        media_type="application/zip",
        headers={"Content-Disposition": _content_disposition(f"{stem}.zip")},
    )


# ---------------------------------------------------------------------------
# POST /api/models/upload（外部音色模型导入）
# ---------------------------------------------------------------------------

# 与 datasets.MAX_FILE_BYTES 同口径的 500MB：最终产物 pth 通常 55~170MB，大实验的
# 索引可到百 MB 级。上限在拷贝循环按实际字节数强制——请求体在 handler 之前已被
# Starlette 整体 spool，这里拦的是「目标目录被污染」，不是带宽。
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# 流式拷贝粒度：与 datasets.CHUNK_BYTES 同值同理由（spool 阈值也是 1MB）
_COPY_CHUNK = 1024 * 1024
# torch>=1.6 的 torch.save 是 zip 容器：魔数把「改后缀的杂鱼文件」拦在上传口。
# torch<1.6 的旧式非 zip 存档不放行——在产的 RVC 社区模型没有这种形态，误放行的
# 代价是坏模型混进列表、推理期才炸，提示远不如上传口明确。
_PTH_MAGIC = b"PK\x03\x04"


def _safe_upload_name(raw: str | None, kind: str) -> str:
    """multipart filename → 严格校验的单段文件名。

    与 datasets._safe_filename 的宽松清洗不同：模型名要往返 URL（删除/下载路由，
    单段路由匹配兜不住分隔符），且必须与 _scan 的 *.pth glob 口径一致，故不做
    静默改写——含路径分隔符（正反斜杠都算，Windows 客户端塞整路径时明确报错而不是
    悄悄换个名字入库）、控制字符、点名的名字一律 400。
    """
    name = raw or ""
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or any(ord(ch) < 0x20 for ch in name)
    ):
        raise HTTPException(400, f"{kind}文件名非法: {name!r}")
    return name


def _save_upload(upload: UploadFile, dest: Path, magic: bytes | None) -> None:
    """把上传 spool 流式落到 dest：同目录隐匿 .part 临时文件 + os.replace 原子改名。

    限额在拷贝循环按实际字节数强制（Content-Length 可谎报）；超限 / 魔数不符 /
    IO 失败一律先删 .part 再上抛——磁盘上只允许出现完整文件（与 datasets._save_stream
    同纪律）。.part 名带 uuid：并发同名上传互不践踏，且后缀不会被 _scan 的 glob
    列出，即便残留也不进列表。"""
    dest.parent.mkdir(parents=True, exist_ok=True)  # 测试把目录指到 tmp 时可能尚不存在
    tmp = dest.with_name(f".{dest.name}.{uuid4().hex}.part")
    try:
        with tmp.open("wb") as out:
            if magic is not None:
                head = upload.file.read(len(magic))
                if head != magic:
                    raise HTTPException(
                        400,
                        f"{dest.name} 不是有效的 .pth 文件（缺少 torch zip 容器头），"
                        "请确认文件完整且由 RVC 导出",
                    )
                out.write(head)
            while chunk := upload.file.read(_COPY_CHUNK):
                if out.tell() + len(chunk) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"{dest.name} 超过大小上限（500MB）")
                out.write(chunk)
        if dest.exists():  # 复核：封住入口查重到落盘之间的竞态窗口
            raise HTTPException(409, f"文件在上传期间已出现: {dest.name}")
        os.replace(tmp, dest)
    except BaseException:  # noqa: BLE001 半成品清理必须覆盖 4xx 与 IO 错误两条路径
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


@router.post("/models/upload")
def upload_model(
    model: UploadFile = File(...),
    index: UploadFile | None = File(None),
):
    """导入外部音色模型：pth 必选、配对索引可选，校验通过后写入 weights/indices。

    入库即生效：写盘成功后 GET /api/models 的既有扫描、索引配对、推理、下载、
    删除全部自然覆盖（推理只认这两个目录，见模块 docstring），不需要任何登记步骤。
    响应是与 GET /api/models 条目同构的对象（含按配对规则回查的 index）。

    重名一律 409 拒绝且两目录独立查重：覆盖藏在「上传」里会绕过显式的彻底删除
    确认；自动改名则扰乱索引配对与训练产物命名。任何拒绝（含 index 一侧非法）
    都发生在落盘之前或触发整体回滚，不产生半成品。"""
    model_name = _safe_upload_name(model.filename, "模型")
    if not model_name.endswith(".pth"):
        raise HTTPException(400, f"模型文件必须是 .pth 后缀（小写）: {model_name}")
    index_name = None
    if index is not None:
        index_name = _safe_upload_name(index.filename, "索引")
        if not index_name.endswith(".index"):
            raise HTTPException(400, f"索引文件必须是 .index 后缀（小写）: {index_name}")
    if (paths.WEIGHTS_DIR / model_name).exists():
        raise HTTPException(409, f"模型已存在: {model_name}，如需替换请先删除")
    if index_name is not None and (paths.INDICES_DIR / index_name).exists():
        raise HTTPException(409, f"索引已存在: {index_name}，如需替换请先删除")

    dest_model = paths.WEIGHTS_DIR / model_name
    dest_index = paths.INDICES_DIR / index_name if index_name is not None else None
    try:
        _save_upload(model, dest_model, _PTH_MAGIC)
        if dest_index is not None:
            _save_upload(index, dest_index, None)
    except BaseException:
        # 整体回滚：一半成功一半失败对「上传成对文件」的意图是谎言，让用户修好
        # 对应一侧后整套重传（「有模型无索引」的合法形态只来自用户本来就只传 pth，
        # 而不是残缺的事务）
        with contextlib.suppress(OSError):
            dest_model.unlink(missing_ok=True)
        raise

    for entry in _scan():
        if entry["name"] == model_name:
            return entry
    raise HTTPException(500, "上传已落盘但模型未出现在列表中")  # 理论不可达：落盘名即 glob 口径
