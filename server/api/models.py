"""扫描 assets/weights 与 assets/indices，返回模型及其索引配对。"""
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from server import paths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_EPOCH_SUFFIX = re.compile(r"_e\d+_s\d+$", re.IGNORECASE)
_SPKID_SUFFIX = re.compile(r"_spkid\d+$", re.IGNORECASE)


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
                }
            )
    return models


@router.get("/models")
def list_models():
    return _scan()


@router.delete("/models/{name}")
def delete_model(name: str):
    """删除模型权重，并联动删除会配对到它的索引。

    联动范围取配对规则的候选全集（_index_matches 命中且非 spkid），而不是只删
    GET /api/models 当前选中的那一个——否则同实验的旧索引会残留成孤儿；相似实验名
    （bob / bobby）因匹配规则带边界不会被牵连。防穿越与 infer.py 同手法：非 basename
    一律 404，绝不触达 weights 目录之外的路径（"." / ".." 单独拒绝：Python 3.13 起
    Path("..").name 返回 ".."，name 比较兜不住；带路径分隔符的形态到不了本 handler，
    路由层只匹配单段）。仅接受 .pth 后缀（与 GET 扫描的 *.pth 一致），weights 下的
    杂物文件（说明文档、备份残片等）不由本接口删除。

    索引联动不具原子性：单个索引删除失败（权限/占用等）不回滚也不 500，收集进
    failed_indices 如实上报；模型权重自身的 unlink 失败仍然抛 500（删除没发生）。
    """
    if (
        name in (".", "..")
        or Path(name).name != name
        or Path(name).suffix != ".pth"
        or not (paths.WEIGHTS_DIR / name).is_file()
    ):
        raise HTTPException(404, f"模型不存在: {name}")
    model_path = paths.WEIGHTS_DIR / name
    exp = experiment_name(model_path.stem).lower()
    paired = _paired_indices(exp, _list_indices())
    model_path.unlink()
    deleted_indices = []
    failed_indices = []
    for index_path in paired:
        try:
            index_path.unlink()
        except OSError:
            logger.exception("删除索引失败：%s", index_path)
            failed_indices.append(index_path.name)
        else:
            deleted_indices.append(index_path.name)
    return {
        "deleted_model": name,
        "deleted_indices": deleted_indices,
        "failed_indices": failed_indices,
    }
