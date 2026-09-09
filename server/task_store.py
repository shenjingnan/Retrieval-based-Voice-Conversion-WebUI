"""训练任务队列的磁盘持久化后端（design docs/plans/2026-09-09-training-queue-design.md §4.2）。

文件内容 = 当前全部 pending（排队中）任务的记录列表；TaskManager 在 pending 集合每次
变化时全量重写（save），服务启动时由 server.api.training.restore_pending_tasks 读取
（load）。运行中任务不入文件——重启后需手动重新提交，train.py 会从最新 checkpoint
自动续训。

- 原子写：临时文件 + os.replace，进程任意时刻被杀都不会留下半截 JSON
- 损坏兜底：解析失败时把坏文件改名保留（.corrupt-<时间戳>）后按空队列处理，
  不静默销毁用户排队的记录

本模块只做字节级读写，不解释记录内容（字段语义见 server.tasks._persist_queue），
也不 import torch/configs（pytest 收集期安全）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_json_list(path: Path, what: str) -> list:
    """读取 JSON 列表文件，损坏/不可读/格式非列表一律按空处理并告警；
    损坏文件改名保留（.corrupt-<时间戳>）供人工找回。"""
    try:
        raw = path.read_text(encoding="utf8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("%s文件不可读（%s）：%s，按空处理", what, path, exc)
        return []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        backup = path.with_name("%s.corrupt-%d" % (path.name, int(time.time())))
        try:
            path.replace(backup)
            logger.warning("%s文件损坏，已改名保留：%s → %s（%s）", what, path, backup, exc)
        except OSError:
            logger.warning("%s文件损坏且无法改名保留：%s（%s）", what, path, exc)
        return []
    if not isinstance(data, list):
        logger.warning("%s文件格式非预期（应为列表，实际 %s），按空处理：%s",
                       what, type(data).__name__, path)
        return []
    return data


def _atomic_write(path: Path, payload) -> None:
    """原子写整个 JSON 快照（临时文件 + os.replace）；目录不存在则创建。
    I/O 失败向上抛出，由调用方兜底降级为日志。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(list(payload), ensure_ascii=False, indent=1), encoding="utf8"
    )
    os.replace(tmp, path)


class TaskStore:
    """JSON 文件存取。鸭子类型接口（load/save），测试可注入内存替身。"""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list:
        """读取 pending 记录列表。文件不存在 / 不可读 / 损坏 / 格式非列表一律按
        空队列处理（损坏文件改名保留，供人工找回）。"""
        return _read_json_list(self.path, "队列")

    def save(self, records) -> None:
        """原子写整个 pending 快照（临时文件 + os.replace）；目录不存在则创建。
        I/O 失败向上抛出，由调用方（TaskManager._persist_queue）兜底降级为日志。"""
        _atomic_write(self.path, records)


class TaskHistoryStore:
    """训练历史存储（design docs/plans/2026-09-09-task-centric-training-ui-design.md
    §3）：任务落终态时由训练记录器追加/替换一条记录，cap 截断保最新。

    与 TaskStore 同构的原子 JSON 语义；额外按条目容错——历史文件存活周期长，
    遇到半旧 schema / 手工编辑的坏条目时剔除该条而不是整文件作废。
    """

    def __init__(self, path: Path, cap: int = 50):
        self.path = Path(path)
        self.cap = cap

    def load(self) -> list:
        """读取全部历史记录（旧→新序）。"""
        return [
            record
            for record in _read_json_list(self.path, "历史")
            if isinstance(record, dict) and record.get("id")
        ]

    def record(self, rec: dict) -> None:
        """追加或按 id 原位替换一条记录，cap 截断砍头部（保最新）。
        I/O 失败向上抛出，由调用方降级为日志。"""
        records = self.load()
        for index, existing in enumerate(records):
            if existing.get("id") == rec.get("id"):
                records[index] = rec
                break
        else:
            records.append(rec)
        if len(records) > self.cap:
            records = records[len(records) - self.cap:]
        _atomic_write(self.path, records)
