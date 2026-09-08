"""训练任务系统：内存任务表 + 全局互斥 + 子进程编排 + 日志环形缓冲。

行为参考（不 import，仅对齐）：webui.py:723-849 的 TRAIN_TASK/互斥锁/start_train_process，
终止复用 tools/process_utils.kill_process_tree（POSIX 进程组 SIGTERM→SIGKILL）。

- 状态机：pending → running → success | failed | cancelled（人为停止判 cancelled，
  不算失败）；终态即释放全局互斥
- 全局互斥：同一时刻仅一个任务处于非终态，重复创建抛 TaskConflictError（API 层转 409）
- 日志：子进程 stdout/stderr 追加写入 log_path（train.log「追加不截断」语义；
  其余阶段的启动前截断由编排层负责）；工作线程按字节偏移增量读新行进环形
  deque(maxlen=1000)，与训练脚本的多进程并发 append 兼容
- 进度：progress 为占位字段（None 表示未知），由 Task 5/6 的编排层写入

延迟导入纪律：本模块顶层不 import torch/configs（与 server.commands 一致，
pytest 收集期不得加载 torch）。
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from server import paths
from server.progress import TailReader
from tools.process_utils import kill_process_tree

logger = logging.getLogger(__name__)

# 状态机
PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({SUCCESS, FAILED, CANCELLED})

# webui.py:786-790：训练子进程注入 RVC_CUDA_GRAPH=0（禁 CUDA graph），其余继承父进程环境
TRAIN_CMD_MARKER = "train/train.py"

# 任务日志环形缓冲上限（设计 §3.1）：够覆盖一次训练的关键日志，又不至于撑爆内存
LOG_BUFFER_MAXLEN = 1000
# 任务失败时 error 携带的日志尾部行数（设计 §4：detail 带日志尾部 + returncode）
ERROR_LOG_TAIL_LINES = 20


class TaskConflictError(RuntimeError):
    """已有任务处于非终态时再次创建；API 层转 409。"""


class _Task:
    """任务运行时对象。

    可变字段（state/progress/error/logs/log_seq/processes/started_at/finished_at）
    都由 TaskManager._lock 保护；id/name/cmds/log_path/setup/truncate 构造后不再修改，
    stop_event 是线程安全的 Event；thread 仅在创建线程时写入一次。
    """

    def __init__(self, task_id, name, cmds, log_path, setup, truncate):
        self.id = task_id
        self.name = name
        self.cmds = list(cmds)
        self.log_path = Path(log_path)
        self.setup = setup
        self.truncate = truncate
        self.state = PENDING
        self.progress = None
        self.error = None
        # 当前正在执行的子命令序号（1-based；None = 尚未启动任何 cmd）。pipeline
        # 之类多 cmd 任务的消费者（前端步骤条阶段映射）靠它知道推进到哪一步；
        # 终态保留最后执行过的序号，失败诊断可据此定位出错的 cmd
        self.current_cmd = None
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.logs = deque(maxlen=LOG_BUFFER_MAXLEN)
        self.log_seq = 0  # 累计入缓冲行数（SSE 增量游标，不随环形淘汰回退）
        self.processes = []
        self.stop_event = threading.Event()
        self.thread = None


def _popen_env(cmd: str, root: Path) -> dict:
    """子进程环境。webui.py:786-790 同源：训练命令注入 RVC_CUDA_GRAPH=0；
    全部命令统一注入 PYTHONIOENCODING=utf-8（防 Windows 子进程按 GBK 写日志造成
    mojibake，TailReader 侧按 utf-8 解码）；其余继承父进程环境。

    另注入两层 sys.path 修复（webui 不需要：它自己跑在仓库根且由 gradio 入口装配路径）：
    - PYTHONPATH=仓库根（已有值则前插）：`python train/xxx.py` 直跑时 sys.path[0] 是
      train/ 而非仓库根（cwd 不参与脚本形态的路径解析），脚本里的 `from infer.audio
      import ...` 等顶层导入会 ModuleNotFoundError（preprocess.py:21 实测）。
    - PYTHONSAFEPATH=1（即 `python -P`，≥3.11；等价地不把脚本目录插进 sys.path）：
      只补 PYTHONPATH 还不够——sys.path[0]=train/ 会让顶层 `train` 被解析成
      train/train.py（同目录同名模块优先于 PYTHONPATH 里的命名空间包），train/preprocess.py:22
      的 `from train.dataset.slicer2 import Slicer` 与 train/train.py:31 的
      `from train import utils` 因此 ImportError。命令模板保持与 webui 逐字一致，
      修复全部收在环境变量层。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONSAFEPATH"] = "1"
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(root), inherited]) if inherited else str(root)
    if TRAIN_CMD_MARKER in cmd.replace("\\", "/"):
        env["RVC_CUDA_GRAPH"] = "0"
    return env


def _failure_message(task: _Task, step: int, cmd: str, returncode: int) -> str:
    """失败详情：标明第几步与命令串（extract 双 cmd 共享日志时前端要能区分是 f0 还是
    hubert 挂了），附返回码与日志尾部（设计 §4）。"""
    tail = list(task.logs)[-ERROR_LOG_TAIL_LINES:]
    detail = "\n".join(tail) if tail else "(暂无日志输出)"
    return "子进程执行失败（第 %d 步：%s），返回码：%s\n日志尾部：\n%s" % (
        step,
        cmd,
        returncode,
        detail,
    )


class TaskManager:
    """内存任务表。互斥在实例内生效，API 层只用模块级单例 task_manager，
    因此进程级等价于「同一时刻仅一个训练任务」；测试可自建实例做隔离。
    """

    def __init__(self, poll_interval: float = 1.0, *, root: Path = paths.ROOT, popen=None):
        """popen 可注入替身（测试隔离）；省略时每次启动才解析 subprocess.Popen，
        因此用 monkeypatch 替换 subprocess.Popen 的测试同样生效。"""
        self.poll_interval = poll_interval
        self._root = Path(root)
        self._popen = popen
        self._lock = threading.Lock()
        self._tasks = {}  # task_id -> _Task（dict 保序，list_tasks 按创建序输出）
        self._active_id = None  # 全局互斥标志：当前非终态任务的 id

    # -- 创建 / 查询 -------------------------------------------------------

    def create_task(self, name, cmds, log_path, truncate: bool = True, *, setup=None) -> str:
        """登记任务并启动后台工作线程，返回 task_id。

        cmds 为按序执行的 shell 命令串（每个 cmd 一次 Popen，前一 cmd 成功才起下一个）；
        setup 为可选 Python 前置步骤（在命令之前于工作线程内执行，异常 → 任务 failed，
        供编排层放 filelist/config 生成与产物校验）。setup 的返回值若为字符串或字符串
        列表，会作为提示行写入任务日志缓冲（编排层用它上报「未使用底模」一类信息——
        只写缓冲不落盘，因为默认 truncate=True 时首个 cmd 启动就会截断日志文件）。
        truncate 控制 log_path 的打开方式：默认 True（每个 cmd 启动前以 "wb" 截断，
        对齐 webui 每阶段截断行为）；若任务日志要跨任务续写（如将来直接接 train.log）
        必须显式传 False，否则会丢上一次的日志。
        """
        cmds = list(cmds)
        if not cmds:
            raise ValueError("cmds 不能为空：任务至少需要一个步骤")
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)  # 失败应快速暴露而非吞成任务失败

        with self._lock:
            if self._active_id is not None:
                active = self._tasks.get(self._active_id)
                raise TaskConflictError(
                    "已有训练任务在运行（%s，id=%s），请等待完成或先停止"
                    % (active.name if active else "未知", self._active_id)
                )
            task = _Task(uuid.uuid4().hex, name, cmds, log_path, setup, truncate)
            self._tasks[task.id] = task
            self._active_id = task.id

        thread = threading.Thread(
            target=self._run, args=(task,), name="rvc-task-%s" % task.name, daemon=True
        )
        with self._lock:
            task.thread = thread
        thread.start()
        logger.info("任务已创建：%s（%s），id=%s", task.name, task.cmds, task.id)
        return task.id

    def get_task(self, task_id):
        """状态快照（副本）；未知 id 返回 None。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return self._snapshot(task)

    def list_tasks(self):
        """全部任务的快照，按创建顺序（仅本进程内的任务，重启即清空）。"""
        with self._lock:
            return [self._snapshot(task) for task in self._tasks.values()]

    def read_logs_since(self, task_id, cursor):
        """SSE 增量游标：返回 (新行, 新游标)。cursor 为累计行数（task 快照里的 log_seq），
        环形缓冲淘汰旧行时自动收敛到缓冲里剩余的部分；未知 id 返回 None。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            seq = task.log_seq
            if seq <= cursor:  # 快路径：无新行就不复制缓冲（SSE 高频调用）
                return (), seq
            buffered = list(task.logs)
        new = seq - cursor
        return tuple(buffered[-new:]), seq

    def update_progress(self, task_id, progress) -> bool:
        """写进度占位字段（0-1，越界钳位；None 表示未知）。Task 5/6 的编排层调用。"""
        if progress is not None:
            progress = max(0.0, min(1.0, float(progress)))
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.progress = progress
        return True

    def cancel(self, task_id) -> bool:
        """请求终止：置停止标志，工作线程在下一个轮询周期内终止当前进程组并跳过后续 cmd。
        实际终止由工作线程执行（进程句柄的唯一所有者），延迟至多一个 poll_interval；
        任务不存在或已终态返回 False。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state in TERMINAL_STATES:
                return False
            task.stop_event.set()
        logger.info("任务收到停止请求：%s", task_id)
        return True

    def dispose(self, timeout: float = 10.0) -> None:
        """终止仍在运行的任务并等待工作线程退出。

        默认超时 10s：kill_process_tree 自身可阻塞约 6s（SIGTERM → 等待 → SIGKILL → wait），
        取更短的值会在线程还没收尾时提前返回。Task 5 必须在 FastAPI lifespan 的 shutdown
        阶段调用（见 docs/plans/2026-09-06-p2-training-plan.md Task 5 第 10 条），
        否则 daemon 线程被硬杀后训练子进程会孤儿化。
        """
        with self._lock:
            all_tasks = list(self._tasks.values())
        for task in all_tasks:
            task.stop_event.set()
        for task in all_tasks:
            if task.thread is not None and task.thread.is_alive():
                task.thread.join(timeout)

    # -- 内部 --------------------------------------------------------------

    @staticmethod
    def _snapshot(task: _Task) -> dict:
        """须持锁调用：返回含环形缓冲副本的快照。"""
        return {
            "id": task.id,
            "name": task.name,
            "state": task.state,
            "progress": task.progress,
            "error": task.error,
            "cmds": list(task.cmds),
            "current_cmd": task.current_cmd,
            "log_path": str(task.log_path),
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "log_seq": task.log_seq,
            "logs": list(task.logs),
        }

    def _run(self, task: _Task) -> None:
        """工作线程主体：按序执行 cmds，任一环节异常都必须落到终态（否则互斥不释放）。"""
        try:
            with self._lock:
                task.state = RUNNING
                task.started_at = time.time()

            if task.setup is not None:
                try:
                    setup_output = task.setup()
                except Exception as exc:
                    self._finish(task, FAILED, "前置步骤失败：%s" % exc)
                    return
                self._append_setup_output(task, setup_output)

            # truncate=False：整任务共用一个增量读取器（追加语义）；truncate=True：文件按 cmd
            # 重建，读取器也按 cmd 重建（offset 归零）——否则截断后新内容长度一旦达到旧偏移，
            # TailReader 会误判为无新行而把整段输出跳过
            reader = None if task.truncate else TailReader(task.log_path)
            log_mode = "wb" if task.truncate else "ab"
            for step, cmd in enumerate(task.cmds, start=1):
                if task.stop_event.is_set():  # 每个 cmd 起前都检查（pipeline 提前收尾）
                    self._finish(task, CANCELLED, None)
                    return
                with self._lock:
                    task.current_cmd = step
                if task.truncate:
                    reader = TailReader(task.log_path)
                # with：无论后续流程如何退出都关闭父进程侧的日志句柄
                with open(task.log_path, log_mode) as log_handle:
                    process = self._start(task, cmd, log_handle)
                    returncode = self._wait(task, process, reader)
                if task.stop_event.is_set():
                    self._finish(task, CANCELLED, None)  # 人为停止优先于非零返回码
                    return
                if returncode != 0:
                    self._finish(task, FAILED, _failure_message(task, step, cmd, returncode))
                    return
            self._finish(task, SUCCESS, None)
        except Exception as exc:  # noqa: BLE001 线程兜底
            logger.exception("任务线程异常：%s", task.id)
            self._finish(task, FAILED, "任务线程异常：%s" % exc)
        finally:
            self._release(task)

    def _start(self, task: _Task, cmd: str, log_handle):
        """启动单个 cmd：stdout/stderr 重定向进日志文件，POSIX 下独立进程组。"""
        kwargs = {
            "shell": True,
            "cwd": self._root,  # 脚本内 ./logs/{exp} 等相对路径按仓库根解析
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":  # webui.py:791-796 同源的平台分支
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        env = _popen_env(cmd, self._root)  # 与 cwd 同源：脚本相对路径与顶层导入都按仓库根解析
        kwargs["env"] = env

        logger.info("任务 %s 执行命令：%s", task.name, cmd)
        process = (self._popen or subprocess.Popen)(cmd, **kwargs)
        with self._lock:
            task.processes.append(process)
        if task.stop_event.is_set():  # webui.py:793-796：封掉「启动瞬间被取消」的窗口
            kill_process_tree(process, task.name, logger)
        return process

    def _wait(self, task: _Task, process, reader: TailReader) -> int:
        """轮询单个进程直至退出；期间增量读日志，收到停止请求则终止进程组。"""
        kill_sent = False
        while True:
            if task.stop_event.is_set() and not kill_sent:
                kill_sent = True
                kill_process_tree(process, task.name, logger)
            self._drain(task, reader)
            returncode = process.poll()
            if returncode is not None:
                self._drain(task, reader)  # 收尾再读一次，拿全最后几行
                return returncode
            time.sleep(self.poll_interval)

    def _drain(self, task: _Task, reader: TailReader) -> None:
        lines = reader.read_new_lines()
        if not lines:
            return
        with self._lock:
            task.logs.extend(lines)
            task.log_seq += len(lines)

    def _append_setup_output(self, task: _Task, output) -> None:
        """setup 返回的提示行进任务日志缓冲（log_seq 一并推进，SSE 游标才不会漏行）。
        仅写缓冲不写文件：truncate=True 时首个 cmd 启动会以 "wb" 截断日志文件，
        落盘内容必被抹掉，而缓冲才是 GET/SSE 暴露的任务日志视图。"""
        if not output:
            return
        lines = [output] if isinstance(output, str) else list(output)
        with self._lock:
            task.logs.extend(lines)
            task.log_seq += len(lines)

    def _finish(self, task: _Task, state: str, error) -> None:
        """落终态并释放全局互斥（同一把锁内完成，避免出现终态但锁未释放的窗口）。"""
        with self._lock:
            if task.state in TERMINAL_STATES:
                return
            task.state = state
            task.error = error
            task.finished_at = time.time()
            if self._active_id == task.id:
                self._active_id = None
        if state == FAILED:  # 失败终态提级为 warning，服务日志里直接带日志尾部便于排障
            logger.warning("任务 %s 失败：%s", task.id, error or "(无错误信息)")
        else:
            logger.info("任务 %s 终态：%s", task.id, state)

    def _release(self, task: _Task) -> None:
        with self._lock:  # 兜底：任何路径退出线程都不能遗留互斥标志
            if self._active_id == task.id:
                self._active_id = None


#: 进程级单例，训练全局互斥的载体（API 层只 import 这个）
task_manager = TaskManager()
