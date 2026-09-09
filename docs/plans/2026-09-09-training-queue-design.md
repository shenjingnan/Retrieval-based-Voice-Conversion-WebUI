# 训练任务队列（排队串行执行）设计

日期：2026-09-09
状态：已确认（需求方逐项确认容错语义与持久化范围）

## 1. 背景与目标

训练一个实验动辄数小时。当前 WebUI 的任务系统（`server/tasks.py`）实现了严格的
「同时仅一个非终态任务」全局互斥，但互斥的表达方式是**提交时拒绝**（409）：
用户必须守在电脑前等上一个任务跑完才能提交下一个。

目标：允许连续提交多个训练任务，服务端按提交顺序串行执行，完成（或失败）一个
自动开始下一个，用户无需守候。

不引入任何队列中间件（Celery/Redis）：延续
`docs/plans/2026-09-06-new-webui-design.md` 的「单用户单机、单命令启动」约束，
队列在本进程内实现。

## 2. 已确认的需求决策

| 决策点 | 结论 |
| --- | --- |
| 队列容错 | 某任务失败或被手动取消后，**自动继续执行下一个**；另提供「停止并清空队列」一次性全停 |
| 持久化 | 排队中的任务落盘 JSON，服务重启后**自动恢复继续跑**；「运行中」任务不做自动续跑（被重启打断的任务需手动重新提交，`train/train.py:246` 会从最新 checkpoint 自动续训） |
| 同名实验 | 唯一保留的 409：同一 `exp_name` 已有非终态任务（运行中或排队中）时拒绝提交，防止共写 `logs/{exp}` 互相覆盖 |
| 队列范围 | 所有任务类型（4 个分步任务、一键 pipeline、数据集人声分离）统一排队，机制位于 `TaskManager` 层，与任务语义无关 |

## 3. 现状分析

- `server/tasks.py`：状态机 `pending → running → success | failed | cancelled`；
  `pending` 目前是转瞬即逝的过渡态（登记后立即启动工作线程）；互斥标志
  `_active_id` 指向当前非终态任务，`create_task` 冲突时抛 `TaskConflictError`。
- 任务自包含：`_Task` 持有 cmds/log_path/setup，创建时不做实际工作；参数校验在
  API 层提交时完成（400 快速失败），产物校验是 setup 钩子、在真正执行时才跑——
  排队数小时后执行反而更正确（校验的是执行时刻的产物状态）。
- 互斥释放点集中：`_finish`（正常路径）与 `_release`（兜底），调度器有明确的挂载点。
- `/train/pipeline` 已把一个完整训练流程打包为单任务多 cmd，正是「一个训练任务」
  的排队粒度。
- 测试护航：`test_tasks.py`（36 例）+ `test_training_api.py`（57 例）。

## 4. 方案

### 4.1 核心机制（server/tasks.py）

- `TaskManager` 增加 `_queue`（待执行任务 id 有序列表）。`create_task` 不再抛
  `TaskConflictError`：登记为 `pending` 入队，然后调用 `_dispatch()`。
- `_dispatch()`：持锁检查 `_active_id`；空闲则从 `_queue` 队头取第一个仍为
  `pending` 的任务（跳过陈旧条目），设为活跃并出队；锁外启动其工作线程。
  首个任务立即启动与「终态后启动下一个」走同一条路径。
- 调度触发：工作线程 `finally` 中 `_release` 之后调用 `_dispatch()`——成功、
  失败、取消三种终态都自动推进队列。
- 排队中的取消：不涉及进程，直接出队并落 `cancelled` 终态。若恰逢调度器刚出队、
  线程未起的窗口，退化为既有逻辑（`stop_event` → 工作线程首个 cmd 前检查 →
  `cancelled`，`tasks.py` `_start` 已封该窗口）。
- `_run` 启动时若发现任务已终态（出队与启动之间被取消）直接返回，不得「复活」。
- `clear_queue()`：停止正在跑的（`stop_event`）+ 全部 `pending` 落 `cancelled`。
- `dispose()`：除既有逻辑外，把 `pending` 任务也落 `cancelled`。
- 快照新增 `queue_position`（`pending` 为 1-based 队列位次，其余为 `None`）。
- `create_task` 扩展可选参数：`definition`（API 层提供的可序列化任务定义，原样
  持久化）、`task_id`/`created_at`（恢复路径保留原标识与提交时间）。
- `TaskConflictError` 删除：TaskManager 层不再有「拒绝提交」语义，同名实验互斥
  上移到 API 层。

### 4.2 持久化（server/task_store.py + 恢复）

- `TaskStore`：JSON 文件（`paths.queue_file()` → `logs/task_queue.json`，gitignored），
  原子写（临时文件 + `os.replace`）；损坏时把坏文件改名保留后按空队列处理。
- 持久化内容 = 当前全部 `pending` 任务的记录：
  `{id, created_at, name, cmds, log_path, truncate, definition}`。
  **cmds 存提交时已解析的最终命令串**（batch_size 自适应、device/is_half、底模
  存在性探测都已定格），恢复路径不触发 torch 加载。
- 持久化时机：每次 pending 集合变化（入队 / 出队启动 / 排队取消 / 清空 /
  dispose / 恢复完成）后全量重写。文件始终是 pending 集合的快照，天然幂等。
- `definition` 由 API 层构造：`{kind, body, batch_note?, pipeline_stages?}`，
  全部 JSON 安全；恢复时按 `kind` 纯函数重建 setup 闭包与 `_TASK_META`
  （total_epoch / pipeline_stages），同样不触 torch。
- `training.restore_pending_tasks()`（`main.py` lifespan startup 调用）：读文件、
  逐条重建并入队（保留原 id 与 created_at）；解析失败 / kind 未知 / 重复 id 的
  记录跳过并 `logger.warning`；恢复完成后由入队路径自动重写文件（剔除坏记录）。
- 恢复顺序 = 文件顺序 = 原提交顺序；首条恢复任务即刻启动。

### 4.3 API 层（server/api/training.py / datasets.py）

- 各训练端点拆出「spec 构造」纯函数（校验 + 命令拼装 + definition 构造），端点
  与恢复路径共用；`_create` 传入 definition，返回
  `{task_id, queued, queue_position}`。
- 新增 `_ensure_exp_idle(exp_name)`：非终态任务中存在相同 `exp_name`（记于
  `_TASK_META`）→ 409。唯一保留的冲突语义。
- 新增 `DELETE /api/tasks`：清空队列（停当前 + 取消全部排队），返回
  `{stopped_task, cancelled_pending}`。
- `datasets.py` 的 separate 任务：删除 `TaskConflictError` 处理，自然进入队列。
- `GET /api/tasks` 紧凑投影增加 `queue_position` 与 `exp_name`（供前端队列面板）。

### 4.4 前端（frontend/src）

- `client.ts`：`TaskCreated`/`TaskSummary` 增加 `queue_position`；新增
  `api.clearQueue()`；`TaskView`（useTask）透传 `queue_position`。
- `Training.tsx`：
  - 提交按钮放开 `running` 禁用（保留表单校验 / busy / 上传中禁用）；提交后排队的
    任务在监视卡显示「排队中（第 N 位）」而非「进行中」。
  - 排队提交不乐观置步骤条 running；监视任务从 `pending` 转 `running` 时再置。
  - 新增任务队列面板：轮询 `GET /api/tasks`（3s，页面存活期常驻），展示运行中 /
    排队中 / 近期终态任务（状态徽标、进度、队位、单任务停止按钮），以及
    「停止并清空队列」入口（两段式确认，与既有停止按钮同款交互）。

## 5. 边界与取舍

- **运行中任务重启即中断**：不自动续跑（避免区分 fit 可续训 vs 前置阶段重跑的
  复杂边界）；用户手动重新提交，checkpoint 自动续训。文档与 UI 不承诺运行中任务
  跨重启存活。
- **持久化写放大**：pending 集合变化是低频事件（提交/终态），全量重写小 JSON
  的开销可忽略；单进程串行写，无并发竞争（锁外写，最坏 momentarily 陈旧）。
- **测试隔离**：conftest 把真实单例的 queue_store 置 None，测试永不触碰真实
  `logs/task_queue.json`；restore 相关测试自建 TaskManager + tmp 文件。
- **不做**：队列长度上限、优先级/重排、任务修改（删了重提）、运行中任务跨重启
  恢复、多说话人任务接入（现有 P3 范围外）。

## 6. 实施阶段与验收

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| P1 | tasks.py 队列核心（排队 / 调度 / pending 取消 / clear / dispose） | pytest：运行中提交第二个任务入队，首个终态后自动启动；失败/取消同样推进；取消排队任务即时生效不启动进程 |
| P2 | API 层（spec 拆分 / 200+queued / 同名 409 / DELETE /api/tasks） | pytest：运行中 POST /train/pipeline 返回 200 与 queue_position；同名 409；清空端点行为正确 |
| P3 | 持久化 + 启动恢复 | pytest：store 往返 / 损坏兜底 / 恢复保留 id 与顺序 / 坏记录跳过；重启服务后排队任务继续跑（手动验收） |
| P4 | 前端队列面板与提交放行 | pnpm build 通过；页面连续提交两个一键任务，第二个显示排队位次并自动接续（手动验收） |
