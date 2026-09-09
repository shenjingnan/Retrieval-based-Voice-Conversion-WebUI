# 训练页任务中心化改造：表单重置 + 队列项内联详情 + 历史持久化

日期：2026-09-09
前置：docs/plans/2026-09-09-training-queue-design.md（串行队列，已上线）
状态：已确认（用户逐项确认三项交互决策，见 §2）

## 1. 背景与问题

队列机制上线后验收暴露的工作流断裂：

1. 提交任务 1 后表单仍保留其音频与参数；DatasetPicker 无重置机制，新一批音频会
   **追加**进同一上传会话数据集（`ds-xxxxxxxx`）。
2. 监视区只盯单个「被监视任务」，多任务时看不同任务的 loss/日志需在队列里来回
   切换监视对象。
3. 用户想删任务 1 的数据集（任务 2 不用它）被 409 拦截。护栏本身正确（运行中
   任务仍引用该数据集，引用级判定见 server/api/datasets.py `_referencing_task`），
   根因是「提交后不重置」迫使新旧任务共用素材语境——重置流程落地后新任务用
   全新数据集，旧数据集等占用任务终态后即可删，**护栏不改**。
4. 历史训练记录只在内存，重启即清空。且**磁盘日志不完整**：pipeline 多 cmd 共用
   `logs/{exp}/pipeline_task.log` 且 truncate=True 每条 cmd 启动前截断
   （server/tasks.py `_run`），任务结束后磁盘只剩最后一段（index 段）；
   train.py 的 loss/epoch 行走 stdout 重定向进该任务日志——**内存环形缓冲
   （deque maxlen=1000）是唯一完整来源**。不落快照就无法跨重启回看。

## 2. 已确认决策

| 决策点 | 结论 |
| --- | --- |
| 表单重置范围 | **全部重置**（音频 + 高级设置全回默认；batch size 回「自动」并同步重置 touched 标记） |
| 步骤条与监视卡 | **全部收进队列项**：移除顶部步骤条与独立监视卡；队列项单开手风琴展开（阶段指示/进度/loss/日志/失败原因/重新提交/成功产物+去试音） |
| 历史持久化 | **落盘跨重启回看**：任务终态时把状态/参数/日志尾部快照（300 行）写入 logs/task_history.json（上限 50 条） |

## 3. 方案概要

### 后端
- `TaskManager.on_finish` 公开回调属性：终态锁内取快照、锁外同步调用、异常降级为
  日志。**必须覆盖全部四条终态路径**：`_finish`（成功/失败/运行中取消）与
  `cancel()` pending 分支、`clear_queue()`、`dispose()` 三条绕过路径。
- `TaskHistoryStore`（server/task_store.py）：与 TaskStore 同构的原子 JSON 存储，
  按 id 替换/追加、cap 截断、损坏兜底、逐条剔除坏记录。
- 记录器（server/api/training.py）：惰性 store + 内存镜像缓存 + 幂等集合；
  `_TASK_META` 无条目（separate 任务）不记录；`GET /api/tasks` 合并内存与历史
  （内存优先去重，历史行不带 logs_tail），`GET /api/tasks/{id}` 历史回退（含
  logs_tail，供展开时一次性拉取）。

### 前端
- 纯函数层：`lib/loss.ts`（LossPoint/parseLossPoint/parseLossPoints，从 useTask 迁出）、
  `trainingSteps.stageStatuses`（阶段状态推导，数据源 pipeline_stages/current_cmd）。
- 组件层：`components/training/`（TaskQueuePanel 手风琴 + TaskDetail 双源收敛 +
  LossChart/LogPanel 迁出）+ `hooks/useProductName`。
- Training.tsx 收敛为「提交器」：表单 + 提交按钮 + 资源卡 + 队列面板；删步骤条、
  监视卡、分步重试（重试由 TaskDetail 的「重新提交」替代——同名实验提交即
  checkpoint 续训）；提交成功 `resetForm()` 全部重置 + DatasetPicker key remount。
- live/history 双源：展开行 `row.history ? null : id` 挂 useTask（历史行绝不挂
  SSE）；历史详情展开时 GET /api/tasks/{id} 一次并缓存。

## 4. 边界与取舍

- 日志尾部快照 300 行：头部缺 epoch/step 锚点，前 1-2 个 loss 点标注为 null
  （LossChart 已兼容）；更早的 loss 趋势不保留（环形缓冲 1000 行上限的既有语义）。
- 历史行上限 50 条（新进旧出）；单条体积 ≈ 300 行日志，文件 ≤ ~2MB。
- separate 任务不进训练历史（无训练语义），但仍出现在内存任务列表（与现状一致）。
- 切 Tab 卸载 TrainingPage：队列轮询与展开状态随之重置（可接受，与现状一致）。
- 前端无测试基建，纯函数用 `node --experimental-strip-types` 自测，页面靠
  tsc + oxlint + 手工走查。
