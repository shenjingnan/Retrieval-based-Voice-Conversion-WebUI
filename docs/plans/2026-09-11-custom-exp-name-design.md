# 自定义训练实验名 — 设计与实施记录

日期：2026-09-11　分支：`feature/train-custom-exp-name`

## 需求

训练界面提供自定义实验名输入：

1. 非必填——留空沿用现有自动生成（`voice-月日时分-随机后缀`），填写则用用户的；
2. 开始训练前检查实验名是否已存在，防止冲突（已确认行为：**拒绝提交并提示**，
   不提供「覆盖/续训」选择——续训走任务历史的显式「重新提交」路径）；
3. 冲突检查时机（已确认）：**输入时实时提示 + 提交时后端兜底**。

## 现状与约束

- 实验名物理上对应 `logs/{exp_name}` 目录；「已存在」以该目录存在为准。
- 已有 `_ensure_exp_idle`（活动任务互斥 409）管「排队/运行中」，与本设计的
  「磁盘已有产物」检查互补，互不替代。
- 两条既有路径**依赖**「已存在实验可写」，不得被新检查误伤：
  - 任务历史「重新提交」按钮：同名实验 = 断点续训（显式意图）；
  - Models 页 `trainIndex`（重建索引）：天然操作已存在实验。
  因此存在性检查**只挂在 pipeline 提交路径**。

## 后端（server/api/training.py）

- `PipelineBody.allow_existing: bool = False`：False = 新提交语义，目录已存在则 409；
  True 仅由「重新提交」携带（失败任务的目录必然已存在，重提交即续训）。
- `_ensure_exp_dir_free`：`allow_existing=False` 且 `logs/{exp}` 存在 → 409，
  文案指引「换名」或「任务历史重新提交续训」。接入 `start_pipeline`。
- 新增只读端点 `GET /api/train/exp-name/exists?name=`：复用 `_check_exp_name`
  （非法名 400，文案与提交一致），合法名返回 `{"exists": bool}`。

## 前端（frontend/src）

- `client.ts`：`TrainParams.allow_existing?`；`trainPipeline` 透传（undefined 省略键）；
  新增 `trainExpNameExists(name)`。
- `Training.tsx`：
  - 表单首字段「实验名（可选）」输入框；
  - 本地格式校验镜像后端 `_check_exp_name` 字符规则（非法名不发检查请求，后端兜 400）；
  - 防抖 400ms 调 exists 接口；结果与查询时的名字绑定存储（`{name, exists}`），
    渲染时按当前名字派生占用态——名字一变提示即时消失，迟到响应由 ref 比对丢弃；
    检查失败静默（后端 409 兜底），不阻塞表单；
  - `currentParams`：用户填写优先；留空则**每次提交现场生成**新自动名、不写回
    state——提交失败重试自然换新名，不会反复撞同一个 409；
  - `resubmitTask` 显式带 `allow_existing: true`；
  - 提交按钮 disabled 追加 `expNameErr === null && !expNameTaken`。

## 测试

后端（tests/test_training_api.py，TDD：先失败后实现）：

- exists 端点：目录缺席/在场、非法名 400（参数化）；
- pipeline：目录已存在 → 409 且不创建任务；`allow_existing=true` 放行；
- 回归：存在性检查不外溢到 preprocess/extract/fit/index（续训与索引重建语义）。

前端无测试设施（仅 tsc + oxlint），靠类型检查、lint 与手动验证。

## 已知边界

- 输入时检查与提交之间存在竞态窗口（TOCTOU），由提交时后端兜底 409 收口；
  单用户本地服务，窗口实际风险可忽略。
- 自动生成名同样受兜底检查保护：极端撞名时用户重试即换新名。
