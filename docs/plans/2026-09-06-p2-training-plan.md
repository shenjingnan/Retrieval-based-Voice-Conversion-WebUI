# P2 训练功能 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.（本会话采用 subagent-driven-development，等效）

**Goal:** 新版 WebUI 补齐单说话人训练全流程：向导式 4 步训练 + 一键 + 实时日志/进度 + 模型管理，产物与旧版 webui 完全兼容。

**Architecture:** 见 `docs/plans/2026-09-06-p2-training-design.md`（本计划的设计依据，命令行契约与解析锚点均已核实）。

**Tech Stack:** 沿用 P1（FastAPI/pytest/React19/shadcn4.x）。新增依赖：无。

**工作区:** `/Users/nemo/github/Retrieval-based-Voice-Conversion-WebUI/.worktrees/new-webui`（feature/new-webui）。所有相对路径基于此。

**关键前置事实（已核实，实现者必读）:**
- `config.python_cmd` 语义 = `sys.executable or "python"`（configs/config.py:191）；server 侧直接用 `sys.executable`，不需要 Config 实例
- `config.preprocess_per` 默认 3.7；`config.n_cpu = cpu_count()`；`config.device`/`is_half` 来自 `configs.config.Config()`（单例开销可接受，延迟导入防 pytest 加载 torch）
- 4 个训练脚本**没有 argparse，全是 sys.argv 位置参数**；布尔传参是 `"True"/"False"` 字符串
- `train.log` 追加不截断；其余三阶段日志在每次启动前由 server 截断（对齐 webui 行为）
- mute 样本已就位于 `logs/mute/`（mute48k.wav / mute.wav.npy / 2a_f0、2b-f0nsf、3_feature256|768 齐全）
- `train_index.py` 会把索引链接到其 argv[3]（outside_index_root）→ 传 `"assets/indices"`，与 P1 `/api/models` 闭环
- 全部子进程 `cwd = server.paths.ROOT`（脚本内 `./logs/{exp}` 是相对路径）
- fit 子进程环境变量：`RVC_CUDA_GRAPH=0`（其余继承 os.environ）
- 现有 7861 服务在运行中——pytest 已验证可共存，但**不要动它**

---

### Task 1: 命令拼装纯函数模块 `server/commands.py`

**Files:**
- Create: `server/commands.py`
- Test: `tests/test_commands.py`

**Step 1: 写失败测试**（快照测试，逐字断言命令串）

```python
import sys

from server.commands import (
    build_extract_f0_cmd, build_extract_hubert_cmd, build_fit_cmd,
    build_index_cmd, build_preprocess_cmd, get_pretrained_paths, SR_DICT,
)


def test_sr_dict():
    assert SR_DICT == {"32k": 32000, "40k": 40000, "48k": 48000}


def test_build_preprocess_cmd():
    cmd = build_preprocess_cmd("/data/ds", 40000, 8, "mi-test", False, 3.7)
    assert cmd == ('"%s" train/preprocess.py "/data/ds" 40000 8 "%s/logs/mi-test" False 3.7'
                   % (sys.executable, str(ROOT := __import__("server.paths", fromlist=["ROOT"]).ROOT)))


def test_build_extract_f0_cmd():
    cmd = build_extract_f0_cmd("mi-test", 8, "rmvpe")
    assert cmd == ('"%s" train/dataset/extract_f0.py cpu "%s/logs/mi-test" 8 rmvpe'
                   % (sys.executable, str(__import__("server.paths", fromlist=["ROOT"]).ROOT)))


def test_build_extract_hubert_cmd():
    cmd = build_extract_hubert_cmd("mi-test", "v2", False)
    # device/is_half 由调用方传入函数参数（见签名），断言 7 参数形态
    assert cmd.count(" ") >= 6 and "extract_hubert_feature.py" in cmd
    assert cmd.endswith("mi-test\" v2 False") or 'v2 False' in cmd


def test_build_fit_cmd_with_pretrained():
    cmd = build_fit_cmd("mi-test", "48k", True, 8, 20, 5, True,
                        "assets/pretrained_v2/f0G48k.pth", "assets/pretrained_v2/f0D48k.pth")
    assert '-e "mi-test" -sr 48k -f0 1 -bs 8' in cmd
    assert "-te 20 -se 5" in cmd
    assert '-pg assets/pretrained_v2/f0G48k.pth' in cmd
    assert "-l 0 -c 0 -sw 1 -v v2" in cmd
    assert "-g" not in cmd  # P2 单进程，无 -g


def test_build_fit_cmd_without_pretrained():
    cmd = build_fit_cmd("e", "48k", False, 8, 20, 5, False, "", "")
    assert "-pg" not in cmd and "-pd" not in cmd and "-f0 0" in cmd


def test_build_index_cmd():
    cmd = build_index_cmd("mi-test", "v2", 8)
    assert cmd == ('"%s" train/train_index.py "mi-test" v2 "assets/indices" 8 single'
                   % sys.executable)


def test_get_pretrained_paths(monkeypatch, tmp_path):
    (tmp_path / "f0G48k.pth").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    g, d = get_pretrained_paths("48k", True, "v2")
    assert g.endswith("pretrained_v2/f0G48k.pth") and d.endswith("f0D48k.pth")
    g2, d2 = get_pretrained_paths("48k", True, "v1")
    assert g2 == "" and d2 == ""  # v1 底模不存在 → 空串
```

（注意：`get_pretrained_paths` 的 v1/v2 分支按设计文档 §3.3——`path_str = "" if v1 else "_v2"`，v1+32k 强制 40k。测试按此语义编写，实现者可按实际调整断言细节但语义不得偏离 webui.py:1083-1146。）

**Step 2: 运行确认失败**

**Step 3: 实现** `server/commands.py`——要点：
- 每个函数返回 shell 字符串（不是列表），路径双引号包裹，与设计文档 §3.3 表格逐字一致
- `build_fit_cmd` 签名建议 `(exp, sr, if_f0, batch_size, total_epoch, save_every, save_every_weights, pretrain_G, pretrain_D)`；`-pg/-pd` 仅在非空串时拼入
- `build_extract_hubert_cmd(exp, version, is_half)` 内部取 `device`（延迟 `from configs.config import Config; Config().device`，模块级缓存避免重复探测）拼 7 参数形态：`"{py}" ... extract_hubert_feature.py {device} 1 0 "{ROOT}/logs/{exp}" {version} {is_half}`
- `get_pretrained_paths(sr, if_f0, version) -> tuple[str, str]`：照 webui `get_pretrained_models`，路径 `assets/pretrained{path_str}/{f0_str}G{sr}.pth`；v1+32k → sr 改 "40k"；不存在返回 `("", "")`

**Step 4: 测试通过 → Step 5: Commit** `feat(server): 训练命令拼装与底模选择`

---

### Task 2: filelist 与 config.json 生成（`server/api/training.py` 纯函数层）

**Files:**
- Create: `server/api/training.py`（本任务只放纯函数，路由 Task 5 加）
- Test: `tests/test_training_prep.py`

**Step 1: 写失败测试**（tmp_path 造假产物目录）

覆盖点：
1. `generate_filelist(exp_dir, sr, if_f0)`：
   - 四目录 stem 交集；行格式 `"{gt}/{n}.wav|{fea}/{n}.npy|{f0}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0"`（路径不做反斜杠转义也行，POSIX 无影响——但保持与 webui 一致的 `\\\\` 替换无副作用，二选一并在注释说明）
   - 追加 2 条 mute 行：`"{ROOT}/logs/mute/0_gt_wavs/mute{sr}k.wav|{ROOT}/logs/mute/3_feature{fea_dim}/mute.npy|{ROOT}/logs/mute/2a_f0/mute.wav.npy|{ROOT}/logs/mute/2b-f0nsf/mute.wav.npy|0"`（fea_dim 256/768 按 version）
   - 写 `logs/{exp}/filelist.txt`，`\n`.join 无结尾换行
   - 交集为空 → 抛 `ValueError("没有可用于训练的有效音频，请先完成数据切分和特征提取")`
2. `write_config(exp_dir, sr, version)`：v2+48k → 模板 `configs/v2/48k.json`；v2+40k → `configs/v1/40k.json`；v1+40k → `configs/v1/40k.json`；已存在 config.json 时沿用不覆盖；写出的 json 含 `model.spk_embed_dim=109` 且无 `speaker_info` 键
3. `validate_preprocess_outputs(exp_dir)` / `validate_feature_outputs(exp_dir, version, if_f0)`：交集非空通过，空则抛 RuntimeError（信息对齐 webui）

**Step 2: 确认失败 → Step 3: 实现**（对照 webui.py:1169-1307 / :694-720 逐字移植；shuffle 用 `random.shuffle`）**→ Step 4: 通过 → Step 5: Commit** `feat(server): filelist 与训练配置生成（移植自 webui）`

---

### Task 3: 任务系统 `server/tasks.py`

**Files:**
- Create: `server/tasks.py`
- Test: `tests/test_tasks.py`

**Step 1: 写失败测试**（monkeypatch `subprocess.Popen` 为 fake，不真跑训练）

```python
class FakeProcess:
    def __init__(self, returncode=0, delay=0):
        self.returncode = None
        self._rc = returncode
        self.pid = 42424
    def poll(self):
        return self.returncode
    def finish(self):  # 测试手动推进
        self.returncode = self._rc
```

覆盖点：
1. `create_task(name, cmds: list[str], exp_dir: Path) -> task_id`：任务表登记、状态 pending→running；全部 returncode 0 → success；任一非零且非人为停止 → failed（error 含 returncode 与日志尾部）
2. **全局互斥**：第一个任务 running 时再 create → 抛 `TaskConflictError`（API 层转 409）
3. `cancel(task_id)`：killpg SIGTERM（测试 monkeypatch `os.killpg` 记录调用）→ 状态 cancelled，后续 cmd 不再启动（pipeline 场景 cmds 列表多元素）
4. 日志：fake Popen 落一个日志文件（任务系统按 cmd 顺序读对应 log_path 尾部进环形 deque，maxlen=1000）——实现可简化为：任务系统直接把 cmds 的输出重定向到 `logs/{exp}/train_task_{name}.log`（Popen stdout=file），轮询时增量读该文件进缓冲。**推荐此法**：与 webui 的多进程并发 append 兼容
5. 重启语义：模块级表随进程消亡即可；提供 `list_tasks()` 只返回本进程任务
6. 轮询间隔 1s 可注入（构造参数 `poll_interval`），测试用 0.01s
7. **【实现补充】`create_task(..., truncate: bool = True)`**：默认 True 时每个 cmd 启动前以 `"wb"` 打开 log_path（对齐 webui 每阶段启动前截断的行为），此时增量读取器按 cmd 重建（offset 归零，否则截断后新内容长度达到旧偏移会被误判为无新行而整段跳过）；只有当任务日志要跨任务续写时才传 `truncate=False`（`"ab"` 追加 + 单一读取器）。fit 若将来要接 `train.log` 场景必须传 `truncate=False`，其余默认截断

**Step 2-4: TDD 循环 → Step 5: Commit** `feat(server): 训练任务系统（互斥/终止/日志缓冲）`

---

### Task 4: 进度解析器 `server/progress.py`

**Files:**
- Create: `server/progress.py`
- Create: `tests/fixtures/train_log_sample.log`（真实格式 3 行：`训练轮次：20 [34%]`、`[1234, 0.0001]`、`loss_disc=12.345, loss_gen=23.456, loss_fm=1.234,loss_mel=45.678, loss_kl=0.912`）
- Test: `tests/test_progress.py`

覆盖点：
1. `parse_train_line(line) -> dict | None`：识别 step/lr 行（`^\[(\d+), ([\d.eE+-]+)\]$`）与 loss 行（五项正则）；中英 epoch 行 `训练轮次：(\d+) \[([\d.]+)%\]` / `Epoch: (\d+) \[([\d.]+)%\]` 识别为 epoch 信息；其他行返回 None
2. `count_dir_progress(product_dir, total) -> float | None`：产物数/分母，越界钳 [0,1]；分母 0 返回 None
3. `TailReader(path)`：记录偏移增量读，文件被截断（变小）时自动复位

**Commit** `feat(server): 训练进度解析器`

---

### Task 5: 训练 API 路由 + pipeline 编排

**Files:**
- Modify: `server/api/training.py`（加路由）
- Modify: `server/main.py`（include_router）
- Test: `tests/test_training_api.py`

**Step 1: 写失败测试**（mock `server.tasks.TaskManager` / `server.commands` 函数，不真跑子进程）

覆盖点：
1. `POST /api/train/preprocess` 合法参数 → 200 + task_id；断言拼出的命令传给 TaskManager 的内容（用 monkeypatch 捕获）
2. 校验：`exp_name` 含 `/` 或 `..` → 400（防穿越，同 P1 model 手法）；`dataset_dir` 不存在 → 400；`f0_method` 不在 {pm, rmvpe} → 400
3. `POST /api/train/fit` → 断言先调了 generate_filelist/write_config（mock 它们）再起任务；产物校验失败 → 任务 failed 而非 500
4. `POST /api/train/pipeline` → cmds 队列为 4 段（preprocess、f0+hubert 合并为一个任务的多 cmd、fit、index）；mock 中第一步 failed → 后续不启动
5. 互斥 → 409；`GET /api/tasks/{id}` 未知 id → 404；`DELETE` → 202/200 + cancel 被调
6. extract 步骤内部结构：if_f0 时 cmds = [f0_cmd, hubert_cmd]（顺序执行，共享 `extract_f0_feature.log`——日志文件在两次启动间由任务系统各自 O_TRUNC 打开即对齐 webui 的截断行为，即 `create_task` 默认 `truncate=True`；fit 若将来要接 `train.log` 场景必须显式传 `truncate=False`，其余默认截断）
7. v1+32k 归一化：API 层 `version=="v1" and sr=="32k"` 时强制 `sr="40k"`（对齐 webui.py:1120-1123 change_version19 的全局强制，train.py/config 模板/filelist 的 mute 文件名都依赖这一致性），并加测试
8. 底模缺失必须可见：`get_pretrained_paths` 返回空串时，把「未使用生成器/判别器预训练模型：<路径>」写入任务日志与 SSE（对齐 webui 的「未使用生成器预训练模型」提示），避免静默从零训练
9. **【硬性要求】fit 的任务日志必须用任务专属文件（如 `train_task_fit.log`），不得复用 `logs/{exp}/train.log`**——否则 FileHandler 与 stdout 重定向双写导致锚点行重复且带 asctime 前缀的行解析失败。依据：`train/train.py:138` 在训练循环里调 `train/utils.py:446-458 get_logger(hps.model_dir)`，后者向 `logs/{exp}/train.log` 另挂 FileHandler（`asctime\t实验名\tLEVEL\t行`），同一行日志会同时写进该文件与 stdout 重定向的任务日志，`server/progress.parse_train_line` 的锚点行因此出现两份（epoch 进度与 loss 曲线数据点重复）。preprocess/extract/index 同理各用专属日志文件，`train.log` 只留给训练脚本自身的 FileHandler；另注意 `server/progress.TailReader` 对截断会复位重读，不要在任务运行中途截断它的 log_path
10. **【硬性要求】FastAPI lifespan 的 shutdown 阶段必须调用 `task_manager.dispose()`**——任务工作线程是 daemon 线程，服务退出时不调用的话线程被硬杀，其训练子进程会孤儿化（继续占 GPU/CPU 且无人能终止）。`dispose` 默认 timeout 10s（覆盖 `kill_process_tree` 自身约 6s 的阻塞：SIGTERM→等待→SIGKILL→wait）

**Step 2-4: TDD → Step 5: Commit** `feat(server): 训练 API（4 步 + 一键 + 校验）`

---

### Task 6: SSE 端点 + GET /api/tasks 列表

**Files:**
- Modify: `server/api/training.py`
- Test: `tests/test_training_sse.py`

**Step 1: 写失败测试**

- `GET /api/tasks/{id}/events` 返回 `text/event-stream`；先推 `status` 事件（当前态+进度快照）再持续推增量；用 TestClient 的 `stream()` + fake 任务对象（可控的日志缓冲注入）验证 2-3 个事件的顺序与格式（`event: log\ndata: {...}\n\n`）
- 断开即清理订阅（实现侧用 finally）
- 心跳注释行 `: keepalive` 每 15s（防代理断连；测试注入小间隔）

**Step 2-4: TDD → Step 5: Commit** `feat(server): 任务 SSE 日志/进度流`

---

### Task 7: 前端 API 扩展 + useTask hook

**Files:**
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/useTask.ts`

要点：
- client.ts 增加 `TrainParams` 类型与 `trainPreprocess/trainExtract/trainFit/trainIndex/trainPipeline/cancelTask/getTasks` 方法（错误处理复用 `errorFrom`）
- `useTask(taskId | null)`：EventSource 订阅 `/api/tasks/{id}/events`；返回 `{status, progress, logs(增量数组), losses(loss 曲线数组), error}`；组件卸载/任务终态时 close；断连自动重连一次
- 验证：`pnpm build` 零错误

**Commit** `feat(frontend): 训练 API client 与 useTask hook`

---

### Task 8: 训练向导页

**Files:**
- Create: `frontend/src/pages/Training.tsx`
- Modify: `frontend/src/App.tsx`（Tab 挂载替换占位）

实现（设计 §3.6）：
- 步骤条组件（5 态：待进行/进行中/成功/失败/已跳过），shadcn Card + Button 即可
- 表单：实验名（Input，前端先做一次 `^[^/\\]+$` 校验）、数据集路径（Input，提示是服务器上的绝对/相对路径）、折叠高级区（采样率 Select 48k/40k/32k、f0 开关、音高算法 rmvpe/pm、版本 v2/v1、总轮次 Number 默认 20、保存间隔默认 5、batch size 默认 8）
- 「开始训练」（pipeline）与「分步执行」两种模式；分步模式下每步按钮独立可用、成功后下一步解锁
- 进行中：进度条（progress 事件）+ 日志滚动区（等宽、自动滚底、最多保留 500 行）+ loss 曲线（轻量：内联 SVG polyline 即可，不引图表库）
- 失败：错误卡片 + 「重试该步」
- 成功：显示产物（weights 目录新模型名由 `api.models()` 重新拉取比对）+「去试音」按钮（App 层切换 Tab 并通过 props/callback 把模型名带到推理页——推理页需支持 `initialModel` prop）
- 验证：`pnpm build` 零错误 + dev 模式手工走查表单交互（后端可不连，mock 态不崩即可）

**Commit** `feat(frontend): 训练向导页`

---

### Task 9: 模型管理页 + 删除模型接口

**Files:**
- Modify: `server/api/models.py`（加 `DELETE /api/models/{name}`）
- Modify: `frontend/src/pages/Models.tsx`、`App.tsx`
- Test: `tests/test_models_api.py`（追加）

要点：
- DELETE：basename 校验防穿越；删除 weights 下 .pth，同时删 assets/indices 下与该模型实验名配对的 .index（复用 Task 2 的配对逻辑）；404 当不存在；测试覆盖 3 分支
- 前端卡片：名称、配对索引名（无→黄色提示 + 「补训索引」按钮：弹输入实验名对应关系——P2 简化：仅当能从模型名推出实验名时可用，调 `/api/train/index` 并用 useTask 显示进度）、删除（确认后调用，成功刷新列表）
- 「去推理」按钮切 Tab 并选中该模型

**Commit** `feat(models): 模型删除与补训索引 + 管理页`

---

### Task 10: 端到端验收（真实训练，CPU 少 epoch）

**Step 0（验收前置检查）:**
- `tests/test_tasks.py::test_training_subprocess_imports_repo_packages`（PYTHONPATH + PYTHONSAFEPATH 注入的集成测试）通过——脚本直跑形态下训练子进程的顶层导入依赖这两层注入
- 真实起一个 preprocess 子进程冒烟：任选一个小数据集目录跑 `python train/preprocess.py <ds> 40000 1 <exp_dir> False 3.7`（tasks.py 会注入 PYTHONPATH/PYTHONSAFEPATH），确认日志出现「数据切分」开始/完成，且无 `ModuleNotFoundError: No module named 'infer'`、无 `cannot import name 'utils' from 'train'`（后者是 train/train.py 遮蔽 train 命名空间包的症状）

**Step 1:** 造小样本数据集：用 ffmpeg 生成或从用户素材中取 2-3 分钟人声（优先问用户要一段真实人声；没有则合成 sine+噪声近似，标注"合成样本仅供流程验证"）
**Step 2:** 浏览器走查：向导完成 4 步（总轮次 3、保存间隔 3）→ 训练完成 → 模型管理页看到新模型 → 推理页选中新模型试音
**Step 3:** 自动化兜底：curl 按步骤调 API 走全流程（同 P1 Task 6 手法），验证产物 `assets/weights/{exp}.pth`（savee 格式）与 `assets/indices/{exp}_added_*.index` 存在且 `/api/models` 配对成功
**Step 4:** 全量 pytest + pnpm build；报告（不合并，分支保持）

**Commit**（如有修复）+ 汇报

---

## 审查要求（沿用 P1 流程）

每任务：实现者 → 规格审查 → 质量审查。特别关注：命令拼装与 webui.py 模板的逐字一致性（Task 1）、任务系统并发正确性（Task 3）、SSE 资源清理（Task 6）。
