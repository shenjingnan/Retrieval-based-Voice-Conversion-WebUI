# P1 新版 WebUI 骨架跑通 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 跑通新版 WebUI 的最小闭环：FastAPI 编排层 + React 前端，用户可在浏览器里 选模型 → 上传音频 → A/B 试听。

**Architecture:** `server/`（FastAPI，只编排不算法：推理 import `infer.vc.modules.VC`）+ `frontend/`（Vite+React+TS+Tailwind+shadcn，构建产物由 FastAPI 静态托管）。核心 `train/ infer/` 零改动。

**Tech Stack:** Python 3.13 / FastAPI / uvicorn / pytest；Node 22 / pnpm / Vite 8 / React 19 / TypeScript 6 / Tailwind v4 / shadcn 4.x（base-nova，Base UI 原语，亮色主题）。

> **执行后版本回写（Task 4 实际落地的代差）：** shadcn 4.x 底层是 Base UI 而非 Radix，Task 5 写推理页时注意两处 API 差异——① `Slider` 的 `value`/`onValueChange` 是**数组**形态（如 `value={[num]}`、`(v: number[]) => ...`），不是单值；② `Select` 系列组件的 props 与 Radix 版不同（自定义触发器用 `render` prop，不支持 `asChild`），以 `frontend/src/components/ui/select.tsx` 实际源码为准。

**工作区:** 全部在 worktree `/Users/nemo/github/Retrieval-based-Voice-Conversion-WebUI/.worktrees/new-webui`（分支 `feature/new-webui`）内进行。下文所有相对路径基于 worktree 根。

**关键前置事实（已核实）:**
- `VC(config)` 在 `infer/vc/modules.py:87`；`vc_single(sid, input_audio_path, f0_up_key, f0_method, file_index, index_rate, resample_sr, rms_mix_rate, protect)` 返回 `(状态字符串, (tgt_sr, int16音频numpy) | (None, None))`
- `VC.get_vc(sid)` 内部用 `os.getenv("weight_root")/{sid}` 拼路径（modules.py:168）→ **server 启动时必须设置 `os.environ["weight_root"] = assets/weights 绝对路径`，并把模型文件名（basename）作为 sid 传入**
- `inference_status` 返回纯字符串
- 索引自动配对规则参考 `infer/vc/utils.py:7`：去掉 `_e<epoch>_s<step>` 后缀做实验名匹配，跳过含 "trained" 的索引
- 旧服务占 7865，新服务用 **7861**

---

### Task 1: server 骨架（健康检查 + 路径常量 + pytest 就绪）

**Files:**
- Create: `server/__init__.py`（空文件）
- Create: `server/paths.py`
- Create: `server/main.py`
- Create: `tests/__init__.py`（空）、`tests/conftest.py`
- Test: `tests/test_main.py`

**Step 1: 安装测试依赖并确认**

```bash
python3 -m pip install fastapi uvicorn httpx 2>&1 | tail -1
python3 -c "from fastapi.testclient import TestClient; print('ok')"
```
预期输出 `ok`。

**Step 2: 写失败测试** `tests/conftest.py`：

```python
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

`tests/test_main.py`：

```python
from fastapi.testclient import TestClient

from server.main import create_app


def test_health():
    client = TestClient(create_app())
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

**Step 3: 运行确认失败**

```bash
cd /Users/nemo/github/Retrieval-based-Voice-Conversion-WebUI/.worktrees/new-webui && python3 -m pytest tests/test_main.py -v
```
预期：FAIL（`No module named server`）。

**Step 4: 最小实现**

`server/paths.py`：

```python
"""运行时路径常量。server 只在此处感知目录布局。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
WEIGHTS_DIR = ASSETS / "weights"
INDICES_DIR = ASSETS / "indices"
LOGS_DIR = ROOT / "logs"
STATIC_DIR = Path(__file__).resolve().parent / "static"  # 前端构建产物，可能不存在
```

`server/main.py`：

```python
"""新版 WebUI 编排层入口。只做编排，算法一律复用 infer/ 与 train/。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server import paths
from server.api import infer, models


def create_app() -> FastAPI:
    os.environ.setdefault("weight_root", str(paths.WEIGHTS_DIR))
    os.environ.setdefault("index_root", str(paths.INDICES_DIR))

    app = FastAPI(title="RVC WebUI", version="2.0.0-p1")
    app.include_router(models.router)
    app.include_router(infer.router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    if paths.STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=paths.STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    import socket
    import threading
    import webbrowser

    import uvicorn

    port = 7861
    while port < 7871:  # 端口被占用时向后找，最多试 5 个
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        port += 1

    threading.Timer(1.0, webbrowser.open, args=(f"http://127.0.0.1:{port}",)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
```

注意：`import os` 需补在 main.py 顶部；`from server.api import infer, models` 要求 Task 2/3 的空路由先存在——本任务先创建最小占位 `server/api/__init__.py`（空）、`server/api/models.py` 与 `server/api/infer.py`（各含 `router = APIRouter()`，无路由），后续任务再填充。

**Step 5: 运行测试通过**

```bash
python3 -m pytest tests/test_main.py -v
```
预期：1 passed。

**Step 6: Commit**

```bash
git add server/ tests/
git commit -m "feat(server): FastAPI 编排层骨架与健康检查"
```

---

### Task 2: GET /api/models 模型与索引扫描

**Files:**
- Modify: `server/api/models.py`
- Test: `tests/test_models_api.py`

**Step 1: 写失败测试** `tests/test_models_api.py`：

```python
from fastapi.testclient import TestClient

from server.main import create_app


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_models_scan_and_pair(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    indices = tmp_path / "indices"
    _touch(weights / "alice_v2.pth")
    _touch(weights / "alice_v2_e20_s100.pth")  # 带 epoch/step 后缀的变体
    _touch(indices / "added_IVFxxx_Flat_nprobe_1_alice_v2.index")
    _touch(indices / "trained_IVFxxx_alice_v2.index")  # trained 必须被忽略
    _touch(weights / "bob.pth")  # 无索引
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", weights)
    monkeypatch.setattr("server.paths.INDICES_DIR", indices)

    client = TestClient(create_app())
    resp = client.get("/api/models")

    assert resp.status_code == 200
    items = resp.json()
    by_name = {m["name"]: m for m in items}
    assert by_name["alice_v2.pth"]["index"] is not None
    assert "trained" not in by_name["alice_v2.pth"]["index"]
    assert by_name["alice_v2_e20_s100.pth"]["index"] is not None
    assert by_name["bob.pth"]["index"] is None


def test_models_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("server.paths.WEIGHTS_DIR", tmp_path / "none")
    monkeypatch.setattr("server.paths.INDICES_DIR", tmp_path / "none")
    client = TestClient(create_app())
    assert client.get("/api/models").json() == []
```

**Step 2: 运行确认失败**（404，路由不存在）

**Step 3: 实现** `server/api/models.py`：

```python
"""扫描 assets/weights 与 assets/indices，返回模型及其索引配对。"""
import re

from fastapi import APIRouter

from server import paths

router = APIRouter(prefix="/api")

_EPOCH_SUFFIX = re.compile(r"_e\d+_s\d+$", re.IGNORECASE)


def experiment_name(model_stem: str) -> str:
    """alice_v2_e20_s100 -> alice_v2，与 infer/vc/utils.py 规则一致。"""
    return _EPOCH_SUFFIX.sub("", model_stem)


def _scan():
    models = []
    indices = [
        p
        for p in paths.INDICES_DIR.glob("*.index")
        if "trained" not in p.name.lower()
    ] if paths.INDICES_DIR.is_dir() else []
    if paths.WEIGHTS_DIR.is_dir():
        for p in sorted(paths.WEIGHTS_DIR.glob("*.pth")):
            exp = experiment_name(p.stem).lower()
            index = next(
                (i for i in indices if experiment_name(i.stem).lower().startswith(exp)),
                None,
            ) if exp else None
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
```

**Step 4: 运行测试通过**：`python3 -m pytest tests/test_models_api.py -v`，预期 2 passed。

**Step 5: Commit**

```bash
git add server/api/models.py tests/test_models_api.py
git commit -m "feat(server): 模型列表与索引自动配对接口"
```

---

### Task 3: POST /api/infer 推理接口

**Files:**
- Modify: `server/api/infer.py`
- Test: `tests/test_infer_api.py`

**Step 1: 写失败测试** `tests/test_infer_api.py`：

```python
import numpy as np
from fastapi.testclient import TestClient

from server.main import create_app


class FakeVC:
    def __init__(self):
        self.loaded = None

    def get_vc(self, sid):
        self.loaded = sid

    def vc_single(self, sid, path, f0_up_key, f0_method, file_index,
                  index_rate, resample_sr, rms_mix_rate, protect):
        return "状态：成功", (32000, np.zeros(3200, dtype=np.int16))


def test_infer_ok(monkeypatch, tmp_path):
    fake = FakeVC()
    monkeypatch.setattr("server.api.infer.get_vc_cached", lambda name: fake)

    client = TestClient(create_app())
    resp = client.post(
        "/api/infer",
        files={"audio": ("in.wav", b"RIFFfakewav", "audio/wav")},
        data={"model": "m.pth", "transpose": "0", "f0_method": "rmvpe",
              "index_rate": "0.75", "resample_sr": "0",
              "rms_mix_rate": "0.25", "protect": "0.33", "index_path": ""},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert fake.loaded == "m.pth"


def test_infer_model_not_found(monkeypatch, tmp_path):
    from server.api import infer as infer_api

    monkeypatch.setattr(
        "server.api.infer.get_vc_cached",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )
    client = TestClient(create_app())
    resp = client.post(
        "/api/infer",
        files={"audio": ("in.wav", b"x", "audio/wav")},
        data={"model": "ghost.pth", "transpose": "0", "f0_method": "rmvpe",
              "index_rate": "0.75", "resample_sr": "0",
              "rms_mix_rate": "0.25", "protect": "0.33", "index_path": ""},
    )
    assert resp.status_code == 404


def test_infer_pipeline_failure(monkeypatch):
    class FailVC:
        def get_vc(self, sid):
            pass

        def vc_single(self, *a, **k):
            return "【单次推理】\n状态：失败\nboom", (None, None)

    monkeypatch.setattr("server.api.infer.get_vc_cached", lambda name: FailVC())
    client = TestClient(create_app())
    resp = client.post(
        "/api/infer",
        files={"audio": ("in.wav", b"x", "audio/wav")},
        data={"model": "m.pth", "transpose": "0", "f0_method": "rmvpe",
              "index_rate": "0.75", "resample_sr": "0",
              "rms_mix_rate": "0.25", "protect": "0.33", "index_path": ""},
    )
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]
```

**Step 2: 运行确认失败**

**Step 3: 实现** `server/api/infer.py`：

```python
"""推理编排：缓存 VC 实例，透传参数给 infer.vc.modules.VC.vc_single。"""
import os
import tempfile
import threading
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException
from fastapi.responses import FileResponse

from server import paths

router = APIRouter(prefix="/api")

_vc_lock = threading.Lock()
_vc_cache: dict = {}


def get_vc(name: str):
    """按模型文件名缓存 VC 实例；VC.get_vc 内部用 weight_root 拼路径。"""
    with _vc_lock:
        vc = _vc_cache.get(name)
        if vc is None:
            from infer.vc.modules import VC  # 延迟导入：避免 pytest 收集期加载 torch
            from configs.config import Config

            vc = VC(Config())
            _vc_cache[name] = vc
        vc.get_vc(name)
        return vc


get_vc_cached = get_vc  # 测试 monkeypatch 点


@router.post("/infer")
def infer(
    audio: bytes = File(...),
    model: str = Form(...),
    transpose: int = Form(0),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    resample_sr: int = Form(0),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33),
    index_path: str = Form(""),
):
    model_path = paths.WEIGHTS_DIR / model
    if not model_path.is_file():
        raise HTTPException(404, f"模型不存在: {model}")

    src = Path(tempfile.gettempdir()) / f"rvc_in_{uuid.uuid4().hex}{Path(model).suffix or '.wav'}"
    src.write_bytes(audio)
    dst = Path(tempfile.gettempdir()) / f"rvc_out_{uuid.uuid4().hex}.wav"
    try:
        vc = get_vc_cached(model)
        index = index_path if index_path else None
        info, result = vc.vc_single(
            0, str(src), transpose, f0_method, index,
            index_rate, resample_sr, rms_mix_rate, protect,
        )
        if result is None or result[1] is None:
            raise HTTPException(500, f"推理失败: {info}")
        sr, audio_opt = result
        sf.write(dst, audio_opt, sr)
        return FileResponse(dst, media_type="audio/wav", filename="converted.wav")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
```

实现时注意两处需现场核实（计划编写时未读到该代码）：
1. `Config()` 构造是否需要参数：看 `configs/config.py` 的 `__init__`（webui.py 顶部有现成用法）
2. `get_vc` 传文件名即可（`weight_root` 已在 create_app 里设好）

**Step 4: 运行测试通过**：`python3 -m pytest tests/ -v`，预期全部 passed。

**Step 5: Commit**

```bash
git add server/api/infer.py tests/test_infer_api.py
git commit -m "feat(server): 推理接口（VC 实例缓存 + 临时文件管理）"
```

---

### Task 4: 前端脚手架（Vite + React + TS + Tailwind + shadcn，亮色）

**Files:** `frontend/` 整个目录（脚手架生成 + 少量修改）

**Step 1: 生成脚手架**

```bash
cd /Users/nemo/github/Retrieval-based-Voice-Conversion-WebUI/.worktrees/new-webui
pnpm create vite frontend --template react-ts
cd frontend && pnpm install
pnpm add tailwindcss @tailwindcss/vite
pnpm dlx shadcn@latest init -y   # 组件基色选 neutral，CSS variables 默认（亮色）
pnpm dlx shadcn@latest add button select slider collapsible card input
```

**Step 2: vite.config.ts 配置代理与构建输出**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: { proxy: { '/api': 'http://127.0.0.1:7861' } },
  build: { outDir: '../server/static', emptyOutDir: true },
})
```

`tsconfig` 按 shadcn init 提示补 `baseUrl/paths`。

**Step 3: 页面骨架与 API client**

`frontend/src/api/client.ts`：

```typescript
export interface RvcModel { name: string; path: string; index: string | null }

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail ?? `HTTP ${resp.status}`)
  return resp.json()
}

export const api = {
  models: () => fetch('/api/models').then(r => handle<RvcModel[]>(r)),
  infer: (form: FormData): Promise<Blob> =>
    fetch('/api/infer', { method: 'POST', body: form }).then(async r => {
      if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `HTTP ${r.status}`)
      return r.blob()
    }),
}
```

`frontend/src/App.tsx`：顶部三个 Tab（推理 / 训练 / 模型管理）。训练、模型管理渲染占位卡片「P2 敬请期待」。推理 Tab 引 `<InferencePage />`（Task 5 实现，本任务先占位空组件）。

**Step 4: 验证 dev 起得来**

```bash
pnpm dev   # 5173 端口，亮色界面 + 三个 Tab，手动确认
```

**Step 5: 首次构建验证**

```bash
pnpm build   # 产物出现在 ../server/static/
```

**Step 6: Commit**

```bash
cd .. && git add frontend server/static/.gitkeep 2>/dev/null; git add -A frontend
printf 'server/static/\nnode_modules/\n' >> .git/info/exclude
git commit -m "feat(frontend): Vite+React+shadcn 脚手架与 Tab 壳"
```

注意：`frontend/` 的源码要提交；`server/static/`（构建产物）与 `node_modules` 只进 worktree 本地 exclude，不进 .gitignore（是否全局忽略留给用户决定）。

---

### Task 5: 推理页 UI

**Files:**
- Create: `frontend/src/pages/Inference.tsx`
- Modify: `frontend/src/App.tsx`（挂载该页）

**Step 1: 实现页面**（设计见 `docs/plans/2026-09-06-new-webui-design.md` §5.1）

组件结构（单文件即可，不过度拆分）：

```typescript
// 状态: models(列表) / selected / transpose(-12..12) / f0Method('rmvpe') / indexRate(0.75)
//       expert{resampleSr, rmsMixRate, protect} 折叠 / file(File|null) / srcUrl / outUrl / busy / error
```

交互要点：
- 挂载时 `api.models()` 填充模型下拉；选中模型即确定其配对索引（列表已含 `index` 字段，提交时作为 `index_path` 传给后端）
- 基础参数仅 4 个：变调 Slider(-12~12, 步长1, 显示当前值)、音高算法 Select(rmvpe/pm/fcpe)、音色相似度 Slider(0~1, 步长0.05)、其余进 `Collapsible` 专家区（resample_sr=0、rms_mix_rate=0.25、protect=0.33）
- `<input type="file" accept="audio/*">` 选完立即 `URL.createObjectURL` 生成原声试听
- 点击「开始转换」：FormData 组装 → `api.infer` → blob → objectURL 到「转换后」播放器 + 下载链接；busy 时按钮转圈禁用；失败展示后端 `detail`（含日志尾部）
- A/B 两个 `<audio controls>` 并排，同一音源便于对比

**Step 2: 手动走查（dev 模式）**

```bash
# 终端A
python3 server/main.py
# 终端B
cd frontend && pnpm dev
```
浏览器 http://localhost:5173 完整走一遍：选模型 → 拖音频 → 默认参数转换 → A/B 试听 → 下载。assets/weights 里若没有模型，先从主工作区找任意 `.pth` 复制到 `assets/weights/`（两个工作区共享真实模型目录，worktree 内 assets 下各子目录是软链）。

**Step 3: Commit**

```bash
git add frontend/src
git commit -m "feat(frontend): 推理页（基础/专家参数分层 + A/B 试听）"
```

---

### Task 6: 生产形态集成验收

**Step 1: 构建 + 单命令启动**

```bash
cd frontend && pnpm build && cd ..
python3 server/main.py   # 自动开 http://127.0.0.1:7861，静态托管前端
```

**Step 2: 走查验收清单**

- [ ] 7861 打开即是新 UI（非 dev 端口）
- [ ] 选模型 → 上传音频 → 转换 → A/B 试听 → 下载，全流程无报错
- [ ] 缺索引模型可推理（index 置空），有索引模型音色明显更贴
- [ ] 旧 webui.py 不受影响（`python3 webui.py --noautoopen` 仍能起在 7865）
- [ ] `python3 -m pytest tests/ -v` 全绿

**Step 3: 收尾 Commit（如有微调）+ 汇报**

向用户演示，确认 P1 验收，再决定是否进入 P2（届时另写 P2 计划文档）。
