# RVC 实时语音合成在 macOS 上的可行性分析

## 1. 环境概况

| 项目 | 详情 |
|------|------|
| 设备 | Apple M1 Max (arm64) |
| 操作系统 | macOS (Darwin 25.6.0) |
| Python 版本 | 3.13.2 (pyenv) |
| 项目 | Retrieval-based-Voice-Conversion-WebUI |
| 目标文件 | `realtime_gui.py` |

## 2. 当前架构分析

### 2.1 Windows 启动流程

```
go-realtime_gui.bat
  → 设置 PATH 指向 runtime/（嵌入式 Python 运行时）
  → runtime\python.exe -I realtime_gui.py
```

`runtime/` 目录是 Windows 专用的嵌入式 Python 发行版，**在 macOS 上不存在且不可用**。

### 2.2 核心调用链

```
realtime_gui.py (GUI 层)
  ├── FreeSimpleGUI        → 跨平台 GUI 框架
  ├── sounddevice          → 跨平台音频 I/O（基于 PortAudio）
  ├── librosa              → 音频处理（RMS、重采样等）
  ├── torch/torchaudio     → 深度学习推理 + 重采样
  ├── tools/torchgate      → 噪声抑制（频谱门控）
  ├── tools/cuda_graph     → CUDA Graph 加速（可选）
  │
  └── infer/rtrvc.py (推理核心)
        ├── infer/hubert.py       → Hubert 特征提取
        ├── infer/rmvpe.py        → RMVPE 音高提取
        ├── infer/fcpe.py         → FCPE 音高提取
        ├── infer/module/models.py → 合成器模型（HiFi-GAN）
        ├── faiss                 → 向量索引检索
        └── parselmouth           → PM 音高提取（Praat）
```

### 2.3 设备选择逻辑（configs/config.py）

```text
1. 检测 CUDA GPU → 满足条件(≥4GB 显存, SM ≥ 5.3) → 使用 CUDA
2. 无 CUDA → 检测 DirectML (Windows) → 使用 DML
3. 无 DML → 回退到 CPU (torch.float32)
```

**关键结论：当没有 CUDA 和 DirectML 时，代码会自动回退到 CPU 推理。**

### 2.4 已有的 macOS 适配

`realtime_gui.py` 第 913-915 行已经包含 macOS 特定代码：

```python
if sys.platform == "darwin":
    _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
    sola_offset = sola_offset.item()
else:
    sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
```

这说明项目开发者已经考虑了 macOS 兼容性，SOLA 波形拼接算法在 macOS 上使用了不同的实现路径。

## 3. 逐项可行性分析

### 3.1 GUI 框架 — ✅ 可行

`FreeSimpleGUI` 是 PySimpleGUI 的 fork，基于 tkinter，在 macOS 上完全可用。

### 3.2 音频 I/O — ✅ 可行

`sounddevice` 基于 PortAudio 库，原生支持 macOS Core Audio。在 M1 Mac 上可以正常枚举输入/输出设备（麦克风、扬声器、虚拟音频设备等）。

### 3.3 CUDA Graph — ✅ 自动降级

`tools/cuda_graph.py` 中的 `cuda_graph_enabled()` 在检测不到 CUDA 设备时返回 `False`，`run_cuda_graph()` 会自动回退到直接函数调用（eager mode），不会报错。

### 3.4 推理引擎 — ⚠️ 可用但性能受限

| 组件 | macOS 兼容性 | 说明 |
|------|-------------|------|
| PyTorch | ✅ 原生支持 | MPS (Metal Performance Shaders) 后端可用于 Apple Silicon |
| Hubert 特征提取 | ✅ | 纯 PyTorch 运算 |
| RMVPE 音高提取 | ✅ | 纯 PyTorch 模型推理 |
| FCPE 音高提取 | ✅ | 纯 PyTorch 模型推理 |
| PM 音高提取 | ✅ | parselmouth 库支持 macOS |
| HiFi-GAN 合成器 | ✅ | 纯 PyTorch 运算 |
| FAISS 索引检索 | ✅ | faiss-cpu 可通过 pip 安装 |
| TorchGate 噪声抑制 | ✅ | 纯 PyTorch 运算 |

### 3.5 Python 依赖安装 — ⚠️ 需要注意

以下依赖在 macOS arm64 上的安装情况：

| 依赖 | 安装方式 | 风险 |
|------|---------|------|
| `torch` + `torchaudio` | `pip install torch torchaudio` | ✅ 官方支持 MPS |
| `sounddevice` | `pip install sounddevice` | ✅ 通过 PortAudio |
| `librosa` | `pip install librosa` | ✅ |
| `FreeSimpleGUI` | `pip install FreeSimpleGUI` | ✅ |
| `faiss-cpu` | `pip install faiss-cpu` | ✅ Apple Silicon 支持 |
| `parselmouth` | `pip install praat-parselmouth` | ⚠️ 可能需要编译 |
| `numpy<2.0` | `pip install numpy` | ⚠️ 项目可能需要 numpy<2.0 |
| `scipy` | `pip install scipy` | ✅ |

### 3.6 性能评估 — 🔴 核心瓶颈

这是最关键的问题。在 **Apple M1 Max** 上运行实时语音合成：

#### CPU 模式（当前默认回退路径）

| 处理阶段 | 预估耗时 (CPU) |
|----------|---------------|
| Hubert 特征提取 | 50-150ms |
| F0 提取 (RMVPE) | 30-80ms |
| 索引检索 (FAISS) | 5-20ms |
| HiFi-GAN 推理 | 80-200ms |
| 重采样 + 噪声抑制 | 10-30ms |
| SOLA 拼接 | 5-10ms |
| **总计** | **180-490ms** |

实时音频回调的 block_time 默认是 250ms，如果推理时间超过 block_time，会导致音频卡顿或丢帧。

**CPU 模式下，默认参数（block_time=0.25s）可能勉强可用，但延迟较高（约 300-500ms 总延迟）。**

#### MPS 模式（潜在优化）

如果用 `torch.device("mps")` 替代 `torch.device("cpu")`：

| 处理阶段 | 预估耗时 (MPS) |
|----------|---------------|
| Hubert 特征提取 | 10-30ms |
| F0 提取 (RMVPE) | 8-20ms |
| 索引检索 (FAISS) | 5-20ms (仍在 CPU) |
| HiFi-GAN 推理 | 20-60ms |
| 重采样 + 噪声抑制 | 5-15ms |
| SOLA 拼接 | 5-10ms |
| **总计** | **53-155ms** |

**MPS 模式下，推理时间可能降到 50-150ms，可以满足实时性要求。**

但需要注意：
- 当前代码**不支持 MPS 后端**，需要修改 `config.py` 的设备选择逻辑
- MPS 后端对某些操作的支持不完整（如某些 `torchaudio` 操作）
- `faiss` 在 MPS 上没有加速，索引检索仍需 CPU

## 4. 综合结论

### 结论：可行，但需要解决以下问题

| 问题 | 严重程度 | 解决方案 |
|------|---------|---------|
| 无 CUDA GPU | 🔴 高 | 修改代码支持 MPS 后端，或接受 CPU 推理 |
| runtime/ 不存在 | 🟡 中 | 手动安装 Python 依赖（pip install） |
| 实时性能 | 🔴 高 | 需启用 MPS 加速，否则延迟较高 |
| 部分依赖安装 | 🟢 低 | 个别库可能需要编译工具 |

### 推荐方案（按优先级）

1. **方案一：修改代码支持 MPS 后端（推荐）**
   - 修改 `configs/config.py`，在无 CUDA/DML 时检测 MPS 可用性
   - 将 `device` 设置为 `torch.device("mps")`
   - 预期效果：推理延迟 50-150ms，可满足实时性
   - 风险：部分 MPS 操作可能不支持，需要逐一验证

2. **方案二：纯 CPU 模式（最简单）**
   - 直接安装依赖后运行，无需修改代码
   - 预期效果：延迟 300-500ms，勉强可用但体验不佳
   - 建议降低 block_time 到 0.5s 以上，调整 `extra_time` 参数

3. **方案三：使用 MPS 后端 + 优化参数（最佳体验）**
   - 在方案一基础上，调整参数：
     - 使用 `fcpe` 或 `pm` 音高算法（比 RMVPE 更快）
     - 关闭索引检索（`index_rate=0`）
     - 关闭输入/输出降噪
     - 增大 `block_time` 到 0.3-0.5s

## 5. 实施步骤建议

如果你决定尝试，建议按以下步骤操作：

1. **安装依赖**
   ```bash
   pip install torch torchaudio sounddevice librosa FreeSimpleGUI faiss-cpu praat-parselmouth numpy scipy
   ```

2. **验证 Python 导入**
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   确认 MPS 可用。

3. **修改设备选择逻辑**（支持 MPS）

4. **运行测试**
   ```bash
   python realtime_gui.py
   ```

5. **加载你的 .pth 和 .index 模型文件**，选择音频设备，开始测试。

---

*分析日期：2026-08-24*
*分析对象：Apple M1 Max (arm64) + macOS*