# 外部音色模型上传（模型导入）设计

日期：2026-09-12
状态：已确认（入库永久可用 / 仅 pth + index 双文件槽）

## 1. 背景与现状

用户希望在推理界面使用自己拿到手的第三方 RVC 音色模型（`.pth` 权重 + 可选
`.index` 特征索引），而不必先经过本机训练流程。

现状（新版 WebUI，`server/main.py` 起 React + FastAPI）：

- `GET /api/models` 扫描 `assets/weights/*.pth` + `assets/indices/*.index`，
  按实验名自动配对（`server/api/models.py` `_scan` / `_pick_index`）。
- `POST /api/infer` 只接受 `WEIGHTS_DIR` 下的 basename 模型名与 `INDICES_DIR`
  下的索引路径（`server/api/infer.py`），防路径穿越。
- 模型管理页已有列表 / 整组删除 / zip 打包下载 / 补训索引；推理页有模型下拉框。

**关键洞察**：推理链路只认这两个目录里的文件。因此「上传」的最小实现是把文件
校验后写入 `assets/weights` 与 `assets/indices`——之后列表展示、索引配对、推理、
zip 下载、删除全部现有逻辑零改动自动生效。

## 2. 方案要点（已确认的决策）

| 决策点 | 结论 | 理由 |
| --- | --- | --- |
| 存储语义 | **入库，永久可用** | 复用全部现有逻辑；临时会话模型需给 /api/infer 开目录外路径口子，改动更大且更不安全 |
| 上传格式 | **仅 pth（必选）+ index（可选）双文件** | 不做 zip 解析（zip-slip 等额外攻击面）；zip 可后续再加 |
| 重名策略 | **409 拒绝**，两目录各自独立查重 | 覆盖藏在「上传」里太危险（删除是显式彻底删除语义）；自动改名会扰乱配对 |
| 上传入口 | 推理页（主）+ 模型管理页，共用同一表单组件与 API | 推理页覆盖「拿到模型直接试」的主场景；模型页是管理语义的自然归属 |
| pth 有效性校验 | 轻校验（zip 魔数），**不做 torch.load 试载** | 完整试载耗时数秒、占 GB 级内存，且 weights_only 对社区模型兼容性不可控；坏模型在首次推理时以显式报错暴露 |

## 3. 后端设计

新增端点：`POST /api/models/upload`（`server/api/models.py`，与列表/删除/下载同文件）。

```
multipart/form-data:
  model: UploadFile        # .pth，必选
  index: UploadFile | None # .index，可选
```

处理流程：

1. **校验链**（任一步失败即拒绝，不落盘）：
   - 文件名清洗：仅接受 basename（`Path(name).name != name` 或 `.` / `..` → 400，
     与 delete/download 同手法）；后缀白名单且**必须小写** `.pth` / `.index`
     （`_scan` 的 glob 与 delete 的 suffix 检查均为大小写敏感，静默改写后缀会
     造出列表里看不见的文件）。
   - 大小上限：`MAX_UPLOAD_BYTES = 500 MB`（与 datasets.MAX_FILE_BYTES 同口径；
     最终产物 pth 通常 55~170 MB，大实验索引可到百 MB 级）。在拷贝循环里按实际
     字节数强制（Content-Length 可谎报；请求体在 handler 前已被 Starlette 整体
     spool，这里拦的是目标目录被污染），超限中止并清理半成品，413。
   - pth 魔数：前 4 字节必须为 `PK\x03\x04`（torch>=1.6 的 zip 容器格式），拦住
     「随便改后缀的文件」；index 文件不做魔数校验（faiss 头部 fourcc 变体不值得
     硬编码，错误在推理期由 faiss 显式抛出）。
   - 重名：`weights/{name}` 或 `indices/{index_name}` 已存在 → 409，两目录独立报错。
2. **原子落盘**：写入同目录下的 `.part` 临时文件（不会被 `*.pth` glob 列出），
   拷贝完成后复核重名（封住校验到落盘的竞态窗口），`os.replace` 原子改名。
   任一文件写失败 → 清理半成品并 500；index 失败而 pth 已落盘时容忍
   「有模型无索引」——这本来就是合法状态。
3. **响应**：复用 `_scan()` 返回与 `GET /api/models` 条目同构的对象
   （`name/path/index/mtime`），前端拿到即知配对结果。

错误码约定：400 非法文件名/后缀/魔数，409 重名，
413 超大小上限，422 缺必选 multipart 字段（FastAPI 校验层默认行为），500 落盘失败。

安全说明：`.pth` 是 torch.save 的序列化容器，`torch.load` 反序列化期间可执行
其中嵌入的代码，因此导入来路不明的模型等同于运行对方提供的程序。本机单用户
场景下风险面与现状相同（用户本可手动放文件进 weights/）；服务无鉴权绑 0.0.0.0
时「局域网他人可注入模型」是新增暴露面，与现有删除/训练接口同级，不在此处单独设防。

## 4. 前端设计

- `client.ts` 新增 `uploadModel(model, index, onProgress)`：XMLHttpRequest 上传
  （仿 `uploadDataset`——fetch 无上传进度，大文件必须给进度条），返回 `RvcModel`。
- 新增共享组件 `components/ModelUploadForm.tsx`：pth 文件槽（必选）+ index 文件槽
  （可选）+ 进度条 + 就地错误展示 + 上传成功回调（携带新模型名）。
  仓库无 Dialog 组件，遵循现有内联表单惯例（同 Models 页补训索引表单）。
- 推理页：模型选择区内以 Collapsible 提供「上传音色模型」入口；上传成功后刷新
  列表并自动选中新模型——上传完即可推理，零跳转。
- 模型管理页：头部「刷新」旁加「导入模型」按钮，展开同一表单组件；成功后刷新列表。

## 5. 实施与验收

1. 后端测试先行（TDD）：`tests/test_models_api.py` 增加用例——成功（pth+index /
   仅 pth）、重名 409（两目录独立）、非 basename 400、大写后缀 400、超限 413、
   魔数 400、上传后 `GET /api/models` 可见且配对正确。
2. 后端实现 `POST /api/models/upload`。
3. 前端实现（client.ts → ModelUploadForm → 两页接入）。
4. 验收：`pytest` 全绿；前端 `vitest` + `tsc` + `vite build` 通过；手工验收为
   上传一个真实 pth+index 后在推理页直接出声。
