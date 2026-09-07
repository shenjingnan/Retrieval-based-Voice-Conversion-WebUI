# RVC WebUI 前端

Vite + React + TypeScript + Tailwind CSS v4 + shadcn/ui（亮色主题）。

```bash
pnpm install
pnpm dev     # http://localhost:5173，/api 代理到 http://127.0.0.1:7861
pnpm build   # 产物输出到 ../server/static/（emptyOutDir）
pnpm lint    # oxlint
```

- `src/App.tsx`：Tab 壳（推理 / 训练 / 模型管理），训练与模型管理为 P2 占位。
- `src/pages/Inference.tsx`：推理页（Task 5 实现）。
- `src/api/client.ts`：后端 API client（`GET /api/models`、`POST /api/infer`）。
- UI 组件在 `src/components/ui/`，由 `pnpm dlx shadcn@latest add <name>` 添加。
