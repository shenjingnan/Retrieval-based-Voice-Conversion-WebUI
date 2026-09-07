/**
 * 错误展示块：标题 + 等宽 <pre>（超长信息内部滚动，保留换行）。
 * 推理失败 / 任务创建失败 / 任务失败等场景共用，避免 <pre> 样式多处漂移。
 */
import type { ReactNode } from 'react'

export function ErrorDetail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div role="alert" className="flex flex-col gap-1">
      <span className="text-sm font-medium text-destructive">{title}</span>
      <pre className="max-h-48 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs break-all whitespace-pre-wrap">
        {children}
      </pre>
    </div>
  )
}
