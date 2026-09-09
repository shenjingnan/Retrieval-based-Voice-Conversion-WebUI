/**
 * 任务日志滚动区（自 Training.tsx 迁出，队列项详情复用）：等宽 + 自动滚底，
 * 用户上滚时暂停（回到底部恢复）；右上角悬浮复制按钮（复制全部可见日志，
 * 成功后短暂变 ✓）。
 */
import { useEffect, useRef, useState } from 'react'
import { CheckIcon, CopyIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'

export function LogPanel({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement | null>(null)
  const stick = useRef(true)
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (el === null || !stick.current) return
    el.scrollTop = el.scrollHeight
  }, [lines])

  async function copyAll() {
    const text = lines.join('\n')
    try {
      if (navigator.clipboard !== undefined) {
        await navigator.clipboard.writeText(text)
      } else {
        // 非 secure context（局域网 http 访问）没有 async Clipboard API：
        // 退回隐藏 textarea + execCommand（已废弃但在所有浏览器仍可用）
        const ta = document.createElement('textarea')
        ta.value = text
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        ta.remove()
      }
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // 复制失败（权限拒绝等）：不弹错误打断看日志的心流，按钮原样保留可重试
    }
  }

  return (
    <div className="relative">
      <pre
        ref={ref}
        onScroll={() => {
          const el = ref.current
          if (el === null) return
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        className="max-h-72 overflow-y-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5 break-all whitespace-pre-wrap"
      >
        {lines.length > 0 ? lines.join('\n') : '（暂无日志输出）'}
      </pre>
      <Button
        variant="outline"
        size="icon-sm"
        className="absolute right-2 top-2 bg-background/80 backdrop-blur"
        disabled={lines.length === 0}
        aria-label="复制日志"
        title="复制日志"
        onClick={() => void copyAll()}
      >
        {copied ? <CheckIcon className="text-emerald-600" /> : <CopyIcon />}
      </Button>
    </div>
  )
}
