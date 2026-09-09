/**
 * loss 曲线（自 Training.tsx 迁出，队列项详情复用）：内联 SVG 双折线 + 悬浮十字线
 * 提示（第几个点 / 所属轮次 / 两条 loss 值）。高亮点与提示用 HTML 绝对定位而不是
 * SVG 图元——svg 的 preserveAspectRatio=none 会在横向拉伸时把圆点变成椭圆；位置按
 * viewBox 百分比换算，与鼠标横坐标一致。
 */
import { useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'

import type { LossPoint } from '@/lib/loss'

export function LossChart({ points }: { points: LossPoint[] }) {
  const [hover, setHover] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  // loss 每 200 步才记录一个点（train/train.py 的 log_interval；小数据集整场训练
  // 可能只有一两个点）——单点构不成线，如实展示数值并说明节奏
  if (points.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        暂无 loss 数据：训练开始后每 200 步记录一个点，产生日志后自动绘制
      </p>
    )
  }
  if (points.length === 1) {
    const p = points[0]
    return (
      <div className="flex flex-col gap-1.5">
        <div className="flex h-24 flex-wrap items-center justify-center gap-4 rounded-lg bg-muted text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1 text-foreground">
            <span className="inline-block size-2 rounded-full bg-chart-1" />
            loss_disc {p.disc.toFixed(3)}
          </span>
          <span className="inline-flex items-center gap-1 text-foreground">
            <span className="inline-block size-2 rounded-full bg-chart-2" />
            loss_gen {p.gen.toFixed(3)}
          </span>
          {p.epoch !== null && <span>轮次 {p.epoch}</span>}
        </div>
        <p className="text-xs text-muted-foreground">
          已记录 1 个点（每 200 步记录一次）；当前数据量小、步数少，构不成曲线属正常，不影响训练
        </p>
      </div>
    )
  }
  const w = 320
  const h = 96
  const pad = 4
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const p of points) {
    min = Math.min(min, p.disc, p.gen)
    max = Math.max(max, p.disc, p.gen)
  }
  const span = max > min ? max - min : 1
  const x = (i: number) => pad + (i / (points.length - 1)) * (w - pad * 2)
  const y = (v: number) => h - pad - ((v - min) / span) * (h - pad * 2)
  const leftPct = (i: number) => (x(i) / w) * 100
  const topPct = (v: number) => (y(v) / h) * 100
  const polyline = (pick: (p: LossPoint) => number) =>
    points.map((p, i) => `${x(i).toFixed(2)},${y(pick(p)).toFixed(2)}`).join(' ')

  function onMove(e: ReactMouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect()
    if (rect === undefined || rect.width === 0) return
    const frac = (e.clientX - rect.left) / rect.width
    setHover(Math.max(0, Math.min(points.length - 1, Math.round(frac * (points.length - 1)))))
  }

  const hp = hover !== null ? points[hover] : null
  // 提示框横向钳位，避免贴边溢出
  const tipLeft = hover !== null ? Math.min(82, Math.max(18, leftPct(hover))) : 0
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${w} ${h}`}
          preserveAspectRatio="none"
          className="h-24 w-full cursor-crosshair rounded-lg bg-muted"
          role="img"
          aria-label="loss 曲线"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          <polyline
            points={polyline((p) => p.disc)}
            fill="none"
            className="stroke-chart-1"
            strokeWidth={1.5}
          />
          <polyline
            points={polyline((p) => p.gen)}
            fill="none"
            className="stroke-chart-2"
            strokeWidth={1.5}
          />
          {hp !== null && hover !== null && (
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={pad}
              y2={h - pad}
              className="stroke-border"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
        {hp !== null && hover !== null && (
          <>
            <span
              className="pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-background bg-chart-1"
              style={{ left: `${leftPct(hover)}%`, top: `${topPct(hp.disc)}%` }}
            />
            <span
              className="pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-background bg-chart-2"
              style={{ left: `${leftPct(hover)}%`, top: `${topPct(hp.gen)}%` }}
            />
            <div
              className="pointer-events-none absolute top-1 flex -translate-x-1/2 items-center gap-2 rounded-md border bg-background/95 px-2 py-1 text-[11px] shadow-sm"
              style={{ left: `${tipLeft}%` }}
            >
              <span className="font-medium">
                点 {hover + 1}/{points.length}
              </span>
              {hp.epoch !== null && (
                <span>
                  轮次 {hp.epoch}
                  {hp.pct !== null ? ` (${hp.pct}%)` : ''}
                </span>
              )}
              {hp.step !== null && <span>步 {hp.step}</span>}
              <span className="inline-flex items-center gap-1">
                <span className="inline-block size-1.5 rounded-full bg-chart-1" />
                {hp.disc.toFixed(3)}
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="inline-block size-1.5 rounded-full bg-chart-2" />
                {hp.gen.toFixed(3)}
              </span>
            </div>
          </>
        )}
      </div>
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2 rounded-full bg-chart-1" />
          loss_disc
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2 rounded-full bg-chart-2" />
          loss_gen
        </span>
        <span>{points.length} 个点</span>
      </div>
    </div>
  )
}
