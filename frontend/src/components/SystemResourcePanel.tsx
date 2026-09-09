/**
 * 系统资源卡（自 Training.tsx 迁出）：GPU/内存/CPU/硬盘一行一条，常驻轮询，
 * 与任务无关。stats 为 null（从未成功拉到过）整卡降级为提示文案；各采集项的
 * null 在行内降级为「不可用」，互不影响。
 */
import { formatGib } from '@/lib/domain'
import type { GpuStats, SystemStats } from '@/api/client'

/** 手写进度条；percent 为 null 表示无数据：不渲染填充、仅脉冲占位 */
function UsageBar({ percent }: { percent: number | null }) {
  const clamped = percent === null ? 0 : Math.min(100, Math.max(0, percent))
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={`h-full bg-primary transition-[width] duration-500 ${
          percent === null ? 'animate-pulse' : ''
        }`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}

/** 一行资源：标签 + 右侧数值 + 占用条；percent 为 null（采集不可用/驱动不支持）时条脉冲 */
function ResourceRow({
  label,
  value,
  percent,
}: {
  label: string
  value: string
  percent: number | null
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{label}</span>
        <span className="tabular-nums">{value}</span>
      </div>
      <UsageBar percent={percent} />
    </div>
  )
}

/** GPU 逐卡一行（多卡时标签带序号）；util 为 null（WDDM 等不支持）只降级该字段文案，
 *  显存数据照常展示——对训练来说显存比利用率更有参考价值（batch size 自适应按显存） */
function GpuRows({ gpus }: { gpus: GpuStats[] }) {
  const multi = gpus.length > 1
  return (
    <>
      {gpus.map((g) => {
        const util = g.utilization_percent === null ? '利用率不可用' : `${g.utilization_percent}%`
        const memory =
          g.memory_used_bytes === null || g.memory_total_bytes === null
            ? null
            : `显存 ${formatGib(g.memory_used_bytes)} / ${formatGib(g.memory_total_bytes)} GiB`
        return (
          <ResourceRow
            key={g.index}
            label={multi ? `GPU ${g.index} · ${g.name}` : `GPU · ${g.name}`}
            value={memory === null ? util : `${util} · ${memory}`}
            percent={g.utilization_percent}
          />
        )
      })}
    </>
  )
}

export function SystemResourcePanel({ stats, stale }: { stats: SystemStats | null; stale: boolean }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">系统资源</span>
        <span className="text-xs text-muted-foreground">
          {stats === null ? '获取中…' : stale ? '数据已过期（服务不可达）' : '每 3 秒刷新'}
        </span>
      </div>
      {stats === null ? (
        <p className="text-xs text-muted-foreground">
          暂时无法获取系统资源，页面其他功能不受影响。
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          {stats.gpus.length === 0 ? (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>GPU</span>
                <span>不可用</span>
              </div>
              <p className="text-xs text-muted-foreground">
                未检测到 NVIDIA 显卡或 nvidia-smi 不可用；训练将以 CPU 模式进行。
              </p>
            </div>
          ) : (
            <GpuRows gpus={stats.gpus} />
          )}
          <ResourceRow
            label="内存"
            value={
              stats.memory === null
                ? '不可用'
                : `${formatGib(stats.memory.used_bytes)} / ${formatGib(stats.memory.total_bytes)} GiB`
            }
            percent={
              stats.memory === null
                ? null
                : (stats.memory.used_bytes / stats.memory.total_bytes) * 100
            }
          />
          <ResourceRow
            label="CPU"
            value={stats.cpu === null ? '不可用' : `${stats.cpu.percent}%（${stats.cpu.count} 核）`}
            percent={stats.cpu === null ? null : stats.cpu.percent}
          />
          <ResourceRow
            label="硬盘"
            value={
              stats.disk === null
                ? '不可用'
                : `剩余 ${formatGib(stats.disk.free_bytes)} / 共 ${formatGib(stats.disk.total_bytes)} GiB`
            }
            percent={
              stats.disk === null ? null : (stats.disk.used_bytes / stats.disk.total_bytes) * 100
            }
          />
        </div>
      )}
    </div>
  )
}
