/**
 * 队列项展开详情（docs/plans/2026-09-09-task-centric-training-ui-design.md §3）：
 * 阶段指示 / 进度条 / loss 图 / 日志 / 失败原因 / 成功产物 + 去试音 / 重新提交。
 * live（内存任务，useTask SSE 数据）与 detail（磁盘历史快照）双源在此收敛：
 * live 存在时一切以 live 为准，历史行只用快照静态渲染。
 */
import { useEffect, useState } from 'react'
import { CheckIcon, LoaderCircleIcon, PauseIcon, XIcon } from 'lucide-react'

import type { TaskHistoryDetail, TaskSummary } from '@/api/client'
import { ErrorDetail } from '@/components/ErrorDetail'
import { Button } from '@/components/ui/button'
import { useProductName } from '@/hooks/useProductName'
import type { LossPoint, TaskView } from '@/hooks/useTask'
import { parseLossPoints } from '@/lib/loss'
import { cmdToStep, stageStatuses, STEP_IDS, type StepId } from '@/lib/trainingSteps'

import { LossChart } from './LossChart'
import { LogPanel } from './LogPanel'

const STAGE_LABELS: Record<StepId, string> = {
  preprocess: '处理数据',
  extract: '特征提取',
  fit: '训练',
  index: '建立索引',
}

function StageBadge({ status }: { status: 'pending' | 'running' | 'done' | 'failed' | 'skipped' }) {
  const icon =
    status === 'done' ? (
      <CheckIcon />
    ) : status === 'failed' ? (
      <XIcon />
    ) : status === 'skipped' ? (
      <PauseIcon />
    ) : status === 'running' ? (
      <LoaderCircleIcon className="animate-spin" />
    ) : null
  const tone =
    status === 'done'
      ? 'bg-emerald-600 text-white'
      : status === 'failed'
        ? 'bg-destructive text-white'
        : status === 'running'
          ? 'bg-primary text-primary-foreground'
          : status === 'skipped'
            ? 'bg-muted-foreground/60 text-white'
            : 'bg-muted text-muted-foreground'
  return (
    <span
      className={`inline-flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium ${tone}`}
    >
      {icon}
    </span>
  )
}

export interface TaskDetailProps {
  row: TaskSummary
  /** 内存任务的实时视图（useTask）；历史行为 null */
  live: TaskView | null
  /** 历史任务详情（父级缓存）；内存行为 null */
  detail: TaskHistoryDetail | null
  detailLoading: boolean
  resubmitBusy: boolean
  onResubmit: (params: Record<string, unknown>) => void
  onGoInfer: (model: string) => void
}

export function TaskDetail({
  row,
  live,
  detail,
  detailLoading,
  resubmitBusy,
  onResubmit,
  onGoInfer,
}: TaskDetailProps) {
  const isLive = live !== null
  const cmds = isLive && live.cmds.length > 0 ? live.cmds : (detail?.cmds ?? [])
  const currentCmd = isLive ? live.current_cmd : (detail?.current_cmd ?? null)
  const error = isLive ? live.error : row.error
  const statuses = stageStatuses({
    stages: row.pipeline_stages,
    cmds,
    currentCmd,
    state: row.state,
    error,
  })

  // 可见的阶段格：按该任务实际包含的阶段去重（pipeline 4 格，单步任务 1 格；
  // separate 等 cmds 无法识别的任务不显示阶段指示）
  const stageList =
    row.pipeline_stages !== null && row.pipeline_stages.length === cmds.length
      ? row.pipeline_stages
      : cmds.map((cmd) => cmdToStep(cmd))
  const visibleSteps = STEP_IDS.filter((step) => stageList.includes(step))

  const progress = isLive ? live.progress : row.progress
  const progressPct = progress === null ? null : Math.round(progress * 100)
  const logs = isLive ? live.logs : (detail?.logs_tail ?? [])
  const losses: LossPoint[] = isLive ? live.losses : parseLossPoints(logs)
  const showLoss = row.kind === 'fit' || row.kind === 'pipeline'
  const resubmittable =
    row.kind === 'pipeline' &&
    row.params !== null &&
    (row.state === 'failed' || row.state === 'cancelled')

  return (
    <div className="flex flex-col gap-3 border-t pt-3">
      {visibleSteps.length > 0 && (
        <ol className="flex flex-wrap items-center gap-x-3 gap-y-2">
          {visibleSteps.map((step, i) => {
            const status = statuses[step]
            return (
              <li key={step} className="inline-flex items-center gap-1.5 text-xs">
                <StageBadge status={status} />
                <span className={status === 'pending' ? 'text-muted-foreground' : 'font-medium'}>
                  {STAGE_LABELS[step]}
                </span>
                {i < visibleSteps.length - 1 && (
                  <span className="ms-1.5 text-muted-foreground" aria-hidden>
                    →
                  </span>
                )}
              </li>
            )
          })}
        </ol>
      )}

      {(row.state === 'running' || row.state === 'pending') && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>进度</span>
            <span className="tabular-nums">
              {row.state === 'pending'
                ? '排队等待中'
                : progressPct === null
                  ? '等待进度…'
                  : `${progressPct}%`}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full bg-primary transition-[width] duration-500 ${
                progressPct === null ? 'animate-pulse' : ''
              }`}
              style={{ width: `${progressPct ?? 0}%` }}
            />
          </div>
        </div>
      )}

      {isLive && live.connectionLost && !live.terminal && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          连接已断开，任务仍在后台运行。
        </p>
      )}

      {showLoss && <LossChart points={losses} />}

      {detailLoading && !isLive && (
        <p className="text-xs text-muted-foreground">正在加载历史日志…</p>
      )}
      <LogPanel lines={logs} />

      {error !== null && row.state === 'failed' && (
        <ErrorDetail title="任务失败">{error}</ErrorDetail>
      )}
      {row.state === 'cancelled' && (
        <p className="text-xs text-muted-foreground">
          任务已停止{row.history ? '（服务重启前被终止）' : ''}；失败或停止的任务可以原参数重新提交，
          训练将从最近的检查点续跑。
        </p>
      )}

      {row.state === 'success' && row.exp_name !== null && (
        <div className="flex flex-col gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-3">
          <ProductRow exp={row.exp_name} onGoInfer={onGoInfer} />
        </div>
      )}

      {resubmittable && (
        <div>
          <Button
            variant="outline"
            size="sm"
            disabled={resubmitBusy}
            onClick={() => onResubmit(row.params as Record<string, unknown>)}
            title="以本任务的原参数重新提交（同名实验，训练将从最近的检查点续跑）"
          >
            {resubmitBusy && <LoaderCircleIcon className="animate-spin" />}
            重新提交（原参数续训）
          </Button>
        </div>
      )}
    </div>
  )
}

/** 成功任务的产物确认 + 去试音（历史成功行同样可用） */
function ProductRow({ exp, onGoInfer }: { exp: string; onGoInfer: (model: string) => void }) {
  const { product, error, refreshAsync } = useProductName(exp, true)
  // pending 只在事件回调里同步置位（初始 true），探测落定后异步清除——避免 effect 内同步 setState
  const [pending, setPending] = useState(true)
  const recheck = () => {
    setPending(true)
    void refreshAsync().finally(() => setPending(false))
  }
  useEffect(() => {
    void refreshAsync().finally(() => setPending(false))
  }, [refreshAsync])

  if (error !== null) {
    return (
      <>
        <p className="text-xs text-destructive">产物确认失败：{error}</p>
        <Button variant="ghost" size="sm" onClick={recheck}>
          重新检查产物
        </Button>
      </>
    )
  }
  if (product === null) {
    return (
      <>
        <p className="text-xs text-muted-foreground">
          {pending ? '正在确认产物…' : '未找到本实验的模型文件（可能已被删除或重命名）'}
        </p>
        <Button variant="ghost" size="sm" onClick={recheck}>
          重新检查产物
        </Button>
      </>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">
        模型 <span className="font-mono text-foreground">{product}</span> 已就绪
      </span>
      <Button size="sm" onClick={() => onGoInfer(product)}>
        去试音
      </Button>
      <Button variant="ghost" size="sm" onClick={recheck}>
        重新检查产物
      </Button>
    </div>
  )
}
