/**
 * 任务队列面板（docs/plans/2026-09-09-task-centric-training-ui-design.md §3）：
 * 串行队列 + 历史的全局视图。单开手风琴：展开项详情由父级经 renderDetail 注入
 * （useTask 订阅与历史详情缓存的归属在页面层）；支持单任务停止与「停止并清空队列」。
 */
import { useState, type ReactNode } from 'react'
import {
  CheckIcon,
  ChevronDownIcon,
  ClockIcon,
  LoaderCircleIcon,
  PauseIcon,
  XIcon,
} from 'lucide-react'

import type { TaskSummary } from '@/api/client'
import { Button } from '@/components/ui/button'
import { TASK_NAME_LABELS } from '@/lib/trainingSteps'

const TERMINAL_STATES: ReadonlySet<TaskSummary['state']> = new Set([
  'success',
  'failed',
  'cancelled',
])

/** 排序：运行中 → 排队中（位次序）→ 终态（完成时间降序，最新在上；Array.sort 稳定） */
const STATE_ORDER: Record<TaskSummary['state'], number> = {
  running: 0,
  pending: 1,
  success: 2,
  failed: 2,
  cancelled: 2,
}

const TERMINAL_SHOW_LIMIT = 8

function rowLabel(task: TaskSummary): string {
  return task.exp_name ?? TASK_NAME_LABELS[task.name] ?? task.name
}

function StateIcon({ state }: { state: TaskSummary['state'] }) {
  if (state === 'running') return <LoaderCircleIcon className="size-4 shrink-0 animate-spin text-primary" />
  if (state === 'pending') return <ClockIcon className="size-4 shrink-0 text-muted-foreground" />
  if (state === 'success') return <CheckIcon className="size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
  if (state === 'failed') return <XIcon className="size-4 shrink-0 text-destructive" />
  return <PauseIcon className="size-4 shrink-0 text-muted-foreground" />
}

function StateText({ task }: { task: TaskSummary }) {
  if (task.state === 'running') {
    return <>{task.progress === null ? '进行中' : `进行中 ${Math.round(task.progress * 100)}%`}</>
  }
  if (task.state === 'pending') {
    return <>{task.queue_position === null ? '排队中' : `排队中 · 第 ${task.queue_position} 位`}</>
  }
  if (task.state === 'success') return <>成功</>
  if (task.state === 'failed') return <>失败</>
  return <>已停止</>
}

export interface TaskQueuePanelProps {
  /** null = 首次轮询未返回 */
  tasks: TaskSummary[] | null
  expandedId: string | null
  onToggle: (taskId: string) => void
  onStopTask: (taskId: string) => void
  onClearQueue: () => void
  /** 展开项详情（由页面层注入：useTask 订阅与历史详情缓存的归属在页面层） */
  renderDetail: (task: TaskSummary) => ReactNode
}

export function TaskQueuePanel({
  tasks,
  expandedId,
  onToggle,
  onStopTask,
  onClearQueue,
  renderDetail,
}: TaskQueuePanelProps) {
  const [confirmStopId, setConfirmStopId] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)

  if (tasks === null) {
    return (
      <div className="rounded-lg border p-4 text-xs text-muted-foreground">正在获取任务队列…</div>
    )
  }

  const sorted = tasks
    .slice()
    .sort(
      (a, b) =>
        STATE_ORDER[a.state] - STATE_ORDER[b.state] ||
        (b.finished_at ?? b.created_at ?? 0) - (a.finished_at ?? a.created_at ?? 0),
    )
  const active = sorted.filter((t) => !TERMINAL_STATES.has(t.state))
  const terminal = sorted.filter((t) => TERMINAL_STATES.has(t.state))
  const folded = terminal.length - TERMINAL_SHOW_LIMIT
  const visible =
    folded > 0 ? [...active, ...terminal.slice(0, TERMINAL_SHOW_LIMIT)] : sorted

  if (sorted.length === 0) {
    return (
      <div className="rounded-lg border p-4 text-xs text-muted-foreground">
        当前没有训练任务。提交后自动依次串行执行，无需等待；历史记录跨重启保留。
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">任务队列</span>
        {active.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (!confirmClear) {
                setConfirmClear(true)
                setConfirmStopId(null)
                window.setTimeout(() => setConfirmClear(false), 3000)
                return
              }
              setConfirmClear(false)
              onClearQueue()
            }}
          >
            {confirmClear ? '确认停止全部？' : '停止并清空队列'}
          </Button>
        )}
      </div>
      <ul className="flex flex-col gap-1">
        {visible.map((task) => {
          const terminal = TERMINAL_STATES.has(task.state)
          const expanded = expandedId === task.id
          return (
            <li
              key={task.id}
              className={`rounded-md ${expanded ? 'bg-muted/60' : ''}`}
            >
              <div
                className={`flex items-center gap-2 px-2 py-1.5 text-sm ${
                  expanded ? '' : 'hover:bg-muted/50'
                } cursor-pointer rounded-md`}
                onClick={() => onToggle(task.id)}
                aria-expanded={expanded}
              >
                <StateIcon state={task.state} />
                <span
                  className={`min-w-0 flex-1 truncate font-mono ${
                    terminal ? 'text-muted-foreground' : ''
                  }`}
                  title={rowLabel(task)}
                >
                  {rowLabel(task)}
                  {task.history && (
                    <span className="ms-2 rounded bg-muted px-1 text-[10px] text-muted-foreground">
                      历史
                    </span>
                  )}
                </span>
                <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                  <StateText task={task} />
                </span>
                {!terminal && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="shrink-0"
                    onClick={(e) => {
                      e.stopPropagation()
                      if (confirmStopId !== task.id) {
                        setConfirmStopId(task.id)
                        setConfirmClear(false)
                        window.setTimeout(
                          () => setConfirmStopId((cur) => (cur === task.id ? null : cur)),
                          3000,
                        )
                        return
                      }
                      setConfirmStopId(null)
                      onStopTask(task.id)
                    }}
                  >
                    {confirmStopId === task.id ? '确认停止？' : '停止'}
                  </Button>
                )}
                <ChevronDownIcon
                  className={`size-4 shrink-0 text-muted-foreground transition-transform ${
                    expanded ? 'rotate-180' : ''
                  }`}
                />
              </div>
              {expanded && renderDetail(task)}
            </li>
          )
        })}
        {folded > 0 && (
          <li className="px-2 py-1 text-xs text-muted-foreground">
            更早的终态任务已折叠（{folded} 条；完整历史保留在服务端日志目录）
          </li>
        )}
      </ul>
    </div>
  )
}
