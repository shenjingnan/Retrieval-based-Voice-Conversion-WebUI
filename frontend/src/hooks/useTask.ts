/**
 * 任务实时状态 hook：EventSource 订阅 GET /api/tasks/{id}/events（cursor=0），
 * 消费 status / log / progress 三类事件（见 server/api/training.py 的 SSE 协议）。
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import type { TaskState } from '@/api/client'
import { parseLossPoints, type LossPoint } from '@/lib/loss'

export type { LossPoint } from '@/lib/loss'

export interface TaskView {
  /** 任务状态；null 表示尚未收到任何 status 事件（连接中 / 后端不可达） */
  status: TaskState | null
  /** 0-1，后端读时换算；null 表示暂无进度锚点 */
  progress: number | null
  /** 当前处理的文件名（切分/提取阶段才有，训练段为 null） */
  current: string | null
  /** 日志行（前端保留最近 MAX_LOG_LINES 行） */
  logs: string[]
  /** 从日志行解析出的 loss 序列（训练段每个日志步一个点） */
  losses: LossPoint[]
  /** 仅 failed 终态携带（后端把日志尾部拼进 error） */
  error: string | null
  /** 已到终态（success/failed/cancelled），SSE 连接已关闭 */
  terminal: boolean
  /**
   * 重连重试已耗尽（自动 + 手动共 MAX_RETRIES 次）且任务未到终态：
   * UI 应提示「连接丢失」，而不是让进度条永远停在不确定态
   */
  connectionLost: boolean
  /** status 事件快照里的命令串（pipeline 失败时用于把失败定位到具体步骤） */
  cmds: string[]
  /**
   * 当前正在执行的子命令序号（1-based；null = 未启动任何 cmd / 事件未携带）。
   * pipeline 的步骤条阶段映射靠它推进：服务端在切换子命令时会补发 status 事件。
   * 字段名沿用服务端快照的 snake_case（与 cmds 同类的原样透传）
   */
  current_cmd: number | null
  /** 队列位次（1-based）；仅任务还在排队（pending）时有值，开始运行后为 null */
  queue_position: number | null
  /** 重新建立订阅（连接丢失后的手动重挂入口：重置重试额度并重建 EventSource） */
  resubscribe: () => void
}

/** 订阅状态本体（不含 resubscribe：它是行为函数，不进状态快照） */
type TaskViewState = Omit<TaskView, 'resubscribe'>

const TERMINAL_STATES: ReadonlySet<string> = new Set(['success', 'failed', 'cancelled'])
const ALL_STATES: ReadonlySet<string> = new Set(['pending', 'running', ...TERMINAL_STATES])

/** 日志保留行数：后端环形缓冲上限（1000）的一半左右，够回看又不撑爆 DOM */
const MAX_LOG_LINES = 500
/** loss 点数上限：与日志同一量级，超出丢弃最早的点 */
const MAX_LOSS_POINTS = 500

/**
 * 浏览器放弃自动重连（readyState === CLOSED）时的手动重试上限与间隔。
 * 限次防止「后端已停 / 任务已被清理」场景下的无限重连循环。
 */
const MAX_RETRIES = 5
const RETRY_DELAY_MS = 2000

/**
 * loss 行锚点与解析已收敛到 lib/loss.ts（与 server/progress.py 同源；
 * 历史日志尾部快照共用同一实现）。
 */

/** SSE data 是 JSON；畸形事件丢弃而不是让页面崩掉 */
function parseData<T>(ev: Event): T | null {
  try {
    return JSON.parse((ev as MessageEvent<string>).data) as T
  } catch {
    return null
  }
}

interface StatusEvent {
  state?: string
  progress?: number | null
  current?: string | null
  error?: string | null
  cmds?: unknown
  current_cmd?: unknown
  queue_position?: unknown
}

interface LogEvent {
  lines?: unknown
}

interface ProgressEvent {
  progress?: number | null
  current?: string | null
}

function trimTail<T>(items: T[], max: number): T[] {
  return items.length > max ? items.slice(items.length - max) : items
}

const EMPTY_VIEW: TaskViewState = {
  status: null,
  progress: null,
  current: null,
  logs: [],
  losses: [],
  error: null,
  terminal: false,
  connectionLost: false,
  cmds: [],
  current_cmd: null,
  queue_position: null,
}

export function useTask(taskId: string | null): TaskView {
  const [view, setView] = useState<TaskViewState>(EMPTY_VIEW)

  // 切换任务时的状态重置放在 render 阶段（React 官方「Adjusting state when a prop
  // changes」模式：上一次的值存进 state 而不是 ref），而不是订阅 effect 的开头——
  // 避免先渲染出一帧上一个任务的残留数据。手动重连（attempt 变化）不走这里：
  // open 事件会清空增量状态，status 事件随后重放。
  const [prevTaskId, setPrevTaskId] = useState(taskId)
  if (prevTaskId !== taskId) {
    setPrevTaskId(taskId)
    setView(EMPTY_VIEW)
  }

  // 手动重试计数放 ref 里跨 effect 保留：若放闭包局部变量，每次重建 effect 都会
  // 归零，「CLOSED → 重建 → 又 CLOSED」会变成无限循环
  const retries = useRef(0)
  // 上一个实际订阅过的任务 id（effect 期比较，用于切任务时重置重试额度）
  const subscribedTaskId = useRef<string | null>(null)
  // 手动重连通过递增 attempt 触发 effect 重跑（重建 EventSource）
  const [attempt, setAttempt] = useState(0)

  // 连接丢失后的手动重挂入口：重置重试额度再重建（否则耗尽的额度会立即标记丢失）。
  // 引用稳定，可直接作为按钮 onClick
  const resubscribe = useCallback(() => {
    retries.current = 0
    setAttempt((a) => a + 1)
  }, [])

  useEffect(() => {
    if (taskId === null) return
    // 组件卸载 / 任务切换时关连接，并撤销尚未触发的手动重试
    let cancelRetry: (() => void) | null = null

    // 重试额度随任务重置（新任务是全新订阅，不继承上一个任务的耗尽额度）。
    // 放在 effect 内而不是 prevTaskId 的 render 分支：effect 期访问 ref 合规，
    // 且只有任务切换才清零，attempt 驱动的重连重建会保留已消耗的额度
    if (subscribedTaskId.current !== taskId) {
      subscribedTaskId.current = taskId
      retries.current = 0
    }

    const es = new EventSource(`/api/tasks/${encodeURIComponent(taskId)}/events?cursor=0`)

    /**
     * 断线重连的幂等渲染方案：**重置重放**。每次连接建立（首次 + 浏览器自动重连 +
     * 手动重试）都清空增量状态，随后服务端按 cursor=0 重放全部日志行，视图天然重建。
     * 选它而不是按 seq 去重的原因：去重游标要与后端环形缓冲（maxlen=1000，旧行会被
     * 淘汰）的收敛语义对齐，一旦错位会永久丢行；重置方案每次都拿到缓冲内的完整剩余
     * 内容，淘汰多少收敛多少，前端不需要维护任何游标。代价是重连瞬间日志区清空后
     * 重填（闪烁），对低频断线可接受。EventSource 规范保证 open 先于本连接的任何
     * message 派发，因此重置不会吞掉重放数据。
     */
    es.addEventListener('open', () => {
      // 重连成功即恢复「连接丢失」提示（若曾触发），随后 cursor=0 重放重建视图
      setView((prev) => ({ ...prev, connectionLost: false, logs: [], losses: [] }))
    })

    es.addEventListener('status', (ev) => {
      const data = parseData<StatusEvent>(ev)
      if (data === null) return
      const state = data.state
      if (typeof state !== 'string' || !ALL_STATES.has(state)) return
      setView((prev) => ({
        ...prev,
        status: state as TaskState,
        progress: typeof data.progress === 'number' ? data.progress : null,
        current: typeof data.current === 'string' ? data.current : null,
        error: state === 'failed' ? (data.error ?? null) : prev.error,
        cmds: Array.isArray(data.cmds)
          ? data.cmds.filter((c): c is string => typeof c === 'string')
          : prev.cmds,
        current_cmd:
          typeof data.current_cmd === 'number' ? data.current_cmd : prev.current_cmd,
        queue_position:
          typeof data.queue_position === 'number' ? data.queue_position : null,
        terminal: TERMINAL_STATES.has(state),
      }))
      if (TERMINAL_STATES.has(state)) {
        // 终态：服务端发完这条 status 就关流，这里提前释放连接并落定终态标记
        es.close()
      }
    })

    // 轮次/步数锚点跨行跟踪收敛到 parseLossPoints（批次内成组跟踪）；断线重连时
    // cursor=0 按原序重放，锚点自然收敛回正确值
    es.addEventListener('log', (ev) => {
      const data = parseData<LogEvent>(ev)
      if (data === null || !Array.isArray(data.lines)) return
      const lines = data.lines.filter((line): line is string => typeof line === 'string')
      if (lines.length === 0) return
      const points = parseLossPoints(lines)
      setView((prev) => ({
        ...prev,
        logs: trimTail([...prev.logs, ...lines], MAX_LOG_LINES),
        losses: points.length > 0 ? trimTail([...prev.losses, ...points], MAX_LOSS_POINTS) : prev.losses,
      }))
    })

    es.addEventListener('progress', (ev) => {
      const data = parseData<ProgressEvent>(ev)
      if (data === null) return
      setView((prev) => ({
        ...prev,
        progress: typeof data.progress === 'number' ? data.progress : null,
        current: typeof data.current === 'string' ? data.current : null,
      }))
    })

    es.addEventListener('error', () => {
      // 网络闪断时浏览器会自动重连（open 处的重置保证重放幂等），这里只兜底
      // 浏览器放弃重连的情况（readyState === CLOSED，典型如后端返回 404 / 非 SSE 响应）
      if (es.readyState !== EventSource.CLOSED) return
      if (retries.current >= MAX_RETRIES) {
        // 额度耗尽：把「连接丢失」外露给 UI（否则任务可能仍在跑，进度条却永远 pulse）
        setView((prev) => ({ ...prev, connectionLost: true }))
        return
      }
      retries.current += 1
      const timer = setTimeout(() => setAttempt((a) => a + 1), RETRY_DELAY_MS)
      cancelRetry = () => clearTimeout(timer)
    })

    return () => {
      cancelRetry?.()
      es.close()
    }
  }, [taskId, attempt])

  return { ...view, resubscribe }
}
