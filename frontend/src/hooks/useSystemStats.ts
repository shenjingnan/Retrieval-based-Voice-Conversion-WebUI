/**
 * 系统资源快照轮询 hook：每 intervalMs 拉一次 GET /api/system/stats。
 * 不复用 useTask 的 SSE——资源信息与任务生命周期无关（无任务也要显示），只能轮询；
 * 端点设计上永不 500（各采集项独立降级，见 server/api/system.py），失败仅需处理网络层。
 */
import { useEffect, useRef, useState } from 'react'

import { api, type SystemStats } from '@/api/client'

const DEFAULT_INTERVAL_MS = 3000
/** 连续失败达到该次数才标记过期：单次失败（网络抖动/服务重启瞬间）不打扰用户 */
const STALE_AFTER_FAILURES = 2

export interface SystemStatsView {
  /** 最近一次成功快照；null = 从未成功（启动中 / 后端不可达 / 后端版本过旧） */
  stats: SystemStats | null
  /** 连续失败 ≥ STALE_AFTER_FAILURES 次：数据可能已过期。保留旧值继续展示，仅提示不冻结 */
  stale: boolean
}

export function useSystemStats(intervalMs: number = DEFAULT_INTERVAL_MS): SystemStatsView {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [stale, setStale] = useState(false)
  // 失败计数跨轮次保留（ref 而非 state）：只用于判定 stale 阈值，不值得触发渲染
  const failures = useRef(0)

  useEffect(() => {
    let cancelled = false
    let timer: number | null = null
    let inFlight = false

    const tick = () => {
      if (cancelled || inFlight) return
      inFlight = true
      // setState 全部在 Promise 回调里：effect 体内同步 setState 会触发
      // react/set-state-in-effect 告警（Training.tsx 的既有注释同因）
      void api
        .systemStats()
        .then((next) => {
          if (cancelled) return
          failures.current = 0
          setStale(false)
          setStats(next)
        })
        .catch(() => {
          if (cancelled) return
          failures.current += 1
          if (failures.current >= STALE_AFTER_FAILURES) setStale(true)
          // 静默降级（口径同 Training.tsx 预填失败的 .catch(() => undefined)）：
          // 资源信息不值得向用户抛错误条，且失败多为瞬时（服务重启/休眠唤醒），
          // 继续轮询自愈比重试按钮 UI 划算
        })
        .finally(() => {
          inFlight = false
          if (cancelled) return
          // 链式调度：上一帧 settle 之后才排下一帧，慢响应不会造成请求堆积
          //（因此 setInterval 不适用；也无须额外的重入锁）
          timer = window.setTimeout(tick, intervalMs)
        })
    }

    tick()
    return () => {
      cancelled = true
      if (timer !== null) window.clearTimeout(timer)
    }
  }, [intervalMs])

  return { stats, stale }
}
