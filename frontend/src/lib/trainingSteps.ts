/**
 * 训练向导的步骤状态机纯函数（无组件、无副作用）：
 * 独立成模块便于直接推演/测试（node --experimental-strip-types 可直接导入），
 * 组件层（pages/Training.tsx）只做消费。
 */
import type { TaskState } from '@/api/client'

export type StepId = 'preprocess' | 'extract' | 'fit' | 'index'

/**
 * 步骤状态：
 * - idle 待进行 / running 进行中 / success 成功 / failed 失败
 * - skipped 已跳过（任务被用户停止 cancelled，可重试，与失败区分展示）
 */
export type StepState = 'idle' | 'running' | 'success' | 'failed' | 'skipped'

/** 监视中的任务归属：pipeline 一键任务驱动全部 4 步 */
export type WatchedStep = StepId | 'pipeline'

export const STEP_IDS: ReadonlyArray<StepId> = ['preprocess', 'extract', 'fit', 'index']

export const INITIAL_STEP_STATES: Record<StepId, StepState> = {
  preprocess: 'idle',
  extract: 'idle',
  fit: 'idle',
  index: 'idle',
}

const FAILED_LIKE: ReadonlySet<StepState> = new Set(['failed', 'skipped'])

/** 后端任务失败信息（server/tasks.py _failure_message）里的步骤序号锚点（1-based） */
const FAILURE_STEP_RE = /第 (\d+) 步/

/**
 * 单条子进程命令 → 所属步骤。与 server/commands.py 的命令模板逐项对应：
 * - train/preprocess.py → 处理数据
 * - train/dataset/{extract_f0,extract_hubert_feature}.py → 特征提取
 * - train/train_index.py → 建立索引
 * - train/train.py → 训练（'train/train_index.py' 不含 'train/train.py' 子串，
 *   但仍按先 index 后 fit 的顺序判断，防将来模板变化踩坑）
 * - `-m server.api.training precheck` → 处理数据（pipeline 的切分产物校验 cmd，
 *   失败时恢复动作是重跑切分）
 * - `-m server.api.training fitprep` → 训练（pipeline 的 fit 前置 cmd）
 */
export function cmdToStep(cmd: string): StepId | null {
  const c = cmd.replace(/\\/g, '/')
  if (c.includes('train/preprocess.py')) return 'preprocess'
  if (c.includes('train/dataset/')) return 'extract'
  if (c.includes('train/train_index.py')) return 'index'
  if (c.includes('train/train.py')) return 'fit'
  if (c.includes('server.api.training precheck')) return 'preprocess'
  if (c.includes('server.api.training fitprep')) return 'fit'
  return null
}

/**
 * 从任务失败信息解析真实失败步骤：error 的「第 N 步」（1-based cmd 序号，见
 * server/tasks.py 的 _failure_message）映射到 status 事件携带的 cmds 数组再分类。
 * 解析不出（格式变化 / 索引越界 / cmd 无法识别）时返回 fallback（调用方回退到
 * 「第一个非 success」推断）。
 */
export function resolveFailedStep(
  error: string | null,
  cmds: ReadonlyArray<string> | undefined,
  fallback: StepId | null,
): StepId | null {
  if (error === null || cmds === undefined) return fallback
  const m = error.match(FAILURE_STEP_RE)
  if (m === null) return fallback
  const cmd = cmds[Number(m[1]) - 1]
  if (cmd === undefined) return fallback
  return cmdToStep(cmd) ?? fallback
}

/**
 * 任务终态 → 步骤状态落定。
 *
 * - 分步：该步 success / failed / skipped（cancelled），其余步骤不受影响
 * - 一键（pipeline）：单任务串行多 cmd，任务状态不直接携带阶段信息。
 *   - failedStep 已解析（resolveFailedStep）：串行语义下失败步骤之前的阶段必然已
 *     成功完成——之前的步骤置 success、失败步骤置失败态、其后置 idle，可精确还原
 *     现场（首次一键没有前置 success 记录时尤其重要）
 *   - failedStep 缺省/无效（解析不到）：回退「第一个非 success」推断，保留已成功
 *     步骤、把第一个未成功步骤标失败态、其后置 idle（与真实失败阶段可能有偏差，
 *     但重试入口仍能走通全流程）
 */
export function settleSteps(
  prev: Record<StepId, StepState>,
  step: WatchedStep,
  status: TaskState,
  failedStep?: StepId | null,
): Record<StepId, StepState> {
  if (step === 'pipeline') {
    if (status === 'success') {
      return { preprocess: 'success', extract: 'success', fit: 'success', index: 'success' }
    }
    const failedLike: StepState = status === 'cancelled' ? 'skipped' : 'failed'
    if (failedStep !== undefined && failedStep !== null) {
      const idx = STEP_IDS.indexOf(failedStep)
      if (idx >= 0) {
        const next = { ...prev }
        for (const [i, s] of STEP_IDS.entries()) {
          next[s] = i < idx ? 'success' : i === idx ? failedLike : 'idle'
        }
        return next
      }
    }
    const next = { ...prev }
    for (const s of STEP_IDS) {
      if (next[s] !== 'success') {
        next[s] = failedLike
        for (const later of STEP_IDS.slice(STEP_IDS.indexOf(s) + 1)) {
          next[later] = 'idle'
        }
        break
      }
    }
    return next
  }
  const state: StepState =
    status === 'success' ? 'success' : status === 'cancelled' ? 'skipped' : 'failed'
  return { ...prev, [step]: state }
}

/**
 * 任务失败/停止后的「重试该步」目标：
 * - 分步：失败的那一步
 * - 一键：settleSteps 标记出的第一个失败/停止步骤（精确解析时即真实失败步骤；
 *   其后步骤锁定，重试成功后按前置状态继续解锁）
 * 无可重试步骤（无终态失败）返回 null。
 */
export function pickRetryStep(states: Record<StepId, StepState>): StepId | null {
  return STEP_IDS.find((s) => FAILED_LIKE.has(states[s])) ?? null
}
