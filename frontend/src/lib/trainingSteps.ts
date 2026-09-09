/**
 * 训练任务的阶段推导纯函数（无组件、无副作用）：独立成模块便于直接推演/测试
 * （node --experimental-strip-types 可直接导入）。任务中心化改造后（设计
 * docs/plans/2026-09-09-task-centric-training-ui-design.md），训练页不再有步骤条，
 * 这里只服务队列项详情的阶段指示与失败定位。
 */
import type { TaskState } from '@/api/client'

export type StepId = 'preprocess' | 'extract' | 'fit' | 'index'

export const STEP_IDS: ReadonlyArray<StepId> = ['preprocess', 'extract', 'fit', 'index']

/** 任务类型 → 展示名（separate 来自数据集页但与训练共用队列，一并列出） */
export const TASK_NAME_LABELS: Record<string, string> = {
  preprocess: '处理数据',
  extract: '特征提取',
  fit: '训练',
  index: '建立索引',
  pipeline: '一键训练',
  separate: '人声分离',
}

/** 队列项详情的阶段状态（任务中心的阶段指示语义，与旧步骤条的五态不同） */
export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

/**
 * 单条子进程命令 → 所属步骤。与 server/commands.py 的命令模板逐项对应：
 * - train/preprocess.py → 处理数据
 * - train/dataset/{extract_f0,extract_hubert_feature}.py → 特征提取
 * - train/train_index.py → 建立索引
 * - train/train.py → 训练（'train/train_index.py' 不含 'train/train.py' 子串，
 *   但仍按先 index 后 fit 的顺序判断，防将来模板变化踩坑）
 * - `-m server.api.training precheck` → 处理数据（pipeline 的切分产物校验 cmd）
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
 * server/tasks.py 的 _failure_message）映射到 cmds 数组再分类。解析不出（任务
 * 未启动任何 cmd / 格式变化 / cmd 无法识别）时返回 fallback。
 */
export function resolveFailedStep(
  error: string | null,
  cmds: ReadonlyArray<string> | undefined,
  fallback: StepId | null,
): StepId | null {
  if (error === null || cmds === undefined) return fallback
  const m = error.match(/第 (\d+) 步/)
  if (m === null) return fallback
  const cmd = cmds[Number(m[1]) - 1]
  if (cmd === undefined) return fallback
  return cmdToStep(cmd) ?? fallback
}

export interface StageInput {
  /** 任务元数据里的阶段归属表（pipeline 提交时登记，与 cmds 等长）；单步任务为 null */
  stages: ReadonlyArray<string> | null
  cmds: ReadonlyArray<string>
  /** 当前正在执行的子命令序号（1-based；null = 尚未启动任何 cmd） */
  currentCmd: number | null
  state: TaskState | null
  error: string | null
}

const FAILURE_STEP_RE = /第 (\d+) 步/

/**
 * 队列项详情的 4 格阶段状态推导：
 * - 每条 cmd 的阶段归属：stages 表（等长时）优先，缺省回退 cmds.map(cmdToStep)
 * - running / pending：当前 cmd 之前 → done、当前 → running、之后 → pending
 * - success 终态：全部 done
 * - cancelled 终态：失败/停止阶段之后与未启动的部分 → skipped（启动过 → skipped，
 *   未启动过 → 全部 skipped）
 * - failed 终态：失败阶段 = error 的「第 N 步」解析，回退到当前（或首个）阶段；
 *   其前 done、其处 failed、其后 pending
 */
export function stageStatuses(input: StageInput): Record<StepId, StageStatus> {
  const { stages, cmds, currentCmd, state, error } = input
  const perCmd: Array<StepId | null> =
    stages !== null && stages.length === cmds.length
      ? (stages as Array<StepId | null>)
      : cmds.map((cmd) => cmdToStep(cmd))
  const effective = perCmd.filter((stage): stage is StepId => stage !== null)
  const next: Record<StepId, StageStatus> = {
    preprocess: 'pending',
    extract: 'pending',
    fit: 'pending',
    index: 'pending',
  }
  if (effective.length === 0) {
    // cmd 无法识别（如历史记录缺 cmds）：无法推导，全部保持 pending
    return next
  }

  if (state === 'success') {
    for (const step of effective) next[step] = 'done'
    return next
  }

  const stageStart = (stage: StepId) => perCmd.indexOf(stage)
  if (state === 'failed') {
    let failedStep: StepId | null = null
    const m = error === null ? null : error.match(FAILURE_STEP_RE)
    if (m !== null) {
      const cmd = cmds[Number(m[1]) - 1]
      failedStep = cmd === undefined ? null : cmdToStep(cmd)
    }
    if (failedStep === null) {
      failedStep =
        currentCmd !== null
          ? (perCmd[Math.min(currentCmd, perCmd.length) - 1] ?? null)
          : (effective[0] ?? null)
    }
    const failedIdx = failedStep !== null ? stageStart(failedStep) : -1
    for (const stage of effective) {
      const idx = stageStart(stage)
      next[stage] = idx < failedIdx ? 'done' : idx === failedIdx ? 'failed' : 'pending'
    }
    return next
  }

  if (state === 'cancelled') {
    if (currentCmd === null || currentCmd <= 0) {
      // 未启动过任何 cmd 的取消（排队中取消 / setup 失败前停止）：全部视为已跳过
      for (const stage of effective) next[stage] = 'skipped'
      return next
    }
    const activeIdx = Math.min(currentCmd, perCmd.length) - 1
    for (const stage of effective) {
      const idx = stageStart(stage)
      next[stage] = idx < activeIdx ? 'done' : idx === activeIdx ? 'skipped' : 'pending'
    }
    return next
  }

  // pending / running：按当前 cmd 推进
  if (currentCmd === null) {
    return next
  }
  const activeIdx = Math.min(currentCmd, perCmd.length) - 1
  for (const stage of effective) {
    const idx = stageStart(stage)
    next[stage] = idx < activeIdx ? 'done' : idx === activeIdx ? 'running' : 'pending'
  }
  return next
}
