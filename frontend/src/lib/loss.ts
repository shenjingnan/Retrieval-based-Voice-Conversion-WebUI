/**
 * loss 日志解析纯函数（从 hooks/useTask.ts 迁出，供 SSE 流与历史日志快照共用，
 * 见 docs/plans/2026-09-09-task-centric-training-ui-design.md §4）。
 * 正则与 server/progress.py 同源；`node --experimental-strip-types` 可直接导入自测。
 */

/** loss 曲线数据点（展示级解析）：disc/gen 画线，epoch/pct/step 供悬浮提示标注
 *  「这是训练的第几轮第几步」 */
export interface LossPoint {
  disc: number
  gen: number
  /** 该点所属的训练轮次（取 loss 行之前最近的轮次锚点；缺失为 null） */
  epoch: number | null
  /** 该轮次内的进度百分比 */
  pct: number | null
  /** 全局训练步数（[step, lr] 锚点；缺失为 null） */
  step: number | null
}

/**
 * loss 行锚点（与 server/progress.py 的五项格式同源；这里只取前两项）。
 * `[-+0-9.eE]` 允许科学计数法；残缺片段（如 "1e+"）交给 Number + isFinite 过滤。
 */
const LOSS_LINE_RE = /loss_disc=([-+0-9.eE]+),\s*loss_gen=([-+0-9.eE]+)/

/**
 * 轮次锚点行（与 server/progress.py 的 EPOCH_PATTERN 同源）：训练脚本按
 * epoch 行 → [step, lr] 行 → loss 行的顺序输出，逐行扫描即可给每个 loss 点
 * 标注它所属的轮次与全局步数。
 */
const EPOCH_LINE_RE = /(?:训练轮次：|Training epoch: |Epoch: )(\d+) \[(-?\d+(?:\.\d+)?)%\]/

/** [step, lr] 行（与 server/progress.py 的 STEP_PATTERN 同源，行尾锚定） */
const STEP_LINE_RE = /\[(-?\d+), (-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\]\s*$/

/** 解析单行；非 loss 行返回 null（epoch/pct/step 传入「当前最近的锚点」）。 */
export function parseLossPoint(
  line: string,
  epoch: number | null,
  pct: number | null,
  step: number | null,
): LossPoint | null {
  const m = line.match(LOSS_LINE_RE)
  if (m === null) return null
  const disc = Number(m[1])
  const gen = Number(m[2])
  if (!Number.isFinite(disc) || !Number.isFinite(gen)) return null
  return { disc, gen, epoch, pct, step }
}

/**
 * 解析一批日志行为 loss 点：轮次/步数锚点在同一批内跨行跟踪（训练日志按
 * epoch → [step, lr] → loss 的顺序成组输出）。历史日志尾部快照与 SSE 增量
 * 批次共用本函数；快照头部的锚点缺失体现为前 1-2 个点的 null 标注。
 */
export function parseLossPoints(lines: ReadonlyArray<string>): LossPoint[] {
  let lastEpoch: number | null = null
  let lastPct: number | null = null
  let lastStep: number | null = null
  const points: LossPoint[] = []
  for (const line of lines) {
    const epochMatch = line.match(EPOCH_LINE_RE)
    if (epochMatch !== null) {
      lastEpoch = Number(epochMatch[1])
      lastPct = Number(epochMatch[2])
    }
    const stepMatch = line.match(STEP_LINE_RE)
    if (stepMatch !== null) lastStep = Number(stepMatch[1])
    const point = parseLossPoint(line, lastEpoch, lastPct, lastStep)
    if (point !== null) points.push(point)
  }
  return points
}
