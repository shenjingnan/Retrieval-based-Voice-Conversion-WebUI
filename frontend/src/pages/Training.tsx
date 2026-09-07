/**
 * 训练向导页（P2）：步骤条 ① 实验配置 → ② 处理数据 → ③ 特征提取 → ④ 训练 → ⑤ 完成。
 * 两种执行方式：一键（trainPipeline，单任务串行 5 段命令）与分步（4 个任务按钮，
 * 前一步成功后解锁下一步，失败可重发同参数请求）。任务监视区由 useTask 驱动：
 * 进度条 + 当前文件 + 日志滚动区 + loss 曲线（内联 SVG，不引图表库）。
 */
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react'
import {
  CheckIcon,
  ChevronDownIcon,
  LoaderCircleIcon,
  PauseIcon,
  PlayIcon,
  XIcon,
} from 'lucide-react'

import { api, type ModelVersion, type SampleRate, type TaskCreated, type TrainF0Method, type TrainParams } from '@/api/client'
import { useTask, type LossPoint } from '@/hooks/useTask'
import { DATASET_BAD_RE, EXP_NAME_RE, pickProductName } from '@/lib/domain'
import { COLLAPSIBLE_TRIGGER_CLASS } from '@/lib/ui'
import { errorMessage } from '@/lib/utils'
import { ErrorDetail } from '@/components/ErrorDetail'
import {
  INITIAL_STEP_STATES,
  pickRetryStep,
  resolveFailedStep,
  settleSteps,
  type StepId,
  type StepState,
  type WatchedStep,
} from '@/lib/trainingSteps'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

const STEP_IDS: ReadonlyArray<StepId> = ['preprocess', 'extract', 'fit', 'index']

/** 步骤条 5 格：第 1 格是表单（非任务），第 5 格由 index 步骤承载（完成后展示产物） */
const STEP_CELLS: ReadonlyArray<{ key: StepId | 'config'; label: string }> = [
  { key: 'config', label: '实验配置' },
  { key: 'preprocess', label: '处理数据' },
  { key: 'extract', label: '特征提取' },
  { key: 'fit', label: '训练' },
  { key: 'index', label: '完成' },
]

/** Base UI 的 Select.Value 默认渲染原始 value（Radix 是渲染选中项文案），
 *  value 与文案不一致的下拉框必须给 Root 传 items 做映射，否则触发器显示 "1"/"0" */
const SR_ITEMS = [
  { value: '48k', label: '48k（推荐）' },
  { value: '40k', label: '40k' },
  { value: '32k', label: '32k' },
]
const IF_F0_ITEMS = [
  { value: '1', label: '开启（歌声/变调推荐）' },
  { value: '0', label: '关闭（说话声更快）' },
]
const F0_METHOD_ITEMS = [
  { value: 'rmvpe', label: 'rmvpe（推荐）' },
  { value: 'pm', label: 'pm（最快，质量一般）' },
]
const VERSION_ITEMS = [
  { value: 'v2', label: 'v2（推荐）' },
  { value: 'v1', label: 'v1（兼容旧模型）' },
]
const SAVE_WEIGHTS_ITEMS = [
  { value: '1', label: '开启（每个保存间隔另存一份）' },
  { value: '0', label: '关闭（只保存最终模型）' },
]

const STEP_TITLES: Record<WatchedStep, string> = {
  preprocess: '处理数据',
  extract: '特征提取',
  fit: '训练',
  index: '建立索引',
  pipeline: '一键训练',
}

const STEP_ACTIONS: Record<StepId, string> = {
  preprocess: '开始切分',
  extract: '开始提取',
  fit: '开始训练',
  index: '建立索引',
}

const STEP_STARTERS: Record<StepId, (p: TrainParams) => Promise<TaskCreated>> = {
  preprocess: api.trainPreprocess,
  extract: api.trainExtract,
  fit: api.trainFit,
  index: api.trainIndex,
}

// 实验名/数据集路径校验规则与产物命名推导见 '@/lib/domain'（与后端同源的领域规则，
// Models 页同样复用）

/** Base UI 的 onValueChange 可能给 null（清空态），本页选项都必选，null 时忽略 */
function applySelect(v: string | null, set: (value: string) => void): void {
  if (v !== null) set(v)
}

interface StepBadgeProps {
  state: StepState
  index: number
}

/** 步骤条格子：序号圆点带状态色 + 标签（skipped = 任务被停止的已跳过态，与失败区分） */
function StepBadge({ state, index }: StepBadgeProps) {
  const icon =
    state === 'success' ? (
      <CheckIcon />
    ) : state === 'failed' ? (
      <XIcon />
    ) : state === 'skipped' ? (
      <PauseIcon />
    ) : state === 'running' ? (
      <LoaderCircleIcon className="animate-spin" />
    ) : (
      <span>{index}</span>
    )
  const tone =
    state === 'success'
      ? 'bg-emerald-600 text-white'
      : state === 'failed'
        ? 'bg-destructive text-white'
        : state === 'running'
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted text-muted-foreground'
  return (
    <span
      className={`inline-flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium ${tone}`}
    >
      {icon}
    </span>
  )
}

/** 日志滚动区：等宽 + 自动滚底，用户上滚时暂停（回到底部恢复） */
function LogPanel({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement | null>(null)
  const stick = useRef(true)
  useEffect(() => {
    const el = ref.current
    if (el === null || !stick.current) return
    el.scrollTop = el.scrollHeight
  }, [lines])
  return (
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
  )
}

/** loss 曲线：内联 SVG 双 polyline（loss_disc / loss_gen），共同 Y 轴范围 */
function LossChart({ points }: { points: LossPoint[] }) {
  if (points.length < 2) {
    return (
      <p className="text-xs text-muted-foreground">暂无 loss 数据，训练产生日志后自动绘制</p>
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
  const polyline = (pick: (p: LossPoint) => number) =>
    points.map((p, i) => `${x(i).toFixed(2)},${y(pick(p)).toFixed(2)}`).join(' ')
  return (
    <div className="flex flex-col gap-1.5">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="none"
        className="h-24 w-full rounded-lg bg-muted"
        role="img"
        aria-label="loss 曲线"
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
      </svg>
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

interface NumberFieldProps {
  id: string
  label: string
  /** null = 未指定（自动）：显示占位文案，提交时不发该字段 */
  value: number | null
  onChange: (value: number | null) => void
  placeholder?: string
  hint?: string
}

/** 数值输入：非法输入（空串等）归 null（由调用方决定 null 语义），数值由调用方校验 */
function NumberField({ id, label, value, onChange, placeholder, hint }: NumberFieldProps) {
  function onInput(e: ChangeEvent<HTMLInputElement>) {
    const n = e.target.valueAsNumber
    onChange(Number.isNaN(n) ? null : n)
  }
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      <Input
        id={id}
        type="number"
        min={1}
        step={1}
        value={value ?? ''}
        onChange={onInput}
        placeholder={placeholder}
      />
      {hint !== undefined && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

export interface TrainingPageProps {
  /** 完成后「去试音」：App 切到推理 Tab 并把模型名带给推理页 */
  onGoInfer: (model: string) => void
}

export function TrainingPage({ onGoInfer }: TrainingPageProps) {
  // -- 表单（基础） ---------------------------------------------------------
  const [expName, setExpName] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  // -- 表单（高级，默认值对齐 webui / server/api/training.py） ----------------
  const [sr, setSr] = useState<SampleRate>('40k')
  const [ifF0, setIfF0] = useState(true)
  const [f0Method, setF0Method] = useState<TrainF0Method>('rmvpe')
  const [version, setVersion] = useState<ModelVersion>('v2')
  const [totalEpoch, setTotalEpoch] = useState(20)
  const [saveEveryEpoch, setSaveEveryEpoch] = useState(5)
  /** null = 自动：提交时不发字段，由后端按设备自适应解析（webui 显存GB÷2，无卡为 1） */
  const [batchSize, setBatchSize] = useState<number | null>(null)
  const [saveEveryWeights, setSaveEveryWeights] = useState(false)
  /** 用户是否手动改过 batch size：改过就不再用 /api/train/defaults 的解析值覆盖 */
  const batchSizeTouched = useRef(false)

  // -- 任务编排 -------------------------------------------------------------
  const [stepStates, setStepStates] = useState<Record<StepId, StepState>>(INITIAL_STEP_STATES)
  const [watched, setWatched] = useState<{ step: WatchedStep; taskId: string } | null>(null)
  const task = useTask(watched?.taskId ?? null)
  const [busyStep, setBusyStep] = useState<StepId | 'pipeline' | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [confirmStop, setConfirmStop] = useState(false)
  /** 提交时冻结的参数（终态映射与产物查询都用当时的实验名，不受表单后续编辑影响） */
  const submitted = useRef<TrainParams | null>(null)

  // -- 完成区（产物确认 + 去试音） -------------------------------------------
  const [product, setProduct] = useState<string | null>(null)
  const [productError, setProductError] = useState<string | null>(null)

  // 异步回调里的 setState 保护：切 Tab 卸载后 React 19 会静默忽略，这里显式短路
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  // 挂载时把设备自适应默认值预填进 batch size（对齐 webui 在滑条上预填解析值的行为）：
  // 用户已手动改过则不覆盖；拉取失败保持「自动」，提交时省略字段，后端按同一逻辑解析
  useEffect(() => {
    void api
      .trainDefaults()
      .then(({ batch_size }) => {
        if (!mounted.current || batchSizeTouched.current) return
        setBatchSize(batch_size)
      })
      .catch(() => undefined) // 预填失败不阻塞页面：自动语义不依赖该请求
  }, [])

  // -- 校验（派生） ---------------------------------------------------------
  const expNameValid = expName !== '.' && expName !== '..' && EXP_NAME_RE.test(expName)
  const datasetValid =
    datasetDir.length > 0 && !DATASET_BAD_RE.test(datasetDir) && !datasetDir.endsWith('\\')
  const epochsValid =
    totalEpoch >= 1 && saveEveryEpoch >= 1 && (batchSize === null || batchSize >= 1)
  const formValid = expNameValid && datasetValid && epochsValid

  // v1 没有 32k 档（server/api/training.py _normalize_sr，webui change_version19 语义）。
  // 分步模式下 preprocess 与 fit 必须用同一采样率，前端在提交前统一归一化
  const normalizedSr: SampleRate = version === 'v1' && sr === '32k' ? '40k' : sr

  function currentParams(): TrainParams {
    return {
      exp_name: expName,
      dataset_dir: datasetDir,
      sr: normalizedSr,
      version,
      if_f0: ifF0,
      f0_method: f0Method,
      total_epoch: totalEpoch,
      save_every_epoch: saveEveryEpoch,
      // 自动（null）时键省略：后端按设备解析，任务日志会记下实际使用的值与来源
      ...(batchSize === null ? {} : { batch_size: batchSize }),
      save_every_weights: saveEveryWeights,
    }
  }

  const running = watched !== null && !task.terminal
  const busy = busyStep !== null

  // -- 启动任务 -------------------------------------------------------------
  /** 新任务启动时的公共复位：停止确认两段式状态、上一轮的完成区产物 */
  function resetTransient() {
    setConfirmStop(false)
    setProduct(null)
    setProductError(null)
  }

  async function startStep(step: StepId) {
    const params = currentParams()
    setBusyStep(step)
    setSubmitError(null)
    try {
      const created = await STEP_STARTERS[step](params)
      if (!mounted.current) return
      submitted.current = params
      resetTransient()
      // 重跑该步会让后续步骤的已有产物过期（数据集/参数可能已改）：级联失效，
      // 完成区随之消失，直到重新走到建立索引
      const idx = STEP_IDS.indexOf(step)
      setStepStates((prev) => {
        const next = { ...prev, [step]: 'running' as StepState }
        for (const later of STEP_IDS.slice(idx + 1)) next[later] = 'idle'
        return next
      })
      setWatched({ step, taskId: created.task_id })
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    } finally {
      if (mounted.current) setBusyStep(null)
    }
  }

  async function startPipeline() {
    const params = currentParams()
    setBusyStep('pipeline')
    setSubmitError(null)
    try {
      const created = await api.trainPipeline(params)
      if (!mounted.current) return
      submitted.current = params
      resetTransient()
      // 未成功的步骤置 running；已 success 的步骤保留——pipeline 幂等重跑这些阶段
      // 通常快速通过，保留它们让失败时的「第一个非 success」推断更贴近真实失败阶段
      setStepStates((prev) => {
        const next = { ...prev }
        for (const s of STEP_IDS) {
          if (next[s] !== 'success') next[s] = 'running'
        }
        return next
      })
      setWatched({ step: 'pipeline', taskId: created.task_id })
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    } finally {
      if (mounted.current) setBusyStep(null)
    }
  }

  async function stopTask() {
    if (watched === null) return
    try {
      await api.cancelTask(watched.taskId)
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    }
  }

  // 任务终态 → 步骤状态落定。用 render 期条件更新（React「Adjusting state when a
  // prop changes」模式）而不是 effect：避免 effect 内同步 setState 的级联渲染
  // （oxlint react/set-state-in-effect），语义上等价于「每个任务只落定一次」。
  // 一键失败时从 error 的「第 N 步」+ cmds 数组解析真实失败步骤，解析不到回退推断
  const [settledTaskId, setSettledTaskId] = useState<string | null>(null)
  if (watched !== null && task.terminal && task.status !== null && settledTaskId !== watched.taskId) {
    setSettledTaskId(watched.taskId)
    const failedStep =
      task.status === 'failed' ? resolveFailedStep(task.error, task.cmds, null) : null
    setStepStates(settleSteps(stepStates, watched.step, task.status, failedStep))
  }

  // -- 完成区：产物确认 ------------------------------------------------------
  const indexDone = stepStates.index === 'success'
  // 状态更新全部放在 Promise 回调里：本函数会被完成区 effect 直接调用，
  // 同步 setState 会触发 set-state-in-effect 告警；「正在确认产物…」由
  // product/productError 双空派生，不再需要独立的 loading 态
  const refreshProduct = useCallback(() => {
    const exp = submitted.current?.exp_name
    if (exp === undefined) return
    void api
      .models()
      .then((list) => {
        if (!mounted.current) return
        setProduct(pickProductName(list, exp))
        setProductError(null)
      })
      .catch((e: unknown) => {
        if (mounted.current) setProductError(errorMessage(e))
      })
  }, [])

  useEffect(() => {
    if (indexDone) refreshProduct()
  }, [indexDone, refreshProduct])

  // -- 渲染 ----------------------------------------------------------------
  const expNameHint =
    expName.length > 0 && !expNameValid
      ? '实验名不能为空，且不得含空格、引号、反斜杠、$、反引号或路径分隔符'
      : null
  const datasetHint =
    datasetDir.length > 0 && !datasetValid
      ? '路径不得含引号、$、反引号、换行，且不能以反斜杠结尾'
      : null
  const epochsHint = !epochsValid
    ? totalEpoch < 1 || saveEveryEpoch < 1
      ? '总轮次与保存间隔都必须至少为 1'
      : 'batch size 必须至少为 1'
    : null

  function cellState(key: StepId | 'config'): StepState {
    if (key === 'config') return formValid ? 'success' : 'idle'
    return stepStates[key]
  }

  /** 分步按钮可用性：前一步成功才解锁；失败步骤自身可重试（前置仍为 success） */
  function stepEnabled(step: StepId): boolean {
    if (!formValid || running || busy) return false
    const idx = STEP_IDS.indexOf(step)
    return idx === 0 || stepStates[STEP_IDS[idx - 1]] === 'success'
  }

  const taskFailed = task.terminal && task.status !== 'success'
  const taskError = taskFailed
    ? (task.error ??
      (task.status === 'cancelled' ? '任务已停止，可在对应步骤重新发起' : '任务失败，原因未知'))
    : null
  // 「重试该步」目标：分步失败重试该步；一键失败重试推断出的第一个失败/停止步骤
  // （复用 startStep 的分步机制，其后步骤保持锁定直到重试成功）
  const retryStep: StepId | null = !taskFailed
    ? null
    : watched !== null && watched.step !== 'pipeline'
      ? (watched.step as StepId)
      : pickRetryStep(stepStates)
  const progressPct = task.progress === null ? null : Math.round(task.progress * 100)
  const showLoss = watched !== null && (watched.step === 'fit' || watched.step === 'pipeline')

  return (
    <Card>
      <CardHeader>
        <CardTitle>训练</CardTitle>
        <CardDescription>
          准备一个含人声 wav 的文件夹路径，配置实验后即可一键训练或分步执行。
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {/* 步骤条 */}
        <ol className="flex flex-wrap items-center gap-x-3 gap-y-2">
          {STEP_CELLS.map((cell, i) => {
            const state = cellState(cell.key)
            return (
              <li
                key={cell.key}
                className="inline-flex items-center gap-1.5 text-sm"
                aria-current={state === 'running' ? 'step' : undefined}
              >
                <StepBadge state={state} index={i + 1} />
                <span
                  className={
                    state === 'idle' ? 'text-muted-foreground' : 'font-medium text-foreground'
                  }
                >
                  {cell.label}
                </span>
                {i < STEP_CELLS.length - 1 && (
                  <span className="ms-1.5 text-muted-foreground" aria-hidden>
                    →
                  </span>
                )}
              </li>
            )
          })}
        </ol>

        {/* 表单：基础 */}
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="train-exp" className="text-sm font-medium">
              实验名
            </label>
            <Input
              id="train-exp"
              value={expName}
              onChange={(e) => setExpName(e.target.value)}
              placeholder="如 my-voice"
              aria-invalid={expNameHint !== null}
            />
            {expNameHint !== null && <p className="text-xs text-destructive">{expNameHint}</p>}
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor="train-dataset" className="text-sm font-medium">
              数据集路径
            </label>
            <Input
              id="train-dataset"
              value={datasetDir}
              onChange={(e) => setDatasetDir(e.target.value)}
              placeholder="服务器上的目录路径，如 /data/dataset"
              aria-invalid={datasetHint !== null}
            />
            {datasetHint !== null ? (
              <p className="text-xs text-destructive">{datasetHint}</p>
            ) : (
              <p className="text-xs text-muted-foreground">
                服务器上的目录路径（浏览器无法选择远端目录）；目录内放要训练的人声 wav
              </p>
            )}
          </div>
        </div>

        {/* 空状态引导 */}
        {datasetDir.length === 0 && (
          <p className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
            还没有数据集：请先在服务器上准备一个文件夹，里面放若干人声 wav（建议 2
            分钟以上、无伴奏无混响），把文件夹路径填到上方即可开始。
          </p>
        )}

        {/* 表单：高级参数（折叠） */}
        <Collapsible>
          <CollapsibleTrigger className={COLLAPSIBLE_TRIGGER_CLASS}>
            高级参数
            <ChevronDownIcon className="size-4 text-muted-foreground" />
          </CollapsibleTrigger>
          <CollapsibleContent className="grid gap-4 pt-4 sm:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">采样率</span>
              <Select
                items={SR_ITEMS}
                value={sr}
                onValueChange={(v) => applySelect(v, (value) => setSr(value as SampleRate))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="40k">40k（推荐）</SelectItem>
                  <SelectItem value="48k">48k</SelectItem>
                  <SelectItem value="32k">32k</SelectItem>
                </SelectContent>
              </Select>
              {normalizedSr !== sr && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  v1 没有 32k 档，训练时将自动使用 40k
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">提取音高（f0）</span>
              <Select
                items={IF_F0_ITEMS}
                value={ifF0 ? '1' : '0'}
                onValueChange={(v) => applySelect(v, (value) => setIfF0(value === '1'))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">开启（歌声/变调推荐）</SelectItem>
                  <SelectItem value="0">关闭（说话声更快）</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">音高算法</span>
              <Select
                items={F0_METHOD_ITEMS}
                value={f0Method}
                disabled={!ifF0}
                onValueChange={(v) => applySelect(v, (value) => setF0Method(value as TrainF0Method))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rmvpe">rmvpe（推荐）</SelectItem>
                  <SelectItem value="pm">pm（最快，质量一般）</SelectItem>
                </SelectContent>
              </Select>
              {!ifF0 && <p className="text-xs text-muted-foreground">已关闭音高提取，无需选择</p>}
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">模型版本</span>
              <Select
                items={VERSION_ITEMS}
                value={version}
                onValueChange={(v) => applySelect(v, (value) => setVersion(value as ModelVersion))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="v2">v2（推荐）</SelectItem>
                  <SelectItem value="v1">v1（兼容旧模型）</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <NumberField
              id="train-total-epoch"
              label="总轮次（total_epoch）"
              value={totalEpoch}
              onChange={(value) => setTotalEpoch(value ?? 0)}
            />
            <NumberField
              id="train-save-epoch"
              label="保存间隔（save_every_epoch）"
              value={saveEveryEpoch}
              onChange={(value) => setSaveEveryEpoch(value ?? 0)}
              hint="每多少轮保存一次检查点"
            />
            <NumberField
              id="train-batch-size"
              label="batch size"
              value={batchSize}
              onChange={(value) => {
                batchSizeTouched.current = true
                setBatchSize(value)
              }}
              placeholder="自动"
              hint="按设备默认：显卡=显存GB÷2，CPU=1；显存不足就调小"
            />
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">逐轮保存权重</span>
              <Select
                items={SAVE_WEIGHTS_ITEMS}
                value={saveEveryWeights ? '1' : '0'}
                onValueChange={(v) => applySelect(v, (value) => setSaveEveryWeights(value === '1'))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">关闭（只保存最终模型）</SelectItem>
                  <SelectItem value="1">开启（每个保存间隔另存一份）</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {epochsHint !== null && <p className="text-xs text-destructive">{epochsHint}</p>}

        {/* 启动按钮：一键 + 分步 */}
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => void startPipeline()} disabled={!formValid || running || busy}>
              {busyStep === 'pipeline' && <LoaderCircleIcon className="animate-spin" />}
              <PlayIcon />
              一键训练
            </Button>
            <span className="text-xs text-muted-foreground">从数据切分到建立索引一次跑完</span>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">分步执行</span>
            <div className="grid gap-2 sm:grid-cols-4">
              {STEP_IDS.map((step) => {
                const state = stepStates[step]
                return (
                  <Button
                    key={step}
                    variant="outline"
                    onClick={() => void startStep(step)}
                    disabled={!stepEnabled(step)}
                  >
                    {busyStep === step && <LoaderCircleIcon className="animate-spin" />}
                    {state === 'failed' || state === 'skipped'
                      ? `重试${STEP_TITLES[step]}`
                      : STEP_ACTIONS[step]}
                  </Button>
                )
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              按顺序执行：处理数据 → 特征提取 → 训练 → 建立索引；前一步成功后下一步解锁。
            </p>
          </div>
        </div>

        {/* 提交错误（400/409/网络失败等） */}
        {submitError !== null && <ErrorDetail title="任务创建失败">{submitError}</ErrorDetail>}

        {/* 任务监视区 */}
        {watched !== null && (
          <div className="flex flex-col gap-3 rounded-lg border p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">{STEP_TITLES[watched.step]}</span>
                {running && (
                  <span className="inline-flex items-center gap-1 text-xs text-primary">
                    <LoaderCircleIcon className="size-3.5 animate-spin" />
                    进行中
                  </span>
                )}
                {task.status === 'success' && task.terminal && (
                  <span className="text-xs text-emerald-600 dark:text-emerald-400">成功</span>
                )}
                {taskFailed && (
                  <span className="text-xs text-destructive">
                    {task.status === 'cancelled' ? '已停止' : '失败'}
                  </span>
                )}
              </div>
              {running && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => {
                    if (!confirmStop) {
                      setConfirmStop(true)
                      // 3s 未确认自动还原，避免误触后一直停留在武装态（卸载后
                      // 的迟到回调被 React 忽略，无需清理）
                      window.setTimeout(() => setConfirmStop(false), 3000)
                      return
                    }
                    setConfirmStop(false)
                    void stopTask()
                  }}
                >
                  {confirmStop ? '确认停止？' : '停止'}
                </Button>
              )}
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>进度</span>
                <span className="tabular-nums">
                  {progressPct === null ? '等待进度…' : `${progressPct}%`}
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
              {task.current !== null && (
                <p
                  className="truncate font-mono text-xs text-muted-foreground"
                  title={task.current}
                >
                  {task.current}
                </p>
              )}
            </div>

            {task.connectionLost && !task.terminal && (
              <div
                role="alert"
                className="flex flex-wrap items-center gap-2 rounded-lg bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-400"
              >
                <span className="flex-1">
                  连接已断开，任务仍在后台运行；点击重新连接恢复进度与日志监视。
                </span>
                <Button variant="outline" size="sm" onClick={task.resubscribe}>
                  重新连接
                </Button>
              </div>
            )}

            {showLoss && <LossChart points={task.losses} />}
            <LogPanel lines={task.logs} />

            {taskError !== null && (
              <div className="flex flex-col gap-1">
                <ErrorDetail title={task.status === 'cancelled' ? '任务已停止' : '任务失败'}>
                  {taskError}
                </ErrorDetail>
                {retryStep !== null && (
                  <div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void startStep(retryStep)}
                    >
                      重试该步（{STEP_TITLES[retryStep]}）
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* 完成区：产物 + 去试音 */}
        {indexDone && (
          <div className="flex flex-col gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-4">
            <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
              训练完成
            </span>
            {productError !== null ? (
              <p role="alert" className="text-xs text-destructive">
                产物确认失败：{productError}
              </p>
            ) : product === null ? (
              <p className="text-xs text-muted-foreground">正在确认产物…</p>
            ) : (
              <p className="text-xs text-muted-foreground">
                模型 <span className="font-mono text-foreground">{product}</span> 已就绪
              </p>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={() => {
                  if (product !== null) onGoInfer(product)
                }}
                disabled={product === null}
              >
                去试音
              </Button>
              <Button variant="ghost" size="sm" onClick={() => refreshProduct()}>
                重新检查产物
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
