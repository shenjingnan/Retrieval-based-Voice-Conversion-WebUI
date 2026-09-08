/**
 * 训练向导页（P2）：步骤条 ① 实验配置 → ② 处理数据 → ③ 特征提取 → ④ 训练 → ⑤ 完成。
 * 两种执行方式：一键（trainPipeline，单任务串行 5 段命令）与分步（4 个任务按钮，
 * 前一步成功后解锁下一步，失败可重发同参数请求）。任务监视区由 useTask 驱动：
 * 进度条 + 当前文件 + 日志滚动区 + loss 曲线（内联 SVG，不引图表库）。
 */
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type MouseEvent as ReactMouseEvent } from 'react'
import {
  CheckIcon,
  ChevronDownIcon,
  CopyIcon,
  LoaderCircleIcon,
  PauseIcon,
  PlayIcon,
  XIcon,
} from 'lucide-react'

import { api, type ModelVersion, type SampleRate, type TaskCreated, type TrainF0Method, type TrainParams } from '@/api/client'
import { DatasetPicker } from '@/components/DatasetPicker'
import { useTask, type LossPoint } from '@/hooks/useTask'
import { pickProductName, randomHex } from '@/lib/domain'
import { errorMessage } from '@/lib/utils'
import { ErrorDetail } from '@/components/ErrorDetail'
import {
  advancePipelineStages,
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

/** 日志滚动区：等宽 + 自动滚底，用户上滚时暂停（回到底部恢复）；右上角悬浮复制
 *  按钮（复制全部可见日志，成功后短暂变 ✓） */
function LogPanel({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement | null>(null)
  const stick = useRef(true)
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (el === null || !stick.current) return
    el.scrollTop = el.scrollHeight
  }, [lines])

  async function copyAll() {
    const text = lines.join('\n')
    try {
      if (navigator.clipboard !== undefined) {
        await navigator.clipboard.writeText(text)
      } else {
        // 非 secure context（局域网 http 访问）没有 async Clipboard API：
        // 退回隐藏 textarea + execCommand（已废弃但在所有浏览器仍可用）
        const ta = document.createElement('textarea')
        ta.value = text
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        ta.remove()
      }
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // 复制失败（权限拒绝等）：不弹错误打断看日志的心流，按钮原样保留可重试
    }
  }

  return (
    <div className="relative">
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
      <Button
        variant="outline"
        size="icon-sm"
        className="absolute right-2 top-2 bg-background/80 backdrop-blur"
        disabled={lines.length === 0}
        aria-label="复制日志"
        title="复制日志"
        onClick={() => void copyAll()}
      >
        {copied ? <CheckIcon className="text-emerald-600" /> : <CopyIcon />}
      </Button>
    </div>
  )
}

/** loss 曲线：内联 SVG 双折线 + 悬浮十字线提示（第几个点 / 所属轮次 / 两条 loss 值）。
 *  高亮点与提示用 HTML 绝对定位而不是 SVG 图元——svg 的 preserveAspectRatio=none
 *  会在横向拉伸时把圆点变成椭圆；位置按 viewBox 百分比换算，与鼠标横坐标一致 */
function LossChart({ points }: { points: LossPoint[] }) {
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
  // 参考音频唯一来源：DatasetPicker（下拉选已有 / 浏览器上传），路径由后端下发
  const [pickedPath, setPickedPath] = useState('')
  // 向导其余逻辑只认 dataset_dir 字符串
  const datasetDir = pickedPath
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
  // 实验名对用户隐藏（第一阶段）：不校验、不展示，提交前自动生成（见 currentParams）
  // 路径由后端下发（参考音频目录的绝对路径），天然合法，不跑字符集正则；只要求已选择
  const datasetValid = pickedPath.length > 0
  const epochsValid =
    totalEpoch >= 1 && saveEveryEpoch >= 1 && (batchSize === null || batchSize >= 1)
  const formValid = datasetValid && epochsValid

  // v1 没有 32k 档（server/api/training.py _normalize_sr，webui change_version19 语义）。
  // 分步模式下 preprocess 与 fit 必须用同一采样率，前端在提交前统一归一化
  const normalizedSr: SampleRate = version === 'v1' && sr === '32k' ? '40k' : sr

  function currentParams(): TrainParams {
    // 实验名隐藏（第一阶段）：首次提交时生成 voice-月日时分-随机后缀 并记进 state。
    // 必须记住而非每次现生成——分步模式的 4 个接口各自调一次 currentParams，若每次
    // 随机会得到不同实验名，流程直接错乱（preprocess 写 logs/A，extract 读 logs/B）。
    // 时间因子保证新实验不会续进旧目录（train.py 会从实验目录里已有的 G/D 权重自动
    // 续训），随机后缀兜掉同分钟内重开的碰撞
    let resolvedExpName = expName
    if (resolvedExpName.length === 0) {
      const now = new Date()
      const pad = (n: number) => String(n).padStart(2, '0')
      resolvedExpName = `voice-${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}-${randomHex(4)}`
      setExpName(resolvedExpName)
    }
    return {
      exp_name: resolvedExpName,
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
  /** 参考音频上传队列在途（DatasetPicker 回报）：期间禁止启动训练 */
  const [uploadBusy, setUploadBusy] = useState(false)

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

  // pipeline 运行中：按服务端「当前子命令」实时推进步骤条（render 期条件更新，与
  // 上面的终态落定同一模式）。启动时乐观置态让全部步骤一起 loading；这里把已过
  // 阶段校正回 success、未到阶段校正回 idle。每对 (taskId, currentCmd) 只应用一次：
  // cmd 数组身份随每个 SSE 事件重建，靠 stagedCmdKey 挡住重复 setState
  const [stagedCmdKey, setStagedCmdKey] = useState<string | null>(null)
  if (
    watched !== null &&
    watched.step === 'pipeline' &&
    !task.terminal &&
    task.current_cmd !== null &&
    task.cmds.length > 0
  ) {
    const key = `${watched.taskId}:${task.current_cmd}`
    if (stagedCmdKey !== key) {
      const next = advancePipelineStages(task.current_cmd, task.cmds)
      if (next !== null && STEP_IDS.some((s) => next[s] !== stepStates[s])) {
        setStagedCmdKey(key)
        setStepStates(next)
      }
    }
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
  const epochsHint = !epochsValid
    ? totalEpoch < 1 || saveEveryEpoch < 1
      ? '总轮次与保存间隔都必须至少为 1'
      : 'batch size 必须至少为 1'
    : null

  function cellState(key: StepId | 'config'): StepState {
    if (key === 'config') return formValid ? 'success' : 'idle'
    return stepStates[key]
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

        {/* 表单：基础。实验名对用户隐藏（第一阶段）：提交时自动生成随机标识，
            训练完成后在完成区 / 模型管理页以产物名（{实验名}.pth）可见 */}
        <div className="flex flex-col gap-4">
          <span className="text-sm font-medium">参考音频</span>

          <DatasetPicker
            onSelect={(ds) => setPickedPath(ds === null ? '' : ds.path)}
            onUploaded={(r) => {
              // 上传成功即把后端下发的绝对路径回填 dataset_dir：表单立即可提交
              setPickedPath(r.path)
            }}
            onUploadBusyChange={setUploadBusy}
            trainingRunning={running}
          />
        </div>

        {epochsHint !== null && <p className="text-xs text-destructive">{epochsHint}</p>}

        {/* 启动区：一键训练（高级设置收在其旁）+ 分步 */}
        <div className="flex flex-col gap-3">
          <Collapsible>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void startPipeline()}
                disabled={!formValid || running || busy || uploadBusy}
                title={uploadBusy ? '参考音频还在上传中，等全部完成再开始训练' : undefined}
              >
                {busyStep === 'pipeline' && <LoaderCircleIcon className="animate-spin" />}
                <PlayIcon />
                一键训练
              </Button>
              <CollapsibleTrigger className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm text-muted-foreground select-none hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none [&_svg]:transition-transform [&[aria-expanded=true]_svg]:rotate-180">
                高级设置
                <ChevronDownIcon className="size-4" />
              </CollapsibleTrigger>
            </div>
            <CollapsibleContent>
              <div className="grid gap-4 pt-4 sm:grid-cols-2">
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
              </div>
            </CollapsibleContent>
          </Collapsible>

          {epochsHint !== null && <p className="text-xs text-destructive">{epochsHint}</p>}
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
