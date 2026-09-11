/**
 * 训练页（任务中心化，docs/plans/2026-09-09-task-centric-training-ui-design.md）：
 * 页面只做「提交器」——配置表单 + 一键提交；提交成功即全部重置表单（音频与高级
 * 设置回默认，DatasetPicker remount 开新上传会话），用户可立即配置下一个任务。
 * 所有任务（运行中 / 排队中 / 历史终态）在下方任务队列面板中独立展示，展开可看
 * 阶段指示、进度、loss 图、日志与产物；失败/停止的任务可原参数重新提交（同名
 * 实验从最近检查点续训）。
 */
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react'
import { ChevronDownIcon, LoaderCircleIcon, PlayIcon } from 'lucide-react'

import {
  api,
  type ModelVersion,
  type SampleRate,
  type TaskHistoryDetail,
  type TaskSummary,
  type TrainF0Method,
  type TrainParams,
} from '@/api/client'
import { DatasetPicker } from '@/components/DatasetPicker'
import { ErrorDetail } from '@/components/ErrorDetail'
import { SystemResourcePanel } from '@/components/SystemResourcePanel'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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
import { TaskDetail } from '@/components/training/TaskDetail'
import { TaskQueuePanel } from '@/components/training/TaskQueuePanel'
import { useSystemStats } from '@/hooks/useSystemStats'
import { useTask } from '@/hooks/useTask'
import { randomHex } from '@/lib/domain'
import { errorMessage } from '@/lib/utils'

/**
 * Base UI 的 Select.Value 默认渲染原始 value（Radix 是渲染选中项文案），
 * value 与文案不一致的下拉框必须给 Root 传 items 做映射，否则触发器显示 "1"/"0"
 */
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

/** Base UI 的 onValueChange 可能给 null（清空态），本页选项都必选，null 时忽略 */
function applySelect(v: string | null, set: (value: string) => void): void {
  if (v !== null) set(v)
}

/**
 * 自动实验名：voice-月日时分-随机后缀。时间因子保证新实验不续进旧目录，
 * 随机后缀兜掉同分钟内重开的碰撞。
 */
function autoExpName(): string {
  const now = new Date()
  const pad = (n: number) => String(n).padStart(2, '0')
  return `voice-${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}-${randomHex(4)}`
}

/**
 * 镜像 server/api/training.py _check_exp_name 的字符规则（空串合法 = 自动生成，
 * 由调用方先行放行；NUL 输入框打不出不查）。true = 含空白/引号/反斜杠/$/反引号/
 * 路径分隔符，或恰为 "." / ".."。规则改动须与后端同步。
 */
function expNameInvalid(name: string): boolean {
  return (
    /[ \t\r\n"\\$`]/.test(name) || name.includes('/') || name === '.' || name === '..'
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
  // 实验名可选：留空自动生成；填写则用用户的（冲突由输入时检查 + 后端 409 兜底拦截）
  const [expName, setExpName] = useState('')
  /** 占用检查结果（与查询时的名字绑定）：渲染时按当前名字派生占用态，名字一变提示即消失 */
  const [expNameCheck, setExpNameCheck] = useState<{ name: string; exists: boolean } | null>(
    null,
  )
  // 参考音频唯一来源：DatasetPicker（上传会话），路径由后端下发
  const [pickedPath, setPickedPath] = useState('')
  // 提交成功后整组重置：remount DatasetPicker 才能丢弃旧上传会话（ds-xxxxxxxx）
  const [pickerKey, setPickerKey] = useState(0)
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

  // -- 任务编排（页面只提交，不再持有单一「被监视任务」） ----------------------
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  /** 参考音频上传队列在途（DatasetPicker 回报）：期间禁止启动训练 */
  const [uploadBusy, setUploadBusy] = useState(false)

  // -- 任务队列（3s 轮询；运行中/排队中/历史在面板内独立展示） ----------------
  const [queueTasks, setQueueTasks] = useState<TaskSummary[] | null>(null)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const refreshQueue = useCallback(() => {
    void api
      .getTasks()
      .then((list) => {
        if (mounted.current) setQueueTasks(list)
      })
      .catch(() => undefined) // 轮询失败静默保留旧值（面板是辅助视图）
  }, [])

  useEffect(() => {
    refreshQueue()
    const timer = window.setInterval(refreshQueue, 3000)
    return () => window.clearInterval(timer)
  }, [refreshQueue])

  // 手风琴：单开。live 订阅只属于展开的内存任务；历史行绝不挂 SSE（404 会误报连接丢失）
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const expandedRow = queueTasks?.find((task) => task.id === expandedId) ?? null
  const live = useTask(
    expandedRow !== null && !expandedRow.history && expandedId !== null ? expandedId : null,
  )
  // 历史详情缓存（展开历史行时拉一次）；失败集合防止轮询周期内的重试风暴；
  // detailLoading 为派生值（展开中的历史行且尚无缓存）——不在 effect 内同步 setState
  const [historyDetails, setHistoryDetails] = useState<Record<string, TaskHistoryDetail>>({})
  const [detailFailedIds, setDetailFailedIds] = useState<ReadonlySet<string>>(new Set())

  useEffect(() => {
    if (expandedId === null) return
    if (historyDetails[expandedId] !== undefined || detailFailedIds.has(expandedId)) return
    const row = queueTasks?.find((task) => task.id === expandedId)
    if (row === undefined || !row.history) return
    let cancelled = false
    api
      .taskDetail(expandedId)
      .then((detail) => {
        if (cancelled || !mounted.current) return
        setHistoryDetails((prev) => ({ ...prev, [expandedId]: detail }))
      })
      .catch((e: unknown) => {
        setDetailFailedIds((prev) => new Set(prev).add(expandedId))
        if (!cancelled && mounted.current) setSubmitError(errorMessage(e))
      })
    return () => {
      cancelled = true
    }
  }, [expandedId, queueTasks, historyDetails, detailFailedIds])

  const detailLoading =
    expandedRow !== null &&
    expandedRow.history &&
    historyDetails[expandedRow.id] === undefined &&
    !detailFailedIds.has(expandedRow.id)

  const [resubmitBusy, setResubmitBusy] = useState(false)

  // -- 系统资源（常驻轮询，与任务无关；失败在 hook 内静默降级） ----------------
  const { stats: sysStats, stale: sysStale } = useSystemStats()

  // -- 校验（派生） ---------------------------------------------------------
  // 实验名：本地格式校验即时反馈（不发非法名请求，后端同一张表兜 400）；
  // 路径由后端下发，天然合法，只要求已选择
  const expNameErr =
    expName.length > 0 && expNameInvalid(expName)
      ? '实验名不能包含空格、引号、$、`、反斜杠或路径分隔符'
      : null
  const expNameTaken =
    expNameCheck !== null && expNameCheck.name === expName && expNameCheck.exists
  const datasetValid = pickedPath.length > 0
  const epochsValid =
    totalEpoch >= 1 && saveEveryEpoch >= 1 && (batchSize === null || batchSize >= 1)
  const formValid = datasetValid && epochsValid && expNameErr === null && !expNameTaken

  // 实验名占用检查：输入停顿 400ms 后查询（防抖）。结果与查询时的名字绑定写入，
  // 迟到响应由 ref 比对丢弃（不覆盖新名字的检查结果）；检查失败静默——提交时
  // 后端 409 兜底，不阻塞表单
  const expNameRef = useRef('')
  useEffect(() => {
    expNameRef.current = expName
    if (expName.length === 0 || expNameInvalid(expName)) return // 空名自动生成；非法名不发请求
    const name = expName
    const timer = window.setTimeout(() => {
      api
        .trainExpNameExists(name)
        .then((r) => {
          if (expNameRef.current === name) setExpNameCheck({ name, exists: r.exists })
        })
        .catch(() => undefined)
    }, 400)
    return () => window.clearTimeout(timer)
  }, [expName])

  // v1 没有 32k 档（server/api/training.py _normalize_sr，webui change_version19 语义）
  const normalizedSr: SampleRate = version === 'v1' && sr === '32k' ? '40k' : sr

  // 挂载时把设备自适应默认值预填进 batch size（对齐 webui 在滑条上预填解析值的行为）；
  // resetForm 复用同一函数
  const prefillBatchSize = useCallback(() => {
    void api
      .trainDefaults()
      .then(({ batch_size }) => {
        if (!mounted.current || batchSizeTouched.current) return
        setBatchSize(batch_size)
      })
      .catch(() => undefined) // 预填失败不阻塞页面：自动语义不依赖该请求
  }, [])
  useEffect(() => {
    prefillBatchSize()
  }, [prefillBatchSize])

  function currentParams(): TrainParams {
    // 实验名可选：填了用用户的；留空则每次提交现场生成全新自动名（排队开放后同一
    // 页面会连续提交多个任务，复用实验名会触发同名互斥/目录冲突）。不写回 state：
    // 提交失败重试自然换新名，不会反复撞同一个 409
    const resolvedExpName = expName.length > 0 ? expName : autoExpName()
    return {
      exp_name: resolvedExpName,
      dataset_dir: pickedPath,
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

  /** 提交成功后的整组重置：新任务 = 全新配置，避免上一任务的参数无意带进下一个 */
  function resetForm() {
    setPickedPath('')
    setSr('40k')
    setIfF0(true)
    setF0Method('rmvpe')
    setVersion('v2')
    setTotalEpoch(20)
    setSaveEveryEpoch(5)
    setBatchSize(null)
    batchSizeTouched.current = false // 必须与 setBatchSize(null) 同步，否则预填被 touched 短路
    setSaveEveryWeights(false)
    setExpName('')
    setPickerKey((k) => k + 1) // remount DatasetPicker：丢弃旧上传会话
    prefillBatchSize()
  }

  async function startPipeline() {
    const params = currentParams()
    setSubmitting(true)
    setSubmitError(null)
    try {
      const created = await api.trainPipeline(params)
      if (!mounted.current) return
      setExpandedId(created.task_id) // 提交即展开新任务（延续「提交即监视」体验）
      resetForm()
      refreshQueue()
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    } finally {
      if (mounted.current) setSubmitting(false)
    }
  }

  async function stopQueueTask(taskId: string) {
    try {
      await api.cancelTask(taskId)
      refreshQueue()
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    }
  }

  async function clearQueue() {
    try {
      await api.clearQueue()
      refreshQueue()
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    }
  }

  /** 失败/停止任务的「重新提交」：同名实验 = 断点续训；不重置表单（与本任务无关）。
   *  失败任务的实验目录必然已存在，显式带 allow_existing 绕过占用检查（409） */
  async function resubmitTask(params: Record<string, unknown>) {
    setResubmitBusy(true)
    setSubmitError(null)
    try {
      const created = await api.trainPipeline({
        ...params,
        allow_existing: true,
      } as unknown as TrainParams)
      if (!mounted.current) return
      setExpandedId(created.task_id)
      refreshQueue()
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    } finally {
      if (mounted.current) setResubmitBusy(false)
    }
  }

  // -- 渲染 ----------------------------------------------------------------
  const epochsHint = !epochsValid
    ? totalEpoch < 1 || saveEveryEpoch < 1
      ? '总轮次与保存间隔都必须至少为 1'
      : 'batch size 必须至少为 1'
    : null

  return (
    <Card>
      <CardHeader>
        <CardTitle>训练</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {/* 表单：实验名（可选）。占用/非法在输入时即时提示，提交时后端 409/400 兜底 */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="train-exp-name" className="text-sm font-medium">
            实验名<span className="font-normal text-muted-foreground">（可选）</span>
          </label>
          <Input
            id="train-exp-name"
            type="text"
            value={expName}
            onChange={(e) => setExpName(e.target.value)}
            placeholder="留空自动生成，如 voice-0911-1430-a1b2"
            autoComplete="off"
          />
          {expNameErr !== null ? (
            <p className="text-xs text-destructive">{expNameErr}</p>
          ) : expNameTaken ? (
            <p className="text-xs text-destructive">
              该实验名已存在，请换一个；要基于已有产物续训，请在下方任务历史中使用「重新提交」
            </p>
          ) : null}
        </div>

        {/* 表单：参考音频。提交成功后整组重置（pickerKey remount 开新上传会话） */}
        <div className="flex flex-col gap-4">
          <span className="text-sm font-medium">参考音频</span>
          <DatasetPicker
            key={pickerKey}
            onSelect={(ds) => setPickedPath(ds === null ? '' : ds.path)}
            onUploaded={(r) => {
              // 上传成功即把后端下发的绝对路径回填 dataset_dir：表单立即可提交
              setPickedPath(r.path)
            }}
            onUploadBusyChange={setUploadBusy}
          />
        </div>

        {epochsHint !== null && <p className="text-xs text-destructive">{epochsHint}</p>}

        {/* 启动区：一键训练（高级设置收在其旁）。任务运行中不锁提交——新任务自动排队 */}
        <div className="flex flex-col gap-3">
          <Collapsible>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void startPipeline()}
                disabled={!formValid || submitting || uploadBusy}
                title={
                  uploadBusy
                    ? '参考音频还在上传中，等全部完成再开始训练'
                    : queueTasks?.some(
                          (t) => t.state === 'running' || t.state === 'pending',
                        ) ?? false
                      ? '当前有任务在队列中，新任务将自动排队，完成后按序执行'
                      : undefined
                }
              >
                {submitting && <LoaderCircleIcon className="animate-spin" />}
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
        {submitError !== null && <ErrorDetail title="任务操作失败">{submitError}</ErrorDetail>}

        {/* 系统资源（常驻；GPU 不可用不报错） */}
        <SystemResourcePanel stats={sysStats} stale={sysStale} />

        {/* 任务队列：运行中/排队中/历史的全局视图，展开项看详情 */}
        <TaskQueuePanel
          tasks={queueTasks}
          expandedId={expandedId}
          onToggle={(taskId) => setExpandedId((prev) => (prev === taskId ? null : taskId))}
          onStopTask={(taskId) => void stopQueueTask(taskId)}
          onClearQueue={() => void clearQueue()}
          renderDetail={(task) => (
            <TaskDetail
              row={task}
              live={task.history ? null : live}
              detail={historyDetails[task.id] ?? null}
              detailLoading={detailLoading}
              resubmitBusy={resubmitBusy}
              onResubmit={(params) => void resubmitTask(params)}
              onGoInfer={onGoInfer}
            />
          )}
        />
      </CardContent>
    </Card>
  )
}
