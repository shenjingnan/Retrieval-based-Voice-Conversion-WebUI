/**
 * 模型管理页（P2）：assets/weights 的模型卡片列表，按实验名聚合——一次训练的
 * 最终模型是组代表项，save_every_weights 存下的中间轮次收进「中间轮次模型（N）」
 * 折叠子列表（groupModels 分组，见 lib/domain）；组卡片按最近训练时间倒序。
 * 代表项操作：去推理（跨 Tab 联动选中）、下载（一键 zip 打包 pth + 配对索引）、
 * 补训索引（实验名默认由模型名推导，可编辑；同组共享一个索引，故为组级操作）、
 * 删除（两段式确认；彻底删除：整组权重 + 配对索引 + logs/{exp} 训练产物）。
 * 中间轮次子项操作：去推理 / 下载（删除是组级操作，只在组卡片上提供）。
 * 补训索引走 POST /api/train/index + useTask 显示进度与结果。
 */
import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import {
  ChevronDownIcon,
  DownloadIcon,
  LoaderCircleIcon,
  RefreshCwIcon,
  UploadIcon,
} from 'lucide-react'

import { api, type ModelVersion, type RvcModel } from '@/api/client'
import { useTask } from '@/hooks/useTask'
import { EXP_NAME_RE, experimentName, groupModels, parseEpochSuffix } from '@/lib/domain'
import { errorMessage } from '@/lib/utils'
import { ErrorDetail } from '@/components/ErrorDetail'
import { ModelUploadForm } from '@/components/ModelUploadForm'
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

/**
 * 补训索引的内联表单：实验名可编辑（默认由模型名推导），版本默认 v2 可改。
 * groupKey 是所属组的键（Models 页以组为卡片单位），不再指向单个模型文件
 */
interface IndexFormState {
  groupKey: string
  expName: string
  version: ModelVersion
}

/** 正在监视的补训索引任务（挂在对应组的卡片下） */
interface WatchedIndexTask {
  groupKey: string
  taskId: string
}

/** 删除成功的回执提示：整组权重 + 配对索引 + logs/{exp} 训练产物的清理结果 */
interface DeletedReceipt {
  model: string
  modelCount: number
  failedModelCount: number
  indexCount: number
  failedCount: number
  logsRemoved: boolean
  logsTarget: string | null
  logsFailedCount: number
}

/** Base UI Select 的 onValueChange 可能给 null（清空态），本页选项都必选，null 时忽略 */
function applySelect(v: string | null, set: (value: string) => void): void {
  if (v !== null) set(v)
}

/** 索引字段存的是绝对路径，卡片里只展示文件名 */
function fileName(path: string): string {
  return path.split('/').pop() ?? path
}

/** 回执里 logs 清理结果的文案：本来就没有产物（target 为 null）不提，未清干净给残留提示 */
function logsReceiptText(receipt: DeletedReceipt): string {
  if (receipt.logsRemoved) return '；训练产物已清除'
  if (receipt.logsTarget === null) return ''
  return `；训练产物未能完全清除（残留 ${receipt.logsFailedCount} 个文件，可手动删除 ${receipt.logsTarget}）`
}

export interface ModelsPageProps {
  /** 「去推理」：App 切到推理 Tab 并选中该模型（pendingModel 联动） */
  onGoInfer: (model: string) => void
  /** 空状态引导：App 切到训练 Tab */
  onGoTrain: () => void
}

export function ModelsPage({ onGoInfer, onGoTrain }: ModelsPageProps) {
  // null 表示模型列表仍在加载；[] 表示已加载但 assets/weights 为空
  const [models, setModels] = useState<RvcModel[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  // 手动刷新 / 删除成功 / 补训索引成功都通过递增它触发重新拉取
  const [reloadTick, setReloadTick] = useState(0)
  // 导入外部模型表单的展开态；上传成功后表单内自行展示回执，列表靠 reloadTick 重取
  const [uploadOpen, setUploadOpen] = useState(false)

  // -- 删除（两段式确认） ---------------------------------------------------
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [receipt, setReceipt] = useState<DeletedReceipt | null>(null)

  // -- 补训索引 -------------------------------------------------------------
  const [indexForm, setIndexForm] = useState<IndexFormState | null>(null)
  const [submitting, setSubmitting] = useState(false)
  // 创建失败在表单内提示（表单仍开着，错误必须可见）；停止失败发生在表单收起后，
  // 独立状态渲染在任务监视区里
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [stopError, setStopError] = useState<string | null>(null)
  const [watched, setWatched] = useState<WatchedIndexTask | null>(null)
  const task = useTask(watched?.taskId ?? null)

  // 异步回调里的 setState 保护：切 Tab 卸载后 React 19 会静默忽略，这里显式短路
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  // 模型列表加载：setState 全部在 Promise 回调里（避免 effect 期同步 setState）
  useEffect(() => {
    let cancelled = false
    api
      .models()
      .then((list) => {
        if (cancelled) return
        setModels(list)
        setLoadError(null)
      })
      .catch((e: unknown) => {
        if (!cancelled) setLoadError(errorMessage(e))
      })
    return () => {
      cancelled = true
    }
  }, [reloadTick])

  // 补训索引任务成功 → 刷新列表（render 期条件更新，同 Training 页的终态落定模式，
  // 避免 effect 内同步 setState 的级联渲染）；每个任务只刷新一次
  const [settledTaskId, setSettledTaskId] = useState<string | null>(null)
  if (
    watched !== null &&
    task.terminal &&
    task.status === 'success' &&
    settledTaskId !== watched.taskId
  ) {
    setSettledTaskId(watched.taskId)
    setIndexForm(null)
    setReloadTick((t) => t + 1)
  }

  async function remove(name: string) {
    setDeleting(name)
    setDeleteError(null)
    try {
      const result = await api.deleteModel(name)
      if (!mounted.current) return
      setReceipt({
        model: result.deleted_model,
        modelCount: result.deleted_models.length,
        failedModelCount: result.failed_models.length,
        indexCount: result.deleted_indices.length,
        failedCount: result.failed_indices.length,
        logsRemoved: result.logs.removed,
        logsTarget: result.logs.target,
        logsFailedCount: result.logs.failed_files.length,
      })
      // 表单/监视区按组键渲染（indexForm.groupKey === g.key），整组删除后渲染条件
      // 自然不成立，残留状态无害（下次打开会覆盖）
      setReloadTick((t) => t + 1)
    } catch (e) {
      if (mounted.current) setDeleteError(errorMessage(e))
    } finally {
      if (mounted.current) {
        setDeleting(null)
        setConfirmDelete(null)
      }
    }
  }

  function onDeleteClick(name: string) {
    if (confirmDelete !== name) {
      setConfirmDelete(name)
      // 3s 未确认自动还原（同 Training 页停止按钮）；卸载后的迟到回调被 React 忽略
      window.setTimeout(() => setConfirmDelete((cur) => (cur === name ? null : cur)), 3000)
      return
    }
    setConfirmDelete(null)
    void remove(name)
  }

  async function startIndex() {
    if (indexForm === null) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const created = await api.trainIndex({
        exp_name: indexForm.expName,
        version: indexForm.version,
      })
      if (!mounted.current) return
      setWatched({ groupKey: indexForm.groupKey, taskId: created.task_id })
      setIndexForm(null) // 任务已受理，收起表单（进度见下方监视区）
    } catch (e) {
      if (mounted.current) setSubmitError(errorMessage(e))
    } finally {
      if (mounted.current) setSubmitting(false)
    }
  }

  async function stopIndex() {
    if (watched === null) return
    setStopError(null)
    try {
      await api.cancelTask(watched.taskId)
    } catch (e) {
      if (mounted.current) setStopError(errorMessage(e))
    }
  }

  const running = watched !== null && !task.terminal
  const progressPct = task.progress === null ? null : Math.round(task.progress * 100)
  const taskFailed = task.terminal && task.status !== 'success'
  // 平铺列表 → 按实验名聚合（最终模型 + 中间轮次折进同组），展示层分组见 groupModels
  const groups = useMemo(
    () => (models === null ? [] : groupModels(models)),
    [models],
  )
  // 补训索引实验名客户端校验（与 Training 页同款：EXP_NAME_RE + 拒 "." / ".."，
  // 后端 _check_exp_name 会再校验一次）
  const expNameValid =
    indexForm !== null &&
    indexForm.expName !== '.' &&
    indexForm.expName !== '..' &&
    EXP_NAME_RE.test(indexForm.expName)

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-col gap-1.5">
            <CardTitle>模型管理</CardTitle>
            <CardDescription>
              查看 assets/weights 中的音色模型，删除或补建检索索引；同一次训练的
              中间轮次产物收进各组卡片下方的折叠列表，最近训练的组排在最前。
              也可导入外部模型文件（pth + 可选索引）。
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setUploadOpen((o) => !o)}
            >
              <UploadIcon />
              {uploadOpen ? '收起导入' : '导入模型'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setReloadTick((t) => t + 1)}
              disabled={models === null && loadError === null}
            >
              <RefreshCwIcon />
              刷新
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {uploadOpen && (
          <div className="flex flex-col gap-3 rounded-lg border p-4">
            <span className="text-sm font-medium">导入外部模型</span>
            <p className="text-xs text-muted-foreground">
              导入后与训练产出的模型同等对待：出现在推理页下拉框，可下载、可删除。
            </p>
            <ModelUploadForm onUploaded={() => setReloadTick((t) => t + 1)} />
          </div>
        )}
        {loadError !== null && (
          <p
            role="alert"
            className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive"
          >
            模型列表加载失败：{loadError}
          </p>
        )}

        {models === null && loadError === null && (
          <p className="text-sm text-muted-foreground">正在加载模型列表…</p>
        )}

        {models !== null && models.length === 0 && (
          <div className="flex flex-col items-start gap-3 rounded-lg bg-muted p-4">
            <p className="text-sm text-muted-foreground">
              还没有模型——去训练页训练第一个音色，或点右上角「导入模型」导入已有的
              pth / index 文件。
            </p>
            <Button size="sm" onClick={onGoTrain}>
              去训练
            </Button>
          </div>
        )}

        {receipt !== null && (
          <div className="flex flex-col gap-1 rounded-lg bg-emerald-500/10 p-3 text-xs text-emerald-700 dark:text-emerald-400">
            <p>
              已删除 {receipt.model} 等共 {receipt.modelCount} 个权重文件（含中间轮次）
              {receipt.indexCount > 0 && `、${receipt.indexCount} 个配对索引`}
              {logsReceiptText(receipt)}。
            </p>
            {receipt.failedModelCount > 0 && (
              <p className="text-amber-600 dark:text-amber-400">
                另有 {receipt.failedModelCount} 个组内权重删除失败。
              </p>
            )}
            {receipt.failedCount > 0 && (
              <p className="text-amber-600 dark:text-amber-400">
                另有 {receipt.failedCount} 个索引删除失败。
              </p>
            )}
          </div>
        )}
        {deleteError !== null && (
          <p role="alert" className="rounded-lg bg-destructive/10 p-3 text-xs text-destructive">
            删除失败：{deleteError}
          </p>
        )}

        {groups.map((g) => {
          const rep = g.representative
          const exp = experimentName(rep.name.replace(/\.pth$/i, ''))
          // 退化文件名（如 _e20_s100.pth）剥不出实验名，补训索引无从谈起
          const canReindex = exp.length > 0
          // 训练未完成（只有中间轮次）或最终模型被单独删除：代表项退化为最大轮次的中间产物
          const noFinal = g.final === null && g.intermediates.length > 0
          // 整组权重文件数（最终模型 + 全部中间轮次），删除确认文案里如实报数
          const memberCount = (g.final !== null ? 1 : 0) + g.intermediates.length
          // 该组补训索引任务进行中：删除任何成员都会让正在建立索引的实验失去主体，全组先禁用
          const reindexingThisGroup =
            watched !== null && watched.groupKey === g.key && running
          return (
            <div key={g.key} className="flex flex-col gap-3 rounded-lg border p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate font-mono text-sm font-medium" title={rep.name}>
                    {rep.name}
                  </span>
                  {noFinal && (
                    <span className="text-xs text-amber-600 dark:text-amber-400">
                      尚无最终产物（训练未完成或已被删除），当前为轮次最大的中间模型
                    </span>
                  )}
                  {rep.index !== null ? (
                    <span
                      className="truncate text-xs text-muted-foreground"
                      title={rep.index}
                    >
                      索引：{fileName(rep.index)}
                    </span>
                  ) : (
                    <span className="text-xs text-amber-600 dark:text-amber-400">
                      缺少索引：推理时音色相似度会打折
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="outline" onClick={() => onGoInfer(rep.name)}>
                    去推理
                  </Button>
                  {/* 下载：a 标签原生下载（服务端 Content-Disposition 定名），
                      缺索引时包内只有 pth，下载本身不被拦截 */}
                  <Button
                    size="sm"
                    variant="outline"
                    render={<a href={api.modelDownloadUrl(rep.name)} />}
                    title="下载模型包（zip，含 pth 与配对索引）"
                  >
                    <DownloadIcon />
                    下载
                  </Button>
                  {rep.index === null && canReindex && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={running}
                      onClick={() =>
                        setIndexForm({ groupKey: g.key, expName: exp, version: 'v2' })
                      }
                    >
                      补训索引
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={deleting !== null || reindexingThisGroup}
                    onClick={() => onDeleteClick(rep.name)}
                    title={
                      reindexingThisGroup
                        ? '该实验正在补训索引，任务结束后再删除'
                        : '彻底删除该组的全部产物（权重 + 索引 + logs 训练产物）'
                    }
                  >
                    {deleting === rep.name && <LoaderCircleIcon className="animate-spin" />}
                    {confirmDelete === rep.name ? '确认删除？' : '删除'}
                  </Button>
                </div>
              </div>

              {/* 两段式确认的说明：删除 = 彻底删除整组（全部轮次 + 配对索引 +
                  logs/{exp} 训练产物，体积大头且不可恢复、无法再补训索引） */}
              {confirmDelete === rep.name && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  将彻底删除该组全部产物：{memberCount} 个权重文件（含最终模型与中间轮次）
                  、按配对规则命中的全部索引
                  {rep.index !== null ? `（当前配对：${fileName(rep.index)}）` : '（当前无配对索引）'}
                  {exp.length > 0 &&
                    `，以及 logs/${exp} 下的训练特征与 checkpoint（体积最大，通常数 GB）`}
                  。删除后无法恢复，也不能再用这些产物补训索引。再次点击按钮确认，3 秒后自动取消。
                </p>
              )}

              {/* 补训索引内联表单：实验名默认推导、可编辑；版本默认 v2 可改（组级操作：
                  同组模型共享同一索引，这里建/补一次即可，中间轮次子项不再各放一份） */}
              {indexForm !== null && indexForm.groupKey === g.key && (
                <div className="flex flex-col gap-2 rounded-lg bg-muted/50 p-3">
                  <span className="text-sm font-medium">补训索引</span>
                  <p className="text-xs text-muted-foreground">
                    用该模型的实验名重新建立检索索引，需要 logs/{indexForm.expName}
                    下仍保留训练特征产物；模型版本与训练时保持一致。
                  </p>
                  <div className="flex flex-wrap items-end gap-2">
                    <div className="flex min-w-48 flex-1 flex-col gap-1.5">
                      <label
                        htmlFor="index-exp"
                        className="text-xs font-medium"
                      >
                        实验名
                      </label>
                      <Input
                        id="index-exp"
                        value={indexForm.expName}
                        aria-invalid={!expNameValid}
                        onChange={(e: ChangeEvent<HTMLInputElement>) =>
                          setIndexForm({ ...indexForm, expName: e.target.value })
                        }
                      />
                      {!expNameValid && (
                        <p className="text-xs text-destructive">
                          实验名不能为空，且不得含空格、引号、反斜杠、$、反引号或路径分隔符
                        </p>
                      )}
                    </div>
                    <div className="flex w-32 flex-col gap-1.5">
                      <span className="text-xs font-medium">模型版本</span>
                      <Select
                        value={indexForm.version}
                        onValueChange={(v) =>
                          applySelect(v, (value) =>
                            setIndexForm({
                              ...indexForm,
                              version: value as ModelVersion,
                            }),
                          )
                        }
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="v2">v2</SelectItem>
                          <SelectItem value="v1">v1</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button
                      size="sm"
                      disabled={submitting || !expNameValid}
                      onClick={() => void startIndex()}
                    >
                      {submitting && <LoaderCircleIcon className="animate-spin" />}
                      开始补训
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={submitting}
                      onClick={() => {
                        setIndexForm(null)
                        setSubmitError(null)
                      }}
                    >
                      取消
                    </Button>
                  </div>
                  {/* 创建失败发生在表单仍打开时（受理失败不收起表单），错误必须就地可见 */}
                  {submitError !== null && (
                    <p role="alert" className="text-xs text-destructive">
                      任务创建失败：{submitError}
                    </p>
                  )}
                </div>
              )}

              {/* 补训索引任务监视区（进度 + 结果） */}
              {watched !== null && watched.groupKey === g.key && (
                <div className="flex flex-col gap-2 rounded-lg border p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">建立索引</span>
                      {running && (
                        <span className="inline-flex items-center gap-1 text-xs text-primary">
                          <LoaderCircleIcon className="size-3.5 animate-spin" />
                          进行中
                        </span>
                      )}
                      {task.status === 'success' && task.terminal && (
                        <span className="text-xs text-emerald-600 dark:text-emerald-400">
                          索引已建立
                        </span>
                      )}
                      {taskFailed && (
                        <span className="text-xs text-destructive">
                          {task.status === 'cancelled' ? '已停止' : '失败'}
                        </span>
                      )}
                    </div>
                    {running && (
                      <Button size="sm" variant="ghost" onClick={() => void stopIndex()}>
                        停止
                      </Button>
                    )}
                  </div>
                  {stopError !== null && (
                    <p role="alert" className="text-xs text-destructive">
                      停止失败：{stopError}
                    </p>
                  )}
                  <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className={`h-full bg-primary transition-[width] duration-500 ${
                        progressPct === null && running ? 'animate-pulse' : ''
                      }`}
                      style={{ width: `${progressPct ?? 0}%` }}
                    />
                  </div>
                  {task.connectionLost && !task.terminal && (
                    <p className="text-xs text-amber-600 dark:text-amber-400">
                      连接已断开，任务仍在后台运行；
                      <button
                        type="button"
                        className="underline underline-offset-4"
                        onClick={task.resubscribe}
                      >
                        重新连接
                      </button>
                    </p>
                  )}
                  {taskFailed && (
                    <ErrorDetail
                      title={task.status === 'cancelled' ? '任务已停止' : '索引建立失败'}
                    >
                      {task.error ?? '任务失败，原因未知'}
                    </ErrorDetail>
                  )}
                </div>
              )}

              {/* 中间轮次折叠子列表：默认收起，epoch 升序（训练演进顺序）。
                  子项操作只有去推理/下载——删除是组级操作（整组彻底删除），
                  只在组卡片上提供。文件名剥不出轮次时原样展示 */}
              {g.intermediates.length > 0 && (
                <Collapsible>
                  <CollapsibleTrigger className="flex items-center gap-1 self-start rounded-lg px-2 py-1.5 text-sm text-muted-foreground select-none hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none [&_svg]:transition-transform [&[aria-expanded=true]_svg]:rotate-180">
                    中间轮次模型（{g.intermediates.length}）
                    <ChevronDownIcon className="size-4" />
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="flex flex-col divide-y pt-1">
                      {g.intermediates.map((m) => {
                        const epoch = parseEpochSuffix(m.name.replace(/\.pth$/i, ''))
                        return (
                          <div
                            key={m.name}
                            className="flex flex-wrap items-center justify-between gap-2 py-2"
                          >
                            <span
                              className="truncate font-mono text-xs"
                              title={m.name}
                            >
                              {epoch === null
                                ? m.name
                                : `第 ${epoch.epoch} 轮 · step ${epoch.step}`}
                            </span>
                            <div className="flex items-center gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => onGoInfer(m.name)}
                              >
                                去推理
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                render={<a href={api.modelDownloadUrl(m.name)} />}
                                title="下载模型包（zip，含 pth 与配对索引）"
                              >
                                <DownloadIcon />
                                下载
                              </Button>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
