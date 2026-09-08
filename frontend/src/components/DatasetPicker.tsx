/**
 * 参考音频上传控件（训练向导「参考音频」字段的主控件；后端与 API 沿用
 * dataset 命名，用户可见文案统一叫「参考音频」）：
 * 唯一主动作是**拖入即传**（顺序逐文件上传，带逐文件状态预览列表），目标目录自动
 * 生成随机编号（ds-xxxxxxxx），用户不需要命名。界面不渲染「已有参考音频」列表——
 * 训练素材就是本次上传的内容（分离完成则自动切到分离副本）；历史目录仍留在服务器
 * datasets/ 下，由 API 管理（DELETE /api/datasets/{name}），界面不再提供入口。
 * 受控组件：本组件只回传意图（onUploaded / onSelect / onUploadBusyChange），
 * dataset_dir 的单一数据源在父级（Training 表单）。
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { LoaderCircleIcon, PauseIcon, PlayIcon, UploadIcon } from 'lucide-react'

import { api, type DatasetSummary, type UploadResult } from '@/api/client'
import {
  AUDIO_SUFFIXES,
  DEFAULT_SEPARATION_MODEL,
  SEPARATION_MODELS,
  derivedDatasetName,
  formatDuration,
  randomHex,
} from '@/lib/domain'
import { errorMessage } from '@/lib/utils'
import { useTask } from '@/hooks/useTask'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export interface DatasetPickerProps {
  /** 选中变化（分离完成后切到衍生数据集时回传给父级） */
  onSelect: (ds: DatasetSummary | null) => void
  /** 上传成功：父级把 result.path 回填 dataset_dir，表单立即可提交 */
  onUploaded: (r: UploadResult) => void
  /**
   * 上传队列忙碌状态变化：true = 还有排队/在途的文件。父级用它禁用「训练」——
   * 否则用户可能在目录只落了一半文件时就开训，训出来的东西缺数据
   */
  onUploadBusyChange?: (busy: boolean) => void
  /** 有训练任务进行中：分离入口被后端 409 互斥（UI 先禁用），上传不拦 */
  trainingRunning: boolean
}

/** 正被监视的分离任务；sourceName/outputName 在页面重挂恢复时无法还原（任务摘要不带
 * 业务字段，从命令串解析过于脆弱），此时只监视进度、终态不自动选中 */
interface SeparationWatch {
  taskId: string
  sourceName: string | null
  outputName: string | null
}

/** 上传会话里的单个文件（状态机：queued → uploading → done | skipped | failed） */
interface UploadItem {
  /** 「名字+大小+修改时间」，同会话去重键 */
  key: string
  file: File
  status: 'queued' | 'uploading' | 'done' | 'skipped' | 'failed'
  /** 0-100，仅 uploading 态有意义（XHR upload.onprogress） */
  progress: number
  /** skipped 的原因（后端拒收明细）/ failed 的错误信息 */
  reason: string | null
  /** 行内操作的瞬时错误（如删除失败），不改变 status，下次操作前清空 */
  notice: string | null
}

function uploadKey(f: File): string {
  return `${f.name}:${f.size}:${f.lastModified}`
}

export function DatasetPicker({
  onSelect,
  onUploaded,
  onUploadBusyChange,
  trainingRunning,
}: DatasetPickerProps) {
  /** null = 首次加载中；[] = 已加载且服务器上还没有数据集 */
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  /** 上传会话的文件列表（uploadsRef 是事实来源，state 供渲染） */
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const uploadsRef = useRef<UploadItem[]>([])
  /** 本次上传会话的目标参考音频名（随机 ID）；null = 还没有任何文件开始上传 */
  const [sessionName, setSessionName] = useState<string | null>(null)
  const sessionNameRef = useRef<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  // -- 人声分离（数据集级衍生操作，设计 docs/plans/2026-09-07-dataset-vocal-separation-design.md §4.4） ------

  const [separating, setSeparating] = useState<SeparationWatch | null>(null)
  /** 确认面板展开中的分离目标：dataset 恒为当前会话；file = null 表示全部分离，
   *  指向文件名表示逐文件分离（同一时刻至多一个面板） */
  const [confirmSeparate, setConfirmSeparate] = useState<{
    dataset: string
    file: string | null
    /** 行级面板的定位键（同文件名不同大小/时间的两行只展开被点的那一行） */
    fileKey?: string
  } | null>(null)
  const [sepModel, setSepModel] = useState<string>(DEFAULT_SEPARATION_MODEL)
  /** 发起失败（409/404 等）与任务失败（终态 error）共用一行展示位 */
  const [separateError, setSeparateError] = useState<string | null>(null)
  /** 正在播放的预览行（去重键）；null = 无。同一时刻只播一个 */
  const [playingKey, setPlayingKey] = useState<string | null>(null)
  const watched = useTask(separating?.taskId ?? null)

  const fileInput = useRef<HTMLInputElement | null>(null)
  // 异步回调里的 setState 保护：切 Tab 卸载后 React 19 会静默忽略，这里显式短路
  // （与 pages/Training.tsx 同一纪律）。上传中的 XHR 无法随卸载安全中止（abort 会把
  // 用户切 Tab 误当成失败），让它照常传完、迟到回调在这里被丢弃即可
  const mounted = useRef(true)
  // 试听用的单一音频元素（懒创建）：换曲即换 src，天然互斥
  const audioRef = useRef<HTMLAudioElement | null>(null)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      audioRef.current?.pause()
    }
  }, [])

  // 最新列表的 ref：refresh 失败兜底要用。不能把 datasets 收进 refresh 的依赖——
  // 那会让每次列表变化都重建 refresh，触发下方 mount effect 重复拉取
  const datasetsRef = useRef<DatasetSummary[] | null>(null)
  useEffect(() => {
    datasetsRef.current = datasets
  }, [datasets])

  const refresh = useCallback((): Promise<DatasetSummary[]> => {
    return api
      .datasets()
      .then((list) => {
        if (!mounted.current) return list
        setDatasets(list)
        setLoadError(null)
        return list
      })
      .catch((e: unknown) => {
        // 刷新失败保留旧列表（可能是过期数据），错误行提示；首次加载失败时列表仍为
        // null → 显示重试入口。返回旧列表（自动选中等调用方据此继续，拿不到就空）
        if (mounted.current) setLoadError(errorMessage(e))
        return datasetsRef.current ?? []
      })
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // 重挂恢复：服务端有进行中的分离任务（切 Tab / 刷新页面后回来）→ 恢复进度监视，
  // 让全局互斥对用户可见（否则分离在后台跑，页面却像什么都没发生）
  useEffect(() => {
    let cancelled = false
    api
      .getTasks()
      .then((tasks) => {
        if (cancelled) return
        const running = tasks.find(
          (t) =>
            t.name === 'separate' && (t.state === 'pending' || t.state === 'running'),
        )
        if (running !== undefined) {
          setSeparating({ taskId: running.id, sourceName: null, outputName: null })
        }
      })
      .catch(() => {
        // 任务列表不可达：不打断数据集主流程（与 loadError 同口径，静默降级）
      })
    return () => {
      cancelled = true
    }
  }, [])

  // 分离任务终态收尾：成功 → 刷新并自动选中衍生数据集（dataset_dir 回填，表单立即可
  // 提交）；失败 → 透出后端 error（含日志尾部）；取消 → 静默。收尾后解除监视
  useEffect(() => {
    if (separating === null || !watched.terminal) return
    const outputName = separating.outputName
    const status = watched.status
    const error = watched.error
    setSeparating(null)
    if (status === 'success') {
      void refresh().then((list) => {
        if (!mounted.current) return
        const derived = list.find((d) => d.name === outputName)
        if (derived !== undefined) onSelect(derived)
      })
    } else if (status === 'failed') {
      setSeparateError(error ?? '人声分离任务失败')
    }
  }, [separating, watched.terminal, watched.status, watched.error, refresh, onSelect])

  // -- 派生 -----------------------------------------------------------------

  const doneCount = uploads.filter((i) => i.status === 'done').length
  const skippedCount = uploads.filter((i) => i.status === 'skipped').length
  const failedCount = uploads.filter((i) => i.status === 'failed').length
  /** 队列里还有没落地的文件（父级据此禁用「训练」） */
  const uploadActive = uploads.some((i) => i.status === 'queued' || i.status === 'uploading')

  // -- 上传会话 -------------------------------------------------------------

  /** 泵送中标志：保证同一时刻只有一个 pump 循环在跑（addFiles/重试并发触发时，
   *  后来者直接返回——循环每轮都会重新扫描 queued，不会漏文件） */
  const pumping = useRef(false)

  function patchUpload(key: string, patch: Partial<UploadItem>) {
    // uploadsRef 是事实来源（pump 的循环靠它读最新状态），state 只是渲染镜像
    uploadsRef.current = uploadsRef.current.map((i) => (i.key === key ? { ...i, ...patch } : i))
    setUploads(uploadsRef.current)
  }

  /** 本次上传会话的目标参考音频名：首个文件启动上传时生成随机 ID。ref 同步读写
   *  （pump 循环内要立刻可见），state 供渲染。真与既有参考音频重名 = 追加，无害
   *  （8 位十六进制 ≈ 43 亿取值，单机场景撞名可忽略）。 */
  function ensureSessionName(): string {
    if (sessionNameRef.current === null) {
      sessionNameRef.current = `ds-${randomHex(8)}`
      setSessionName(sessionNameRef.current)
    }
    return sessionNameRef.current
  }

  /** 顺序上传泵：同一时刻只有一个 XHR 在途——逐文件独立进度与 500MB 限额，单个
   *  失败不拖垮整批。addFiles / 重试都会触发；已在泵送时直接返回（循环自己会捞
   *  到新排队的文件）。卸载后在途 XHR 照常传完（abort 会把切 Tab 误当成失败），
   *  迟到的 patch 由 mounted 短路丢弃。 */
  async function pump() {
    if (pumping.current) return
    pumping.current = true
    onUploadBusyChange?.(true)
    try {
      while (true) {
        const next = uploadsRef.current.find((i) => i.status === 'queued')
        if (next === undefined) break
        patchUpload(next.key, { status: 'uploading', progress: 0, reason: null })
        try {
          const r = await api.uploadDataset(ensureSessionName(), [next.file], (pct) => {
            patchUpload(next.key, { progress: pct })
          })
          if (!mounted.current) return
          if (r.added.length > 0) {
            patchUpload(next.key, { status: 'done', progress: 100 })
            onUploaded(r) // 回填 dataset_dir（同路径重复回填幂等）
            void refresh() // 供分离确认面板读取文件数/时长（失败只走 loadError，不算上传失败）
          } else {
            patchUpload(next.key, {
              status: 'skipped',
              reason: r.skipped[0]?.reason ?? '服务器拒收',
            })
          }
        } catch (e) {
          if (!mounted.current) return
          // 保留 failed 行 + 重试入口：网络闪断/超限都不丢用户已选的文件
          patchUpload(next.key, { status: 'failed', reason: errorMessage(e) })
        }
      }
    } finally {
      pumping.current = false
      onUploadBusyChange?.(false)
    }
  }

  // -- 交互 -----------------------------------------------------------------

  /** 合并新选/拖入的文件并入队上传：按「名字+大小+修改时间」粗去重，同一批文件可
   *  放心重复拖。参数必须是快照（Array.from 的结果）而非 FileList：清空 input.value
   *  会连带清空 FileList，而 setState 的 updater 可能延迟到事件处理器结束后才执行，
   *  届时再读 FileList 已是空——这是「选了文件但列表不出现」的偶发丢失源 */
  function addFiles(snapshot: File[]) {
    const seen = new Set(uploadsRef.current.map((i) => i.key))
    const additions: UploadItem[] = []
    for (const f of snapshot) {
      const key = uploadKey(f)
      if (seen.has(key)) continue
      seen.add(key)
      additions.push({ key, file: f, status: 'queued', progress: 0, reason: null, notice: null })
    }
    if (additions.length === 0) return
    uploadsRef.current = [...uploadsRef.current, ...additions]
    setUploads(uploadsRef.current)
    void pump()
  }

  /** 把一个文件从预览列表移除。列表清空时会话一并重置：下一个拖入的文件会生成
   *  新的随机编号，而不是接着一个（可能已删空的）旧目录传 */
  function removeUpload(key: string) {
    uploadsRef.current = uploadsRef.current.filter((i) => i.key !== key)
    setUploads(uploadsRef.current)
    if (playingKey === key) {
      audioRef.current?.pause()
      setPlayingKey(null)
    }
    if (uploadsRef.current.length === 0) {
      // 会话清空：下一个拖入的文件生成新随机编号
      sessionNameRef.current = null
      setSessionName(null)
    }
  }

  /** 删除已上传到服务器的文件（上传错了就地纠错）。后端 409（训练中）等失败
   *  以行内 notice 透出，不改变该行的 done 状态。 */
  function deleteUploaded(item: UploadItem) {
    patchUpload(item.key, { notice: null })
    void api
      .deleteDatasetFile(ensureSessionName(), item.file.name)
      .then(() => {
        if (!mounted.current) return
        removeUpload(item.key)
        void refresh() // 列表的文件数/时长随之更新
      })
      .catch((e: unknown) => {
        if (mounted.current) patchUpload(item.key, { notice: errorMessage(e) })
      })
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    // 先在事件处理器内同步快照（见 addFiles 注释），再清空 input
    const files = e.target.files
    if (files !== null) addFiles(Array.from(files))
    // 立即清空 input：同一批文件处理后还想再传也能再次触发 change
    e.target.value = ''
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  function retryUpload(key: string) {
    patchUpload(key, { status: 'queued', progress: 0, reason: null })
    void pump()
  }

  /** 播放/暂停预览行音频。同一元素换曲即互斥（播 B 自动停 A）；播完/出错复位
   *  playingKey。仅 done 状态的行有播放入口（文件已确认在服务器上）。 */
  function togglePlay(item: UploadItem) {
    if (audioRef.current === null) {
      const el = new Audio()
      el.addEventListener('ended', () => setPlayingKey(null))
      el.addEventListener('error', () => setPlayingKey(null))
      audioRef.current = el
    }
    const el = audioRef.current
    if (playingKey === item.key) {
      el.pause()
      setPlayingKey(null)
      return
    }
    if (sessionName === null) return
    el.src = `/api/datasets/${encodeURIComponent(sessionName)}/files/${encodeURIComponent(item.file.name)}/content`
    el.play()
      .then(() => {
        if (mounted.current) setPlayingKey(item.key)
      })
      .catch((e: unknown) => {
        // 自动播放策略/解码失败等：行内透出，不复位文件状态
        if (mounted.current) {
          setPlayingKey(null)
          patchUpload(item.key, { notice: `播放失败：${errorMessage(e)}` })
        }
      })
  }

  // -- 人声分离交互 -----------------------------------------------------------

  /** 发起分离（file = null 分离全部，否则只分离该文件）：POST 成功后进入监视态；
   *  全局互斥由后端裁决（409 走 separateError）。 */
  function startSeparation(dataset: string, file: string | null = null) {
    if (separating !== null) return
    setSeparateError(null)
    void api
      .separateDataset(dataset, sepModel, file === null ? undefined : [file])
      .then((r) => {
        if (!mounted.current) return
        setConfirmSeparate(null)
        setSeparating({ taskId: r.task_id, sourceName: dataset, outputName: r.output_dataset })
      })
      .catch((e: unknown) => {
        if (mounted.current) setSeparateError(errorMessage(e))
      })
  }

  /** 请求停止分离。DELETE /tasks 对已终态任务是幂等 202（与任务自然结束竞态安全）。 */
  function stopSeparation() {
    if (separating === null) return
    void api.cancelTask(separating.taskId).catch((e: unknown) => {
      if (mounted.current) setSeparateError(errorMessage(e))
    })
  }

  // -- 渲染 -----------------------------------------------------------------

  return (
    <div className="flex flex-col gap-3">
      {/* 上传区：拖拽/多选即传，目标目录自动创建（随机编号），无需命名 */}
      <div className="flex flex-col gap-2.5 rounded-lg border border-dashed p-3">
        <div
          onDragEnter={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={(e) => {
            // 拖进子元素也会触发 dragleave：还留在拖拽区内就不熄灭高亮
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false)
          }}
          onDrop={onDrop}
          className={`flex flex-wrap items-center justify-center gap-2 rounded-lg p-4 text-center transition-colors ${
            dragOver ? 'border-border bg-muted' : ''
          }`}
        >
          <UploadIcon className="size-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">拖拽音频文件到此处，或</span>
          <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
            选择文件
          </Button>
          {/* 隐藏的真实 input：accept 与后端 AUDIO_SUFFIXES 同源（domain.ts） */}
          <input
            ref={fileInput}
            type="file"
            multiple
            accept={AUDIO_SUFFIXES.join(',')}
            onChange={onPick}
            className="hidden"
          />
        </div>
        <p className="text-xs text-muted-foreground">
          支持 {AUDIO_SUFFIXES.join(' / ')}，可多选；单个文件 ≤ 500 MB
        </p>
      </div>

      {/* 上传会话预览列表：逐文件状态（排队 / 进度 / 完成 / 跳过原因 / 失败重试） */}
      {uploads.length > 0 && (
        <div className="flex flex-col gap-1.5 rounded-lg border px-3 py-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="min-w-0 truncate text-muted-foreground">
              本次上传{sessionName !== null ? ` ${sessionName}` : ''}：{doneCount}/
              {uploads.length} 完成
              {skippedCount > 0 && ` · 跳过 ${skippedCount}`}
              {failedCount > 0 && ` · 失败 ${failedCount}`}
            </span>
            {uploadActive && (
              <LoaderCircleIcon className="size-3.5 shrink-0 animate-spin text-muted-foreground" />
            )}
          </div>
          <ul className="flex max-h-48 flex-col gap-1 overflow-y-auto">
            {uploads.map((item) => (
              <li key={item.key} className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2">
                  {item.status === 'done' && (
                    <Button
                      variant="outline"
                      size="xs"
                      aria-label={playingKey === item.key ? `暂停 ${item.file.name}` : `播放 ${item.file.name}`}
                      title={playingKey === item.key ? '暂停' : '播放'}
                      onClick={() => togglePlay(item)}
                    >
                      {playingKey === item.key ? <PauseIcon /> : <PlayIcon />}
                    </Button>
                  )}
                  <span
                    className="min-w-0 flex-1 truncate font-mono text-foreground"
                    title={item.file.name}
                  >
                    {item.file.name}
                  </span>
                  {item.status === 'queued' && (
                    <span className="shrink-0 text-muted-foreground">排队中</span>
                  )}
                  {item.status === 'uploading' && (
                    <span className="shrink-0 tabular-nums text-muted-foreground">
                      {item.progress}%
                    </span>
                  )}
                  {item.status === 'done' && (
                    <span className="shrink-0 text-emerald-700 dark:text-emerald-400">
                      已完成
                    </span>
                  )}
                  {item.status === 'skipped' && (
                    <span
                      className="shrink-0 truncate text-amber-600 dark:text-amber-400"
                      title={item.reason ?? ''}
                    >
                      跳过：{item.reason}
                    </span>
                  )}
                  {item.status === 'failed' && (
                    <>
                      <span
                        className="min-w-0 shrink truncate text-destructive"
                        title={item.reason ?? ''}
                      >
                        {item.reason}
                      </span>
                      <Button variant="outline" size="xs" onClick={() => retryUpload(item.key)}>
                        重试
                      </Button>
                      {/* 失败文件不在服务器上：移除只是清出列表 */}
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => removeUpload(item.key)}
                      >
                        移除
                      </Button>
                    </>
                  )}
                  {item.status === 'skipped' && (
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => removeUpload(item.key)}
                    >
                      移除
                    </Button>
                  )}
                  {item.status === 'done' && (
                    <>
                      {/* 逐文件分离：只分离这一个文件（产物进同一衍生目录，幂等衔接
                          后续的全部分离）。与训练/其他分离共用全局互斥，进行中禁用 */}
                      <Button
                        variant="outline"
                        size="xs"
                        disabled={separating !== null || trainingRunning}
                        title={
                          separating !== null
                            ? '已有分离任务进行中'
                            : trainingRunning
                              ? '训练任务进行中，后端拒绝并发任务'
                              : '分离该文件的人声'
                        }
                        onClick={() => {
                          setSeparateError(null)
                          if (sessionName !== null) {
                            setConfirmSeparate({
                              dataset: sessionName,
                              file: item.file.name,
                              fileKey: item.key,
                            })
                          }
                        }}
                      >
                        分离
                      </Button>
                      <Button
                        variant="outline"
                        size="xs"
                        disabled={trainingRunning}
                        title={
                          trainingRunning
                            ? '训练任务进行中，后端拒绝删除'
                            : '从服务器删除该文件（不可恢复，可重新上传）'
                        }
                        onClick={() => deleteUploaded(item)}
                      >
                        删除
                      </Button>
                    </>
                  )}
                </div>
                {item.status === 'uploading' && (
                  <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary transition-[width] duration-300"
                      style={{ width: `${item.progress}%` }}
                    />
                  </div>
                )}
                {item.notice !== null && (
                  <span className="text-destructive">{item.notice}</span>
                )}
                {confirmSeparate !== null &&
                  confirmSeparate.fileKey === item.key &&
                  sessionName === confirmSeparate.dataset && (
                    <SeparateConfirm
                      name={confirmSeparate.dataset}
                      summary={
                        datasets?.find((d) => d.name === confirmSeparate.dataset) ?? null
                      }
                      onlyFile={item.file.name}
                      model={sepModel}
                      onModelChange={setSepModel}
                      onConfirm={() =>
                        startSeparation(confirmSeparate.dataset, confirmSeparate.file)
                      }
                      onCancel={() => setConfirmSeparate(null)}
                    />
                  )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 分离任务监视条：全局互斥保证同一时刻至多一个分离任务。页面重挂恢复的监视
          不带源数据集名（只显示进度）；进度来自 SSE 的读时换算（切分同款进度行协议） */}
      {separating !== null && (
        <div className="flex flex-col gap-1.5 rounded-lg border px-3 py-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="flex min-w-0 items-center gap-1.5">
              <LoaderCircleIcon className="size-3.5 shrink-0 animate-spin" />
              <span className="truncate">
                人声分离中{separating.sourceName !== null ? `：${separating.sourceName}` : ''}
                {watched.current !== null && ` · ${watched.current}`}
              </span>
            </span>
            <span className="flex shrink-0 items-center gap-2">
              <span className="tabular-nums text-muted-foreground">
                {watched.progress === null ? '…' : `${Math.round(watched.progress * 100)}%`}
              </span>
              <Button variant="outline" size="xs" onClick={stopSeparation}>
                停止
              </Button>
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full bg-primary transition-[width] duration-500 ${
                watched.progress === null ? 'w-full animate-pulse' : ''
              }`}
              style={{
                width:
                  watched.progress === null ? undefined : `${Math.round(watched.progress * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {separateError !== null && (
        <p role="alert" className="text-xs text-destructive">
          人声分离失败：{separateError}
        </p>
      )}

      {loadError !== null && (
        <p role="alert" className="text-xs text-destructive">
          参考音频列表加载失败：{loadError}
        </p>
      )}
    </div>
  )
}


/**
 * 分离确认面板（行内展开，非模态）：模型选择 + 产物说明 + 开始/取消。
 * 不做耗时预估——没有可靠的每模型速度系数，宁缺毋滥（设计 §2）；
 * 模型 label 与后端 SEPARATION_MODELS 逐字同源（domain.ts），顺序即下拉顺序。
 */
function SeparateConfirm(props: {
  name: string
  summary: DatasetSummary | null
  /** 提供时为逐文件分离：面板文案注明只处理这一个文件 */
  onlyFile?: string | null
  model: string
  onModelChange: (m: string) => void
  onConfirm: () => void
  onCancel: () => void
}) {
  const modelItems = useMemo(
    () => SEPARATION_MODELS.map((m) => ({ value: m, label: m })),
    [],
  )
  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-muted/40 px-3 py-2">
      <span className="text-muted-foreground">
        {props.onlyFile != null && `仅处理 ${props.onlyFile}；`}
        产物写入分离副本 {derivedDatasetName(props.name)}
        {props.summary !== null &&
          `（${props.summary.file_count} 个文件 · ${formatDuration(props.summary.total_duration)}）`}
        ，仅保留人声、伴奏丢弃；耗时可能较长，可随时停止。
      </span>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <Select
          items={modelItems}
          value={props.model}
          onValueChange={(v) => {
            // Select 的值协议含 null（清选）；模型下拉恒有合法选中，null 直接忽略
            if (v !== null) props.onModelChange(v)
          }}
        >
          <SelectTrigger className="w-full sm:flex-1" aria-label="分离模型">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {modelItems.map((m) => (
              <SelectItem key={m.value} value={m.value}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex gap-2">
          <Button size="sm" onClick={props.onConfirm}>
            开始分离
          </Button>
          <Button variant="outline" size="sm" onClick={props.onCancel}>
            取消
          </Button>
        </div>
      </div>
    </div>
  )
}
