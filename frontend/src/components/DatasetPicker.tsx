/**
 * 数据集选择与上传控件（训练向导「数据集」字段的主控件）：
 * 下拉选已有 + 拖拽/多选上传（填新名字=新建，留空=追加到当前选中）+ 逐项追加/删除。
 * 受控组件：选中路径由父级持有（selectedPath），本组件只回传意图（onSelect /
 * onUploaded），保证向导 dataset_dir 的单一数据源；上传/删除成功后自行刷新列表，
 * 选中态跟随父级回填的 path 走。
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { LoaderCircleIcon, RefreshCwIcon, Trash2Icon, UploadIcon } from 'lucide-react'

import { api, type DatasetSummary, type UploadResult } from '@/api/client'
import { AUDIO_SUFFIXES, formatDuration, isValidDatasetName } from '@/lib/domain'
import { errorMessage } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export interface DatasetPickerProps {
  /** 当前选中的数据集目录绝对路径（null = 未选择）；持有方是父级（Training 表单） */
  selectedPath: string | null
  /** 选中变化（含删除选中项后用 null 清空父级选中） */
  onSelect: (ds: DatasetSummary | null) => void
  /** 上传成功：父级把 result.path 回填 dataset_dir，表单立即可提交 */
  onUploaded: (r: UploadResult) => void
  /** 有训练任务进行中：删除被后端 409 互斥（UI 先禁用），上传不拦 */
  trainingRunning: boolean
}

export function DatasetPicker({ selectedPath, onSelect, onUploaded, trainingRunning }: DatasetPickerProps) {
  /** null = 首次加载中；[] = 已加载且服务器上还没有数据集 */
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  /** null = 未在上传；0-100 = 上传进度（XHR upload.onprogress） */
  const [progress, setProgress] = useState<number | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [lastUpload, setLastUpload] = useState<UploadResult | null>(null)
  /** 两段式删除确认：处于武装态的数据集名（3s 未确认自动还原） */
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const fileInput = useRef<HTMLInputElement | null>(null)
  // 异步回调里的 setState 保护：切 Tab 卸载后 React 19 会静默忽略，这里显式短路
  // （与 pages/Training.tsx 同一纪律）。上传中的 XHR 无法随卸载安全中止（abort 会把
  // 用户切 Tab 误当成失败），让它照常传完、迟到回调在这里被丢弃即可
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const refresh = useCallback((): Promise<void> => {
    return api
      .datasets()
      .then((list) => {
        if (!mounted.current) return
        setDatasets(list)
        setLoadError(null)
      })
      .catch((e: unknown) => {
        // 刷新失败保留旧列表（可能是过期数据），错误行提示；首次加载失败时列表仍为
        // null → 显示重试入口
        if (mounted.current) setLoadError(errorMessage(e))
      })
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // -- 派生 -----------------------------------------------------------------

  /** 选中项：以 path 对齐（name 可能被外部重建，path 才是 dataset_dir 的事实来源） */
  const selected = datasets?.find((d) => d.path === selectedPath) ?? null
  const selectItems = useMemo(
    () => (datasets ?? []).map((d) => ({ value: d.name, label: d.name })),
    [datasets],
  )

  // 输入值去首尾空白后再校验/上传：拖进来的尾随空格不该把整个名字判成非法
  const name = newName.trim()
  const nameValid = name.length === 0 || isValidDatasetName(name)
  const existingByName = datasets?.find((d) => d.name === name) ?? null
  /** 上传目标：填了新名字用新名字（重名=追加），否则落到当前选中的数据集 */
  const targetName = name.length > 0 ? name : (selected?.name ?? null)
  const canUpload = pendingFiles.length > 0 && nameValid && targetName !== null && progress === null

  const nameHint =
    name.length === 0
      ? null
      : !nameValid
        ? '数据集名不能以点开头，且不得含空格、引号、反斜杠、$、反引号或路径分隔符'
        : existingByName !== null
          ? `将追加到已有数据集 ${existingByName.name}（现有 ${existingByName.file_count} 个文件）`
          : `将新建数据集 ${name}`

  // -- 交互 -----------------------------------------------------------------

  /** 合并新选/拖入的文件：按「名字+大小+修改时间」粗去重，同一批文件可放心重复拖。
   *  参数必须是快照（Array.from 的结果）而非 FileList：清空 input.value 会连带清空
   *  FileList，而 setState 的 updater 可能延迟到事件处理器结束后才执行，届时再读
   *  FileList 已是空——这是「选了文件但列表不出现」的偶发丢失源 */
  function addFiles(snapshot: File[]) {
    if (progress !== null || snapshot.length === 0) return
    setPendingFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}:${f.lastModified}`))
      const merged = [...prev]
      for (const f of snapshot) {
        const key = `${f.name}:${f.size}:${f.lastModified}`
        if (seen.has(key)) continue
        seen.add(key)
        merged.push(f)
      }
      return merged
    })
    setUploadError(null)
    setLastUpload(null)
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    // 先在事件处理器内同步快照（见 addFiles 注释），再清空 input
    const files = e.target.files
    if (files !== null) addFiles(Array.from(files))
    // 立即清空 input：同一批文件处理后还想再传（比如换目标数据集）也能再次触发 change
    e.target.value = ''
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  function startUpload() {
    const target = targetName
    if (target === null || pendingFiles.length === 0 || progress !== null) return
    setProgress(0)
    setUploadError(null)
    setLastUpload(null)
    void api
      .uploadDataset(target, pendingFiles, (pct) => {
        if (mounted.current) setProgress(pct)
      })
      .then((r) => {
        if (!mounted.current) return
        // 先回填父级选中（表单可提交），再刷新列表——新数据集/新文件数都来自这次刷新。
        // 刷新失败只走列表区的 loadError 提示，不能算作「上传失败」（文件已落盘成功）
        onUploaded(r)
        setLastUpload(r)
        setPendingFiles([])
        void refresh()
      })
      .catch((e: unknown) => {
        // 失败保留 pendingFiles：改个名字或换个目标可以直接重传，不用重新选文件
        if (mounted.current) setUploadError(errorMessage(e))
      })
      .finally(() => {
        if (mounted.current) setProgress(null)
      })
  }

  function requestDelete(name: string) {
    setDeleteError(null)
    if (confirmDelete !== name) {
      setConfirmDelete(name)
      // 3s 未确认自动还原，避免误触后一直停留在武装态（与 Training 停止按钮同一模式；
      // 带比较的函数式更新：A 项的迟到回调不会误清 B 项刚武装的确认态）
      window.setTimeout(() => setConfirmDelete((cur) => (cur === name ? null : cur)), 3000)
      return
    }
    setConfirmDelete(null)
    setDeleting(name)
    void api
      .deleteDataset(name)
      .then(() => {
        if (!mounted.current) return
        // 删掉的是当前选中 → 清空父级选中，别让表单指向已消失的目录。
        // 刷新失败只走列表区的 loadError 提示，不能算作「删除失败」（目录已删除成功）
        if (selected?.name === name) onSelect(null)
        void refresh()
      })
      .catch((e: unknown) => {
        // 后端 detail（含训练中 409 的「请先停止或等待任务完成」）原样透出
        if (mounted.current) setDeleteError(errorMessage(e))
      })
      .finally(() => {
        if (mounted.current) setDeleting(null)
      })
  }

  // -- 渲染 -----------------------------------------------------------------

  return (
    <div className="flex flex-col gap-3">
      {/* 下拉 + 刷新 */}
      <div className="flex items-center gap-2">
        <Select
          items={selectItems}
          value={selected?.name ?? null}
          onValueChange={(v) => {
            const ds = datasets?.find((d) => d.name === v)
            if (ds !== undefined) onSelect(ds)
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={datasets === null ? '加载数据集…' : '选择已有数据集'} />
          </SelectTrigger>
          <SelectContent>
            {(datasets ?? []).map((d) => (
              <SelectItem key={d.name} value={d.name}>
                {d.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          size="icon"
          onClick={() => void refresh()}
          aria-label="刷新数据集列表"
          title="刷新数据集列表"
        >
          <RefreshCwIcon />
        </Button>
      </div>

      {/* 选中项信息行 */}
      {selected !== null && (
        <div className="flex flex-col gap-1 rounded-lg bg-muted/60 px-3 py-2 text-xs text-muted-foreground">
          <span>
            {selected.name} · {selected.file_count} 个文件 · {formatDuration(selected.total_duration)}
          </span>
          {selected.other_count > 0 && (
            <span className="text-amber-600 dark:text-amber-400">
              另有 {selected.other_count} 个非音频文件也会被切分脚本处理
            </span>
          )}
        </div>
      )}

      {datasets !== null && datasets.length === 0 && (
        <p className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
          服务器上还没有数据集：把人声音频拖进下方上传区即可自动创建。
        </p>
      )}
      {loadError !== null && (
        <p role="alert" className="text-xs text-destructive">
          数据集列表加载失败：{loadError}
        </p>
      )}

      {/* 上传区：拖拽 + 多选 */}
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
          } ${progress !== null ? 'pointer-events-none opacity-60' : ''}`}
        >
          <UploadIcon className="size-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">拖拽音频文件到此处，或</span>
          <Button
            variant="outline"
            size="sm"
            disabled={progress !== null}
            onClick={() => fileInput.current?.click()}
          >
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
          支持 {AUDIO_SUFFIXES.join(' / ')}，可多选；同名文件自动加序号，不会覆盖
        </p>

        <div className="flex flex-col gap-1.5 sm:flex-row sm:items-start">
          <div className="flex-1">
            <Input
              aria-label="新数据集名"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="新数据集名，留空则追加到当前选中"
              aria-invalid={!nameValid}
            />
            {nameHint !== null ? (
              <p
                className={`mt-1 text-xs ${nameValid ? 'text-muted-foreground' : 'text-destructive'}`}
              >
                {nameHint}
              </p>
            ) : (
              targetName === null &&
              pendingFiles.length > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  先填一个新数据集名，或在上方选中一个已有数据集
                </p>
              )
            )}
          </div>
          <Button onClick={startUpload} disabled={!canUpload}>
            {progress !== null && <LoaderCircleIcon className="animate-spin" />}
            <UploadIcon />
            上传
          </Button>
        </div>

        {pendingFiles.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              已选 {pendingFiles.length} 个文件（共{' '}
              {(pendingFiles.reduce((n, f) => n + f.size, 0) / (1024 * 1024)).toFixed(1)} MB）
            </span>
            <ul className="max-h-24 overflow-y-auto rounded-lg bg-muted/60 p-2 font-mono text-xs text-muted-foreground">
              {pendingFiles.map((f) => (
                <li key={`${f.name}:${f.size}:${f.lastModified}`} className="truncate" title={f.name}>
                  {f.name}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* 上传进度（复用 Training 任务监视区的进度条 DOM 模式） */}
      {progress !== null && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>上传中…</span>
            <span className="tabular-nums">{progress}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full bg-primary transition-[width] duration-500 ${
                progress === 0 ? 'animate-pulse' : ''
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {uploadError !== null && (
        <p role="alert" className="text-xs text-destructive">
          上传失败：{uploadError}
        </p>
      )}

      {/* 上传结果（含被跳过文件的原因；发起新上传时清空） */}
      {lastUpload !== null && (
        <div className="flex flex-col gap-1 rounded-lg border border-emerald-500/40 bg-emerald-500/5 px-3 py-2 text-xs">
          <span className="text-emerald-700 dark:text-emerald-400">
            {lastUpload.created ? '已创建' : '已追加'}数据集 {lastUpload.dataset}：
            本次 +{lastUpload.added.length} 个文件，共 {lastUpload.file_count} 个
          </span>
          {lastUpload.skipped.length > 0 && (
            <span className="text-amber-600 dark:text-amber-400">
              跳过 {lastUpload.skipped.length} 个：
              {lastUpload.skipped.map((s) => `${s.name}（${s.reason}）`).join('；')}
            </span>
          )}
        </div>
      )}

      {trainingRunning && (
        <p className="rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
          训练任务进行中：上传可用，删除会被拒绝（请先停止或等待任务完成）。
        </p>
      )}

      {/* 已有数据集列表：选中 / 追加 / 删除 */}
      {datasets !== null && datasets.length > 0 && (
        <ul className="flex flex-col gap-1.5">
          {datasets.map((d) => {
            const isConfirm = confirmDelete === d.name
            return (
              <li
                key={d.name}
                className={`flex flex-col gap-1.5 rounded-lg border px-3 py-2 text-xs ${
                  isConfirm ? 'border-destructive/40 bg-destructive/5' : ''
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onSelect(d)}
                    className="flex-1 truncate text-left font-mono text-sm text-foreground hover:underline"
                    title="选中该数据集"
                  >
                    {d.name}
                    {selected?.name === d.name && (
                      <span className="ms-2 font-sans text-xs text-primary">已选中</span>
                    )}
                  </button>
                  <span className="tabular-nums text-muted-foreground">
                    {d.file_count} 个文件 · {formatDuration(d.total_duration)}
                  </span>
                  <Button
                    variant="outline"
                    size="xs"
                    onClick={() => {
                      // 先选中（上传目标随之落到该数据集）再打开文件选择；选完文件由
                      // 「留空则追加到当前选中」承接，无需在这里填名字
                      onSelect(d)
                      fileInput.current?.click()
                    }}
                  >
                    追加
                  </Button>
                  <Button
                    variant="destructive"
                    size="xs"
                    disabled={trainingRunning || deleting !== null}
                    title={
                      trainingRunning
                        ? '训练任务进行中，后端拒绝删除数据集'
                        : '删除服务器上的数据集目录'
                    }
                    onClick={() => requestDelete(d.name)}
                  >
                    {deleting === d.name && <LoaderCircleIcon className="animate-spin" />}
                    <Trash2Icon />
                    {isConfirm ? '确认删除？' : '删除'}
                  </Button>
                </div>
                {isConfirm && (
                  <p className="text-destructive">
                    将删除服务器上的目录与其中全部音频，不可恢复；已有实验的 filelist 仍指向该路径，
                    删除后重跑处理数据会失败。
                  </p>
                )}
              </li>
            )
          })}
        </ul>
      )}
      {deleteError !== null && (
        <p role="alert" className="text-xs text-destructive">
          删除失败：{deleteError}
        </p>
      )}
    </div>
  )
}
