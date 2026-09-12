/**
 * 外部音色模型上传表单（推理页 / 模型管理页共用）：pth 必选 + index 可选双文件槽。
 * 客户端先做后缀预检（与服务端 upload_model 同口径，免得几百 MB 传完才被 400 打回）；
 * 上传走 api.uploadModel 的 XHR 进度，成功后回调 onUploaded——父级负责刷新列表 /
 * 选中新模型，本组件只负责把文件安全送上去并如实展示结果。
 */
import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { LoaderCircleIcon } from 'lucide-react'

import { api, type RvcModel } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { errorMessage } from '@/lib/utils'

/** 后缀预检与服务端一致地大小写敏感：_scan 的 glob 口径不认大写后缀 */
function hasSuffix(file: File, suffix: string): boolean {
  return file.name.endsWith(suffix)
}

export function ModelUploadForm({ onUploaded }: { onUploaded: (model: RvcModel) => void }) {
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [indexFile, setIndexFile] = useState<File | null>(null)
  // null = 空闲；上传中是 0-100 的百分比
  const [progress, setProgress] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploaded, setUploaded] = useState<RvcModel | null>(null)

  // 异步回调里的 setState 保护：切 Tab 卸载后 React 19 会静默忽略，这里显式短路
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  function pickFile(
    e: ChangeEvent<HTMLInputElement>,
    suffix: string,
    set: (file: File | null) => void,
  ) {
    const picked = e.target.files?.[0] ?? null
    // 清空后再次选择同一文件也能触发 onChange
    e.target.value = ''
    setUploaded(null)
    setError(null)
    if (picked === null) return
    if (!hasSuffix(picked, suffix)) {
      set(null)
      setError(`${suffix === '.pth' ? '模型' : '索引'}文件必须是 ${suffix} 后缀（小写）：${picked.name}`)
      return
    }
    set(picked)
  }

  async function upload() {
    if (modelFile === null || busy) return
    setBusy(true)
    setError(null)
    setUploaded(null)
    setProgress(0)
    try {
      const model = await api.uploadModel(modelFile, indexFile, (pct) => {
        if (mounted.current) setProgress(pct)
      })
      if (!mounted.current) return
      setUploaded(model)
      setModelFile(null)
      setIndexFile(null)
      onUploaded(model)
    } catch (e) {
      if (mounted.current) setError(errorMessage(e))
    } finally {
      if (mounted.current) {
        setBusy(false)
        setProgress(null)
      }
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <label htmlFor="upload-model-pth" className="text-sm font-medium">
          模型权重（.pth，必选）
        </label>
        <Input
          id="upload-model-pth"
          type="file"
          accept=".pth"
          disabled={busy}
          onChange={(e) => pickFile(e, '.pth', setModelFile)}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <label htmlFor="upload-model-index" className="text-sm font-medium">
          特征索引（.index，可选）
        </label>
        <Input
          id="upload-model-index"
          type="file"
          accept=".index"
          disabled={busy}
          onChange={(e) => pickFile(e, '.index', setIndexFile)}
        />
        <p className="text-xs text-muted-foreground">
          没有索引也能推理，但音色相似度会打折；索引文件名需与模型实验名对应才会自动配对。
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Button size="sm" disabled={modelFile === null || busy} onClick={() => void upload()}>
          {busy && <LoaderCircleIcon className="animate-spin" />}
          {busy ? `上传中 ${progress ?? 0}%` : '导入'}
        </Button>
        {!busy && modelFile === null && (
          <span className="text-xs text-muted-foreground">请先选择 .pth 文件</span>
        )}
      </div>

      {progress !== null && (
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-primary transition-[width] duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {error !== null && (
        <p role="alert" className="rounded-lg bg-destructive/10 p-3 text-xs text-destructive">
          {error}
        </p>
      )}
      {uploaded !== null && (
        <div className="rounded-lg bg-emerald-500/10 p-3 text-xs text-emerald-700 dark:text-emerald-400">
          已导入 {uploaded.name}
          {uploaded.index !== null
            ? `，配对索引：${uploaded.index.split('/').pop() ?? uploaded.index}。`
            : '，未提供索引（推理时音色相似度会打折）。'}
        </div>
      )}
    </div>
  )
}
