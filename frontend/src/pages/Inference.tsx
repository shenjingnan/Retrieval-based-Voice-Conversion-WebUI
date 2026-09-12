/**
 * 推理页（P1）：模型选择 + 基础/专家参数分层 + 转换前后 A/B 对比试听。
 */
import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { ChevronDownIcon, DownloadIcon, LoaderCircleIcon } from 'lucide-react'

import { api, type RvcModel } from '@/api/client'
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
import { Slider } from '@/components/ui/slider'
import { COLLAPSIBLE_TRIGGER_CLASS } from '@/lib/ui'
import { errorMessage } from '@/lib/utils'

/**
 * 推理端点的音高算法（含 fcpe）。与训练端的 TrainF0Method（仅 rmvpe/pm）取值不同，
 * 故本页保留本地类型，避免与 client.ts 的训练用类型同名不同形。
 */
type F0Method = 'rmvpe' | 'pm' | 'fcpe'

const F0_METHODS: ReadonlyArray<{ value: F0Method; label: string }> = [
  { value: 'rmvpe', label: 'rmvpe（推荐）' },
  { value: 'pm', label: 'pm（最快，质量一般）' },
  { value: 'fcpe', label: 'fcpe（抗变速）' },
]

/** 0 表示按模型原始采样率输出，不做重采样 */
const RESAMPLE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 0, label: '不重采样' },
  { value: 16000, label: '16000 Hz' },
  { value: 44100, label: '44100 Hz' },
  { value: 48000, label: '48000 Hz' },
]

/** 变调展示：0 -> "0"，正数补 "+" */
function signed(n: number): string {
  return n > 0 ? `+${n}` : String(n)
}

/** Base UI Slider 的回调值是 `number | readonly number[]` 联合，本页都是单滑块，取首值即可 */
function firstValue(v: number | readonly number[]): number {
  return typeof v === 'number' ? v : v[0]
}

export interface InferencePageProps {
  /** 训练页「去试音」联动：模型列表就绪后自动选中该模型（消费一次后由 App 清空） */
  initialModel?: string | null
  /** initialModel 处理完成后回调（App 据此清空 pendingModel，避免 Tab 来回时重复选中） */
  onModelConsumed?: () => void
}

export function InferencePage({ initialModel = null, onModelConsumed }: InferencePageProps) {
  // null 表示模型列表仍在加载；[] 表示已加载但 assets/weights 为空
  const [models, setModels] = useState<RvcModel[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [transpose, setTranspose] = useState(0)
  const [f0Method, setF0Method] = useState<F0Method>('rmvpe')
  const [indexRate, setIndexRate] = useState(0.75)
  const [resampleSr, setResampleSr] = useState(0)
  const [rmsMixRate, setRmsMixRate] = useState(0.25)
  const [protect, setProtect] = useState(0.33)
  const [file, setFile] = useState<File | null>(null)
  const [srcUrl, setSrcUrl] = useState<string | null>(null)
  const [outUrl, setOutUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 训练页「去试音」联动：模型列表就绪后选中指定模型并通知 App 消费。
  // 只在挂载后的首次加载消费——「去试音」必然伴随本组件重新挂载（跨 Tab 切换），
  // 因此用 ref 快照捕获首帧 prop 而不进依赖数组。模型不在列表里（跳转后产物被删的
  // 极端竞态）静默跳过，不覆盖正常的加载提示
  const linkedModel = useRef(initialModel)
  const linkedCallback = useRef(onModelConsumed)
  useEffect(() => {
    let cancelled = false
    api
      .models()
      .then((list) => {
        if (cancelled) return
        setModels(list)
        const model = linkedModel.current
        if (model !== null) {
          if (list.some((m) => m.name === model)) setSelected(model)
          linkedCallback.current?.()
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setLoadError(errorMessage(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  // objectURL 生命周期：各自独立 effect——依赖变化时 revoke 旧值，卸载时 revoke 当前值。
  // 不能合并成一个 effect：共享 deps 会让其中一者的变化触发 cleanup，误 revoke 另一个仍在使用的 URL
  useEffect(() => {
    if (srcUrl === null) return
    return () => URL.revokeObjectURL(srcUrl)
  }, [srcUrl])

  useEffect(() => {
    if (outUrl === null) return
    return () => URL.revokeObjectURL(outUrl)
  }, [outUrl])

  // 转换是异步的，中途切 Tab 卸载后 setState 会被忽略，blob 已创建却无人引用 → 泄漏
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const paired = models?.find((m) => m.name === selected) ?? null
  const missingHint =
    selected === null ? '请先选择模型' : file === null ? '请先选择音频文件' : null
  const canConvert = !busy && missingHint === null && models !== null

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0] ?? null
    if (picked === null) return // 取消选择：保持现状，不清结果
    setFile(picked)
    setError(null)
    setOutUrl(null) // 换音频后旧结果不再适用
    setSrcUrl(URL.createObjectURL(picked))
    // 清空后再次选择同一文件也能触发 onChange
    e.target.value = ''
  }

  // 上传导入成功：上传响应即服务端扫描条目，本地并入列表（去重防同名竞态）并即时
  // 选中——上传完即可推理。与下次进页 GET /api/models 的差异只有下拉顺序，不做整表重取
  function onModelUploaded(model: RvcModel) {
    setModels((cur) => [...(cur ?? []).filter((m) => m.name !== model.name), model])
    setSelected(model.name)
  }

  async function convert() {
    if (selected === null || file === null) return
    setBusy(true)
    setError(null)
    // 字段名与 server/api/infer.py 的 Form 参数逐字对齐
    const form = new FormData()
    form.append('audio', file)
    form.append('model', selected)
    form.append('transpose', String(transpose))
    form.append('f0_method', f0Method)
    form.append('index_rate', String(indexRate))
    form.append('resample_sr', String(resampleSr))
    form.append('rms_mix_rate', String(rmsMixRate))
    form.append('protect', String(protect))
    form.append('index_path', paired?.index ?? '')
    try {
      const blob = await api.infer(form)
      const url = URL.createObjectURL(blob)
      if (mounted.current) setOutUrl(url)
      else URL.revokeObjectURL(url) // 已卸载：立即回收，避免悬挂 blob
    } catch (e) {
      if (mounted.current) setError(errorMessage(e))
    } finally {
      if (mounted.current) setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>推理</CardTitle>
        <CardDescription>选模型、传音频，转换前后 A/B 对比试听。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        {loadError !== null && (
          <p
            role="alert"
            className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive"
          >
            模型列表加载失败：{loadError}
          </p>
        )}

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium">模型</span>
          {models === null && loadError === null ? (
            <p className="text-sm text-muted-foreground">正在加载模型列表…</p>
          ) : models !== null && models.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              还没有音色模型——可以去训练，或在下方导入已有模型文件。
            </p>
          ) : (
            <>
              <Select value={selected} onValueChange={setSelected}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="选择要使用的音色模型" />
                </SelectTrigger>
                <SelectContent>
                  {(models ?? []).map((m) => (
                    <SelectItem key={m.name} value={m.name}>
                      {m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {paired !== null && paired.index === null && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  该模型没有配对索引，音色相似度会打折
                </p>
              )}
            </>
          )}
          {/* 上传入口常驻（列表为空/加载失败时也能导入）；成功后列表就地并入并选中新模型 */}
          <Collapsible>
            <CollapsibleTrigger className={COLLAPSIBLE_TRIGGER_CLASS}>
              上传音色模型
              <ChevronDownIcon className="size-4 text-muted-foreground" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4">
              <ModelUploadForm onUploaded={onModelUploaded} />
            </CollapsibleContent>
          </Collapsible>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">变调（半音）</span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {signed(transpose)}
            </span>
          </div>
          <Slider
            value={[transpose]}
            onValueChange={(v) => setTranspose(firstValue(v))}
            min={-12}
            max={12}
            step={1}
            aria-label="变调（半音）"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium">音高算法</span>
          <Select value={f0Method} onValueChange={(v) => v !== null && setF0Method(v)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {F0_METHODS.map(({ value, label }) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">音色相似度</span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {indexRate.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[indexRate]}
            onValueChange={(v) => setIndexRate(firstValue(v))}
            min={0}
            max={1}
            step={0.05}
            aria-label="音色相似度"
          />
        </div>

        <Collapsible>
          <CollapsibleTrigger className={COLLAPSIBLE_TRIGGER_CLASS}>
            专家参数
            <ChevronDownIcon className="size-4 text-muted-foreground" />
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-4 pt-4">
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">输出重采样</span>
              <Select
                value={String(resampleSr)}
                onValueChange={(v) => v !== null && setResampleSr(Number(v))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RESAMPLE_OPTIONS.map(({ value, label }) => (
                    <SelectItem key={value} value={String(value)}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">音量包络混合（rms_mix_rate）</span>
                <span className="text-sm tabular-nums text-muted-foreground">
                  {rmsMixRate.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[rmsMixRate]}
                onValueChange={(v) => setRmsMixRate(firstValue(v))}
                min={0}
                max={1}
                step={0.05}
                aria-label="音量包络混合（rms_mix_rate）"
              />
              <p className="text-xs text-muted-foreground">
                0 完全用转换后的响度，1 完全跟随原音频的响度起伏
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">保护清辅音（protect）</span>
                <span className="text-sm tabular-nums text-muted-foreground">
                  {protect.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[protect]}
                onValueChange={(v) => setProtect(firstValue(v))}
                min={0}
                max={0.5}
                step={0.01}
                aria-label="保护清辅音（protect）"
              />
              <p className="text-xs text-muted-foreground">
                小于 0.5 才生效，越小保护越强，可减少齿音失真
              </p>
            </div>
          </CollapsibleContent>
        </Collapsible>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="infer-audio" className="text-sm font-medium">
            待转换音频
          </label>
          <Input
            id="infer-audio"
            type="file"
            accept="audio/*"
            onChange={onFileChange}
          />
        </div>

        <div className="flex items-center gap-3">
          <Button onClick={convert} disabled={!canConvert}>
            {busy && <LoaderCircleIcon className="animate-spin" />}
            {busy ? '转换中…' : '开始转换'}
          </Button>
          {!busy && missingHint !== null && (
            <span className="text-xs text-muted-foreground">{missingHint}</span>
          )}
        </div>

        {error !== null && <ErrorDetail title="转换失败">{error}</ErrorDetail>}

        {srcUrl !== null && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">原声 A</span>
              <audio controls src={srcUrl} className="w-full" />
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium">转换后 B</span>
              {outUrl === null ? (
                <p className="text-sm text-muted-foreground">尚未转换</p>
              ) : (
                <>
                  <audio controls src={outUrl} className="w-full" />
                  <a
                    href={outUrl}
                    download="rvc_converted.wav"
                    className="inline-flex items-center gap-1 text-sm text-primary underline-offset-4 hover:underline"
                  >
                    <DownloadIcon className="size-3.5" />
                    下载转换结果
                  </a>
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
