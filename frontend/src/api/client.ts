export interface RvcModel {
  name: string
  path: string
  index: string | null
  /** 文件修改时间（epoch 秒），模型页据此把最近训练的组排前面 */
  mtime: number
}

/** server/tasks.py 的状态机（pending → running → success | failed | cancelled） */
export type TaskState = 'pending' | 'running' | 'success' | 'failed' | 'cancelled'

/**
 * GET /api/tasks 的行投影（内存任务与磁盘历史两类行字段形状一致，history 区分）。
 * 历史行是终态快照投影：queue_position 恒为 null、progress 为记录时值。
 */
export interface TaskSummary {
  id: string
  name: string
  state: TaskState
  progress: number | null
  error: string | null
  /** 1-based 队列位次；仅排队（pending）任务有值，运行中/终态为 null */
  queue_position: number | null
  /** 训练任务的实验名（任务元数据登记）；separate 等无实验语义的任务为 null */
  exp_name: string | null
  /** 任务类型（preprocess/extract/fit/index/pipeline/separate） */
  kind: string | null
  /** 提交时的请求体（重新提交 = 同名实验断点续训） */
  params: Record<string, unknown> | null
  /** pipeline 的阶段归属表（与 cmds 等长）；单步任务为 null */
  pipeline_stages: string[] | null
  created_at: number | null
  finished_at: number | null
  /** true = 来自磁盘历史（服务重启前完成的任务） */
  history: boolean
}

/** GET /api/tasks/{id} 对历史任务的详情（内存任务走 SSE，不需要它） */
export interface TaskHistoryDetail extends Omit<TaskSummary, 'history'> {
  history: true
  /** 任务结束时的日志尾部快照（磁盘任务日志逐 cmd 截断，以此为准） */
  logs_tail: string[]
  /** 命令串（解析失败阶段用） */
  cmds: string[]
  current_cmd: number | null
  started_at: number | null
  log_path: string | null
}

export type SampleRate = '48k' | '40k' | '32k'
export type ModelVersion = 'v2' | 'v1'

/**
 * 训练端点的音高算法（server/api/training.py _F0_METHODS：仅 rmvpe / pm）。
 * 带 Train 前缀与推理端取值消歧——/api/infer 还支持 fcpe（Inference 页本地类型），
 * 两者同名不同形容易混用。
 */
export type TrainF0Method = 'rmvpe' | 'pm'

/**
 * 训练参数（字段并集）。与 server/api/training.py 的 PreprocessBody / ExtractBody /
 * FitBody / PipelineBody 逐字段对齐；分步接口只取各自需要的子集。
 * n_p（进程数）不在其中：后端固定取 os.cpu_count()，前端没有调参入口。
 * batch_size 可选：省略（自动）时后端按设备自适应解析（webui 显存GB÷2，无卡为 1）。
 * allow_existing 仅在 pipeline 生效（默认 false = 实验目录已存在则 409）；任务历史
 * 「重新提交」显式传 true——失败任务的目录必然已存在，重提交是续训意图。
 */
export interface TrainParams {
  exp_name: string
  dataset_dir: string
  sr: SampleRate
  version: ModelVersion
  if_f0: boolean
  f0_method: TrainF0Method
  total_epoch: number
  save_every_epoch: number
  batch_size?: number
  save_every_weights: boolean
  allow_existing?: boolean
}

/** 训练接口的统一返回体。已有任务运行中时新任务自动排队：queued=true 且
 *  queue_position 给出当前位次（立即可跑时为 false/null） */
export interface TaskCreated {
  task_id: string
  queued: boolean
  queue_position: number | null
}

/** DELETE /api/tasks（停止并清空队列）的返回体 */
export interface ClearQueueResult {
  /** 被停止的运行中任务 id（无则为 null） */
  stopped_task: string | null
  /** 被取消的排队任务数 */
  cancelled_pending: number
}

/** DELETE /api/models/{name} 的返回体：彻底删除整组（权重 + 索引 + logs），字段见 server/api/models.py */
export interface DeleteModelResult {
  deleted_model: string
  /** 全组被删的权重文件名（含最终模型与中间轮次，含 deleted_model 本身） */
  deleted_models: string[]
  /** 删除失败（权限/占用等）的组内权重文件名；不回滚，如实上报 */
  failed_models: string[]
  deleted_indices: string[]
  /** 删除失败（权限/占用等）的索引文件名；联动不回滚，如实上报 */
  failed_indices: string[]
  /** logs/{exp} 训练产物清理结果：target 为绝对路径（本来就没有产物时为 null） */
  logs: { target: string | null; removed: boolean; failed_files: string[] }
}

/** GET /api/datasets 的条目（server/api/datasets.py _scan_dataset 的 summary，逐字段对齐） */
export interface DatasetSummary {
  name: string
  /** 数据集目录在服务器上的绝对路径（后端下发，前端原样回填 dataset_dir，不自行拼路径） */
  path: string
  /** 音频文件数（点开头文件不计入） */
  file_count: number
  /** 非音频文件数：preprocess 遍历目录不过滤扩展名，这些文件也会被切分脚本处理 */
  other_count: number
  total_bytes: number
  /** 秒；时长探测全失败（含空数据集）为 null，前端显示「未知」而非误导性的 0 秒 */
  total_duration: number | null
  /** 人声分离的源数据集名；非衍生数据集为 null（.meta 衍生标记缺失/损坏也按 null） */
  derived_from: string | null
}

/** GET /api/datasets/{name}：概览字段 + files 明细（音频文件，按名排序） */
export interface DatasetDetail extends DatasetSummary {
  files: Array<{ name: string; size: number; duration: number | null }>
}

/** 上传响应里被后端拒收的单个文件（非音频 / 0 字节 / 点开头 / 空文件名） */
export interface SkippedFile {
  name: string
  reason: string
}

/** POST /api/datasets/{name}/files 的返回体（multipart `files`，可多文件） */
export interface UploadResult {
  dataset: string
  path: string
  /** true = 本次新建了数据集目录；false = 追加到已有目录（重名数据集即追加，非错误） */
  created: boolean
  /** 实际落盘的文件名（与目录内已有文件重名时已按 stem_1.ext 改写） */
  added: string[]
  skipped: SkippedFile[]
  file_count: number
  total_duration: number | null
}

/** DELETE /api/datasets/{name} 的返回体：删除失败（占用/权限）的文件如实上报，不回滚 */
export interface DeleteDatasetResult {
  deleted: boolean
  failed_files: string[]
}

/**
 * POST /api/datasets/{name}/separate 的返回体：分离任务已创建（与训练共用串行队列，
 * 训练任务运行中时自动排队）。output_dataset 恒为 {name}_vocals，前端在任务成功后
 * 按名刷新并选中它。
 */
export interface SeparateDatasetResult {
  task_id: string
  queued: boolean
  queue_position: number | null
  output_dataset: string
  output_path: string
}

/**
 * DELETE /api/datasets/{name}/files/{filename} 的返回体：删除数据集内的单个音频文件
 * （训练任务进行中 409 拒删，与整目录删除同一互斥口径）。
 */
export interface DeleteDatasetFileResult {
  deleted: boolean
}

/** GET /api/system/stats 单块 GPU 快照（server/api/system.py _parse_nvidia_smi 逐字段对齐） */
export interface GpuStats {
  /** 设备序号（nvidia-smi index，从 0 起） */
  index: number
  /** 产品名，如 "NVIDIA GeForce RTX 4090" */
  name: string
  /** 0-100；WDDM 等驱动不支持该计数器时为 null（逐字段降级，非整卡降级） */
  utilization_percent: number | null
  memory_used_bytes: number | null
  memory_total_bytes: number | null
}

/** CPU 快照；psutil 缺失时整个对象为 null */
export interface CpuStats {
  /** 0-100，1 位小数；服务启动后的首个采样帧可能读得偏低（psutil 差分采样） */
  percent: number
  /** 逻辑核数 */
  count: number
}

/** 内存快照（used = total - available，与任务管理器同口径）；psutil 缺失时为 null */
export interface MemoryStats {
  used_bytes: number
  total_bytes: number
}

/** 磁盘快照（仓库根目录所在卷，logs/ 与 datasets/ 都在其下）；stat 失败时为 null */
export interface DiskStats {
  used_bytes: number
  free_bytes: number
  total_bytes: number
}

/**
 * GET /api/system/stats：整机资源快照。cpu/memory/disk 为 null 表示该采集项不可用；
 * gpus 为空数组表示无可用 NVIDIA GPU（无驱动 / nvidia-smi 不在 PATH / 采集失败），
 * 两者是刻意的不同形降级语义。端点设计上永不 500，前端失败仅需处理网络层。
 */
export interface SystemStats {
  /** 服务端生成时刻（unix 秒） */
  timestamp: number
  cpu: CpuStats | null
  memory: MemoryStats | null
  disk: DiskStats | null
  gpus: GpuStats[]
}

/**
 * 从错误响应提取可读信息。detail 仅在为字符串时使用——
 * FastAPI 422 校验错误的 detail 是对象数组，直接塞给 Error 会得到 "[object Object]"，
 * 此情况下回退到 HTTP 状态码。
 */
async function errorFrom(resp: Response): Promise<Error> {
  const body: unknown = await resp.json().catch(() => null)
  const detail = (body as { detail?: unknown } | null)?.detail
  return new Error(typeof detail === 'string' ? detail : `HTTP ${resp.status}`)
}

/**
 * errorFrom 的同步版（XMLHttpRequest 拿到的是 .responseText 字符串，没有 Response.json）：
 * detail 提取语义与 errorFrom 完全一致——仅字符串 detail 可用，否则回退 HTTP 状态码。
 */
function errorFromText(text: string, status: number): Error {
  let detail: unknown = null
  try {
    detail = (JSON.parse(text) as { detail?: unknown } | null)?.detail
  } catch {
    detail = null // 非 JSON body（网关错误页等）：与解析失败同义，回退状态码
  }
  return new Error(typeof detail === 'string' ? detail : `HTTP ${status}`)
}

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) throw await errorFrom(resp)
  return resp.json() as Promise<T>
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((r) => handle<T>(r))
}

export const api = {
  models: (): Promise<RvcModel[]> =>
    fetch('/api/models').then((r) => handle<RvcModel[]>(r)),

  /** 删除模型。后端联动删除会配对到它的索引（server/api/models.py delete_model） */
  deleteModel: (name: string): Promise<DeleteModelResult> =>
    fetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' }).then((r) =>
      handle<DeleteModelResult>(r),
    ),

  /**
   * 模型一键打包下载地址（GET，返回含 pth + 配对索引的 zip，server/api/models.py
   * download_model）。下载不走 fetch——大文件进 JS 内存没有意义，直接给 a 标签
   * 让浏览器原生下载，三端拿到 zip 后自行解压。
   */
  modelDownloadUrl: (name: string): string =>
    `/api/models/${encodeURIComponent(name)}/download`,

  infer: (form: FormData): Promise<Blob> =>
    fetch('/api/infer', { method: 'POST', body: form }).then(async (r) => {
      if (!r.ok) throw await errorFrom(r)
      return r.blob()
    }),

  // -- 数据集：列表 / 详情 / 上传 / 删除（server/api/datasets.py） ----------------

  datasets: (): Promise<DatasetSummary[]> =>
    fetch('/api/datasets').then((r) => handle<DatasetSummary[]>(r)),

  dataset: (name: string): Promise<DatasetDetail> =>
    fetch(`/api/datasets/${encodeURIComponent(name)}`).then((r) => handle<DatasetDetail>(r)),

  deleteDataset: (name: string): Promise<DeleteDatasetResult> =>
    fetch(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }).then((r) =>
      handle<DeleteDatasetResult>(r),
    ),

  /**
   * 上传音频到数据集（目录不存在则建、存在则追加），onProgress 上报 0-100 的百分比。
   *
   * 必须用 XMLHttpRequest 而不是 fetch：fetch 的流式能力只覆盖「下载」方向
   * （Response.body），请求体的上传进度只有 XHR 的 xhr.upload.onprogress 能拿到
   * （ProgressEvent.loaded/total），fetch 没有对应的可观测接口。组件侧卸载后无法
   * 中止（abort 会把用户切 Tab 误当成上传失败），由调用方自行丢弃迟到回调。
   */
  uploadDataset: (
    name: string,
    files: File[],
    onProgress: (pct: number) => void,
  ): Promise<UploadResult> =>
    new Promise<UploadResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `/api/datasets/${encodeURIComponent(name)}/files`)
      xhr.upload.onprogress = (e) => {
        // lengthComputable 为 false（通常是请求体大小未知）时保持上一次的进度值
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
      }
      // 网络层失败（断网 / 中断 / 被代理拒绝）：status 恒为 0，detail 无从解析
      xhr.onerror = () => reject(new Error('网络错误，上传中断'))
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as UploadResult)
          } catch {
            reject(new Error(`HTTP ${xhr.status}`))
          }
          return
        }
        reject(errorFromText(xhr.responseText, xhr.status))
      }
      const form = new FormData()
      // 字段名固定 `files`（后端 list[UploadFile]），逐个 append 即多文件
      for (const f of files) form.append('files', f, f.name)
      xhr.send(form)
    }),

  /**
   * 发起人声分离：datasets/{name}/ → 衍生数据集 {name}_vocals/（仅人声 stem，伴奏
   * 丢弃）。与训练共用全局互斥（409）；产物已存在的文件由 runner 幂等跳过，重发即续跑。
   * model 省略取后端默认（去伴奏）；可选值见 domain.ts 的 SEPARATION_MODELS。
   * files 提供时只分离这些文件（逐文件分离），省略 = 整目录。
   */
  separateDataset: (name: string, model?: string, files?: string[]): Promise<SeparateDatasetResult> =>
    postJson(`/api/datasets/${encodeURIComponent(name)}/separate`, {
      ...(model === undefined ? {} : { model }),
      ...(files === undefined ? {} : { files }),
    }),

  /**
   * 删除数据集内的单个音频文件（上传错了就地纠错）。filename 经 encodeURIComponent
   * 编码（服务端文件名允许空格/中文）；训练任务进行中后端 409 拒删。
   */
  deleteDatasetFile: (name: string, filename: string): Promise<DeleteDatasetFileResult> =>
    fetch(
      `/api/datasets/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`,
      { method: 'DELETE' },
    ).then((r) => handle<DeleteDatasetFileResult>(r)),

  // -- 训练：4 步 + 一键（字段子集见 server/api/training.py 各 Body 模型） --------

  trainPreprocess: (p: TrainParams): Promise<TaskCreated> =>
    postJson('/api/train/preprocess', {
      exp_name: p.exp_name,
      dataset_dir: p.dataset_dir,
      sr: p.sr,
    }),

  trainExtract: (p: TrainParams): Promise<TaskCreated> =>
    postJson('/api/train/extract', {
      exp_name: p.exp_name,
      f0_method: p.f0_method,
      version: p.version,
      if_f0: p.if_f0,
    }),

  trainFit: (p: TrainParams): Promise<TaskCreated> =>
    postJson('/api/train/fit', {
      exp_name: p.exp_name,
      sr: p.sr,
      version: p.version,
      if_f0: p.if_f0,
      total_epoch: p.total_epoch,
      save_every_epoch: p.save_every_epoch,
      // 自动（undefined）时键省略：后端按设备自适应解析并在任务日志里记来源
      ...(p.batch_size === undefined ? {} : { batch_size: p.batch_size }),
      save_every_weights: p.save_every_weights,
    }),

  trainIndex: (p: { exp_name: string; version: ModelVersion }): Promise<TaskCreated> =>
    postJson('/api/train/index', { exp_name: p.exp_name, version: p.version }),

  /** GET /api/train/defaults：与 webui 滑条同源的自适应 batch_size（显卡=显存GB÷2，无卡=1） */
  trainDefaults: (): Promise<{ batch_size: number }> =>
    fetch('/api/train/defaults').then((r) => handle<{ batch_size: number }>(r)),

  /**
   * GET /api/train/exp-name/exists：实验名占用检查（输入框防抖查询用）。非法名后端
   * 400——调用方须先做本地校验；检查失败（网络等）由调用方静默降级，提交时后端兜底。
   */
  trainExpNameExists: (name: string): Promise<{ exists: boolean }> =>
    fetch(`/api/train/exp-name/exists?name=${encodeURIComponent(name)}`).then((r) =>
      handle<{ exists: boolean }>(r),
    ),

  // 注意 pipeline 不传 n_p：后端 PipelineBody 未声明该字段（内部固定 os.cpu_count()）
  trainPipeline: (p: TrainParams): Promise<TaskCreated> =>
    postJson('/api/train/pipeline', {
      exp_name: p.exp_name,
      dataset_dir: p.dataset_dir,
      sr: p.sr,
      version: p.version,
      if_f0: p.if_f0,
      f0_method: p.f0_method,
      total_epoch: p.total_epoch,
      save_every_epoch: p.save_every_epoch,
      // 自动（undefined）时键省略：后端按设备自适应解析并在任务日志里记来源
      ...(p.batch_size === undefined ? {} : { batch_size: p.batch_size }),
      save_every_weights: p.save_every_weights,
      ...(p.allow_existing === undefined ? {} : { allow_existing: p.allow_existing }),
    }),

  getTasks: (): Promise<TaskSummary[]> =>
    fetch('/api/tasks').then((r) => handle<TaskSummary[]>(r)),

  /** 任务详情：内存任务返回实时快照，历史任务返回含 logs_tail 的记录（展开时拉一次） */
  taskDetail: (taskId: string): Promise<TaskHistoryDetail> =>
    fetch(`/api/tasks/${encodeURIComponent(taskId)}`).then((r) => handle<TaskHistoryDetail>(r)),

  /** 终止任务（运行中 → 进程组终止；排队中 → 即时出队取消）。
   *  后端已终态时返回幂等 no-op 的 202，2xx（resp.ok）一并视为成功 */
  cancelTask: async (taskId: string): Promise<void> => {
    const resp = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`, {
      method: 'DELETE',
    })
    if (!resp.ok) throw await errorFrom(resp)
    await resp.json().catch(() => null)
  },

  /** 停止并清空队列：终止当前任务 + 取消全部排队任务（幂等，空队列是 no-op） */
  clearQueue: async (): Promise<ClearQueueResult> => {
    const resp = await fetch('/api/tasks', { method: 'DELETE' })
    if (!resp.ok) throw await errorFrom(resp)
    return resp.json() as Promise<ClearQueueResult>
  },

  // -- 系统：整机资源快照（server/api/system.py） --------------------------------

  systemStats: (): Promise<SystemStats> =>
    fetch('/api/system/stats').then((r) => handle<SystemStats>(r)),
}
