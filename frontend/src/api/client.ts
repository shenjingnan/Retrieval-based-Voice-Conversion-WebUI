export interface RvcModel {
  name: string
  path: string
  index: string | null
}

/** server/tasks.py 的状态机（pending → running → success | failed | cancelled） */
export type TaskState = 'pending' | 'running' | 'success' | 'failed' | 'cancelled'

/** GET /api/tasks 的紧凑投影 */
export interface TaskSummary {
  id: string
  name: string
  state: TaskState
  progress: number | null
  error: string | null
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
}

/** 训练接口的统一返回体：{task_id} */
export interface TaskCreated {
  task_id: string
}

/** DELETE /api/models/{name} 的返回体：联动删除的索引文件名列表见 server/api/models.py */
export interface DeleteModelResult {
  deleted_model: string
  deleted_indices: string[]
  /** 删除失败（权限/占用等）的索引文件名；联动不回滚，如实上报 */
  failed_indices: string[]
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

  infer: (form: FormData): Promise<Blob> =>
    fetch('/api/infer', { method: 'POST', body: form }).then(async (r) => {
      if (!r.ok) throw await errorFrom(r)
      return r.blob()
    }),

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
    }),

  getTasks: (): Promise<TaskSummary[]> =>
    fetch('/api/tasks').then((r) => handle<TaskSummary[]>(r)),

  /** 终止任务。后端已终态时返回幂等 no-op 的 202，2xx（resp.ok）一并视为成功 */
  cancelTask: async (taskId: string): Promise<void> => {
    const resp = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`, {
      method: 'DELETE',
    })
    if (!resp.ok) throw await errorFrom(resp)
    await resp.json().catch(() => null)
  },
}
