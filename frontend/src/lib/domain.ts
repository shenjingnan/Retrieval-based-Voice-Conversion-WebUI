/**
 * 领域规则纯函数/常量（无组件、无副作用）：从 pages/{Training,Models}.tsx 收敛而来，
 * 每条规则注明对应的 server 侧同源实现，两侧需同步修改。
 */

/**
 * 实验名校验（与 server/api/training.py 的 _EXP_FORBIDDEN 同源）：
 * 拒绝空白、双引号、反斜杠、$、反引号与路径分隔符，另拒绝 "." / ".."（后者由
 * 调用方单独比较，正则本身已放行它们）
 */
export const EXP_NAME_RE = /^[^\s"\\$`/]+$/

/**
 * 数据集音频口径（与 server/api/datasets.py 的 AUDIO_SUFFIXES 同源）：preprocess
 * 遍历数据集目录不过滤扩展名（load_audio 走 ffmpeg），这里只列常见音频后缀，
 * 其余文件由后端计入 other_count 并提示。
 */
export const AUDIO_SUFFIXES: readonly string[] = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

/**
 * 人声分离模型 label（与 server/api/datasets.py 的 SEPARATION_MODELS 逐字同源，
 * 顺序即下拉顺序；server 侧注释同 tools.pymss_webui.MODEL_SPECS，改动须三处同步）。
 */
export const SEPARATION_MODELS: readonly string[] = [
  '去混响',
  '去混响（激进）',
  '去伴奏',
  '去伴奏（激进）',
  '提主旋律',
]

/** 默认分离模型（与 server/api/datasets.py 的 DEFAULT_SEPARATION_MODEL 同源） */
export const DEFAULT_SEPARATION_MODEL = '去伴奏'

/** 分离产物数据集固定后缀（与 server/api/datasets.py 的 DERIVED_SUFFIX 同源） */
export const DERIVED_SUFFIX = '_vocals'

/** 分离产物数据集名：{source}_vocals（后端同名规则，前端预显示用） */
export function derivedDatasetName(source: string): string {
  return source + DERIVED_SUFFIX
}

/**
 * 数据集名校验（与 server/api/datasets.py 的 _check_name 同源：字符集同 EXP_NAME_RE
 * + 拒前导点 + 拒 NUL）。前导点被拒是因为 .meta 侧车目录与隐藏目录不能被当成数据集
 * 访问；NUL 不是 \s，EXP_NAME_RE 放行它，但漏到后端 mkdir/unlink 层是 ValueError，
 * 两侧都在入口先拒。server 改规则必须双侧同步。
 */
export function isValidDatasetName(s: string): boolean {
  return s.length > 0 && !s.startsWith('.') && !s.includes('\0') && EXP_NAME_RE.test(s)
}

/**
 * 秒 → 「12 分 34 秒」；null（后端时长探测全失败）→ 「未知」。
 * 不换算小时：数据集以分钟计，与后端不做的规范化保持一致（超大值显示总分钟数）。
 */
export function formatDuration(sec: number | null): string {
  if (sec === null) return '未知'
  const total = Math.round(sec)
  const min = Math.floor(total / 60)
  const s = total % 60
  return min > 0 ? `${min} 分 ${s} 秒` : `${s} 秒`
}

/** 字节 → GiB 文本（1 位小数）。单位「GiB」由调用方拼接，null 由调用方先行降级 */
export function formatGib(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1)
}

/**
 * 模型名 stem → 实验名（与 server/api/models.py 的 experiment_name 同源，含
 * IGNORECASE）：剥训练产物的 epoch/step 后缀，alice_v2_e20_s100 → alice_v2
 */
export const EPOCH_SUFFIX_RE = /_e\d+_s\d+$/i

export function experimentName(modelStem: string): string {
  return modelStem.replace(EPOCH_SUFFIX_RE, '')
}

export function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * 随机十六进制串（小写，定长）。crypto.randomUUID 只在 secure context 存在
 * （localhost 算，局域网 http 不算），getRandomValues 则到处可用——统一走后者。
 * 消费方：上传会话 ID（ds-xxxxxxxx）、隐藏实验名的随机后缀等
 */
export function randomHex(len: number): string {
  const bytes = new Uint8Array(Math.ceil(len / 2))
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('').slice(0, len)
}

/**
 * 训练产物命名（train/train.py savee）：最终模型 {exp}.pth；开启 save_every_weights
 * 时另存 {exp}_e{n}_s{n}.pth，取 epoch 最大的一份；两者都没有返回 null
 */
export function pickProductName(models: Array<{ name: string }>, exp: string): string | null {
  const exact = `${exp}.pth`
  if (models.some((m) => m.name === exact)) return exact
  const re = new RegExp(`^${escapeRegExp(exp)}_e(\\d+)_s\\d+\\.pth$`)
  let best: { name: string; epoch: number } | null = null
  for (const m of models) {
    const m1 = m.name.match(re)
    if (m1 === null) continue
    const epoch = Number(m1[1])
    if (best === null || epoch > best.epoch) best = { name: m.name, epoch }
  }
  return best?.name ?? null
}

/** 从训练产物 stem 解析 epoch/step（alice_v2_e20_s100 → {epoch:20, step:…}）；非该命名返回 null */
export function parseEpochSuffix(
  stem: string,
): { epoch: number; step: number } | null {
  const m = stem.match(/_e(\d+)_s(\d+)$/i)
  return m === null ? null : { epoch: Number(m[1]), step: Number(m[2]) }
}

/**
 * 模型文件名 → 分组键（groupModels 与推理页级联选择共用的键口径）：{exp}.pth 与
 * {exp}_e{n}_s{n}.pth 同归实验名一组；剥不出实验名的退化文件名（如裸的
 * _e20_s100.pth）用原始 stem 自成一组。
 */
export function modelGroupKey(modelName: string): string {
  const stem = modelName.replace(/\.pth$/i, '')
  const exp = experimentName(stem)
  return exp.length > 0 ? exp : stem
}

/**
 * 权重文件名的展示文案（推理页级联下拉与最近使用区共用）：最终产物显示「最终产物」，
 * 中间轮次显示「第 N 轮 · step M」（与 Models 页子项文案同源）；命名不合规的
 * 退化文件名原样展示。
 */
export function weightLabel(modelName: string, isFinal: boolean): string {
  if (isFinal) return '最终产物'
  const p = parseEpochSuffix(modelName.replace(/\.pth$/i, ''))
  return p === null ? modelName : `第 ${p.epoch} 轮 · step ${p.step}`
}

/**
 * 推理页「最近使用」模型条目：只在转换成功后记录（点击选择不算），浏览器
 * localStorage 持久化，无 server 侧同源实现。
 */
export interface RecentModelEntry {
  /** 权重文件名（weights 目录 basename，与 GET /api/models 条目 name 同口径） */
  name: string
  /** 最近一次转换成功时刻（epoch 毫秒） */
  at: number
}

export const RECENT_MODELS_STORAGE_KEY = 'rvc.recent-models'
/** 上限裁剪：最近使用是快捷入口不是档案，塞满一屏反而难选 */
export const RECENT_MODELS_LIMIT = 8

/**
 * 纯函数：把 name 记到最前（已存在则摘除旧条目并更新时间），按上限裁剪。
 * 不修改入参数组，返回新数组。
 */
export function recordRecentModel(
  entries: RecentModelEntry[],
  name: string,
  at: number,
): RecentModelEntry[] {
  return [{ name, at }, ...entries.filter((e) => e.name !== name)].slice(
    0,
    RECENT_MODELS_LIMIT,
  )
}

function isRecentEntry(e: unknown): e is RecentModelEntry {
  if (typeof e !== 'object' || e === null) return false
  const o = e as Record<string, unknown>
  return (
    typeof o.name === 'string' &&
    o.name.length > 0 &&
    typeof o.at === 'number' &&
    Number.isFinite(o.at)
  )
}

/**
 * 从 localStorage 读取最近使用列表：JSON 解析失败 / 非数组 / 条目形状不对的
 * 一律丢弃，返回值保证是合法数组——持久化数据损坏绝不能让推理页挂掉。
 * storage 参数注入（而非直接摸 window.localStorage）是为了 node 环境可测。
 */
export function loadRecentModels(storage: Pick<Storage, 'getItem'>): RecentModelEntry[] {
  try {
    const raw = storage.getItem(RECENT_MODELS_STORAGE_KEY)
    if (raw === null) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isRecentEntry)
  } catch {
    return []
  }
}

/**
 * 模型列表按实验名聚合的分组视图（Models 页展示层专用，无 server 侧同源实现）：
 * 一次训练的全部产物——最终模型 {exp}.pth 与 save_every_weights 存下的中间轮次
 * {exp}_e{n}_s{n}.pth——折进同一个组。分组键见 modelGroupKey（剥 _eX_sY
 * 后缀取实验名）；剥不出后缀的退化文件名（如裸的 _e20_s100.pth）用原始 stem
 * 自成一组（该成员落 intermediates，组内无最终产物，代表项退化为它自身）。
 * 命名不合规的手动模型同理。
 * 组间按最近文件时间（latest）倒序——最近训练的排最前；mtime 同秒时按 key
 * 字典序兜底，保证顺序稳定。组内仍按训练演进序（中间轮次 epoch 升序）。
 */
export interface ModelGroup<T extends { name: string; mtime: number }> {
  /** 分组键：实验名；退化文件名时为原始 stem */
  key: string
  /** 最终产物 {exp}.pth；训练未完成（只有中间轮次）或已被单独删除时为 null */
  final: T | null
  /** 中间轮次产物，epoch 升序、同轮按 step 升序 */
  intermediates: T[]
  /** 组卡片代表项：final ?? epoch 最大的中间产物（与 pickProductName 同口径） */
  representative: T
  /** 组内所有产物中最新的文件修改时间（epoch 秒），组间倒序的排序键 */
  latest: number
}

export function groupModels<T extends { name: string; mtime: number }>(
  models: T[],
): ModelGroup<T>[] {
  const stemOf = (name: string): string => name.replace(/\.pth$/i, '')
  const groups = new Map<string, { final: T | null; intermediates: T[] }>()
  for (const m of models) {
    const stem = stemOf(m.name)
    const key = modelGroupKey(m.name)
    const entry = groups.get(key) ?? { final: null, intermediates: [] }
    // 剥得动后缀（stem !== experimentName(stem)）= 中间轮次；剥不动 = 最终产物。
    // 注意退化文件名（剥后为空串）这里落 intermediates，与键的「自成一组」是两回事
    if (stem === experimentName(stem)) entry.final = m
    else entry.intermediates.push(m)
    groups.set(key, entry)
  }
  const byEpochStep = (a: T, b: T): number => {
    const pa = parseEpochSuffix(stemOf(a.name))
    const pb = parseEpochSuffix(stemOf(b.name))
    return (
      (pa?.epoch ?? -1) - (pb?.epoch ?? -1) || (pa?.step ?? -1) - (pb?.step ?? -1)
    )
  }
  return [...groups.entries()]
    .map(([key, g]) => {
      const intermediates = [...g.intermediates].sort(byEpochStep)
      const members = g.final === null ? intermediates : [g.final, ...intermediates]
      return {
        key,
        final: g.final,
        intermediates,
        // 组内至少有一个模型（final 或 intermediates 二者必有其一），不会越界
        representative: g.final ?? intermediates[intermediates.length - 1]!,
        latest: Math.max(...members.map((m) => m.mtime)),
      }
    })
    .sort((a, b) => b.latest - a.latest || a.key.localeCompare(b.key))
}
