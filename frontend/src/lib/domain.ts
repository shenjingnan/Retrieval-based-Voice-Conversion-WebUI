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

/** 数据集路径（后端 _check_dataset_dir 拒绝引号、$、反引号、换行与结尾反斜杠） */
export const DATASET_BAD_RE = /["`$\r\n]/

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
