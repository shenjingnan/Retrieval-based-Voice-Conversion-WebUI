/**
 * 跨页面复用的 UI 样式常量（只收敛逐字重复的长 className，不做组件抽象）。
 */

/** 可折叠区的触发器（标题 + 展开箭头）：Inference / Training 的高级参数折叠头共用 */
export const COLLAPSIBLE_TRIGGER_CLASS =
  'flex w-full items-center justify-between rounded-lg border px-3 py-2 text-sm font-medium select-none hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none [&_svg]:transition-transform [&[aria-expanded=true]_svg]:rotate-180'
