export { cn } from "cn"

/** 任意抛出值 → 可读信息：fetch 封装抛的都是 Error，但 catch 到的类型是 unknown */
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}
