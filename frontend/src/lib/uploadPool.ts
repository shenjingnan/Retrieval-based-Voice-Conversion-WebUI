/**
 * uploadPool：并发池泵——至多 concurrency 个 worker 同时处理队列里的项，每轮各自
 * 重扫队列（takeNext），直到取空。供 DatasetPicker 的上传泵使用。
 *
 * 认领原子性靠调用方保证：takeNext → begin 之间没有 await，begin 把状态同步改写为
 * 「进行中」后，其他 worker 的 takeNext 就不会再取到同一项。process 约定自行处理
 * 单项错误；即便抛出也只影响当前项，不拖垮池里的其他 worker。
 */
export interface PumpPoolOptions<T> {
  /** 同一时刻的最大在途处理数（≥1） */
  concurrency: number
  /** 取下一个待处理项；没有则返回 null（该 worker 退出，池在全部 worker 退出后 resolve） */
  takeNext: () => T | null
  /** 认领项：takeNext 之后、process 之前同步调用 */
  begin: (item: T) => void
  /** 处理单项（如一次上传 XHR） */
  process: (item: T) => Promise<void>
}

export async function pumpPool<T>(options: PumpPoolOptions<T>): Promise<void> {
  async function worker() {
    for (;;) {
      const item = options.takeNext()
      if (item === null) return
      options.begin(item)
      try {
        await options.process(item)
      } catch {
        // 单项失败不拖垮池；错误信息已由 process 内部上报到调用方自己的状态里
      }
    }
  }
  await Promise.all(Array.from({ length: options.concurrency }, () => worker()))
}
