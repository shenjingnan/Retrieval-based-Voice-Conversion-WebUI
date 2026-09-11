/**
 * uploadPool 并发池的契约测试：并发上限、逐项恰好一次、途中入队拾取、
 * 单项失败隔离、边界（项数少于并发数 / 空队列）。
 */
import { describe, expect, it } from 'vitest'

import { pumpPool } from './uploadPool'

/** 测试队列：记录在途峰值与完成顺序，process 人为延时模拟网络传输。
 *  takeNext 从 items 头部取（shift），addItem 尾部追加（cursor 递增编号）——
 *  与 DatasetPicker 的 uploadsRef「每轮重扫 queued」在语义上同构。 */
function makeQueue(initialCount: number, delayMs: number) {
  let cursor = initialCount
  let inFlight = 0
  let peakInFlight = 0
  const items: number[] = Array.from({ length: initialCount }, (_, i) => i)
  const completed: number[] = []
  return {
    addItem: () => items.push(cursor++),
    completed: () => completed,
    peakInFlight: () => peakInFlight,
    takeNext: () => {
      const next = items.shift()
      return next === undefined ? null : next
    },
    process: async (item: number) => {
      inFlight += 1
      peakInFlight = Math.max(peakInFlight, inFlight)
      await new Promise((resolve) => setTimeout(resolve, delayMs))
      inFlight -= 1
      completed.push(item)
    },
  }
}

describe('pumpPool', () => {
  it('在途处理数不超过 concurrency，且能打满', async () => {
    const q = makeQueue(10, 10)
    await pumpPool({ concurrency: 3, takeNext: q.takeNext, begin: () => {}, process: q.process })
    // < 3 说明串行化了（没打满），> 3 说明超并发——两种实现错误都要抓
    expect(q.peakInFlight()).toBe(3)
  })

  it('每个项恰好被处理一次，且全部完成', async () => {
    const q = makeQueue(12, 2)
    await pumpPool({ concurrency: 5, takeNext: q.takeNext, begin: () => {}, process: q.process })
    expect(q.completed().sort((a, b) => a - b)).toEqual(Array.from({ length: 12 }, (_, i) => i))
  })

  it('泵送途中加入的项也会被拾取（模拟 addFiles / 重试并发触发）', async () => {
    const q = makeQueue(1, 5)
    const original = q.process
    let first = true
    const wrapped = async (item: number) => {
      if (first) {
        first = false
        q.addItem() // 第一个项处理启动的同时再入队一项
      }
      await original(item)
    }
    await pumpPool({ concurrency: 5, takeNext: q.takeNext, begin: () => {}, process: wrapped })
    expect(q.completed().sort((a, b) => a - b)).toEqual([0, 1])
  })

  it('单项抛错不拖垮池：其余项照常处理完', async () => {
    const q = makeQueue(4, 2)
    const original = q.process
    const wrapped = async (item: number) => {
      if (item === 1) throw new Error('boom')
      await original(item)
    }
    await pumpPool({ concurrency: 2, takeNext: q.takeNext, begin: () => {}, process: wrapped })
    expect(q.completed().sort((a, b) => a - b)).toEqual([0, 2, 3])
  })

  it('项数少于 concurrency：全部完成后正常返回', async () => {
    const q = makeQueue(3, 5)
    await pumpPool({ concurrency: 5, takeNext: q.takeNext, begin: () => {}, process: q.process })
    expect(q.peakInFlight()).toBe(3)
    expect(q.completed().sort((a, b) => a - b)).toEqual([0, 1, 2])
  })

  it('空队列：立即返回，不处理任何项', async () => {
    const q = makeQueue(0, 5)
    await pumpPool({ concurrency: 5, takeNext: q.takeNext, begin: () => {}, process: q.process })
    expect(q.completed()).toEqual([])
  })
})
