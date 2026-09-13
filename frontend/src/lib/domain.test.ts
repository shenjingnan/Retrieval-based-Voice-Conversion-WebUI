/**
 * 领域纯函数的契约测试：模型分组键（modelGroupKey）、权重展示文案（weightLabel）、
 * 推理页「最近使用」列表的记录与读取（recordRecentModel / loadRecentModels，
 * 含 localStorage 损坏数据容错）。
 */
import { describe, expect, it } from 'vitest'

import {
  loadRecentModels,
  modelGroupKey,
  RECENT_MODELS_LIMIT,
  recordRecentModel,
  weightLabel,
  type RecentModelEntry,
} from './domain'

/** 最小 storage 桩：只实现 getItem（loadRecentModels 的全部依赖） */
function stubStorage(get: (key: string) => string | null) {
  return { getItem: get }
}

describe('modelGroupKey', () => {
  it('最终产物与中间轮次同归实验名一组', () => {
    expect(modelGroupKey('alice_v2.pth')).toBe('alice_v2')
    expect(modelGroupKey('alice_v2_e20_s100.pth')).toBe('alice_v2')
  })

  it('后缀匹配不区分大小写，无扩展名的名字同样剥得动', () => {
    expect(modelGroupKey('Alice_E20_S100.PTH')).toBe('Alice')
    expect(modelGroupKey('bob_e1_s5000')).toBe('bob')
  })

  it('剥不出实验名的退化文件名用原始 stem 自成一组', () => {
    expect(modelGroupKey('_e20_s100.pth')).toBe('_e20_s100')
  })
})

describe('weightLabel', () => {
  it('最终产物固定显示「最终产物」，与文件名无关', () => {
    expect(weightLabel('alice_v2.pth', true)).toBe('最终产物')
  })

  it('中间轮次显示「第 N 轮 · step M」', () => {
    expect(weightLabel('alice_v2_e20_s100.pth', false)).toBe('第 20 轮 · step 100')
  })

  it('命名不合规的退化文件名原样展示', () => {
    expect(weightLabel('weird-name.pth', false)).toBe('weird-name.pth')
  })
})

describe('recordRecentModel', () => {
  it('空列表写入后成为唯一条目', () => {
    expect(recordRecentModel([], 'a.pth', 100)).toEqual([{ name: 'a.pth', at: 100 }])
  })

  it('已存在的条目被摘除并移到最前，时间更新为本次', () => {
    const cur: RecentModelEntry[] = [
      { name: 'a.pth', at: 300 },
      { name: 'b.pth', at: 200 },
      { name: 'c.pth', at: 100 },
    ]
    expect(recordRecentModel(cur, 'c.pth', 400)).toEqual([
      { name: 'c.pth', at: 400 },
      { name: 'a.pth', at: 300 },
      { name: 'b.pth', at: 200 },
    ])
  })

  it('超过上限时裁掉最旧的条目', () => {
    let cur: RecentModelEntry[] = []
    for (let i = 0; i < RECENT_MODELS_LIMIT + 3; i++) {
      cur = recordRecentModel(cur, `m${i}.pth`, i)
    }
    // 共写入 LIMIT+3 个（m0..m10），裁掉最旧 3 个后剩 m3..m10，最新在最前
    expect(cur).toHaveLength(RECENT_MODELS_LIMIT)
    expect(cur[0]).toEqual({ name: 'm10.pth', at: 10 })
    expect(cur[RECENT_MODELS_LIMIT - 1]).toEqual({ name: 'm3.pth', at: 3 })
  })

  it('纯函数：不修改入参数组', () => {
    const cur: RecentModelEntry[] = [{ name: 'a.pth', at: 1 }]
    recordRecentModel(cur, 'b.pth', 2)
    expect(cur).toEqual([{ name: 'a.pth', at: 1 }])
  })
})

describe('loadRecentModels', () => {
  it('无记录（null）返回空数组', () => {
    expect(loadRecentModels(stubStorage(() => null))).toEqual([])
  })

  it('合法 JSON 数组原样返回', () => {
    const raw = JSON.stringify([
      { name: 'a.pth', at: 1 },
      { name: 'b.pth', at: 2 },
    ])
    expect(loadRecentModels(stubStorage(() => raw))).toEqual([
      { name: 'a.pth', at: 1 },
      { name: 'b.pth', at: 2 },
    ])
  })

  it('JSON 解析失败返回空数组', () => {
    expect(loadRecentModels(stubStorage(() => '{oops'))).toEqual([])
  })

  it('非数组 JSON（对象/标量）返回空数组', () => {
    expect(loadRecentModels(stubStorage(() => '{"name":"a.pth"}'))).toEqual([])
    expect(loadRecentModels(stubStorage(() => '"a.pth"'))).toEqual([])
  })

  it('形状不对的条目逐条丢弃，合法条目保留', () => {
    const raw = JSON.stringify([
      { name: 'a.pth', at: 1 },
      { name: '', at: 2 }, // 空文件名
      { name: 3, at: 3 }, // name 非字符串
      { name: 'b.pth', at: 'x' }, // at 非数字
      { name: 'c.pth', at: Number.NaN }, // at 非有限数
      'a.pth', // 裸字符串
      null,
    ])
    expect(loadRecentModels(stubStorage(() => raw))).toEqual([{ name: 'a.pth', at: 1 }])
  })

  it('storage.getItem 抛异常（隐私模式等）返回空数组而非崩溃', () => {
    expect(
      loadRecentModels(
        stubStorage(() => {
          throw new Error('SecurityError')
        }),
      ),
    ).toEqual([])
  })
})
