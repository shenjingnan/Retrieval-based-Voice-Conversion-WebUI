import { defineConfig } from 'vitest/config'

// 测试跑 node 环境：uploadPool 是纯异步编排，不碰 DOM；将来组件级测试再引 jsdom
export default defineConfig({
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
