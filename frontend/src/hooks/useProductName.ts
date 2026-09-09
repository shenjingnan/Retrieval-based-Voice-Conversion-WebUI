/**
 * 训练产物名探测：按实验名从模型列表匹配 {exp}.pth（或最大 epoch 权重），
 * 队列项成功详情的「去试音」用。enabled 门控：仅成功态且展开时才发请求。
 * loading 不作为状态外露——「确认中」由调用方以 product/error 双空派生
 * （与训练页完成区的既有派生方式一致），避免在 effect 内同步 setState。
 */
import { useCallback, useEffect, useRef, useState } from 'react'

import { api } from '@/api/client'
import { pickProductName } from '@/lib/domain'

export interface ProductProbe {
  /** 匹配到的模型文件名；null = 未匹配（模型尚未生成 / 已被删除重命名） */
  product: string | null
  /** 模型列表拉取失败（网络等）；与「未匹配」区分展示 */
  error: string | null
  /** 发起/重发一次探测；Promise 在结果落定（含失败）后 resolve，供调用方管理 pending 态 */
  refreshAsync: () => Promise<void>
}

export function useProductName(exp: string | null, enabled: boolean): ProductProbe {
  const [product, setProduct] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const refreshAsync = useCallback((): Promise<void> => {
    if (exp === null || exp.length === 0) return Promise.resolve()
    return api
      .models()
      .then((list) => {
        if (!mounted.current) return
        setProduct(pickProductName(list, exp))
        setError(null)
      })
      .catch((e: unknown) => {
        if (!mounted.current) return
        setError(e instanceof Error ? e.message : String(e))
      })
  }, [exp])

  useEffect(() => {
    if (!enabled || exp === null || exp.length === 0) return
    void refreshAsync()
  }, [enabled, exp, refreshAsync])

  return { product, error, refreshAsync }
}
