import { useCallback, useState } from 'react'
import { Button } from '@/components/ui/button'
import { InferencePage } from '@/pages/Inference'
import { ModelsPage } from '@/pages/Models'
import { TrainingPage } from '@/pages/Training'

type TabKey = 'inference' | 'train' | 'models'

const TABS: ReadonlyArray<{ key: TabKey; label: string }> = [
  { key: 'inference', label: '推理' },
  { key: 'train', label: '训练' },
  { key: 'models', label: '模型管理' },
]

export default function App() {
  const [tab, setTab] = useState<TabKey>('inference')
  // 训练 → 推理的联动载荷：训练完成后「去试音」把模型名带给推理页；
  // 推理页消费后回调清空，避免 Tab 来回切换时重复强制选中
  const [pendingModel, setPendingModel] = useState<string | null>(null)

  const goToInference = useCallback((model: string) => {
    setPendingModel(model)
    setTab('inference')
  }, [])

  const consumePendingModel = useCallback(() => setPendingModel(null), [])

  // 模型管理页空状态引导：切到训练 Tab（无载荷）
  const goToTrain = useCallback(() => setTab('train'), [])

  return (
    <div className="flex min-h-svh flex-col">
      <header className="border-b">
        <div className="mx-auto flex h-14 w-full max-w-5xl items-center gap-6 px-6">
          <span className="text-sm font-semibold">RVC WebUI</span>
          <nav className="flex items-center gap-1" role="tablist">
            {TABS.map(({ key, label }) => (
              <Button
                key={key}
                role="tab"
                aria-selected={tab === key}
                variant={tab === key ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => setTab(key)}
              >
                {label}
              </Button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-6">
        {tab === 'inference' && (
          <InferencePage initialModel={pendingModel} onModelConsumed={consumePendingModel} />
        )}
        {tab === 'train' && <TrainingPage onGoInfer={goToInference} />}
        {tab === 'models' && (
          <ModelsPage onGoInfer={goToInference} onGoTrain={goToTrain} />
        )}
      </main>
    </div>
  )
}
