import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { get, put } from './api'
import { clearPendingFeeDefaults, defaultConfig, markFeeDefaultsPending, readFeeDefaults, readPendingFeeDefaults, validFeeDefaults, writeFeeDefaults } from './components'
import type { FeeDefaults } from './components'
import { rememberWorkbenchGroup, useWorkbenchDefaults } from './useWorkbenchDefaults'
import { strategyDefaults, workbenchConfig } from './workbenchDefaults'
import type { ResearchDefaults, StrategyDefaultsGroup } from './workbenchDefaults'
import type { BacktestConfig } from './types'

type ConfigUpdate = BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)
type Workspace = {
  backtest: BacktestConfig | null; research: BacktestConfig | null; researchSettings: ResearchDefaults | null; error: string
  updateBacktest: (next: ConfigUpdate) => void; updateResearch: (next: ConfigUpdate) => void
  updateResearchSettings: (next: ResearchDefaults) => void
  rememberFees: (next: FeeDefaults) => void; rememberStrategy: (group: StrategyDefaultsGroup, next: BacktestConfig) => void
  restore: (page: 'backtest' | 'research') => void
  transfer: (page: 'backtest' | 'research', config: BacktestConfig) => void
}
const WorkspaceContext = createContext<Workspace | null>(null)
export function useWorkspaces() {
  const value = useContext(WorkspaceContext)
  if (!value) throw new Error('工作区尚未初始化')
  return value
}

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const preferences = useWorkbenchDefaults()
  const [backtest, setBacktest] = useState<BacktestConfig | null>(null)
  const [research, setResearch] = useState<BacktestConfig | null>(null)
  const [researchSettings, setResearchSettings] = useState<ResearchDefaults | null>(null)
  const [feesLoaded, setFeesLoaded] = useState(false)
  const [error, setError] = useState('')
  const feeDefaults = useRef(readFeeDefaults())
  const feeRevision = useRef(0)
  const feeDirty = useRef(false)
  const saveChain = useRef<Promise<void>>(Promise.resolve())

  const saveFees = useCallback((next: FeeDefaults) => {
    if (!validFeeDefaults(next)) return
    const revision = ++feeRevision.current
    feeDefaults.current = { ...next }; feeDirty.current = true
    writeFeeDefaults(next); markFeeDefaultsPending()
    saveChain.current = saveChain.current.catch(() => undefined).then(async () => {
      if (revision !== feeRevision.current) return
      await put('/settings', { fee_defaults: next })
      if (revision === feeRevision.current) { feeDirty.current = false; clearPendingFeeDefaults(); setError('') }
    }).catch(cause => { if (revision === feeRevision.current) setError('默认佣金未能保存，将在下次启动时重试：' + String(cause)) })
  }, [])
  useEffect(() => {
    let active = true
    get<{ fee_defaults?: unknown }>('/settings').then(settings => {
      if (!active) return
      const pending = readPendingFeeDefaults()
      feeDefaults.current = pending ?? (validFeeDefaults(settings.fee_defaults) ? settings.fee_defaults : readFeeDefaults())
      writeFeeDefaults(feeDefaults.current)
      if (pending) saveFees(pending)
    }).catch(() => {
      if (!active) return
      const pending = readPendingFeeDefaults()
      if (pending) saveFees(pending)
    }).finally(() => { if (active) setFeesLoaded(true) })
    return () => { active = false }
  }, [saveFees])
  useEffect(() => {
    if (!feesLoaded || preferences.loading) return
    // Initialize both copies together, so editing either page cannot initialize the other with its draft.
    const initial = workbenchConfig(defaultConfig(feeDefaults.current), preferences.value)
    setBacktest(current => current ?? structuredClone(initial))
    setResearch(current => current ?? structuredClone(initial))
    setResearchSettings(current => current ?? structuredClone(preferences.value.research))
  }, [feesLoaded, preferences.loading, preferences.value])
  useEffect(() => {
    const flush = () => {
      if (!feeDirty.current) return
      void fetch('/api/v1/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fee_defaults: feeDefaults.current }), keepalive: true }).catch(() => undefined)
    }
    window.addEventListener('pagehide', flush)
    return () => { window.removeEventListener('pagehide', flush); flush() }
  }, [])
  const updateBacktest = useCallback((next: ConfigUpdate) => setBacktest(current => current ? typeof next === 'function' ? next(current) : next : current), [])
  const updateResearch = useCallback((next: ConfigUpdate) => setResearch(current => current ? typeof next === 'function' ? next(current) : next : current), [])
  const updateResearchSettings = useCallback((next: ResearchDefaults) => { setResearchSettings(next); rememberWorkbenchGroup('research', next) }, [])
  const rememberStrategy = useCallback((group: StrategyDefaultsGroup, next: BacktestConfig) => { rememberWorkbenchGroup(group, strategyDefaults(next)[group]) }, [])
  const restore = (page: 'backtest' | 'research') => {
    const initial = workbenchConfig(defaultConfig(feeDefaults.current), preferences.value)
    if (page === 'backtest') setBacktest(initial)
    else { setResearch(initial); setResearchSettings(structuredClone(preferences.value.research)) }
  }
  const transfer = (page: 'backtest' | 'research', config: BacktestConfig) => {
    if (page === 'backtest') setBacktest(structuredClone(config))
    else setResearch(structuredClone(config))
  }
  return <WorkspaceContext.Provider value={{ backtest, research, researchSettings, error: error || preferences.error, updateBacktest, updateResearch, updateResearchSettings, rememberFees: saveFees, rememberStrategy, restore, transfer }}>{children}</WorkspaceContext.Provider>
}
