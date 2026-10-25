import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { defaultConfig, validFeeDefaults } from './components'
import type { FeeDefaults } from './components'
import { flushConfiguration, rememberConfigurationPatch, rememberWorkbenchGroup, saveConfigurationPatch, useWorkbenchDefaults } from './useWorkbenchDefaults'
import { mergeEditedResearchDefaults, strategyDefaults, workbenchConfig } from './workbenchDefaults'
import type { ResearchDefaults, StrategyDefaultsGroup } from './workbenchDefaults'
import type { BacktestConfig } from './types'

type ConfigUpdate = BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)
type Workspace = {
  backtest: BacktestConfig | null; research: BacktestConfig | null; researchSettings: ResearchDefaults | null; error: string
  updateBacktest: (next: ConfigUpdate) => void; updateResearch: (next: ConfigUpdate) => void
  updateResearchSettings: (next: ResearchDefaults, remember?: boolean) => void
  loadResearchDraft: (config: BacktestConfig, settings: ResearchDefaults) => void
  rememberFees: (next: FeeDefaults) => void; rememberStrategy: (group: StrategyDefaultsGroup, next: BacktestConfig) => void
  configuration: Record<string, unknown>; pendingRestart: string[]; flushConfiguration: () => Promise<string>
  rememberConfiguration: (patch: Record<string, unknown>) => void
  saveConfiguration: (patch: Record<string, unknown>) => Promise<string>
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
  const [error, setError] = useState('')
  const feeDefaults = useRef<FeeDefaults>({ commission_rate: .0005, min_commission: .1 })
  const rememberedResearch = useRef<ResearchDefaults | null>(null)

  const saveFees = useCallback((next: FeeDefaults) => {
    if (!validFeeDefaults(next)) return
    feeDefaults.current = { ...next }
    rememberConfigurationPatch({ backtest: { fees: next } })
    setError('')
  }, [])
  useEffect(() => {
    if (preferences.loading) return
    const fees = (preferences.document.backtest as Record<string, unknown> | undefined)?.fees
    if (validFeeDefaults(fees)) feeDefaults.current = fees
    // Initialize both copies together, so editing either page cannot initialize the other with its draft.
    const initial = workbenchConfig(defaultConfig(feeDefaults.current), preferences.value)
    setBacktest(current => current ?? structuredClone(initial))
    setResearch(current => current ?? structuredClone(initial))
    setResearchSettings(current => current ?? structuredClone(preferences.value.research))
  }, [preferences.loading, preferences.document, preferences.value])
  useEffect(() => {
    if (!preferences.loading) rememberedResearch.current = structuredClone(preferences.value.research)
  }, [preferences.loading, preferences.value.research])
  const updateBacktest = useCallback((next: ConfigUpdate) => setBacktest(current => current ? typeof next === 'function' ? next(current) : next : current), [])
  const updateResearch = useCallback((next: ConfigUpdate) => setResearch(current => current ? typeof next === 'function' ? next(current) : next : current), [])
  // Protocol v5 owns a single research shape, so every edit is stored as the
  // active ``research.defaults`` document instead of a legacy archive copy.
  const updateResearchSettings = useCallback((next: ResearchDefaults, remember = true) => {
    setResearchSettings(current => {
      if (remember && current) {
        const merged = mergeEditedResearchDefaults(current, next, rememberedResearch.current ?? current)
        rememberedResearch.current = merged
        rememberWorkbenchGroup('research', merged)
      }
      return next
    })
  }, [])
  // Source imports replace the active draft only. They deliberately skip the
  // personal-default autosave so ETF constraints do not leak into later drafts.
  const loadResearchDraft = useCallback((config: BacktestConfig, settings: ResearchDefaults) => {
    setResearch(structuredClone(config))
    setResearchSettings(structuredClone(settings))
  }, [])
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
  return <WorkspaceContext.Provider value={{
    backtest, research, researchSettings, error: error || preferences.error,
    updateBacktest, updateResearch, updateResearchSettings, loadResearchDraft, rememberFees: saveFees, rememberStrategy, restore, transfer,
    configuration: preferences.document, pendingRestart: preferences.pendingRestart,
    flushConfiguration, rememberConfiguration: rememberConfigurationPatch,
    saveConfiguration: saveConfigurationPatch,
  }}>{children}</WorkspaceContext.Provider>
}
