import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  clearResearchDraftSource, defaultResearchDraftContext, setResearchPriorPolicy,
} from './researchDraft.ts'
import type { ResearchDraftContextState, ResearchPriorPolicy } from './researchDraft.ts'

export interface ResearchDraftContextValue {
  state: ResearchDraftContextState
  replace: (state: ResearchDraftContextState) => void
  clearSource: () => void
  setPriorPolicy: (policy: ResearchPriorPolicy) => void
}

const ResearchDraftContext = createContext<ResearchDraftContextValue | null>(null)

export function useResearchDraft(): ResearchDraftContextValue {
  const value = useContext(ResearchDraftContext)
  if (!value) throw new Error('研究草稿上下文尚未初始化')
  return value
}

/**
 * Source identity and the prior switch live beside the numeric research draft.
 * The provider stays mounted above page routing, so moving between ETF and
 * research pages keeps the context while restoration and explicit transfers
 * can clear it.
 */
export function ResearchDraftProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ResearchDraftContextState>(() => defaultResearchDraftContext())
  const replace = useCallback((next: ResearchDraftContextState) => setState(next), [])
  const clearSource = useCallback(() => setState(clearResearchDraftSource()), [])
  const setPrior = useCallback((policy: ResearchPriorPolicy) => {
    setState(current => setResearchPriorPolicy(current, policy))
  }, [])
  const value = useMemo(() => ({ state, replace, clearSource, setPriorPolicy: setPrior }), [state, replace, clearSource, setPrior])
  return <ResearchDraftContext.Provider value={value}>{children}</ResearchDraftContext.Provider>
}
