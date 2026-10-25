import { useEffect, useRef, useState } from 'react'
import { del, get, isAbortError, post } from './api'
import { normalizeResearchEstimateResponse, researchEstimateCleanupPath, researchEstimateFailure, researchEstimateSubmission, tagResearchEstimateKey } from './researchEstimateLifecycle'
export { normalizeResearchEstimateResponse, researchEstimateCleanupPath, researchEstimateFailure, researchEstimateSubmission, tagResearchEstimateKey } from './researchEstimateLifecycle'

type Dict = Record<string, unknown>
export { researchEstimateKey } from './researchEstimateKey'
import type { ResearchEstimateState } from './researchEstimateLifecycle'
export type { ResearchEstimateState } from './researchEstimateLifecycle'
type EstimateResponse = Dict & { revision?: number; status?: string; message?: string; error?: unknown }

/** Owns a preview-sessions lifecycle. The caller supplies its completed, date-ready request. */
export function useResearchEstimate(
  request: Dict | null | undefined,
  key: string,
  options: { enabled?: boolean; delayMs?: number; pollingMs?: number; retryToken?: unknown } = {},
): ResearchEstimateState {
  const { enabled = true, delayMs = 1500, pollingMs = 1500, retryToken } = options
  const session = useRef(`research_estimate_${Math.random().toString(36).slice(2)}`)
  const revision = useRef(0)
  const requestRef = useRef(request)
  requestRef.current = request
  const [state, setState] = useState<ResearchEstimateState>({ status: 'idle', preview: null, message: '', key: '' })

  useEffect(() => {
    const current = ++revision.current
    let active = true
    let timer: number | undefined
    let poll: number | undefined
    let controller: AbortController | undefined
    const cleanupUrl = researchEstimateCleanupPath(session.current, current)
    if (!enabled || !requestRef.current || !key) {
      setState({ status: 'idle', preview: null, message: '', key })
      return () => { active = false }
    }
    setState({ status: 'loading', preview: null, message: '', key })
    const fail = (error: unknown) => {
      if (!active || isAbortError(error)) return
      const detail = error && typeof error === 'object' && 'detail' in error ? (error as { detail: unknown }).detail : error
      const status = error && typeof error === 'object' && 'status' in error ? (error as { status: unknown }).status : undefined
      setState(researchEstimateFailure(key, error instanceof Error ? error.message : String(error), status, detail))
    }
    const accept = (value: EstimateResponse) => {
      if (!active || revision.current !== current || (value.revision != null && value.revision !== current)) return
      const normalized = normalizeResearchEstimateResponse(value)
      if (normalized) {
        setState(tagResearchEstimateKey(normalized, key))
        return
      }
      setState({ status: 'loading', preview: value, message: String(value.message ?? ''), key })
      poll = window.setTimeout(() => {
        controller = new AbortController()
        get<EstimateResponse>(`/optimizations/preview-sessions/${encodeURIComponent(session.current)}`, { signal: controller.signal }).then(accept).catch(fail)
      }, pollingMs)
    }
    timer = window.setTimeout(() => {
      controller = new AbortController()
      const latestRequest = requestRef.current
      if (!latestRequest) return
      post<EstimateResponse>('/optimizations/preview-sessions', researchEstimateSubmission(session.current, current, latestRequest, key), { signal: controller.signal }).then(accept).catch(fail)
    }, Math.max(0, delayMs))
    return () => {
      active = false
      window.clearTimeout(timer)
      window.clearTimeout(poll)
      controller?.abort()
      void del(cleanupUrl).catch(() => undefined)
    }
  }, [key, enabled, delayMs, pollingMs, retryToken])

  return state
}
