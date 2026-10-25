type Dict = Record<string, unknown>
export type ResearchEstimateState = {
  status: 'idle' | 'loading' | 'ready' | 'blocked'
  preview: Dict | null
  message: string
  /** Estimate key that produced this state; consumers must compare it with their current key. */
  key: string
  error?: unknown
}
export type NormalizedResearchEstimate = Omit<ResearchEstimateState, 'key'>
type EstimateResponse = Dict & { revision?: number; status?: string; message?: string; error?: unknown }

export function researchEstimateCleanupPath(sessionId: string, revision: number): string {
  return `/optimizations/preview-sessions/${encodeURIComponent(sessionId)}?through_revision=${revision}`
}

export function researchEstimateSubmission(sessionId: string, revision: number, config: Dict, estimateKey: string) {
  return { session_id: sessionId, revision, config, estimate_key: estimateKey }
}

export function researchEstimateFailure(key: string, message: string, status: unknown, detail: unknown): ResearchEstimateState {
  return { status: 'blocked', preview: null, message, key, error: { status, detail } }
}

export function tagResearchEstimateKey(state: NormalizedResearchEstimate, key: string): ResearchEstimateState {
  return { ...state, key }
}

export function normalizeResearchEstimateResponse(value: EstimateResponse): NormalizedResearchEstimate | null {
  const managerStatus = String(value.status ?? '')
  const isCompleted = managerStatus === 'completed'
  if (!isCompleted && managerStatus !== 'ready' && managerStatus !== 'blocked' && managerStatus !== 'failed' && managerStatus !== 'cancelled') return null
  const wrappedResult = isCompleted && value.result && typeof value.result === 'object'
    ? value.result as EstimateResponse
    : null
  const result = wrappedResult ?? value
  const preview = result.preview && typeof result.preview === 'object' ? result.preview as Dict : result
  const error = result.error ?? result.detail
  const reasons = Array.isArray(result.reasons) ? result.reasons.map(String).filter(Boolean).join('；') : ''
  const fallback = reasons || (isCompleted ? (wrappedResult ? String(result.status ?? '研究预检未通过') : '研究预检已完成但未返回结果') : managerStatus === 'cancelled' ? '研究预检已取消' : '')
  const message = String(result.message ?? (preview.message as string | undefined) ?? fallback)
  return { status: isCompleted ? result.status === 'ready' ? 'ready' : 'blocked' : managerStatus === 'ready' ? 'ready' : 'blocked', preview, message, ...(error !== undefined ? { error } : {}) }
}
