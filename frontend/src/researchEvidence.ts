type EvidenceRecord = Record<string, unknown>

function record(value: unknown): EvidenceRecord | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as EvidenceRecord : undefined
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export type BootstrapEvidenceReview = {
  state: 'insufficient' | 'not_positive' | 'positive'
  completed: boolean
  pValue: number | null
  lowerBound: number | null
  alpha: number
}

/** Interpret a family-level paired bootstrap without treating it as a win probability. */
export function reviewBootstrapEvidence(value: unknown): BootstrapEvidenceReview {
  const result = record(value) ?? {}
  const bootstrap = record(result.bootstrap) ?? result
  const primary = record(bootstrap.primary) ?? {}
  const pValue = finite(primary.p_value) ? primary.p_value : null
  const lowerBound = finite(primary.selected_simultaneous_lower_log_excess) ? primary.selected_simultaneous_lower_log_excess : null
  const alpha = finite(primary.alpha) && primary.alpha > 0 && primary.alpha < 1 ? primary.alpha : 0.05
  const completedReps = primary.completed_reps
  const requestedReps = primary.requested_reps
  const repCountComplete = finite(completedReps) && finite(requestedReps) ? completedReps >= requestedReps : true
  const status = String(result.status ?? bootstrap.status ?? '')
  const completed = status === 'completed' && primary.cancelled !== true && bootstrap.cancelled !== true
    && repCountComplete && pValue != null && lowerBound != null
  if (!completed) return { state: 'insufficient', completed: false, pValue, lowerBound, alpha }
  return { state: pValue! <= alpha && lowerBound! > 0 ? 'positive' : 'not_positive', completed: true, pValue, lowerBound, alpha }
}

export function formatPValue(value: unknown): string {
  if (!finite(value)) return '—'
  return value < 0.0001 ? '<0.0001' : value.toFixed(4)
}
