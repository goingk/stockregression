type Dict = Record<string, unknown>

export type V4ValidationLoadInfo =
  | { kind: 'fresh'; dateRange: { start_date: string; end_date: string } }
  | { kind: 'legacy' }
  | { kind: 'incomplete' }

function record(value: unknown): Dict | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Dict : undefined
}

function isoDate(value: unknown): value is string {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
}

/** Resolve whether a v4 recommendation region can be loaded with its exact validation window. */
export function v4ValidationLoadInfo(diagnostics: unknown): V4ValidationLoadInfo {
  const value = record(diagnostics)
  if (!value || !Object.prototype.hasOwnProperty.call(value, 'validation_account_mode')) return { kind: 'legacy' }
  if (value.validation_account_mode !== 'fresh_initial_config') return { kind: 'incomplete' }

  const startDate = value.validation_start
  const endDate = value.validation_end
  if (!isoDate(startDate) || !isoDate(endDate) || startDate > endDate) return { kind: 'incomplete' }
  return { kind: 'fresh', dateRange: { start_date: startDate, end_date: endDate } }
}
