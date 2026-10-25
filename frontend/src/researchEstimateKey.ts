type Dict = Record<string, unknown>
const pick = (source: unknown, fields: string[]): Dict => {
  if (!source || typeof source !== 'object') return {}
  const value = source as Dict
  return Object.fromEntries(fields.filter(field => Object.prototype.hasOwnProperty.call(value, field)).map(field => [field, value[field]]))
}

/** Stable identity for inputs that change required market data or search work. */
export function researchEstimateKey(fullRequest: Dict, workspaceBudget?: unknown): string {
  const base = (fullRequest.base && typeof fullRequest.base === 'object' ? fullRequest.base : {}) as Dict
  const strategy = (fullRequest.strategy && typeof fullRequest.strategy === 'object' ? fullRequest.strategy : {}) as Dict
  return JSON.stringify({
    base: pick(base, ['symbol', 'start_date', 'end_date', 'frequency', 'selection_asof', 'parameter_links']),
    ranges: fullRequest.ranges ?? null,
    strategy: pick(strategy, [
      'granularity', 'window', 'window_days', 'training_window', 'training_window_mode', 'training_min_days', 'training_max_days', 'search_method',
      'hypercube_edge', 'training_budget', 'validation_budget', 'reporting_budget',
      'total_evaluation_budget', 'exhaustive_memory_limit_mb', 'estimated_bytes_per_point', 'stress_region_count',
    ]),
    selection_target: fullRequest.selection_target ?? null,
    rebalance_days: fullRequest.rebalance_days ?? null,
    prior_policy: fullRequest.prior_policy ?? null,
    source_etf: fullRequest.source_etf ?? null,
    requested_data_version: fullRequest.requested_data_version ?? null,
    workspace_budget: workspaceBudget ?? null,
  })
}
