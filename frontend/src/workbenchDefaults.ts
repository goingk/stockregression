import type { BacktestConfig } from './types'

export type GridDefaults = Pick<BacktestConfig, 'grid_mode' | 'buy_step' | 'sell_step' | 'lower_price' | 'upper_price'>
export type PositionDefaults = Pick<BacktestConfig, 'initial_shares' | 'min_shares' | 'max_shares' | 'buy_shares' | 'sell_shares'>
export type SearchRange = { low: number; high: number; step?: number | null }
export interface ResearchDefaults {
  strategy_id: 'geometric_excess_neighborhood' | 'geometric_robust_j'
  protocol_version: 1 | 2 | 3 | 4
  selection_target: 'fixed' | 'adaptive' | 'both'
  rebalance_days: number
  bootstrap_reps: number
  bootstrap_block_lengths: number[]
  significance_alpha: number
  outer_validation_days: number
  mode: 'historical' | 'robust' | 'both'
  n_trials: number
  local_search_budget: number
  seed: number
  top_candidates: number
  max_drawdown: number
  min_trades: number
  min_history_days: number
  min_minute_ratio: number
  ranking_mode: 'max_volume' | 'fixed_mean'
  strategy_version: 1 | 2 | 3 | 4
  hypercube_edge: number
  granularity: Record<string, number>
  search_method: 'auto' | 'hierarchical' | 'exhaustive'
  training_budget: number
  validation_budget: number
  recommendation_count: number
  max_shares_mode: 'search' | 'unlimited'
  slippage_mode: 'configured' | 'zero'
  neighborhood_fields: string[]
  fixed_half_widths: Record<string, number>
  sigma: number
  alpha: number
  beta: number
  tau: number
  eta_limit: number
  eta_hard_cap: number | null
  gamma_max: number
  eta_floor: number
  gamma_min: number
  training_min_days: number
  training_max_days: number
  peak_tolerance: number
  start_min: number
  start_max: number
  starts_per_candidate: number
  initial_strides: number[]
  stride_multiplier: number
  max_stride_fraction: number
  validation_reserve_fraction: number
  exhaustive_memory_limit_mb: number
  estimated_bytes_per_point: number
  stress_cost_multiplier: number
  stress_region_count: number
  ranges: Record<string, SearchRange>
}
export interface WorkbenchDefaults {
  grid: GridDefaults
  position: PositionDefaults
  dca: BacktestConfig['dca']
  research: ResearchDefaults
}
export type StrategyDefaultsGroup = 'grid' | 'position' | 'dca'
export type WorkbenchGroup = keyof WorkbenchDefaults

export const percentageRanges = new Set(['buy_step', 'sell_step'])
export const rangeNames = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'] as const
export const legacyRangeNames = [...rangeNames, 'initial_shares', 'min_shares', 'max_shares'] as const
const v4NeighborhoodFields = ['buy_step', 'sell_step', 'sell_shares', 'buy_shares', 'min_shares', 'max_shares'] as const
const v4RobustFields = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'] as const
const legacyRangeFallbacks: Record<string, SearchRange> = {
  initial_shares: { low: 0, high: 2000, step: 100 },
  min_shares: { low: 0, high: 1000, step: 100 },
  max_shares: { low: 1000, high: 5000, step: 100 },
}

export function defaultSearchRanges(): Record<string, SearchRange> {
  return {
    buy_step: { low: .005, high: .1 }, sell_step: { low: .005, high: .1 },
    buy_shares: { low: 100, high: 1000 }, sell_shares: { low: 100, high: 1000 },
  }
}

export function researchRangesForSymbol(ranges: Record<string, SearchRange>, symbol: string): Record<string, SearchRange> {
  const minimum = symbol.startsWith('688') || symbol.startsWith('689') ? 200 : 100
  return ranges.buy_shares.low < minimum
    ? { ...ranges, buy_shares: { ...ranges.buy_shares, low: minimum, high: Math.max(minimum, ranges.buy_shares.high) } }
    : ranges
}

export function researchConfigForSubmission(value: BacktestConfig, protocolVersion: number, maxSharesMode: ResearchDefaults['max_shares_mode']): BacktestConfig {
  if (protocolVersion !== 4) return value
  return {
    ...value,
    grid_mode: 'geometric',
    max_shares: maxSharesMode === 'unlimited' ? null : Math.max(value.initial_shares, value.max_shares ?? value.initial_shares),
  }
}

export function defaultWorkbenchDefaults(): WorkbenchDefaults {
  return {
    grid: { grid_mode: 'trailing', buy_step: .03, sell_step: .03, lower_price: null, upper_price: null },
    position: { initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 100 },
    dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
    research: { strategy_id: 'geometric_robust_j', protocol_version: 4, strategy_version: 4, selection_target: 'both', rebalance_days: 63, bootstrap_reps: 10000, bootstrap_block_lengths: [20, 10, 40], significance_alpha: .05, outer_validation_days: 126, mode: 'both', n_trials: 1000, local_search_budget: 3000, seed: 42, top_candidates: 20, max_drawdown: .2, min_trades: 10, min_history_days: 250, min_minute_ratio: .95, ranking_mode: 'fixed_mean', hypercube_edge: 3, granularity: { buy_step: .0001, sell_step: .0001, sell_shares: 100, buy_shares: 100 }, search_method: 'auto', training_budget: 20000, validation_budget: 100000, recommendation_count: 10, max_shares_mode: 'unlimited', slippage_mode: 'configured', neighborhood_fields: [...v4RobustFields], fixed_half_widths: { buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100, min_shares: 500, max_shares: 500 }, sigma: 1, alpha: .3, beta: 1.2, tau: .01, eta_limit: 2, eta_hard_cap: 3.5, gamma_max: .4, eta_floor: 0, gamma_min: 0, training_min_days: 250, training_max_days: 500, peak_tolerance: 1e-12, start_min: 20, start_max: 64, starts_per_candidate: 4, initial_strides: [1, 2], stride_multiplier: 2, max_stride_fraction: .25, validation_reserve_fraction: .25, exhaustive_memory_limit_mb: 512, estimated_bytes_per_point: 512, stress_cost_multiplier: 2, stress_region_count: 3, ranges: defaultSearchRanges() },
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
const n = (value: unknown, fallback = 0) => { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback }
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const integer = (value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number => finite(value) && Number.isSafeInteger(value) && value >= min && value <= max
export const researchQuantityUnit = (_name: string) => 100
export const legacyHalfWidthUnit = (name: string) => name === 'min_shares' || name === 'max_shares' ? 500 : 100
const quantity = (value: unknown, min = 0) => integer(value, min) && value % 100 === 0
const fraction = (value: unknown, min = 0) => finite(value) && value >= min && value <= 1
const step = (value: unknown) => finite(value) && value > 0 && value <= 1
const positive = (value: unknown) => finite(value) && value > 0
const nonnegative = (value: unknown) => finite(value) && value >= 0
const price = (value: unknown) => value === null || finite(value) && value > 0

export function validResearchDefaults(value: unknown): value is ResearchDefaults {
  const item = record(value)
  if (!item || ![1, 2, 3, 4].includes(Number(item.protocol_version)) || !['fixed', 'adaptive', 'both'].includes(String(item.selection_target))
    || !['geometric_excess_neighborhood', 'geometric_robust_j'].includes(String(item.strategy_id))
    || !integer(item.rebalance_days, 1, 504) || !integer(item.bootstrap_reps, 100, 100000)
    || !Array.isArray(item.bootstrap_block_lengths) || !item.bootstrap_block_lengths.length || !item.bootstrap_block_lengths.every(value => integer(value, 1, 10000))
    || !fraction(item.significance_alpha) || n(item.significance_alpha) <= 0 || n(item.significance_alpha) >= 1
    || !integer(item.outer_validation_days, 20, 1000)
    || !['historical', 'robust', 'both'].includes(String(item.mode)) || !integer(item.n_trials, 1, 5000) || !integer(item.local_search_budget, 1, 5000)
    || !integer(item.seed, Number.MIN_SAFE_INTEGER) || !integer(item.top_candidates, 1, 100) || !step(item.max_drawdown)
    || !integer(item.min_trades, 0) || !integer(item.min_history_days, 20) || !fraction(item.min_minute_ratio)
    || !['max_volume', 'fixed_mean'].includes(String(item.ranking_mode))
    || ![1, 2, 3, 4].includes(Number(item.strategy_version))
    || (Number(item.strategy_version) === 2 && String(item.ranking_mode) !== 'max_volume')
    || ([3, 4].includes(Number(item.strategy_version)) && String(item.ranking_mode) !== 'fixed_mean')
    || !integer(item.hypercube_edge, 1, 99) || Number(item.hypercube_edge) % 2 === 0
    || !['auto', 'hierarchical', 'exhaustive'].includes(String(item.search_method))
    || !integer(item.training_budget, 1, 1000000) || !integer(item.validation_budget, 1, 2000000)
    || !integer(item.recommendation_count, 1, 100)
    || !['search', 'unlimited'].includes(String(item.max_shares_mode))
    || !['configured', 'zero'].includes(String(item.slippage_mode))
    || !Array.isArray(item.neighborhood_fields)
    || (Number(item.strategy_version) === 4
      ? item.neighborhood_fields.length !== v4RobustFields.length || !v4RobustFields.every(name => (item.neighborhood_fields as unknown[]).includes(name))
      : !item.neighborhood_fields.length || !item.neighborhood_fields.every(name => v4NeighborhoodFields.includes(String(name) as typeof v4NeighborhoodFields[number])))
    || !fraction(item.alpha) || !positive(item.sigma) || !positive(item.beta)
    || !finite(item.tau) || n(item.tau) < -1 || n(item.tau) > 10
    || !nonnegative(item.eta_limit) || !nonnegative(item.gamma_max) || !fraction(item.eta_floor) || !nonnegative(item.gamma_min)
    || n(item.eta_floor) > n(item.eta_limit)
    || (item.eta_hard_cap != null && (!positive(item.eta_hard_cap) || n(item.eta_hard_cap) < n(item.eta_limit)))
    || !integer(item.training_min_days, 20, 5000) || !integer(item.training_max_days, 20, 5000) || n(item.training_min_days) > n(item.training_max_days)
    || !positive(item.peak_tolerance) || !integer(item.start_min, 1, 1000) || !integer(item.start_max, 1, 1000) || n(item.start_min) > n(item.start_max)
    || !integer(item.starts_per_candidate, 1, 100) || !Array.isArray(item.initial_strides) || !item.initial_strides.length || !item.initial_strides.every(value => integer(value, 1, 1000000))
    || !integer(item.stride_multiplier, 2, 16) || !step(item.max_stride_fraction)
    || !fraction(item.validation_reserve_fraction) || !integer(item.exhaustive_memory_limit_mb, 16, 65536)
    || !integer(item.estimated_bytes_per_point, 64, 65536)
    || !finite(item.stress_cost_multiplier) || n(item.stress_cost_multiplier) <= 1 || !integer(item.stress_region_count, 0, 50)
    || new Set(item.neighborhood_fields).size !== item.neighborhood_fields.length) return false
  const halfWidths = record(item.fixed_half_widths)
  if (!halfWidths || !v4NeighborhoodFields.every(name => finite(halfWidths[name]) && Number(halfWidths[name]) > 0)) return false
  if (['buy_step', 'sell_step'].some(name => Math.abs(Number(halfWidths[name]) * 10000 - Math.round(Number(halfWidths[name]) * 10000)) > 1e-8)) return false
  if (['sell_shares', 'buy_shares', 'min_shares', 'max_shares'].some(name => Math.abs(Number(halfWidths[name]) / legacyHalfWidthUnit(name) - Math.round(Number(halfWidths[name]) / legacyHalfWidthUnit(name))) > 1e-8)) return false
  const granularity = record(item.granularity)
  const granularityFields = Number(item.strategy_version) === 4 ? v4RobustFields : v4NeighborhoodFields
  if (!granularity || Object.keys(granularity).length !== granularityFields.length || !granularityFields.every(name => finite(granularity[name]) && Number(granularity[name]) > 0)) return false
  const minimumUnit = (name: string) => ['buy_step', 'sell_step'].includes(name) ? .0001 : 100
  if (!granularityFields.every(name => Math.abs(Number(granularity[name]) / minimumUnit(name) - Math.round(Number(granularity[name]) / minimumUnit(name))) < 1e-8)) return false
  const ranges = record(item.ranges)
  const expectedRanges = Number(item.strategy_version) === 4 ? rangeNames.length : 7
  if (!ranges || Object.keys(ranges).length !== expectedRanges) return false
  const fields: readonly string[] = Number(item.strategy_version) === 4
    ? rangeNames
    : [...rangeNames, 'initial_shares', 'min_shares', 'max_shares']
  return fields.every(name => {
    const range = record(ranges[name])
    if (!range || !finite(range.low) || !finite(range.high) || range.low > range.high) return false
    if (Number(item.strategy_version) !== 4 && (!finite(range.step) || range.step <= 0)) return false
    if (Number(item.protocol_version) === 4 && ['min_shares', 'max_shares'].includes(name)
      && (range.low % 100 !== 0 || range.high % 100 !== 0)) return false
    if (Number(item.protocol_version) === 4 && percentageRanges.has(name)
      && (Math.abs(range.low * 10000 - Math.round(range.low * 10000)) > 1e-8 || Math.abs(range.high * 10000 - Math.round(range.high * 10000)) > 1e-8)) return false
    return percentageRanges.has(name)
      ? step(range.low) && step(range.high)
      : quantity(range.low, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
        && quantity(range.high, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
        && (Number(item.strategy_version) === 4 || quantity(range.step, 100))
  })
}

/** Fill fields added by newer research protocols without discarding remembered legacy ranges. */
function normalizeResearchDefaults(value: unknown): ResearchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults().research
  if (!item) return defaults
  const savedVersion = integer(item.strategy_version, 1, 4)
    ? Number(item.strategy_version)
    : (item.ranking_mode === 'fixed_mean' ? 4 : 3)
  // Connected-component v2 is no longer offered for new studies; remembered
  // settings move to the hypercube-mean strategy v3.
  const strategyVersion = savedVersion === 2 ? 3 : savedVersion
  const activeRangeNames = strategyVersion === 4 ? rangeNames : legacyRangeNames
  const ranges = record(item.ranges)
  const legacyRange = record(ranges?.grid_step)
  const normalizedRanges = Object.fromEntries(activeRangeNames.map((name) => {
    const saved = name === 'buy_step' || name === 'sell_step' ? legacyRange ?? record(ranges?.[name]) : record(ranges?.[name])
    const fallback = defaults.ranges[name] ?? legacyRangeFallbacks[name]
    const legacyStep = percentageRanges.has(name) ? .005 : 100
    const candidate = saved && finite(saved.low) && finite(saved.high)
      ? { low: saved.low, high: saved.high, ...(strategyVersion === 4 ? {} : { step: finite(saved.step) && saved.step > 0 ? saved.step : legacyStep }) }
      : fallback
    return [name, candidate]
  })) as Record<string, SearchRange>
  const savedHalfWidths = record(item.fixed_half_widths)
  const legacyHalfWidth = savedHalfWidths?.grid_step
  const normalizedHalfWidths = Object.fromEntries(Object.keys(defaults.fixed_half_widths).map((name) => {
    const saved = (name === 'buy_step' || name === 'sell_step') ? legacyHalfWidth ?? savedHalfWidths?.[name] : savedHalfWidths?.[name]
    return [name, finite(saved) && saved > 0 ? (['min_shares', 'max_shares'].includes(name) ? Math.ceil(saved / 500) * 500 : saved) : defaults.fixed_half_widths[name]]
  })) as Record<string, number>
  const savedNeighborhood = Array.isArray(item.neighborhood_fields)
    ? item.neighborhood_fields.flatMap(name => String(name) === 'grid_step' ? ['buy_step', 'sell_step'] : [String(name)])
    : null
  const rankingMode = [1, 3, 4].includes(strategyVersion) ? 'fixed_mean' : 'max_volume'
  const merged: ResearchDefaults = {
    ...defaults,
    strategy_id: strategyVersion === 4 ? 'geometric_robust_j' : 'geometric_excess_neighborhood',
    strategy_version: strategyVersion as ResearchDefaults['strategy_version'],
    protocol_version: defaults.protocol_version,
    selection_target: ['fixed', 'adaptive', 'both'].includes(String(item.selection_target)) ? item.selection_target as ResearchDefaults['selection_target'] : defaults.selection_target,
    rebalance_days: integer(item.rebalance_days, 1, 504) ? item.rebalance_days : defaults.rebalance_days,
    bootstrap_reps: integer(item.bootstrap_reps, 100, 100000) ? item.bootstrap_reps : defaults.bootstrap_reps,
    bootstrap_block_lengths: Array.isArray(item.bootstrap_block_lengths) && item.bootstrap_block_lengths.length && item.bootstrap_block_lengths.every(value => integer(value, 1, 10000)) ? item.bootstrap_block_lengths as number[] : defaults.bootstrap_block_lengths,
    significance_alpha: fraction(item.significance_alpha) && n(item.significance_alpha) > 0 && n(item.significance_alpha) < 1 ? item.significance_alpha as number : defaults.significance_alpha,
    outer_validation_days: integer(item.outer_validation_days, 20, 1000) ? item.outer_validation_days : defaults.outer_validation_days,
    mode: ['historical', 'robust', 'both'].includes(String(item.mode)) ? item.mode as ResearchDefaults['mode'] : defaults.mode,
    n_trials: integer(item.n_trials, 1, 5000) ? item.n_trials : defaults.n_trials,
    local_search_budget: integer(item.local_search_budget, 1, 5000) ? item.local_search_budget : defaults.local_search_budget,
    seed: integer(item.seed, Number.MIN_SAFE_INTEGER) ? item.seed : defaults.seed,
    top_candidates: integer(item.top_candidates, 1, 100) ? item.top_candidates : defaults.top_candidates,
    max_drawdown: step(item.max_drawdown) ? item.max_drawdown as number : defaults.max_drawdown,
    min_trades: integer(item.min_trades, 0) ? item.min_trades : defaults.min_trades,
    min_history_days: integer(item.min_history_days, 20) ? item.min_history_days : defaults.min_history_days,
    min_minute_ratio: fraction(item.min_minute_ratio) ? item.min_minute_ratio as number : defaults.min_minute_ratio,
    ranking_mode: rankingMode,
    hypercube_edge: integer(item.hypercube_edge, 1, 99) && Number(item.hypercube_edge) % 2 === 1 ? Number(item.hypercube_edge) : defaults.hypercube_edge,
    granularity: Object.fromEntries((strategyVersion === 4 ? v4RobustFields : v4NeighborhoodFields).map(name => {
      const saved = record(item.granularity)?.[name]
      const minimum = ['buy_step', 'sell_step'].includes(name) ? .0001 : 100
      const valid = finite(saved) && Number(saved) > 0
        && Math.abs(Number(saved) / minimum - Math.round(Number(saved) / minimum)) < 1e-8
      return [name, valid ? Number(saved) : (defaults.granularity[name] ?? 500)]
    })),
    search_method: ['auto', 'hierarchical', 'exhaustive'].includes(String(item.search_method)) ? item.search_method as ResearchDefaults['search_method'] : defaults.search_method,
    training_budget: integer(item.training_budget, 1, 1000000) ? item.training_budget : defaults.training_budget,
    validation_budget: integer(item.validation_budget, 1, 2000000) ? item.validation_budget : defaults.validation_budget,
    recommendation_count: integer(item.recommendation_count, 1, 100) ? item.recommendation_count : defaults.recommendation_count,
    max_shares_mode: ['search', 'unlimited'].includes(String(item.max_shares_mode)) ? item.max_shares_mode as ResearchDefaults['max_shares_mode'] : defaults.max_shares_mode,
    slippage_mode: ['configured', 'zero'].includes(String(item.slippage_mode)) ? item.slippage_mode as ResearchDefaults['slippage_mode'] : defaults.slippage_mode,
    neighborhood_fields: savedNeighborhood?.length
      && savedNeighborhood.every(name => v4NeighborhoodFields.includes(name as typeof v4NeighborhoodFields[number]))
      && new Set(savedNeighborhood).size === savedNeighborhood.length
      ? savedNeighborhood : defaults.neighborhood_fields,
    fixed_half_widths: normalizedHalfWidths,
    sigma: positive(item.sigma) ? item.sigma as number : defaults.sigma,
    alpha: fraction(item.alpha) ? item.alpha as number : defaults.alpha,
    beta: positive(item.beta) ? item.beta as number : defaults.beta,
    tau: finite(item.tau) && n(item.tau) >= -1 && n(item.tau) <= 10 ? item.tau as number : defaults.tau,
    eta_limit: nonnegative(item.eta_limit) ? item.eta_limit as number : defaults.eta_limit,
    eta_hard_cap: item.eta_hard_cap == null ? defaults.eta_hard_cap : positive(item.eta_hard_cap) ? item.eta_hard_cap as number : defaults.eta_hard_cap,
    gamma_max: nonnegative(item.gamma_max) ? item.gamma_max as number : defaults.gamma_max,
    eta_floor: fraction(item.eta_floor) ? item.eta_floor as number : defaults.eta_floor,
    gamma_min: nonnegative(item.gamma_min) ? item.gamma_min as number : defaults.gamma_min,
    training_min_days: integer(item.training_min_days, 20, 5000) ? item.training_min_days as number : defaults.training_min_days,
    training_max_days: integer(item.training_max_days, 20, 5000) ? item.training_max_days as number : defaults.training_max_days,
    peak_tolerance: positive(item.peak_tolerance) ? item.peak_tolerance as number : defaults.peak_tolerance,
    start_min: integer(item.start_min, 1, 1000) ? item.start_min as number : defaults.start_min,
    start_max: integer(item.start_max, 1, 1000) ? item.start_max as number : defaults.start_max,
    starts_per_candidate: integer(item.starts_per_candidate, 1, 100) ? item.starts_per_candidate as number : defaults.starts_per_candidate,
    initial_strides: Array.isArray(item.initial_strides) && item.initial_strides.length && item.initial_strides.every(value => integer(value, 1, 1000000)) ? item.initial_strides as number[] : defaults.initial_strides,
    stride_multiplier: integer(item.stride_multiplier, 2, 16) ? item.stride_multiplier as number : defaults.stride_multiplier,
    max_stride_fraction: step(item.max_stride_fraction) ? item.max_stride_fraction as number : defaults.max_stride_fraction,
    validation_reserve_fraction: fraction(item.validation_reserve_fraction) ? item.validation_reserve_fraction as number : defaults.validation_reserve_fraction,
    exhaustive_memory_limit_mb: integer(item.exhaustive_memory_limit_mb, 16, 65536) ? item.exhaustive_memory_limit_mb as number : defaults.exhaustive_memory_limit_mb,
    estimated_bytes_per_point: integer(item.estimated_bytes_per_point, 64, 65536) ? item.estimated_bytes_per_point as number : defaults.estimated_bytes_per_point,
    stress_cost_multiplier: finite(item.stress_cost_multiplier) && n(item.stress_cost_multiplier) > 1 ? item.stress_cost_multiplier as number : defaults.stress_cost_multiplier,
    stress_region_count: integer(item.stress_region_count, 0, 50) ? item.stress_region_count as number : defaults.stress_region_count,
    ranges: normalizedRanges,
  }
  // An invalid individual old range should fall back independently, while valid saved ranges survive.
  for (const name of activeRangeNames) {
    const candidate = { ...merged, ranges: { ...merged.ranges } }
    if (!validResearchDefaults(candidate)) {
      const range = merged.ranges[name]
      const valid = percentageRanges.has(name)
        ? step(range.low) && step(range.high) && range.low <= range.high
        : quantity(range.low, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
          && quantity(range.high, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
          && range.low <= range.high
      if (!valid) merged.ranges[name] = { ...(defaults.ranges[name] ?? legacyRangeFallbacks[name]) }
    }
  }
  // Preserve valid legacy boundaries that only need the user to correct a v4
  // lattice mismatch. The form reports them as invalid instead of silently
  // replacing a saved custom range with defaults.
  return merged
}

export function validWorkbenchGroup<K extends WorkbenchGroup>(group: K, value: unknown): value is WorkbenchDefaults[K] {
  const item = record(value)
  if (!item) return false
  if (group === 'research') return validResearchDefaults(item)
  if (group === 'grid') return ['trailing', 'arithmetic', 'geometric'].includes(String(item.grid_mode))
    && step(item.buy_step) && step(item.sell_step) && price(item.lower_price) && price(item.upper_price)
    && (item.lower_price == null || item.upper_price == null || (item.lower_price as number) < (item.upper_price as number))
  if (group === 'position') return quantity(item.initial_shares) && quantity(item.min_shares) && (item.max_shares === null || quantity(item.max_shares))
    && quantity(item.buy_shares, 100) && quantity(item.sell_shares, 100)
    && (item.min_shares as number) <= (item.initial_shares as number) && (item.max_shares === null || (item.initial_shares as number) <= (item.max_shares as number))
  return ['daily', 'weekly', 'monthly'].includes(String(item.frequency)) && quantity(item.shares, 100) && quantity(item.initial_shares)
}

export function normalizeWorkbenchDefaults(value: unknown): WorkbenchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults()
  return {
    grid: validWorkbenchGroup('grid', item?.grid) ? structuredClone(item.grid) : defaults.grid,
    position: validWorkbenchGroup('position', item?.position) ? structuredClone(item.position) : defaults.position,
    dca: validWorkbenchGroup('dca', item?.dca) ? structuredClone(item.dca) : defaults.dca,
    research: structuredClone(normalizeResearchDefaults(item?.research)),
  }
}

export function normalizeRememberedResearchDefaults(value: unknown): ResearchDefaults {
  return structuredClone(normalizeResearchDefaults(value))
}

export function strategyDefaults(config: BacktestConfig): Pick<WorkbenchDefaults, StrategyDefaultsGroup> {
  const { grid_mode, buy_step, sell_step, lower_price, upper_price, initial_shares, min_shares, max_shares, buy_shares, sell_shares } = config
  return {
    grid: { grid_mode, buy_step, sell_step, lower_price, upper_price },
    position: { initial_shares, min_shares, max_shares, buy_shares, sell_shares },
    dca: { ...config.dca },
  }
}

export function workbenchConfig(base: BacktestConfig, defaults: WorkbenchDefaults): BacktestConfig {
  return { ...base, ...defaults.grid, ...defaults.position, dca: { ...defaults.dca }, symbol: '', reference_price: null, reference_price_mode: 'auto' }
}
