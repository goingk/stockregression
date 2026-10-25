import type { BacktestConfig } from './types'

export type GridDefaults = Pick<BacktestConfig, 'grid_mode' | 'buy_step' | 'sell_step' | 'lower_price' | 'upper_price'> & { link_spacing: boolean }
export type PositionDefaults = Pick<BacktestConfig, 'initial_shares' | 'min_shares' | 'max_shares' | 'buy_shares' | 'sell_shares'> & { link_shares: boolean }
export type SearchRange = { low: number; high: number; step?: number | null }

/** Research defaults of the v5 robust-J research protocol. */
export interface ResearchDefaults {
  strategy_id: 'geometric_robust_j'
  protocol_version: 5
  strategy_version: 5
  ranking_mode: 'robust_j'
  selection_target: 'fixed' | 'adaptive' | 'both'
  rebalance_days: number
  seed: number
  top_candidates: number
  max_principal_drawdown: number
  min_trades: number
  min_history_days: number
  min_minute_ratio: number
  hypercube_edge: number
  granularity: Record<string, number>
  search_method: 'auto' | 'hierarchical' | 'exhaustive'
  training_budget: number
  validation_budget: number
  reporting_budget: number
  total_evaluation_budget: number | null
  max_runtime_seconds: number
  recommendation_count: number
  max_shares_mode: 'unlimited'
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
  floor_k_ref: number
  training_min_days: number
  training_max_days: number
  training_window_mode: 'same_price' | 'fixed'
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
/** The four search axes of the v5 robust-J protocol. */
const robustFields = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'] as const
export const rangeNames = robustFields
/** Smallest configurable unit of every search axis. */
const granularityMinimums: Record<string, number> = {
  buy_step: .0001, sell_step: .0001, sell_shares: 100, buy_shares: 100,
}
/** Lattice the fixed neighbourhood half widths snap to. */
const halfWidthUnits: Record<string, number> = {
  buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100,
}

export function defaultSearchRanges(): Record<string, SearchRange> {
  return {
    buy_step: { low: .005, high: .1, step: null }, sell_step: { low: .005, high: .1, step: null },
    buy_shares: { low: 100, high: 1000, step: null }, sell_shares: { low: 100, high: 1000, step: null },
  }
}

export function researchRangesForSymbol(ranges: Record<string, SearchRange>, symbol: string): Record<string, SearchRange> {
  const minimum = symbol.startsWith('688') || symbol.startsWith('689') ? 200 : 100
  return ranges.buy_shares.low < minimum
    ? { ...ranges, buy_shares: { ...ranges.buy_shares, low: minimum, high: Math.max(minimum, ranges.buy_shares.high) } }
    : ranges
}

/** Research always submits an unlimited geometric grid. */
export function researchConfigForSubmission(value: BacktestConfig): BacktestConfig {
  return { ...value, grid_mode: 'geometric', max_shares: null }
}

export function defaultWorkbenchDefaults(): WorkbenchDefaults {
  return {
    grid: { grid_mode: 'trailing', buy_step: .03, sell_step: .03, lower_price: null, upper_price: null, link_spacing: false },
    position: { initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 100, link_shares: false },
    dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
    research: {
      strategy_id: 'geometric_robust_j', protocol_version: 5, strategy_version: 5, ranking_mode: 'robust_j',
      selection_target: 'both', rebalance_days: 63, seed: 42, top_candidates: 20, max_principal_drawdown: .2,
      min_trades: 10, min_history_days: 250, min_minute_ratio: .95, hypercube_edge: 3,
      granularity: { buy_step: .0001, sell_step: .0001, sell_shares: 100, buy_shares: 100 },
      search_method: 'auto', training_budget: 20000, validation_budget: 100000, reporting_budget: 1000,
      total_evaluation_budget: null, max_runtime_seconds: 3600, recommendation_count: 10,
      max_shares_mode: 'unlimited', slippage_mode: 'configured',
      neighborhood_fields: ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'],
      fixed_half_widths: { buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100 },
      sigma: 1, alpha: .3, beta: 1.2, tau: .01, eta_limit: 2, eta_hard_cap: 3.5, gamma_max: .4,
      eta_floor: 0, gamma_min: 0, floor_k_ref: 10, training_min_days: 250, training_max_days: 500,
      training_window_mode: 'same_price', peak_tolerance: 1e-12, start_min: 20, start_max: 64,
      starts_per_candidate: 4, initial_strides: [1, 2], stride_multiplier: 2, max_stride_fraction: .25,
      validation_reserve_fraction: .25, exhaustive_memory_limit_mb: 512, estimated_bytes_per_point: 512,
      stress_cost_multiplier: 2, stress_region_count: 3, ranges: defaultSearchRanges(),
    },
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
const n = (value: unknown, fallback = 0) => { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback }
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const integer = (value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number => finite(value) && Number.isSafeInteger(value) && value >= min && value <= max
export const researchQuantityUnit = (_name: string) => 100
const quantity = (value: unknown, min = 0) => integer(value, min) && value % 100 === 0
const fraction = (value: unknown, min = 0) => finite(value) && value >= min && value <= 1
const step = (value: unknown) => finite(value) && value > 0 && value <= 1
const positive = (value: unknown) => finite(value) && value > 0
const nonnegative = (value: unknown) => finite(value) && value >= 0
const price = (value: unknown) => value === null || finite(value) && value > 0

function hasRobustFields(value: unknown): boolean {
  return Array.isArray(value) && value.length === robustFields.length && robustFields.every(name => value.includes(name))
}

function hasRobustKeys(value: Record<string, unknown> | null): value is Record<string, unknown> {
  return !!value && Object.keys(value).length === robustFields.length && robustFields.every(name => name in value)
}

function alignedTo(value: unknown, unit: number): boolean {
  if (!finite(value) || value <= 0) return false
  return Math.abs(Number(value) / unit - Math.round(Number(value) / unit)) <= 1e-7
}

function validRangeValue(name: string, value: unknown): boolean {
  const range = record(value)
  if (!range || !finite(range.low) || !finite(range.high) || range.low > range.high) return false
  if (percentageRanges.has(name)) return range.low > 0 && range.high <= 1
  return range.low >= 100 && range.low % 100 === 0 && range.high % 100 === 0
}

export function validResearchDefaults(value: unknown): value is ResearchDefaults {
  const item = record(value)
  if (!item) return false
  if (Number(item.protocol_version) !== 5 || Number(item.strategy_version) !== 5) return false
  if (String(item.ranking_mode) !== 'robust_j' || String(item.strategy_id) !== 'geometric_robust_j') return false
  if (String(item.max_shares_mode) !== 'unlimited') return false
  if (!['fixed', 'adaptive', 'both'].includes(String(item.selection_target))) return false
  if (!integer(item.rebalance_days, 1) || !integer(item.seed, Number.MIN_SAFE_INTEGER)) return false
  if (!integer(item.top_candidates, 1, 100) || !step(item.max_principal_drawdown)) return false
  if (!integer(item.min_trades, 0) || !integer(item.min_history_days, 20) || !fraction(item.min_minute_ratio)) return false
  if (!integer(item.hypercube_edge, 1, 99) || Number(item.hypercube_edge) % 2 === 0) return false
  if (!['auto', 'hierarchical', 'exhaustive'].includes(String(item.search_method))) return false
  if (!integer(item.training_budget, 1, 1000000) || !integer(item.validation_budget, 1, 2000000)) return false
  if (!integer(item.reporting_budget, 0, 2000000)) return false
  if (item.total_evaluation_budget !== null && !integer(item.total_evaluation_budget, 1, 10000000)) return false
  if (!integer(item.max_runtime_seconds, 1, 86400) || !integer(item.recommendation_count, 1, 100)) return false
  if (!['configured', 'zero'].includes(String(item.slippage_mode))) return false
  if (!['same_price', 'fixed'].includes(String(item.training_window_mode))) return false
  if (!hasRobustFields(item.neighborhood_fields)) return false
  if (!finite(item.sigma) || n(item.sigma) > 100 || !fraction(item.alpha)) return false
  if (!finite(item.beta) || n(item.beta) < 0 || n(item.beta) > 100) return false
  if (!finite(item.tau) || n(item.tau) < -1 || n(item.tau) > 10) return false
  if (!nonnegative(item.eta_limit) || n(item.eta_limit) > 1000 || !nonnegative(item.gamma_max) || n(item.gamma_max) > 1000) return false
  if (!fraction(item.eta_floor) || !nonnegative(item.gamma_min) || n(item.gamma_min) > 1000) return false
  if (!positive(item.floor_k_ref) || n(item.floor_k_ref) > 1000000) return false
  if (n(item.eta_floor) > n(item.eta_limit)) return false
  if (item.eta_hard_cap !== null) {
    if (!finite(item.eta_hard_cap) || n(item.eta_hard_cap) < 0 || n(item.eta_hard_cap) > 1000) return false
    if (n(item.eta_hard_cap) < n(item.eta_limit)) return false
  }
  if (!integer(item.training_min_days, 20, 5000) || !integer(item.training_max_days, 20, 5000)) return false
  if (n(item.training_min_days) > n(item.training_max_days)) return false
  if (!positive(item.peak_tolerance) || n(item.peak_tolerance) > .1) return false
  if (!integer(item.start_min, 1, 1000) || !integer(item.start_max, 1, 1000) || n(item.start_min) > n(item.start_max)) return false
  if (!integer(item.starts_per_candidate, 1, 100)) return false
  if (!Array.isArray(item.initial_strides) || !item.initial_strides.length || !item.initial_strides.every(value => integer(value, 1))) return false
  if (!integer(item.stride_multiplier, 2, 16) || !step(item.max_stride_fraction)) return false
  if (!fraction(item.validation_reserve_fraction) || !integer(item.exhaustive_memory_limit_mb, 16, 65536)) return false
  if (!integer(item.estimated_bytes_per_point, 64, 65536)) return false
  if (!finite(item.stress_cost_multiplier) || n(item.stress_cost_multiplier) <= 1 || n(item.stress_cost_multiplier) > 20) return false
  if (!integer(item.stress_region_count, 0, 50)) return false
  const halfWidths = record(item.fixed_half_widths)
  if (!hasRobustKeys(halfWidths)) return false
  if (!robustFields.every(name => alignedTo(halfWidths[name], halfWidthUnits[name]))) return false
  const granularity = record(item.granularity)
  if (!hasRobustKeys(granularity)) return false
  if (!robustFields.every(name => alignedTo(granularity[name], granularityMinimums[name]))) return false
  const ranges = record(item.ranges)
  if (!hasRobustKeys(ranges)) return false
  return robustFields.every(name => {
    const range = record(ranges[name])
    if (!range) return false
    if (range.step !== null && range.step !== undefined) return false
    return validRangeValue(name, range)
  })
}

/** Migrate remembered research settings of any earlier protocol to the v5 shape. */
function normalizeResearchDefaults(value: unknown): ResearchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults().research
  if (!item) return defaults
  const savedRanges = record(item.ranges)
  const savedGranularity = record(item.granularity)
  const savedHalfWidths = record(item.fixed_half_widths)
  return {
    ...defaults,
    strategy_id: 'geometric_robust_j',
    protocol_version: 5,
    strategy_version: 5,
    ranking_mode: 'robust_j',
    max_shares_mode: 'unlimited',
    neighborhood_fields: [...robustFields],
    selection_target: ['fixed', 'adaptive', 'both'].includes(String(item.selection_target))
      ? item.selection_target as ResearchDefaults['selection_target'] : defaults.selection_target,
    rebalance_days: integer(item.rebalance_days, 1) ? item.rebalance_days : defaults.rebalance_days,
    seed: integer(item.seed, Number.MIN_SAFE_INTEGER) ? item.seed : defaults.seed,
    top_candidates: integer(item.top_candidates, 1, 100) ? item.top_candidates : defaults.top_candidates,
    // The retired unit-NAV drawdown uses a different denominator than the peak
    // principal drawdown and is therefore never promoted to an equivalent.
    max_principal_drawdown: step(item.max_principal_drawdown) ? item.max_principal_drawdown as number : defaults.max_principal_drawdown,
    min_trades: integer(item.min_trades, 0) ? item.min_trades : defaults.min_trades,
    min_history_days: integer(item.min_history_days, 20) ? item.min_history_days : defaults.min_history_days,
    min_minute_ratio: fraction(item.min_minute_ratio) ? item.min_minute_ratio as number : defaults.min_minute_ratio,
    hypercube_edge: integer(item.hypercube_edge, 1, 99) && Number(item.hypercube_edge) % 2 === 1
      ? Number(item.hypercube_edge) : defaults.hypercube_edge,
    granularity: Object.fromEntries(robustFields.map(name => [
      name, alignedTo(savedGranularity?.[name], granularityMinimums[name])
        ? Number(savedGranularity?.[name]) : defaults.granularity[name],
    ])),
    search_method: ['auto', 'hierarchical', 'exhaustive'].includes(String(item.search_method))
      ? item.search_method as ResearchDefaults['search_method'] : defaults.search_method,
    training_budget: integer(item.training_budget, 1, 1000000) ? item.training_budget : defaults.training_budget,
    validation_budget: integer(item.validation_budget, 1, 2000000) ? item.validation_budget : defaults.validation_budget,
    reporting_budget: integer(item.reporting_budget, 0, 2000000) ? item.reporting_budget : defaults.reporting_budget,
    total_evaluation_budget: integer(item.total_evaluation_budget, 1, 10000000) ? item.total_evaluation_budget : null,
    max_runtime_seconds: integer(item.max_runtime_seconds, 1, 86400) ? item.max_runtime_seconds : defaults.max_runtime_seconds,
    recommendation_count: integer(item.recommendation_count, 1, 100) ? item.recommendation_count : defaults.recommendation_count,
    slippage_mode: ['configured', 'zero'].includes(String(item.slippage_mode))
      ? item.slippage_mode as ResearchDefaults['slippage_mode'] : defaults.slippage_mode,
    fixed_half_widths: Object.fromEntries(robustFields.map(name => [
      name, alignedTo(savedHalfWidths?.[name], halfWidthUnits[name])
        ? Number(savedHalfWidths?.[name]) : defaults.fixed_half_widths[name],
    ])),
    sigma: positive(item.sigma) && n(item.sigma) <= 100 ? item.sigma as number : defaults.sigma,
    alpha: fraction(item.alpha) ? item.alpha as number : defaults.alpha,
    beta: nonnegative(item.beta) && n(item.beta) <= 100 ? item.beta as number : defaults.beta,
    tau: finite(item.tau) && n(item.tau) >= -1 && n(item.tau) <= 10 ? item.tau as number : defaults.tau,
    eta_limit: nonnegative(item.eta_limit) && n(item.eta_limit) <= 1000 ? item.eta_limit as number : defaults.eta_limit,
    eta_hard_cap: item.eta_hard_cap === null ? null
      : nonnegative(item.eta_hard_cap) && n(item.eta_hard_cap) <= 1000 ? item.eta_hard_cap as number : defaults.eta_hard_cap,
    gamma_max: nonnegative(item.gamma_max) && n(item.gamma_max) <= 1000 ? item.gamma_max as number : defaults.gamma_max,
    eta_floor: fraction(item.eta_floor) ? item.eta_floor as number : defaults.eta_floor,
    gamma_min: nonnegative(item.gamma_min) && n(item.gamma_min) <= 1000 ? item.gamma_min as number : defaults.gamma_min,
    floor_k_ref: positive(item.floor_k_ref) && n(item.floor_k_ref) <= 1000000 ? item.floor_k_ref as number : defaults.floor_k_ref,
    training_min_days: integer(item.training_min_days, 20, 5000) ? item.training_min_days : defaults.training_min_days,
    training_max_days: integer(item.training_max_days, 20, 5000) ? item.training_max_days : defaults.training_max_days,
    training_window_mode: ['same_price', 'fixed'].includes(String(item.training_window_mode))
      ? item.training_window_mode as ResearchDefaults['training_window_mode'] : defaults.training_window_mode,
    peak_tolerance: positive(item.peak_tolerance) && n(item.peak_tolerance) <= .1 ? item.peak_tolerance as number : defaults.peak_tolerance,
    start_min: integer(item.start_min, 1, 1000) ? item.start_min : defaults.start_min,
    start_max: integer(item.start_max, 1, 1000) ? item.start_max : defaults.start_max,
    starts_per_candidate: integer(item.starts_per_candidate, 1, 100) ? item.starts_per_candidate : defaults.starts_per_candidate,
    initial_strides: Array.isArray(item.initial_strides) && item.initial_strides.length && item.initial_strides.every(value => integer(value, 1))
      ? item.initial_strides as number[] : defaults.initial_strides,
    stride_multiplier: integer(item.stride_multiplier, 2, 16) ? item.stride_multiplier : defaults.stride_multiplier,
    max_stride_fraction: step(item.max_stride_fraction) ? item.max_stride_fraction as number : defaults.max_stride_fraction,
    validation_reserve_fraction: fraction(item.validation_reserve_fraction) ? item.validation_reserve_fraction as number : defaults.validation_reserve_fraction,
    exhaustive_memory_limit_mb: integer(item.exhaustive_memory_limit_mb, 16, 65536) ? item.exhaustive_memory_limit_mb : defaults.exhaustive_memory_limit_mb,
    estimated_bytes_per_point: integer(item.estimated_bytes_per_point, 64, 65536) ? item.estimated_bytes_per_point : defaults.estimated_bytes_per_point,
    stress_cost_multiplier: finite(item.stress_cost_multiplier) && n(item.stress_cost_multiplier) > 1 && n(item.stress_cost_multiplier) <= 20
      ? item.stress_cost_multiplier as number : defaults.stress_cost_multiplier,
    stress_region_count: integer(item.stress_region_count, 0, 50) ? item.stress_region_count : defaults.stress_region_count,
    // The v5 protocol configures the sampling step through granularity only.
    ranges: Object.fromEntries(robustFields.map(name => {
      const saved = savedRanges?.[name]
      const range = validRangeValue(name, saved) ? record(saved) as SearchRange : defaults.ranges[name]
      return [name, { low: range.low, high: range.high, step: null }]
    })),
  }
}

export function validWorkbenchGroup<K extends WorkbenchGroup>(group: K, value: unknown): value is WorkbenchDefaults[K] {
  const item = record(value)
  if (!item) return false
  if (group === 'research') return validResearchDefaults(item)
  if (group === 'grid') return ['trailing', 'arithmetic', 'geometric'].includes(String(item.grid_mode))
    && step(item.buy_step) && step(item.sell_step) && price(item.lower_price) && price(item.upper_price)
    && (item.lower_price == null || item.upper_price == null || (item.lower_price as number) < (item.upper_price as number))
    && (item.link_spacing === undefined || typeof item.link_spacing === 'boolean')
  if (group === 'position') return quantity(item.initial_shares) && quantity(item.min_shares) && (item.max_shares === null || quantity(item.max_shares))
    && quantity(item.buy_shares, 100) && quantity(item.sell_shares, 100)
    && (item.link_shares === undefined || typeof item.link_shares === 'boolean')
    && (item.min_shares as number) <= (item.initial_shares as number) && (item.max_shares === null || (item.initial_shares as number) <= (item.max_shares as number))
  return ['daily', 'weekly', 'monthly'].includes(String(item.frequency)) && quantity(item.shares, 100) && quantity(item.initial_shares)
}

export function normalizeWorkbenchDefaults(value: unknown): WorkbenchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults()
  return {
    grid: validWorkbenchGroup('grid', item?.grid) ? { ...defaults.grid, ...structuredClone(item.grid) } : defaults.grid,
    position: validWorkbenchGroup('position', item?.position) ? { ...defaults.position, ...structuredClone(item.position) } : defaults.position,
    dca: validWorkbenchGroup('dca', item?.dca) ? structuredClone(item.dca) : defaults.dca,
    research: structuredClone(normalizeResearchDefaults(item?.research)),
  }
}

export function normalizeRememberedResearchDefaults(value: unknown): ResearchDefaults {
  return structuredClone(normalizeResearchDefaults(value))
}

/**
 * Merge only fields the user actually changed into the remembered defaults.
 * Source drafts may temporarily carry ETF constraints, but unrelated edits
 * must not persist those imported values as personal defaults.
 */
export function mergeEditedResearchDefaults(
  current: ResearchDefaults,
  next: ResearchDefaults,
  remembered: ResearchDefaults,
): ResearchDefaults {
  const merged = structuredClone(remembered)
  for (const key of Object.keys(next) as Array<keyof ResearchDefaults>) {
    if (JSON.stringify(current[key]) !== JSON.stringify(next[key])) {
      (merged as unknown as Record<string, unknown>)[key] = structuredClone(next[key])
    }
  }
  return merged
}

export function strategyDefaults(config: BacktestConfig): Pick<WorkbenchDefaults, StrategyDefaultsGroup> {
  const { grid_mode, buy_step, sell_step, lower_price, upper_price, initial_shares, min_shares, max_shares, buy_shares, sell_shares } = config
  const links = config.parameter_links ?? { spacing: false, shares: false }
  return {
    grid: { grid_mode, buy_step, sell_step, lower_price, upper_price, link_spacing: links.spacing },
    position: { initial_shares, min_shares, max_shares, buy_shares, sell_shares, link_shares: links.shares },
    dca: { ...config.dca },
  }
}

export function workbenchConfig(base: BacktestConfig, defaults: WorkbenchDefaults): BacktestConfig {
  const { link_spacing, ...grid } = defaults.grid
  const { link_shares, ...position } = defaults.position
  return { ...base, ...grid, ...position, parameter_links: { spacing: link_spacing, shares: link_shares }, dca: { ...defaults.dca }, symbol: '', reference_price: null, reference_price_mode: 'auto' }
}
