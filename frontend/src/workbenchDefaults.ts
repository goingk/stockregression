import type { BacktestConfig } from './types'

export type GridDefaults = Pick<BacktestConfig, 'grid_mode' | 'buy_step' | 'sell_step' | 'lower_price' | 'upper_price'>
export type PositionDefaults = Pick<BacktestConfig, 'initial_shares' | 'min_shares' | 'max_shares' | 'buy_shares' | 'sell_shares'>
export type SearchRange = { low: number; high: number; step: number }
export interface ResearchDefaults {
  protocol_version: 1 | 2 | 3 | 4
  selection_target: 'fixed' | 'adaptive' | 'both'
  rebalance_days: number
  bootstrap_reps: number
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
  strategy_version: 1 | 2
  search_method: 'auto' | 'hierarchical' | 'exhaustive'
  training_budget: number
  validation_budget: number
  recommendation_count: number
  max_shares_mode: 'search' | 'unlimited'
  slippage_mode: 'configured' | 'zero'
  neighborhood_fields: string[]
  fixed_half_widths: Record<string, number>
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
export const rangeNames = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares', 'initial_shares', 'min_shares', 'max_shares'] as const
const v4NeighborhoodFields = ['buy_step', 'sell_step', 'sell_shares', 'buy_shares', 'min_shares', 'max_shares'] as const

export function defaultSearchRanges(): Record<string, SearchRange> {
  return {
    buy_step: { low: .005, high: .1, step: .005 }, sell_step: { low: .005, high: .1, step: .005 },
    buy_shares: { low: 100, high: 1000, step: 100 }, sell_shares: { low: 100, high: 1000, step: 100 },
    initial_shares: { low: 0, high: 2000, step: 100 }, min_shares: { low: 0, high: 1000, step: 100 }, max_shares: { low: 1000, high: 5000, step: 100 },
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
    research: { protocol_version: 4, strategy_version: 2, selection_target: 'both', rebalance_days: 63, bootstrap_reps: 10000, mode: 'both', n_trials: 1000, local_search_budget: 3000, seed: 42, top_candidates: 20, max_drawdown: .2, min_trades: 10, min_history_days: 504, min_minute_ratio: .95, ranking_mode: 'max_volume', search_method: 'auto', training_budget: 20000, validation_budget: 100000, recommendation_count: 10, max_shares_mode: 'search', slippage_mode: 'configured', neighborhood_fields: [...v4NeighborhoodFields], fixed_half_widths: { buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100, min_shares: 500, max_shares: 500 }, ranges: defaultSearchRanges() },
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const integer = (value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number => finite(value) && Number.isSafeInteger(value) && value >= min && value <= max
export const researchQuantityUnit = (name: string) => name === 'min_shares' || name === 'max_shares' ? 500 : 100
const quantity = (value: unknown, min = 0) => integer(value, min) && value % 100 === 0
const fraction = (value: unknown, min = 0) => finite(value) && value >= min && value <= 1
const step = (value: unknown) => finite(value) && value > 0 && value <= 1
const price = (value: unknown) => value === null || finite(value) && value > 0

export function validResearchDefaults(value: unknown): value is ResearchDefaults {
  const item = record(value)
  if (!item || ![1, 2, 3, 4].includes(Number(item.protocol_version)) || !['fixed', 'adaptive', 'both'].includes(String(item.selection_target))
    || !integer(item.rebalance_days, 1, 504) || !integer(item.bootstrap_reps, 100, 100000)
    || !['historical', 'robust', 'both'].includes(String(item.mode)) || !integer(item.n_trials, 1, 5000) || !integer(item.local_search_budget, 1, 5000)
    || !integer(item.seed, Number.MIN_SAFE_INTEGER) || !integer(item.top_candidates, 1, 100) || !step(item.max_drawdown)
    || !integer(item.min_trades, 0) || !integer(item.min_history_days, 20) || !fraction(item.min_minute_ratio)
    || !['max_volume', 'fixed_mean'].includes(String(item.ranking_mode))
    || ![1, 2].includes(Number(item.strategy_version))
    || (String(item.ranking_mode) === 'fixed_mean' && Number(item.strategy_version) !== 1)
    || !['auto', 'hierarchical', 'exhaustive'].includes(String(item.search_method))
    || !integer(item.training_budget, 1, 1000000) || !integer(item.validation_budget, 1, 2000000)
    || !integer(item.recommendation_count, 1, 100)
    || !['search', 'unlimited'].includes(String(item.max_shares_mode))
    || !['configured', 'zero'].includes(String(item.slippage_mode))
    || !Array.isArray(item.neighborhood_fields) || !item.neighborhood_fields.length
    || !item.neighborhood_fields.every(name => v4NeighborhoodFields.includes(String(name) as typeof v4NeighborhoodFields[number]))
    || new Set(item.neighborhood_fields).size !== item.neighborhood_fields.length) return false
  const halfWidths = record(item.fixed_half_widths)
  if (!halfWidths || !v4NeighborhoodFields.every(name => finite(halfWidths[name]) && Number(halfWidths[name]) > 0)) return false
  if (['buy_step', 'sell_step'].some(name => Math.abs(Number(halfWidths[name]) * 10000 - Math.round(Number(halfWidths[name]) * 10000)) > 1e-8)) return false
  if (['sell_shares', 'buy_shares', 'min_shares', 'max_shares'].some(name => Math.abs(Number(halfWidths[name]) / researchQuantityUnit(name) - Math.round(Number(halfWidths[name]) / researchQuantityUnit(name))) > 1e-8)) return false
  const ranges = record(item.ranges)
  return ranges != null && rangeNames.every(name => {
    const range = record(ranges[name])
    if (!range || !finite(range.low) || !finite(range.high) || !finite(range.step) || range.low > range.high || range.step <= 0) return false
    if (Number(item.protocol_version) === 4 && ['min_shares', 'max_shares'].includes(name)
      && (range.low % 500 !== 0 || range.high % 500 !== 0)) return false
    if (Number(item.protocol_version) === 4 && percentageRanges.has(name)
      && (Math.abs(range.low * 10000 - Math.round(range.low * 10000)) > 1e-8 || Math.abs(range.high * 10000 - Math.round(range.high * 10000)) > 1e-8)) return false
    return percentageRanges.has(name)
      ? step(range.low) && step(range.high)
      : quantity(range.low, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0) && quantity(range.high, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0) && quantity(range.step, 100)
  })
}

/** Fill fields added by newer research protocols without discarding remembered legacy ranges. */
function normalizeResearchDefaults(value: unknown): ResearchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults().research
  if (!item) return defaults
  const ranges = record(item.ranges)
  const legacyRange = record(ranges?.grid_step)
  const normalizedRanges = Object.fromEntries(rangeNames.map((name) => {
    const saved = name === 'buy_step' || name === 'sell_step' ? legacyRange ?? record(ranges?.[name]) : record(ranges?.[name])
    const fallback = defaults.ranges[name]
    const candidate = saved && finite(saved.low) && finite(saved.high) && finite(saved.step)
      ? { low: saved.low, high: saved.high, step: saved.step }
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
  const merged: ResearchDefaults = {
    ...defaults,
    strategy_version: item.ranking_mode === 'fixed_mean' ? 1 : 2,
    protocol_version: defaults.protocol_version,
    selection_target: ['fixed', 'adaptive', 'both'].includes(String(item.selection_target)) ? item.selection_target as ResearchDefaults['selection_target'] : defaults.selection_target,
    rebalance_days: integer(item.rebalance_days, 1, 504) ? item.rebalance_days : defaults.rebalance_days,
    bootstrap_reps: integer(item.bootstrap_reps, 100, 100000) ? item.bootstrap_reps : defaults.bootstrap_reps,
    mode: ['historical', 'robust', 'both'].includes(String(item.mode)) ? item.mode as ResearchDefaults['mode'] : defaults.mode,
    n_trials: integer(item.n_trials, 1, 5000) ? item.n_trials : defaults.n_trials,
    local_search_budget: integer(item.local_search_budget, 1, 5000) ? item.local_search_budget : defaults.local_search_budget,
    seed: integer(item.seed, Number.MIN_SAFE_INTEGER) ? item.seed : defaults.seed,
    top_candidates: integer(item.top_candidates, 1, 100) ? item.top_candidates : defaults.top_candidates,
    max_drawdown: step(item.max_drawdown) ? item.max_drawdown as number : defaults.max_drawdown,
    min_trades: integer(item.min_trades, 0) ? item.min_trades : defaults.min_trades,
    min_history_days: integer(item.min_history_days, 20) ? item.min_history_days : defaults.min_history_days,
    min_minute_ratio: fraction(item.min_minute_ratio) ? item.min_minute_ratio as number : defaults.min_minute_ratio,
    ranking_mode: ['max_volume', 'fixed_mean'].includes(String(item.ranking_mode)) ? item.ranking_mode as ResearchDefaults['ranking_mode'] : defaults.ranking_mode,
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
    ranges: normalizedRanges,
  }
  // An invalid individual old range should fall back independently, while valid saved ranges survive.
  for (const name of rangeNames) {
    const candidate = { ...merged, ranges: { ...merged.ranges } }
    if (!validResearchDefaults(candidate)) {
      const range = merged.ranges[name]
      const valid = percentageRanges.has(name)
        ? step(range.low) && step(range.high) && range.low <= range.high && step(range.step)
        : quantity(range.low, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
          && quantity(range.high, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0)
          && quantity(range.step, 100) && range.low <= range.high
      if (!valid) merged.ranges[name] = { ...defaults.ranges[name] }
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
