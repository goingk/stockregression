import type { BacktestConfig } from './types'

export type GridDefaults = Pick<BacktestConfig, 'grid_mode' | 'buy_step' | 'sell_step' | 'grid_step' | 'lower_price' | 'upper_price'>
export type PositionDefaults = Pick<BacktestConfig, 'initial_shares' | 'min_shares' | 'max_shares' | 'buy_shares' | 'sell_shares'>
export type SearchRange = { low: number; high: number; step: number }
export interface ResearchDefaults {
  mode: 'historical' | 'robust' | 'both'
  n_trials: number
  seed: number
  top_candidates: number
  max_drawdown: number
  min_trades: number
  min_history_days: number
  min_minute_ratio: number
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

export const percentageRanges = new Set(['buy_step', 'sell_step', 'grid_step'])
export const rangeNames = ['buy_step', 'sell_step', 'grid_step', 'buy_shares', 'sell_shares', 'initial_shares', 'min_shares', 'max_shares'] as const

export function defaultSearchRanges(): Record<string, SearchRange> {
  return {
    buy_step: { low: .005, high: .1, step: .005 }, sell_step: { low: .005, high: .1, step: .005 }, grid_step: { low: .005, high: .1, step: .005 },
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

export function defaultWorkbenchDefaults(): WorkbenchDefaults {
  return {
    grid: { grid_mode: 'trailing', buy_step: .03, sell_step: .03, grid_step: .03, lower_price: null, upper_price: null },
    position: { initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 100 },
    dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
    research: { mode: 'both', n_trials: 200, seed: 42, top_candidates: 20, max_drawdown: .2, min_trades: 10, min_history_days: 504, min_minute_ratio: .95, ranges: defaultSearchRanges() },
  }
}

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const integer = (value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number => finite(value) && Number.isSafeInteger(value) && value >= min && value <= max
const quantity = (value: unknown, min = 0) => integer(value, min) && value % 100 === 0
const fraction = (value: unknown, min = 0) => finite(value) && value >= min && value <= 1
const step = (value: unknown) => finite(value) && value > 0 && value <= 1
const price = (value: unknown) => value === null || finite(value) && value > 0

export function validResearchDefaults(value: unknown): value is ResearchDefaults {
  const item = record(value)
  if (!item || !['historical', 'robust', 'both'].includes(String(item.mode)) || !integer(item.n_trials, 1, 5000)
    || !integer(item.seed, Number.MIN_SAFE_INTEGER) || !integer(item.top_candidates, 1, 100) || !step(item.max_drawdown)
    || !integer(item.min_trades, 0) || !integer(item.min_history_days, 20) || !fraction(item.min_minute_ratio)) return false
  const ranges = record(item.ranges)
  return ranges != null && rangeNames.every(name => {
    const range = record(ranges[name])
    if (!range || !finite(range.low) || !finite(range.high) || !finite(range.step) || range.low > range.high || range.step <= 0) return false
    return percentageRanges.has(name)
      ? step(range.low) && step(range.high)
      : quantity(range.low, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0) && quantity(range.high, name === 'buy_shares' || name === 'sell_shares' ? 100 : 0) && quantity(range.step, 100)
  })
}

export function validWorkbenchGroup<K extends WorkbenchGroup>(group: K, value: unknown): value is WorkbenchDefaults[K] {
  const item = record(value)
  if (!item) return false
  if (group === 'research') return validResearchDefaults(item)
  if (group === 'grid') return ['trailing', 'arithmetic', 'geometric'].includes(String(item.grid_mode))
    && step(item.buy_step) && step(item.sell_step) && step(item.grid_step) && price(item.lower_price) && price(item.upper_price)
    && (item.lower_price == null || item.upper_price == null || (item.lower_price as number) < (item.upper_price as number))
  if (group === 'position') return quantity(item.initial_shares) && quantity(item.min_shares) && quantity(item.max_shares)
    && quantity(item.buy_shares, 100) && quantity(item.sell_shares, 100)
    && (item.min_shares as number) <= (item.initial_shares as number) && (item.initial_shares as number) <= (item.max_shares as number)
  return ['daily', 'weekly', 'monthly'].includes(String(item.frequency)) && quantity(item.shares, 100) && quantity(item.initial_shares)
}

export function normalizeWorkbenchDefaults(value: unknown): WorkbenchDefaults {
  const item = record(value)
  const defaults = defaultWorkbenchDefaults()
  return {
    grid: validWorkbenchGroup('grid', item?.grid) ? structuredClone(item.grid) : defaults.grid,
    position: validWorkbenchGroup('position', item?.position) ? structuredClone(item.position) : defaults.position,
    dca: validWorkbenchGroup('dca', item?.dca) ? structuredClone(item.dca) : defaults.dca,
    research: validWorkbenchGroup('research', item?.research) ? structuredClone(item.research) : defaults.research,
  }
}

export function strategyDefaults(config: BacktestConfig): Pick<WorkbenchDefaults, StrategyDefaultsGroup> {
  const { grid_mode, buy_step, sell_step, grid_step, lower_price, upper_price, initial_shares, min_shares, max_shares, buy_shares, sell_shares } = config
  return {
    grid: { grid_mode, buy_step, sell_step, grid_step, lower_price, upper_price },
    position: { initial_shares, min_shares, max_shares, buy_shares, sell_shares },
    dca: { ...config.dca },
  }
}

export function workbenchConfig(base: BacktestConfig, defaults: WorkbenchDefaults): BacktestConfig {
  return { ...base, ...defaults.grid, ...defaults.position, dca: { ...defaults.dca }, symbol: '', reference_price: null, reference_price_mode: 'auto' }
}
