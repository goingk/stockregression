export type Frequency = '1m' | '5m' | '15m' | '30m' | '60m' | '1d'
export type QuantityMode = 'shares' | 'cash' | 'equity_ratio'
export interface FeeConfig {
  commission_rate: number
  min_commission: number
  slippage_bps: number
  share_face_value?: number
  legacy_sh_transfer_rate?: number
  legacy_sh_min_transfer?: number
  legacy_sz_transfer_rate?: number
}

export interface QuantityConfig {
  mode: QuantityMode
  value: number
}

export interface BacktestConfig {
  schema_version: 2
  symbol: string
  start_date: string
  end_date: string | null
  frequency: Frequency
  initial_shares: number
  min_shares: number
  max_shares: number
  buy_shares: number
  sell_shares: number
  dca: { frequency: 'daily' | 'weekly' | 'monthly'; shares: number; initial_shares: number }
  reference_price_mode: 'auto' | 'manual'
  initial_cash?: number
  /** Cash reserved outside the strategy; fixed when loading a research candidate. */
  reserve_cash?: number | null
  /** Parameter selection cutoff associated with a loaded research candidate. */
  selection_asof?: string | null
  grid_mode: 'trailing' | 'arithmetic' | 'geometric'
  buy_step: number
  sell_step: number
  grid_step: number
  reference_price: number | null
  lower_price: number | null
  upper_price: number | null
  fees: FeeConfig
}

export interface Instrument {
  symbol: string
  name: string
  exchange: string
  board: string
  asset_type: 'stock' | 'etf'
  price_tick?: number
  settlement?: 'T0' | 'T1'
}

export interface WatchItem extends Instrument {
  coverage?: Record<string, unknown>
}

export interface DataSourceSpan {
  source?: string | null
  source_kind?: string | null
  start?: string | null
  end?: string | null
  days?: number | null
  verified?: boolean | null
}

export interface DataSourceEvent {
  kind?: string | null
  source?: string | null
  message?: string | null
  status?: 'historical' | 'resolved' | string | null
  at?: string | null
}

export interface DataDiagnostics {
  schema_version?: number
  status?: 'complete' | 'partial' | 'unavailable' | string
  requested_frequency?: string | null
  minute_quality_applicable?: boolean | null
  checked_start?: string | null
  checked_end?: string | null
  daily_days?: number
  minute_days?: number
  total_days?: number
  pending_calibration_days?: number
  minute_ratio?: number | null
  minute_through?: string | null
  daily_through?: string | null
  auction_placeholders?: number
  missing_raw_minutes?: number
  repaired_days?: number
  raw_complete?: boolean | null
  source_spans?: DataSourceSpan[]
  current_warnings?: string[]
}

export interface Job {
  id: string
  kind: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string
  progress: number
  message: string
  error?: unknown
  result_id?: string
  created_at?: string
  updated_at?: string
  cancel_requested?: boolean
  payload?: Record<string, unknown>
}

export interface ConfigRecord {
  id: string
  name: string
  config: BacktestConfig
}

export interface ResultSummary {
  id: string
  kind: string
  symbol: string
  created_at: string
  title: string
}

export interface EquityPoint {
  timestamp: string
  equity: number
  cash?: number
  position?: number
  position_ratio?: number
  drawdown?: number | null
  normalized_equity?: number | null
  total_invested?: number
}

export interface Trade {
  timestamp: string
  side: string
  quantity: number
  price: number
  fees?: number
  cash?: number
  shares?: number
  reason?: string
}

export interface StrategyResult {
  metrics: Record<string, number | null>
  return_basis?: string
  equity: EquityPoint[]
  equity_frequency?: string
  trades: Trade[]
  rejections: Array<Record<string, unknown>>
  period_returns: Array<Record<string, unknown>>
}

export interface BacktestResult {
  id?: string
  symbol: string
  is_demo?: boolean
  config: BacktestConfig
  data_version?: string
  diagnostics?: DataDiagnostics
  source_events?: DataSourceEvent[]
  equity_frequency?: string
  quality?: Record<string, unknown>
  warnings?: string[]
  strategies: Record<'grid' | 'dca' | 'buy_hold' | string, StrategyResult>
  schema_version?: number
  funding?: { reserve_cash: number; initial_equity: number; valuation_price: number; reference_price: number; reference_date: string; dca_initial_equity?: number; buy_hold_initial_shares?: number; buy_hold_initial_equity?: number }
  effective_end_date?: string
  accounting_basis?: string
}

export interface Coverage {
  symbol: string
  version?: string
  rows?: number
  first_date?: string | null
  last_date?: string | null
  minute_days?: number
  daily_days?: number
  total_days?: number
  suspended_days?: number
  minute_ratio?: number
  synthetic_bars?: number
  is_demo?: boolean
  diagnostics?: DataDiagnostics
  source_events?: DataSourceEvent[]
  days?: Array<{ date: string; mode?: string; missing_minutes?: number; missing_data_minutes?: number; auction_placeholders?: number; reason?: string | null }>
  warnings?: string[]
  profiles?: unknown[] | Record<string, unknown>
  bytes?: number
  [key: string]: unknown
}
