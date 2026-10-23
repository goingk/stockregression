export type Frequency = '1m' | '5m' | '15m' | '30m' | '60m' | '1d'
export type DataTimeframe = '1min' | '5min' | '15min' | '30min' | '60min' | '1day' | '1week' | '1month'

export interface DataSourceHealth {
  source: string
  capability: string
  market: 'stock' | 'etf'
  status: 'unknown' | 'passed' | 'failed' | 'cancelled'
  rows?: number | null
  coverage_start?: string | null
  coverage_end?: string | null
  duration_seconds?: number | null
  last_checked_at?: string | null
  last_success_at?: string | null
  last_failure_at?: string | null
  cooldown_until?: string | null
  error_category?: string | null
  error?: string | null
}

export interface DataSourceOption {
  source: string
  label: string
  role: 'default' | 'optional' | 'fallback'
  selected: boolean
  selectable: boolean
  states: DataSourceHealth[]
}

export interface DataSourceCapability {
  capability: 'recent_minute' | 'etf_daily'
  label: string
  description: string
  configured: string
  default: string
  optional: string
  routes: Array<{ market: 'stock' | 'etf'; configured: string; effective: string; fallback: boolean; reason?: string | null }>
  sources: DataSourceOption[]
}

export interface DataSourcesResponse {
  policy_version: string
  cooldown_hours: number
  capabilities: DataSourceCapability[]
}

export interface DataDataset {
  symbol: string
  timeframe: DataTimeframe
  adjustment: string
  current_version: number
  first_time?: string | null
  last_time?: string | null
  completed_through?: string | null
  latest_record_time?: string | null
  verified_from?: string | null
  verified_through?: string | null
  row_count: number
  raw_rows: number
  synthetic_rows: number
  status: 'ready' | 'partial' | 'pending_validation' | 'legacy' | string
  reason?: string | null
  updated_at?: string | null
  metadata?: Record<string, unknown>
  storage?: string
}

export interface DataPreparationRequest {
  symbol?: string
  frequency?: string
  start?: string | null
  end?: string | null
  gaps?: Array<{ kind?: string; timeframe?: string; message?: string }>
  config?: BacktestConfig
  checked_start?: string | null
  checked_end?: string | null
  package_id?: string
  origin?: 'backtest' | 'research'
}

export interface DataPeriodSummary {
  symbol: string
  timeframe: DataTimeframe
  current_version: number
  first_time?: string | null
  last_time?: string | null
  completed_through?: string | null
  latest_record_time?: string | null
  verified_through?: string | null
  row_count: number
  synthetic_rows: number
  status: string
  quality_status: 'passed' | 'pending' | 'failed' | string
  coverage_status: 'complete' | 'partial' | 'unknown' | string
  file_status: 'readable' | 'missing' | 'changed' | string
  ready: boolean
  reason?: string | null
  updated_at?: string | null
  metadata?: Record<string, unknown>
  validation?: {
    rules_version?: number
    validated_days?: number
    pending?: string[]
    failed?: string[]
    calendar_available?: boolean
  }
  shard_count?: number
  archive?: {
    protected: boolean
    sources: Array<{ path: string; size: number; archived: boolean; archive_path?: string | null; archived_at?: string | null }>
  }
}

export interface DataInventoryItem {
  symbol: string
  name: string
  asset_type: 'stock' | 'etf' | string
  periods: Partial<Record<DataTimeframe, DataPeriodSummary>>
  updated_at?: string | null
}

export interface DataInventoryResponse {
  items: DataInventoryItem[]
  count: number
  dataset_count: number
  total_symbols: number
  stats: { symbols: number; datasets: number; ready: number; pending: number; rows: number; bytes: number }
  limit: number
  offset: number
}

export interface DataPackageInfo {
  package_id: string | null
  generation: string | null
  package_version: number
  schema_version: number | null
  storage_layout: string
  index_status: string
  path: string
  status: string
  readable: boolean
  manifest_ready?: boolean
  shard_count: number
  shard_bytes: number
  bytes: number
  file_count: number
  updated_at?: string | null
  pending_recovery?: number
}

export interface DataPreparationStep {
  kind: 'auxiliary' | 'bars' | 'validate' | string
  symbol: string
  timeframe?: string
  mode?: string
  start?: string
  end?: string
  reason: string
  status?: string
  error?: string
  result?: unknown
}

export interface DataPreparationPlan {
  symbol: string
  frequency: string
  asset_type?: string
  ready: boolean
  checked_start?: string | null
  checked_end?: string | null
  package_id: string
  package_version: number
  gaps: Array<{ kind?: string; timeframe?: string; message?: string }>
  warnings?: string[]
  degraded_days?: Array<{ date: string; timeframe?: string; reason?: string }>
  steps: DataPreparationStep[]
  config: BacktestConfig
}
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

export interface FloorPenaltyConfig {
  eta_floor: number
  gamma_min: number
  floor_k_ref: number
}

export interface BacktestConfig {
  schema_version: number
  symbol: string
  start_date: string
  end_date: string | null
  frequency: Frequency
  initial_shares: number
  min_shares: number
  max_shares: number | null
  buy_shares: number
  sell_shares: number
  dca: { frequency: 'daily' | 'weekly' | 'monthly'; shares: number; initial_shares: number }
  reference_price_mode: 'auto' | 'manual'
  initial_cash?: number
  /** Legacy v1/v2 only; new v3 configs do not send this field. */
  reserve_cash?: number | null
  /** Parameter selection cutoff associated with a loaded research candidate. */
  selection_asof?: string | null
  grid_mode: 'trailing' | 'arithmetic' | 'geometric'
  buy_step: number
  sell_step: number
  /** Legacy schema 1 result field. New share-account configs always send buy_step and sell_step. */
  grid_step?: number
  reference_price: number | null
  lower_price: number | null
  upper_price: number | null
  floor_penalty_config?: FloorPenaltyConfig | null
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
  has_data?: boolean
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
  status: 'queued' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled' | string
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
  total_invested?: number
  /** v5 peak-principal accounting: the principal denominator the ratio is measured against. */
  peak_principal?: number | null
  /** v5 peak-principal accounting: profit accumulated since the run started. */
  cumulative_profit?: number | null
  /** v5 peak-principal accounting: 1 + cumulative profit / peak principal. */
  principal_curve?: number | null
  /** v5 peak-principal accounting: drawdown of the principal profit curve. */
  principal_drawdown?: number | null
  /** Legacy unit-NAV fields; only archived pre-v5 backtest results still carry them. */
  normalized_equity?: number | null
  /** Legacy unit-NAV drawdown; v5 reports `principal_drawdown` instead. */
  drawdown?: number | null
  /** Legacy position ratio; v5 reports the share count only. */
  position_ratio?: number | null
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
  floor_episode_count?: number
  floor_episode_grids?: number[]
  floor_penalty?: number
  floor_penalty_basis?: string
  return_basis?: string
  equity: EquityPoint[]
  equity_frequency?: string
  trades: Trade[]
  rejections: Array<Record<string, unknown>>
}

export interface BacktestResult {
  id?: string
  symbol: string
  is_demo?: boolean
  config: BacktestConfig
  data_version?: string
  data_version_available?: boolean
  data_version_warning?: string | null
  diagnostics?: DataDiagnostics
  source_events?: DataSourceEvent[]
  equity_frequency?: string
  quality?: Record<string, unknown>
  warnings?: string[]
  strategies: Record<'grid' | 'dca' | 'buy_hold' | 'matched_funding_hold' | string, StrategyResult>
  schema_version?: number
  result_schema_version?: number
  funding?: { reserve_cash?: number; initial_equity: number; initial_principal?: number; initial_cash?: number; funding_mode?: string; topups?: number; external_topups?: number; peak_invested?: number; peak_principal?: number; peak_invested_date?: string | null; final_invested?: number; funding_flows?: Array<{ timestamp: string; amount: number }>; valuation_price: number; reference_price: number; reference_date: string; dca_initial_equity?: number; buy_hold_initial_shares?: number; buy_hold_initial_equity?: number; matched_funding_hold_invested?: number }
  effective_end_date?: string
  accounting_basis?: string
  floor_episode_count?: number
  floor_episode_grids?: number[]
  floor_penalty?: number
  floor_penalty_basis?: string
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
