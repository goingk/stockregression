import type { BacktestConfig, FeeConfig } from './types.ts'

export type EtfScope = 'watchlist' | 'local' | 'selected'
export type EtfPreflightStatus = 'ready' | 'limited' | 'pending_validation' | 'insufficient_history' | 'data_not_ready' | string
export type EtfDataStatus = EtfPreflightStatus | 'pending' | 'unavailable' | 'completed' | 'failed'

export type EtfResolutionMode = 'minute' | 'daily' | 'skip' | 'suspended' | 'no_trades'
export interface EtfDataResolution {
  policy: string
  plan_hash: string
  days: Array<{ date: string; mode: EtfResolutionMode; reason_code?: string | null; reason?: string | null }>
  counts: Record<EtfResolutionMode, number>
  first_evaluable_date: string | null
  last_evaluable_date: string | null
}

export interface EtfSampling {
  expected_days: number
  valid_days: number
  coverage_ratio: number
  excluded_days: string[]
  statistics_excluded_days: string[]
  statistics_days: number
  comparable: boolean
  minute_complete_days?: number
  gss_days?: number
}

export interface EtfExecution {
  initial_shares: number
  min_shares: number
  max_shares: number | null
  buy_shares: number
  sell_shares: number
  fees: FeeConfig
}

export interface EtfResearchSettings {
  scope: EtfScope
  symbolsText: string
  as_of: string
  window_days: 120 | 250
  validation_top_k: number
  min_trades: number
  max_principal_drawdown: number
  max_runtime_seconds: number
}

export interface EtfResearchRequest extends Omit<EtfResearchSettings, 'symbolsText' | 'as_of'> {
  symbols: string[]
  as_of: string | null
  execution: EtfExecution
  configuration_revision?: string | null
}

export interface EtfGap {
  kind: string
  message: string
  timeframe?: string
}

export interface EtfPreflightRow {
  symbol: string
  symbol_name: string
  status: EtfPreflightStatus
  window_start: string | null
  window_end: string
  version: string | null
  gaps: EtfGap[]
  warnings: string[]
  sampling?: EtfSampling
  data_resolution?: EtfDataResolution
  replay_allowed?: boolean
}

export interface EtfPreflightCounts {
  total: number
  ready: number
  replay_allowed?: number
  limited?: number
  pending_validation: number
  insufficient_history: number
  data_not_ready: number
}

export interface EtfUniverseSymbolVersion {
  version: string
  market_versions?: Record<string, unknown>
  auxiliary_versions?: Record<string, unknown>
}

export interface EtfUniverse {
  package_id: string
  generation: string
  versions: Record<string, EtfUniverseSymbolVersion>
}

export interface EtfPreflight {
  as_of: string
  requested_as_of: string | null
  window_days: number
  scope: EtfScope
  symbols: string[]
  counts: EtfPreflightCounts
  rows: EtfPreflightRow[]
  universe: EtfUniverse
  data_resolution?: EtfDataResolution
  replay_allowed?: boolean
}

export interface EtfMetricDetail {
  status: string
  reason?: string
  sample_count?: number
  required_samples?: number
  reason_code?: string | null
}

export interface EtfMarketState {
  method_version?: 'daily_state_v1' | string
  as_of?: string | null
  status?: string
  reason?: string | null
  empty_reason?: string | null
  message?: string | null
  sample_count?: number | null
  samples?: {
    daily_rows?: number | null
    selected_start?: string | null
    selected_end?: string | null
    prior_close_date?: string | null
    gap_before_start?: string | null
    invalid_daily_dates?: string[]
    [key: string]: unknown
  } | null
  metrics?: Partial<Record<'natr14' | 'hv_ratio' | 'adx14' | 'plus_di14' | 'minus_di14' | 'amplitude_q75', number | null>> & Record<string, number | null | undefined>
  metric_details?: Record<string, EtfMetricDetail>
  diagnostics?: Record<string, unknown>
  [key: string]: unknown
}

export interface EtfGridBasis {
  reference_price: number | null
  lower_price: number | null
  upper_price: number | null
  current_raw_price: number | null
  current_price_date?: string | null
  current_price_source?: 'minute' | 'daily' | null
  current_price_basis?: string | null
  current_raw_price_unadjusted?: number | null
  atr30_pct: number | null
  svr: number | null
  realized_skewness: number | null
  realized_kurtosis: number | null
  price_tick: number | null
}

export interface EtfCandidateMetrics {
  peak_principal_return: number | null
  matched_funding_hold_return: number | null
  excess_return: number | null
  max_principal_drawdown: number | null
  trade_count: number | null
  buy_count: number | null
  sell_count: number | null
  total_fees: number | null
}

export interface EtfCandidate {
  id: string | number
  base_step: number
  buy_step: number
  sell_step: number
  reference_price: number
  lower_price: number
  upper_price: number
  grids_down_count: number
  grids_up_count: number
  asymmetric: boolean
  metrics: EtfCandidateMetrics
  qualified: boolean
  reasons: string[]
  replay_config: BacktestConfig
}

export interface EtfValidation {
  status: 'not_selected' | 'completed' | 'no_qualified' | 'no_candidates' | 'unavailable' | 'partial' | 'cancelled'
  candidates: EtfCandidate[]
  recommendation: EtfCandidate | null
  forward_config: BacktestConfig | null
  reason?: string
  stop_reason?: string
  reason_code?: string | null
  constraints?: Record<string, unknown> | null
}

export interface EtfRow {
  symbol: string
  symbol_name: string
  status: string
  data_status: string
  replay_allowed?: boolean
  data_resolution?: EtfDataResolution
  sampling?: EtfSampling
  version: string
  window_start: string
  window_end: string
  metrics: Record<string, number | null>
  metric_details: Record<string, EtfMetricDetail>
  /** Added by daily-state-aware ETF results; historical rows legitimately omit it. */
  market_state?: EtfMarketState | null
  gaps: EtfGap[]
  eligible: boolean
  filter_reasons: string[]
  warnings: string[]
  diagnostics: Record<string, unknown>
  grid_basis: EtfGridBasis
  gss_score: number | null
  rank: number | null
  score_cohort_size: number
  score_contributions: Record<string, number>
  validation: EtfValidation
}

export interface EtfResearchLinkSummary {
  fixed?: Record<string, unknown> | null
  adaptive?: Record<string, unknown> | null
  current?: Record<string, unknown> | null
  [key: string]: unknown
}

export interface EtfResearchLink {
  job_id: string
  result_id: string | null
  status: string
  created_at: string
  updated_at: string
  error?: unknown
  summary?: EtfResearchLinkSummary | null
}

export interface EtfResearchLinksResponse {
  items: EtfResearchLink[]
  total: number
  offset: number
  limit: number
}

export interface EtfResearchResultCounts {
  total: number
  scored: number
  eligible: number
  validated: number
  recommended: number
  ready: number
  limited?: number
  pending_validation: number
  insufficient_history: number
  data_not_ready: number
  [dataStatus: string]: number | undefined
}

export interface EtfResearchResult {
  id: string
  kind: 'etf_research'
  method_version: 'etf_min_grid_v1' | 'etf_min_grid_v2' | 'etf_min_grid_v3' | 'etf_min_grid_v4' | 'etf_min_grid_v5'
  result_schema_version: 1
  status: 'completed' | 'partial' | 'cancelled'
  as_of: string
  request: EtfResearchRequest
  universe: EtfUniverse
  counts: EtfResearchResultCounts
  ranking_complete: boolean
  rows_count: number
  stop_reason?: string
  warnings: string[]
}

export interface EtfValidationRow {
  symbol: string
  symbol_name: string
  status: string
  reason?: string
  version: string | null
  window_start: string | null
  window_end: string | null
  gaps: EtfGap[]
  warnings: string[]
  reports: unknown[]
}

export interface EtfValidationResultCounts {
  total: number
  [status: string]: number
}

export interface EtfValidationResult {
  id: string
  kind: 'etf_validation'
  method_version: 'etf_min_grid_v1' | 'etf_min_grid_v2' | 'etf_min_grid_v3' | 'etf_min_grid_v4' | 'etf_min_grid_v5'
  result_schema_version: 1
  status: 'completed' | 'partial' | 'cancelled' | 'failed'
  as_of: string
  request: EtfResearchRequest
  universe: EtfUniverse
  counts: EtfValidationResultCounts
  ranking_complete: boolean
  rows_count: number
  stop_reason?: string
  warnings: string[]
  message?: string
}

export type EtfResult = EtfResearchResult | EtfValidationResult

export interface EtfItemsResponse<Item = EtfRow> {
  items: Item[]
  total: number
  offset: number
  limit: number
}

export function isEtfValidationResult(value: Pick<EtfResult, 'kind'> | null | undefined): value is EtfValidationResult {
  return value?.kind === 'etf_validation'
}

export function isEtfResearchResult(value: Pick<EtfResult, 'kind'> | null | undefined): value is EtfResearchResult {
  return value?.kind === 'etf_research'
}

export interface EtfMetricDefinition {
  key: string
  label: string
  unit: 'percent' | 'ratio' | 'days' | 'price' | 'number'
  meaning: string
  usage: string
}

export const ETF_METRIC_DEFINITIONS: readonly EtfMetricDefinition[] = [
  { key: 'rv_annualized', label: '年化价格波动率（RV）', unit: 'percent', meaning: '根据完整分钟日内收益估算价格波动，并按 252 个交易日换算为年化水平，午休与隔夜不计。它不是年化收益率。', usage: '描述价格波动幅度；越高通常意味着更多网格触发机会，也伴随更大的波动风险。' },
  { key: 'rv_5m_annualized', label: '5 分钟年化已实现波动率', unit: 'percent', meaning: '根据 5 分钟日内收益估算价格波动，并按 252 个交易日换算为年化水平，午休与隔夜不计。', usage: '作为 1 分钟波动率的采样对照，数值越高表示观测到的日内波动越大。' },
  { key: 'path_redundancy', label: '历史全窗口价格往返程度（PR，1m）', unit: 'percent', meaning: '旧版 v1/v2 在全 120/250 日窗口上计算的价格往返程度。', usage: '仅作历史结果展示；不同于 v3 的逐日及滚动窗口均值，不可当作新日均指标。' },
  { key: 'path_redundancy_daily_mean', label: '日均价格往返程度（PR，5m）', unit: 'percent', meaning: '逐日计算首个采样 bar 开盘至当日收盘的路径冗余，再对有效日等权平均。', usage: '计入 v3 GSS 10%。先算各日 PR 再等权平均，不是对日 PR 平滑；高 PR 不代表盈利。' },
  { key: 'path_redundancy_5d_mean', label: '5 日价格往返程度（PR，5m）', unit: 'percent', meaning: '每日推进一个窗口，计算连续 5 个交易日 PR 后对有效窗口等权平均；跨缺口窗口整窗剔除，隔夜保留真实位移。', usage: 'v3 GSS 路径分项的一半权重。先算各窗口 PR 再平均，不是日 PR 平滑；高 PR 不代表盈利。' },
  { key: 'path_redundancy_20d_mean', label: '20 日价格往返程度（PR，5m）', unit: 'percent', meaning: '每日推进一个窗口，计算连续 20 个交易日 PR 后对有效窗口等权平均；跨缺口窗口整窗剔除，隔夜保留真实位移。', usage: '描述较长窗口，不计入 GSS；高 PR 不代表盈利。' },
  { key: 'path_redundancy_daily_mean_1m', label: '日均价格往返程度（PR，1m 对照）', unit: 'percent', meaning: '按逐日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'path_redundancy_5d_mean_1m', label: '5 日价格往返程度（PR，1m 对照）', unit: 'percent', meaning: '按滚动 5 日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'path_redundancy_20d_mean_1m', label: '20 日价格往返程度（PR，1m 对照）', unit: 'percent', meaning: '按滚动 20 日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'iovr', label: '日内／隔夜方差比（IOVR）', unit: 'ratio', meaning: '日内完整分钟实现方差均值除以官方开盘相对前收盘隔夜平方收益均值。', usage: '比较波动发生在日内还是隔夜；12 表示日内方差约为隔夜的 12 倍。' },
  { key: 'variance_ratio_60m', label: '5 分钟采样 · 60 分钟方差比', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 60 分钟的方差与基频方差的线性缩放值。', usage: '描述该时间尺度的收益延续性，约为 1 接近随机游走，高于 1 偏持续、低于 1 偏反转。' },
  { key: 'variance_ratio_240m', label: '趋势／反转特征（VR 240m）', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 240 分钟的方差与基频方差的线性缩放值。', usage: '约为 1 接近随机游走；大于 1 偏趋势延续，小于 1 偏反转。' },
  { key: 'variance_ratio_1200m', label: '5 分钟采样 · 1200 分钟方差比', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 1200 分钟的方差与基频方差的线性缩放值。', usage: '描述跨日较长尺度的收益延续性，约为 1 接近随机游走，高于 1 偏持续、低于 1 偏反转。' },
  { key: 'ou_half_life_days', label: '30 分钟采样 · OU 半衰期', unit: 'days', meaning: '用 30 分钟价格拟合均值回归模型，估计价格偏离中枢的幅度缩小一半所需的交易日数。', usage: '用于描述拟合模型中的回归速度，半衰期越短表示回归越快，模型无效时不提供数值。' },
  { key: 'zero_crossing_rate', label: '5 分钟采样 · 滚动中枢过零率', unit: 'number', meaning: '5 分钟价格从过去约 5 日均线的一侧穿到另一侧的次数，占有效相邻价格对的比例。', usage: '越高表示围绕中枢往返更频繁，可辅助判断网格交易机会；显示为 0–1 比率。' },
  { key: 'jump_ratio', label: '突发跳动方差占比（跳跃）', unit: 'percent', meaning: '按日计算完整 5 分钟数据实现方差中超出双幂变差估计部分的占比，不跨午休、隔夜或缺失段。', usage: '衡量突发价格跳动对波动的贡献；越高表示极端跳动贡献越大。' },
  { key: 'max_drawdown_duration_days', label: '最长水下持续期', unit: 'days', meaning: '取独立最近 250 日日线窗口内，从峰值到恢复此前高点的最长交易日数，期末未恢复时计入未完成区间。', usage: '衡量回撤后恢复所需时间，数值越大表示历史水下持续更久。' },
  { key: 'tick_to_price_ratio', label: '最小报价 / 均价', unit: 'percent', meaning: '最小价格单位除以最近 20 个有分钟数据交易日的未复权分钟收盘均价。', usage: '衡量最小报价单位相对价格的粗细，越高表示报价刻度相对更粗。' },
  { key: 'mean_raw_price', label: '平均不复权价格', unit: 'price', meaning: '最近 20 个有分钟数据交易日的未复权分钟收盘均价。', usage: '用于描述近期原始成交价格水平并参与适用性筛选，低于 0.5 元会被过滤。' },
]

const defaultFees: Required<FeeConfig> = {
  commission_rate: .0005,
  min_commission: .1,
  slippage_bps: 5,
  share_face_value: 1,
  legacy_sh_transfer_rate: .001,
  legacy_sh_min_transfer: 1,
  legacy_sz_transfer_rate: .0000255,
}

export function normalizeFeeConfig(value?: Partial<FeeConfig> | null): FeeConfig {
  return {
    commission_rate: finite(value?.commission_rate) ? value!.commission_rate! : defaultFees.commission_rate,
    min_commission: finite(value?.min_commission) ? value!.min_commission! : defaultFees.min_commission,
    slippage_bps: finite(value?.slippage_bps) ? value!.slippage_bps! : defaultFees.slippage_bps,
    share_face_value: finite(value?.share_face_value) ? value!.share_face_value! : defaultFees.share_face_value,
    legacy_sh_transfer_rate: finite(value?.legacy_sh_transfer_rate) ? value!.legacy_sh_transfer_rate! : defaultFees.legacy_sh_transfer_rate,
    legacy_sh_min_transfer: finite(value?.legacy_sh_min_transfer) ? value!.legacy_sh_min_transfer! : defaultFees.legacy_sh_min_transfer,
    legacy_sz_transfer_rate: finite(value?.legacy_sz_transfer_rate) ? value!.legacy_sz_transfer_rate! : defaultFees.legacy_sz_transfer_rate,
  }
}

export function executionFromBacktest(config: BacktestConfig): EtfExecution {
  return {
    initial_shares: config.initial_shares,
    min_shares: config.min_shares,
    max_shares: config.max_shares,
    buy_shares: config.buy_shares,
    sell_shares: config.sell_shares,
    fees: normalizeFeeConfig(config.fees),
  }
}

export function executionError(value: EtfExecution | null | undefined): string {
  if (!value) return '当前回测草稿尚未载入。'
  for (const [key, label] of [
    ['initial_shares', '初始持股'], ['min_shares', '最低持股'], ['buy_shares', '每次买入'], ['sell_shares', '每次卖出'],
  ] as const) {
    const quantity = value[key]
    if (!Number.isSafeInteger(quantity) || quantity < 0 || quantity % 100 !== 0) return `${label}应为 100 的整数倍。`
  }
  if (value.buy_shares < 100 || value.sell_shares < 100) return '每次买入和每次卖出至少为 100 份。'
  if (value.min_shares > value.initial_shares) return '最低持股不能高于初始持股。'
  if (value.max_shares != null && (!Number.isSafeInteger(value.max_shares) || value.max_shares < 0 || value.max_shares % 100 !== 0)) return '最高持股留空表示不限，填写时须为 100 的整数倍。'
  if (value.max_shares != null && value.initial_shares > value.max_shares) return '初始持股不能高于最高持股。'
  const fees = value.fees
  if (!finite(fees.commission_rate) || fees.commission_rate < 0 || fees.commission_rate > .05) return '佣金费率应在 0%–5% 之间。'
  if (!finite(fees.min_commission) || fees.min_commission < 0) return '最低佣金不能为负。'
  if (!finite(fees.slippage_bps) || fees.slippage_bps < 0 || fees.slippage_bps > 1000) return '滑点应在 0–1000 基点之间。'
  if (!finite(fees.share_face_value) || fees.share_face_value <= 0) return '股票面值必须为正数。'
  if (!finite(fees.legacy_sh_transfer_rate) || fees.legacy_sh_transfer_rate < 0 || fees.legacy_sh_transfer_rate > .05) return '沪市历史过户费率应在 0%–5% 之间。'
  if (!finite(fees.legacy_sh_min_transfer) || fees.legacy_sh_min_transfer < 0) return '沪市历史最低过户费不能为负。'
  if (!finite(fees.legacy_sz_transfer_rate) || fees.legacy_sz_transfer_rate < 0 || fees.legacy_sz_transfer_rate > .05) return '深市历史过户费率应在 0%–5% 之间。'
  return ''
}

export function parseSelectedEtfSymbols(text: string): { symbols: string[]; invalid: string[] } {
  const invalid: string[] = []
  const symbols: string[] = []
  for (const raw of text.split(/[\s,，;；]+/).map(part => part.trim()).filter(Boolean)) {
    const upper = raw.toUpperCase()
    let symbol = ''
    if (/^[15]\d{5}$/.test(upper)) symbol = upper + (upper.startsWith('5') ? '.SH' : '.SZ')
    else if (/^[15]\d{5}\.(SH|SZ)$/.test(upper)) {
      const validExchange = upper.startsWith('5') ? upper.endsWith('.SH') : upper.endsWith('.SZ')
      if (validExchange) symbol = upper
    }
    if (!symbol) duplicateSafePush(invalid, raw)
    else duplicateSafePush(symbols, symbol)
  }
  return { symbols, invalid }
}

export function settingsError(settings: EtfResearchSettings, execution: EtfExecution | null | undefined): string {
  const parsed = parseSelectedEtfSymbols(settings.symbolsText)
  if (settings.scope === 'selected' && !parsed.symbols.length) return '自定义范围至少需要一只 ETF。'
  if (settings.scope === 'selected' && parsed.invalid.length) return '无法识别 ETF 代码：' + parsed.invalid.join('、')
  if (settings.as_of && !/^\d{4}-\d{2}-\d{2}$/.test(settings.as_of)) return '截止日应为 YYYY-MM-DD。'
  if (settings.window_days !== 120 && settings.window_days !== 250) return '统计窗口只能为 120 或 250 个交易日。'
  if (!Number.isSafeInteger(settings.validation_top_k) || settings.validation_top_k < 0 || settings.validation_top_k > 100) return '自动核验标的数应在 0–100 之间。'
  if (!Number.isSafeInteger(settings.min_trades) || settings.min_trades < 0 || settings.min_trades > 1_000_000) return '最少成交笔数应在 0–1000000 之间。'
  if (!finite(settings.max_principal_drawdown) || settings.max_principal_drawdown <= 0 || settings.max_principal_drawdown > 1) return '最大本金回撤应大于 0% 且不超过 100%。'
  if (!Number.isSafeInteger(settings.max_runtime_seconds) || settings.max_runtime_seconds < 1 || settings.max_runtime_seconds > 86400) return '最长运行时间应在 1–86400 秒之间。'
  return executionError(execution)
}

export function buildEtfResearchRequest(
  settings: EtfResearchSettings,
  execution: EtfExecution,
  configurationRevision?: string | null,
): EtfResearchRequest {
  const parsed = parseSelectedEtfSymbols(settings.symbolsText)
  return {
    scope: settings.scope,
    symbols: settings.scope === 'selected' ? parsed.symbols : [],
    as_of: settings.as_of.trim() || null,
    window_days: settings.window_days,
    validation_top_k: settings.validation_top_k,
    execution: {
      initial_shares: execution.initial_shares,
      min_shares: execution.min_shares,
      max_shares: execution.max_shares,
      buy_shares: execution.buy_shares,
      sell_shares: execution.sell_shares,
      fees: { ...execution.fees },
    },
    min_trades: settings.min_trades,
    max_principal_drawdown: settings.max_principal_drawdown,
    max_runtime_seconds: settings.max_runtime_seconds,
    ...(configurationRevision ? { configuration_revision: configurationRevision } : {}),
  }
}

export function etfRequestKey(settings: EtfResearchSettings, execution: EtfExecution | null | undefined): string {
  const request = execution ? buildEtfResearchRequest(settings, execution) : settings
  return JSON.stringify(request)
}

export function etfMetricRows(metrics: Record<string, number | null>, details: Record<string, EtfMetricDetail>) {
  const known = new Set(ETF_METRIC_DEFINITIONS.map(item => item.key))
  const definitions = [
    ...ETF_METRIC_DEFINITIONS,
    ...Object.keys(metrics).filter(key => !known.has(key)).sort().map(key => ({ key, label: key.replaceAll('_', ' '), unit: 'number' as const, meaning: '', usage: '' })),
  ]
  return definitions.filter(definition => Object.prototype.hasOwnProperty.call(metrics, definition.key)).map(definition => ({
    ...definition,
    value: etfMetricText(definition.key, metrics[definition.key]),
    detail: details[definition.key],
  }))
}

export function etfMetricText(key: string, value: number | null | undefined): string {
  if (!finite(value)) return '—'
  const definition = ETF_METRIC_DEFINITIONS.find(item => item.key === key)
  if (definition?.unit === 'percent') return `${(value * 100).toFixed(2)}%`
  if (definition?.unit === 'days') return `${value.toFixed(1)} 日`
  if (definition?.unit === 'price') return value.toFixed(3) + ' 元'
  if (definition?.unit === 'ratio') return value.toFixed(3)
  return Number(value.toPrecision(5)).toString()
}

export const ETF_MARKET_STATE_METRICS = [
  { key: 'natr14', label: 'NATR14', unit: 'percent' },
  { key: 'hv_ratio', label: 'HV 比率', unit: 'ratio' },
  { key: 'adx14', label: 'ADX14', unit: 'number' },
  { key: 'plus_di14', label: '+DI14', unit: 'number' },
  { key: 'minus_di14', label: '-DI14', unit: 'number' },
  { key: 'amplitude_q75', label: '振幅 Q75', unit: 'percent' },
] as const

export function etfMarketStateMetricText(key: string, value: number | null | undefined): string {
  if (!finite(value)) return '—'
  if (key === 'natr14' || key === 'amplitude_q75') return Math.abs(value) <= 1.5
    ? `${(value * 100).toFixed(2)}%`
    : `${value.toFixed(2)}%`
  if (key === 'hv_ratio') return value.toFixed(3)
  return value.toFixed(2)
}

export function marketStateStatusLabel(status?: string | null): string {
  if (!status) return '未提供状态'
  return ({
    ready: '状态可用', available: '状态可用', completed: '状态可用',
    insufficient_history: '历史不足', insufficient: '样本不足', empty: '无有效样本',
    unavailable: '状态不可用', failed: '状态计算失败', pending: '待计算',
  } as Record<string, string>)[status] ?? status
}

export function marketStateStatusTone(status?: string | null): 'neutral' | 'blue' | 'green' | 'red' | 'amber' {
  if (status === 'ready' || status === 'available' || status === 'completed') return 'green'
  if (status === 'pending') return 'blue'
  if (status === 'insufficient_history' || status === 'insufficient' || status === 'empty') return 'amber'
  if (status === 'unavailable' || status === 'failed') return 'red'
  return 'neutral'
}

export function researchLinkActive(link: Pick<EtfResearchLink, 'status'>): boolean {
  return ['queued', 'running'].includes(link.status)
}

export function researchLinkStatusLabel(status: string): string {
  return ({
    queued: '排队中', running: '运行中', completed: '已完成', partial: '部分完成',
    failed: '失败', cancelled: '已取消',
  } as Record<string, string>)[status] ?? status
}

export function researchLinkStatusTone(status: string): 'neutral' | 'blue' | 'green' | 'red' | 'amber' {
  if (status === 'completed') return 'green'
  if (status === 'running' || status === 'queued') return 'blue'
  if (status === 'partial') return 'amber'
  if (status === 'failed') return 'red'
  return 'neutral'
}

/** Use the immutable result when available, otherwise the task's research resource. */
export function researchLinkResourceId(link: Pick<EtfResearchLink, 'job_id' | 'result_id'>): string {
  return link.result_id || link.job_id
}

function linkBranchText(value: unknown): string {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  const item = value as Record<string, unknown>
  const parts: string[] = []
  if (typeof item.status === 'string' && item.status) {
    const status = ['queued', 'running'].includes(item.status)
      ? researchLinkStatusLabel(item.status)
      : ['no_qualified', 'not_selected', 'unavailable'].includes(item.status)
      ? validationStatusLabel(item.status)
      : resultStatusLabel(item.status)
    parts.push(status)
  }
  if (typeof item.prior_policy === 'string' && item.prior_policy) parts.push('先验 ' + item.prior_policy)
  if (finite(item.score)) parts.push(`评分 ${item.score.toFixed(2)}`)
  if (typeof item.eligible === 'boolean') parts.push(item.eligible ? '合格' : '未合格')
  for (const key of ['selected_count', 'selection_count', 'eligible_count', 'candidate_count']) {
    if (finite(item[key])) {
      const label = ({ selected_count: '入选', selection_count: '入选', eligible_count: '合格', candidate_count: '候选' } as Record<string, string>)[key]
      parts.push(`${label} ${item[key]}`); break
    }
  }
  return parts.join(' · ')
}

export function researchLinkSummaryText(summary?: EtfResearchLinkSummary | null): string {
  if (!summary || typeof summary !== 'object') return '尚无摘要'
  const labels: Record<string, string> = { fixed: '固定', adaptive: '自适应', current: '当前建议' }
  const parts = Object.entries(labels).map(([key, label]) => {
    const text = linkBranchText(summary[key])
    return text ? `${label} ${text}` : ''
  }).filter(Boolean)
  return parts.length ? parts.join(' · ') : '尚无先验摘要'
}

export function dataStatusLabel(status: string): string {
  return ({
    ready: '数据就绪', limited: '统计可用（受限）', pending_validation: '待校验', insufficient_history: '历史不足',
    data_not_ready: '数据未就绪', pending: '待计算', unavailable: '统计不可用',
    completed: '统计完成', failed: '失败',
  } as Record<string, string>)[status] ?? status
}

export function dataStatusTone(status: string): 'neutral' | 'blue' | 'green' | 'red' | 'amber' {
  if (status === 'ready' || status === 'completed') return 'green'
  if (status === 'limited' || status === 'pending_validation' || status === 'pending') return 'amber'
  if (['insufficient_history', 'data_not_ready', 'unavailable', 'failed'].includes(status)) return 'red'
  return 'neutral'
}

export function filterReasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    'ou_invalid': 'OU 半衰期无效或不可估计',
    'variance_ratio_240m_unavailable': '240 分钟方差比不可用',
    'variance_ratio_240m_gt_1.10': '240 分钟方差比高于 1.10',
    'mean_raw_price_unavailable': '平均不复权价格不可用',
    'mean_raw_price_lt_0.5': '平均不复权价格低于 0.5 元',
    'status_insufficient_history': '历史窗口不足',
    'status_data_not_ready': '分钟或日线数据未就绪',
    'status_unavailable': '统计计算不可用',
    'status_pending': '尚未完成统计',
    'status_pending_validation': '数据待校验',
    'status_completed': '统计完成',
  }
  if (labels[reason]) return labels[reason]
  if (reason.startsWith('status_')) return '数据状态：' + dataStatusLabel(reason.slice('status_'.length))
  return reason
}

export function filterReasonsText(reasons: string[]): string {
  return reasons.length ? reasons.map(filterReasonLabel).join('；') : '—'
}

export function gapText(gap: EtfGap): string {
  const context = [gap.timeframe, gap.kind].filter(Boolean).join(' · ')
  return [context, gap.message].filter(Boolean).join('：') || '未提供缺口说明'
}

export function etfSymbolValidationPath(resultId: string, symbol: string): string {
  return `/etf-research/${encodeURIComponent(resultId)}/validate/${encodeURIComponent(symbol)}`
}

export function validationStatusLabel(status: string): string {
  return ({
    not_selected: '尚未试算', completed: '找到历史达标参数', no_qualified: '已试算但未达标', no_candidates: '没有生成有效网格', unavailable: '暂时无法试算', partial: '部分完成', cancelled: '试算已取消',
  } as Record<string, string>)[status] ?? status
}

export function validationSummary(validation?: Pick<EtfValidation, 'status' | 'reason' | 'reason_code' | 'constraints' | 'recommendation'> | null): string {
  if (!validation) return '尚未试算'
  if (validation.status === 'completed' && validation.recommendation) {
    const metrics = validation.recommendation.metrics
    const pct = (value: number | null) => finite(value) ? `${(value * 100).toFixed(2)}%` : '—'
    return `历史本金收益 ${pct(metrics.peak_principal_return)} · 最大回撤 ${pct(metrics.max_principal_drawdown)} · 成交 ${metrics.trade_count ?? '—'} 笔`
  }
  if (validation.status === 'no_candidates') return candidateReasonText(validation.reason_code, validation.constraints) || validation.reason || '没有生成有效网格'
  if (validation.status === 'no_qualified') return validation.reason || '已试算候选，但没有达到成交、收益、超额收益和回撤条件。'
  if (validation.status === 'unavailable') return validation.reason || '相关样本不足，暂时无法完成可靠试算。'
  if (validation.status === 'not_selected') return '尚未自动试算；查看详情了解筛选条件和下一步。'
  return validation.reason || validationStatusLabel(validation.status)
}

export function candidateReasonText(reasonCode?: string | null, constraints?: Record<string, unknown> | null): string {
  const current = constraints?.current_price
  const lower = constraints?.lower_price
  if (reasonCode === 'current_at_or_below_lower' && typeof current === 'number' && typeof lower === 'number') {
    const date = typeof constraints?.current_price_date === 'string' ? constraints.current_price_date : null
    const label = date ? `${date} 最近可信收盘价` : '现价'
    return `${label} ${current.toFixed(4)} 元低于建议区间下界约 ${lower.toFixed(4)} 元，因此未生成网格。`
  }
  return ({
    current_at_or_below_lower: '现价不高于建议区间下界，因此未生成网格。',
    current_at_or_above_upper: '现价不低于建议区间上界，因此未生成网格。',
    reference_outside_band: '参考价超出建议区间，因此未生成网格。',
    step_inputs_invalid: '网格步长所需的样本或波幅数据无效。',
    minimum_step_exceeds_maximum: '所需最小网格步长超过允许上限。',
    tick_grid_invalid: '按最小报价单位对齐后无法形成有效网格。',
    insufficient_samples: '相关分钟样本不足，无法可靠生成网格。',
    grid_basis_missing: '现有样本尚未形成可靠的价格区间，暂未生成网格。',
  } as Record<string, string>)[reasonCode ?? ''] ?? ''
}

export function dataResolutionSummary(resolution?: EtfDataResolution | null): string {
  if (!resolution) return '逐日数据使用情况未提供（历史结果）'
  const counts = resolution.counts
  return `分钟 ${counts.minute ?? 0} 日 · 日线降级 ${counts.daily ?? 0} 日 · 跳过 ${counts.skip ?? 0} 日`
}

export function minuteAndGssSampleText(sampling?: EtfSampling): string {
  if (!sampling) return '完整分钟日 — · GSS 样本 —'
  const own = sampling.minute_complete_days ?? sampling.valid_days
  if (sampling.comparable === false) return `完整分钟日 ${own} 日 · 未参与横向评分`
  const gss = sampling.gss_days ?? sampling.statistics_days
  return `完整分钟日 ${own} 日 · GSS 样本 ${gss} 日`
}

export function resultStatusTone(status: string): 'neutral' | 'blue' | 'green' | 'red' | 'amber' {
  if (status === 'completed') return 'green'
  if (status === 'partial') return 'amber'
  if (status === 'cancelled') return 'neutral'
  return 'red'
}

export function resultStatusLabel(status: string): string {
  return ({ completed: '已完成', partial: '部分完成', cancelled: '已取消', failed: '失败' } as Record<string, string>)[status] ?? status
}

export function stopReasonLabel(reason: string): string {
  return ({
    runtime_limit: '达到最长运行时间', cancelled: '任务已取消',
    data_generation_changed: '数据版本在运行期间发生变化',
    workspace_limit: '工作内存达到预算',
  } as Record<string, string>)[reason] ?? reason
}

const NON_STATUS_COUNT_KEYS = new Set(['total', 'scored', 'eligible', 'validated', 'recommended', 'no_candidates'])
const DATA_STATUS_ORDER = [
  'ready', 'limited', 'pending_validation', 'insufficient_history', 'data_not_ready',
  'pending', 'unavailable', 'completed', 'failed',
]

export function etfDataStatusCounts(counts: Record<string, number | undefined>) {
  const order = new Map(DATA_STATUS_ORDER.map((status, index) => [status, index]))
  return Object.entries(counts)
    .filter(([key, count]) => !NON_STATUS_COUNT_KEYS.has(key) && finite(count))
    .sort(([left], [right]) => {
      const leftOrder = order.get(left) ?? Number.MAX_SAFE_INTEGER
      const rightOrder = order.get(right) ?? Number.MAX_SAFE_INTEGER
      return leftOrder - rightOrder || left.localeCompare(right)
    })
    .map(([status, count]) => ({
      status, count, label: dataStatusLabel(status), tone: dataStatusTone(status),
    }))
}

export function etfResearchStartCount(counts: Pick<EtfPreflightCounts, 'ready' | 'limited'> & { replay_allowed?: number }): number {
  return counts.replay_allowed ?? ((counts.ready ?? 0) + (counts.limited ?? 0))
}

export function samplingCoverageText(sampling?: EtfSampling): string {
  if (!sampling || !finite(sampling.coverage_ratio)) return '覆盖率 —'
  return `完整覆盖率 ${(sampling.coverage_ratio * 100).toFixed(1)}%`
}

export function samplingStatisticsText(sampling?: EtfSampling): string {
  return sampling && Number.isFinite(sampling.statistics_days) ? `统计样本 ${sampling.statistics_days} 日` : '统计样本 —'
}

export function itemsPageCount(total: number, limit: number): number {
  return Math.max(1, Math.ceil(Math.max(0, total) / Math.max(1, limit)))
}

export function safeItemsOffset(offset: number, total: number, limit: number): number {
  const size = Math.max(1, limit)
  const maxOffset = Math.max(0, (itemsPageCount(total, size) - 1) * size)
  const normalized = Math.max(0, Math.floor(offset / size) * size)
  return Math.min(normalized, maxOffset)
}

export function scoreDisplay(row: Pick<EtfRow, 'gss_score' | 'score_cohort_size'>): string {
  if (finite(row.gss_score)) return row.gss_score.toFixed(2)
  if (row.score_cohort_size === 1) return '单标的无横截面分数'
  if (row.score_cohort_size <= 0) return '评分样本不足'
  return '—'
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function duplicateSafePush(values: string[], value: string) {
  if (!values.includes(value)) values.push(value)
}
