import type { BacktestConfig, FeeConfig } from './types.ts'

export type EtfScope = 'watchlist' | 'local' | 'selected'
export type EtfPreflightStatus = 'ready' | 'limited' | 'pending_validation' | 'insufficient_history' | 'data_not_ready'
export type EtfDataStatus = EtfPreflightStatus | 'pending' | 'unavailable' | 'completed' | 'failed'

export interface EtfSampling {
  expected_days: number
  valid_days: number
  coverage_ratio: number
  excluded_days: string[]
  statistics_excluded_days: string[]
  statistics_days: number
  comparable: boolean
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
}

export interface EtfPreflightCounts {
  total: number
  ready: number
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
}

export interface EtfMetricDetail {
  status: string
  reason?: string
  sample_count?: number
}

export interface EtfGridBasis {
  reference_price: number | null
  lower_price: number | null
  upper_price: number | null
  current_raw_price: number | null
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
  status: 'not_selected' | 'completed' | 'no_qualified' | 'unavailable' | 'partial' | 'cancelled'
  candidates: EtfCandidate[]
  recommendation: EtfCandidate | null
  forward_config: BacktestConfig | null
  reason?: string
  stop_reason?: string
}

export interface EtfRow {
  symbol: string
  symbol_name: string
  status: string
  data_status: string
  sampling?: EtfSampling
  version: string
  window_start: string
  window_end: string
  metrics: Record<string, number | null>
  metric_details: Record<string, EtfMetricDetail>
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
  method_version: 'etf_min_grid_v1' | 'etf_min_grid_v2' | 'etf_min_grid_v3'
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
  method_version: 'etf_min_grid_v1' | 'etf_min_grid_v2' | 'etf_min_grid_v3'
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
  { key: 'rv_annualized', label: '1 分钟年化已实现波动率', unit: 'percent', meaning: '根据 1 分钟日内收益估算价格波动，并按 252 个交易日换算为年化水平，午休与隔夜不计。', usage: '衡量短周期价格波动，越高通常意味着更多网格触发机会，也伴随更大的波动风险。' },
  { key: 'rv_5m_annualized', label: '5 分钟年化已实现波动率', unit: 'percent', meaning: '根据 5 分钟日内收益估算价格波动，并按 252 个交易日换算为年化水平，午休与隔夜不计。', usage: '作为 1 分钟波动率的采样对照，数值越高表示观测到的日内波动越大。' },
  { key: 'path_redundancy', label: '历史全窗口路径冗余（1 分钟）', unit: 'percent', meaning: '旧版 v1/v2 在全 120/250 日窗口上计算的路径冗余度。', usage: '仅作历史结果展示；不同于 v3 的逐日及滚动窗口均值，不可当作新日均指标。' },
  { key: 'path_redundancy_daily_mean', label: '日均路径冗余（5m 采样）', unit: 'percent', meaning: '逐日计算首个采样 bar 开盘至当日收盘的路径冗余，再对有效日等权平均。', usage: '计入 v3 GSS 10%。先算各日 PR 再等权平均，不是对日 PR 平滑；高 PR 不代表盈利。' },
  { key: 'path_redundancy_5d_mean', label: '5 日均路径冗余（5m 采样）', unit: 'percent', meaning: '每日推进一个窗口，计算连续 5 个交易日 PR 后对有效窗口等权平均；跨缺口窗口整窗剔除，隔夜保留真实位移。', usage: 'v3 GSS 路径分项的一半权重。先算各窗口 PR 再平均，不是日 PR 平滑；高 PR 不代表盈利。' },
  { key: 'path_redundancy_20d_mean', label: '20 日均路径冗余（5m 采样）', unit: 'percent', meaning: '每日推进一个窗口，计算连续 20 个交易日 PR 后对有效窗口等权平均；跨缺口窗口整窗剔除，隔夜保留真实位移。', usage: '描述较长窗口，不计入 GSS；高 PR 不代表盈利。' },
  { key: 'path_redundancy_daily_mean_1m', label: '日均路径冗余（1 分钟对照）', unit: 'percent', meaning: '按逐日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'path_redundancy_5d_mean_1m', label: '5 日均路径冗余（1 分钟对照）', unit: 'percent', meaning: '按滚动 5 日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'path_redundancy_20d_mean_1m', label: '20 日均路径冗余（1 分钟对照）', unit: 'percent', meaning: '按滚动 20 日窗口定义计算的 1 分钟采样对照值。', usage: '用于观察采样敏感性，不计入 GSS；日、5 日、20 日各尺度应分别比较；高 PR 不代表盈利。' },
  { key: 'iovr', label: '日内 / 隔夜方差比', unit: 'ratio', meaning: '日内 1 分钟实现方差均值除以官方开盘相对前收盘隔夜平方收益均值。', usage: '比较波动发生在日内还是隔夜，越高表示相对更多方差来自日内。' },
  { key: 'variance_ratio_60m', label: '5 分钟采样 · 60 分钟方差比', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 60 分钟的方差与基频方差的线性缩放值。', usage: '描述该时间尺度的收益延续性，约为 1 接近随机游走，高于 1 偏持续、低于 1 偏反转。' },
  { key: 'variance_ratio_240m', label: '5 分钟采样 · 240 分钟方差比', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 240 分钟的方差与基频方差的线性缩放值。', usage: '描述较长日内尺度的收益延续性，约为 1 接近随机游走，高于 1 偏持续、低于 1 偏反转。' },
  { key: 'variance_ratio_1200m', label: '5 分钟采样 · 1200 分钟方差比', unit: 'ratio', meaning: '以 5 分钟收益为基频，比较聚合至 1200 分钟的方差与基频方差的线性缩放值。', usage: '描述跨日较长尺度的收益延续性，约为 1 接近随机游走，高于 1 偏持续、低于 1 偏反转。' },
  { key: 'ou_half_life_days', label: '30 分钟采样 · OU 半衰期', unit: 'days', meaning: '用 30 分钟价格拟合均值回归模型，估计价格偏离中枢的幅度缩小一半所需的交易日数。', usage: '用于描述拟合模型中的回归速度，半衰期越短表示回归越快，模型无效时不提供数值。' },
  { key: 'zero_crossing_rate', label: '5 分钟采样 · 滚动中枢过零率', unit: 'number', meaning: '5 分钟价格从过去约 5 日均线的一侧穿到另一侧的次数，占有效相邻价格对的比例。', usage: '越高表示围绕中枢往返更频繁，可辅助判断网格交易机会；显示为 0–1 比率。' },
  { key: 'jump_ratio', label: '5 分钟采样 · 跳跃方差占比', unit: 'percent', meaning: '按日计算 5 分钟实现方差中超出双幂变差估计部分的占比，双幂变差不跨午休、隔夜或缺失段。', usage: '衡量方差中跳跃成分占比，越高表示极端跳动贡献越大。' },
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
    not_selected: '未进入核验', completed: '核验完成', no_qualified: '无合格候选', unavailable: '核验不可用', partial: '部分完成', cancelled: '核验已取消',
  } as Record<string, string>)[status] ?? status
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

const NON_STATUS_COUNT_KEYS = new Set(['total', 'scored', 'eligible', 'validated', 'recommended'])
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

export function etfResearchStartCount(counts: Pick<EtfPreflightCounts, 'ready' | 'limited'>): number {
  return (counts.ready ?? 0) + (counts.limited ?? 0)
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
