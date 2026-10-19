import { useEffect, useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { FlaskConical, RefreshCw, RotateCcw } from 'lucide-react'
import { ApiError, get, post } from '../api'
import ChartView from '../ChartView'
import EquityCurve from '../EquityCurve'
import ResearchLandscape from '../ResearchLandscape'
import DataDiagnosticsPanel from '../DataDiagnosticsPanel'
import Disclosure from '../Disclosure'
import { useTasks } from '../TaskCenter'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, Field, Loading, NumField, PageHeader, SelectInput, configError, configPayload, loadConfig, percent, revealConfigError, useErrorMessage } from '../components'
import { formatPValue, reviewBootstrapEvidence } from '../researchEvidence'
import type { BacktestConfig, BacktestResult, DataDiagnostics, DataPreparationRequest, DataSourceEvent, Job } from '../types'
import { defaultSearchRanges, percentageRanges as pctRange, rangeNames, researchRangesForSymbol, validResearchDefaults } from '../workbenchDefaults'
import type { ResearchDefaults, SearchRange as Range } from '../workbenchDefaults'

type Dict = Record<string, unknown>
type Candidate = Dict & { params?: Dict; metrics?: Record<string, number | null> }
type SearchConfig = { base?: BacktestConfig; mode?: string; protocol_version?: number; n_trials?: number; local_search_budget?: number; seed?: number; ranges?: Dict; [key: string]: unknown }
type TestRun = Partial<BacktestResult> & { strategies?: Record<string, { metrics?: Record<string, number | null>; equity?: Array<{ timestamp: string; equity: number }> }> }
type RecommendationRegion = { region_id?: number; center_params?: Dict; radii?: Dict; normalized_volume?: number; combination_count?: number; completed_count?: number; center_excess_return?: number | null; mean_excess_return?: number | null; min_excess_return?: number | null; complete?: boolean; maximum_search_complete?: boolean; global_exhaustive?: boolean; stress?: Dict }
type Recommendation = { params?: Dict | null; recommended_params?: Dict | null; diagnostic_params?: Dict; as_of?: string; score?: unknown; status?: string; reasons?: unknown; candidates?: Candidate[]; diagnostics?: Dict; recommendation_status?: string; recommendation_regions?: RecommendationRegion[] }
type OptimizationResult = {
  protocol_version?: number; mode?: string; seed?: number; n_trials?: number; data_version?: string; label?: string; warnings?: string[]
  config?: SearchConfig; protocol?: Dict; is_demo?: boolean; quality?: Dict; diagnostics?: DataDiagnostics; source_events?: DataSourceEvent[]; computation?: Dict; locked_funding?: Dict
  fixed?: { status?: string; params?: Dict; as_of?: string; test_result?: TestRun; current?: Recommendation; diagnostics?: Dict; [key: string]: unknown }
  adaptive?: { status?: string; schedule?: Array<Dict & { effective_date?: string; decision_date?: string; params?: Dict }>; test_result?: TestRun; latest_params?: Dict; latest_as_of?: string; [key: string]: unknown }
  evidence?: Dict & { decisions?: Array<Dict>; final_126_day_check?: Dict }
  historical?: { status?: string; best_params?: Dict; candidates?: Candidate[]; best_result?: TestRun & Dict }
  robust?: { status?: string; reason?: string; selected_params?: Dict; candidates?: Candidate[]; split?: Dict; train_metrics?: Record<string, number>; validation_metrics?: Record<string, number>; test_metrics?: Record<string, number>; test_result?: TestRun & Dict }
}
export type Props = {
  base: BacktestConfig
  settings: ResearchDefaults
  onSettingsChange: (settings: ResearchDefaults) => void
  onBaseChange: (config: BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)) => void
  onLoadConfig: (config: BacktestConfig) => void
  requestedJobId?: string | null
  onRequestHandled?: () => void
  onSelectedJobChange?: (id: string) => void
  onFeeDefaultsEdit?: (defaults: { commission_rate: number; min_commission: number }) => void
  onDefaultsEdit?: (group: 'grid' | 'position' | 'dca', config: BacktestConfig) => void
  onRestore?: () => void
  onPrepareData?: (request: DataPreparationRequest) => void
}
const labels: Record<string, string> = { buy_step: '买入步长', sell_step: '卖出步长', grid_step: '网格间距', buy_ratio: '旧版买入比例', sell_ratio: '旧版卖出比例', initial_position: '旧版初始仓位', min_position: '旧版最小仓位', max_position: '旧版最大仓位', buy_shares: '每次买入股数', sell_shares: '每次卖出股数', initial_shares: '初始持股', min_shares: '最低持股', max_shares: '最高持股' }
function revealDisclosure(id: string) {
  const element = document.querySelector<HTMLDetailsElement>(`details[data-disclosure="${id}"]`)
  if (!element) return
  element.open = true
  requestAnimationFrame(() => element.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}
const candidateLabels: Record<string, string> = {
  total_return: '总收益', cagr: '年化收益', max_drawdown: '最大回撤', sharpe: '夏普比率', turnover: '换手', cost_score: '成本得分',
  cost_adjusted_score: '成本后得分', score: '综合得分', raw_score: '原始得分', selection_score: '选择得分', p_value: 'p 值', ci_low: '区间下界', ci_high: '区间上界',
  confidence_interval: '置信区间', bootstrap_reps: '重抽样次数', evaluated_count: '实际评估数', candidate_count: '候选数', eligible_count: '约束合格数',
  neighbor_count: '邻域压力测试数', cost_evaluations: '成本评估数', cost_scores: '成本得分', elapsed_seconds: '计算耗时（秒）', elapsed_time: '计算耗时', backtest_count: '回测次数', evaluation_count: '评估次数',
  decision_count: '决策点数', n_trials_per_decision: '每个决策点基础试验数', lookback_days: '训练回看交易日', fold_count_per_full_window: '每个完整窗口的分窗数', fold_days_target: '每个分窗目标交易日',
  paired_days: '配对交易日数', selected_total_excess_return: '选中策略总超额收益', short_term_insufficient: '末段样本不足', primary: '主检验', sensitivity: '敏感性检验',
  family_count: '完整候选族数量', sample_role: '样本用途', independent_oos: '独立样本外', alpha: '显著性阈值 α', selected_candidate_id: '入选候选编号',
  reserve_cash: '锁定现金', initial_shares: '固定初始持股', initial_cash: '固定初始现金', train_start: '训练开始', train_end: '训练结束',
  evaluation_start: '样本外开始', evaluation_end: '样本外结束', decision_date: '决策日期', effective_date: '生效日期', return: '收益', excess_return: '超额收益',
  status: '状态', reason: '说明', reasons: '说明', eligible: '约束合格', constraint_eligible: '约束合格', neighbor_score: '邻域得分', fold_score: '分窗得分',
  completed_reps: '已完成重采样', requested_reps: '请求重采样', block_length: '平均时间块长度', days: '交易日数',
  family_max_mean_log_excess: '流程族最大日均对数超额收益', selected_mean_log_excess: '该流程日均对数超额收益',
  selected_simultaneous_lower_log_excess: '同时置信下界（日均对数超额）', simultaneous_max_quantile: '多重校正分位值',
  overlapping_prior_studies: '本地重叠区间研究数', exact_repeat_prior_studies: '本地同窗重复研究数',
  interpretation: '检验解释', method: '检验方法', return_basis: '收益口径', selected_index: '参照流程编号',
}
const knownRangeNames = [...rangeNames]
function n(value: unknown, fallback = 0) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback }
function record(value: unknown): Dict | undefined { return value && typeof value === 'object' && !Array.isArray(value) ? value as Dict : undefined }
function list(value: unknown): unknown[] { return Array.isArray(value) ? value : [] }
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
function searchConfigOf(value: unknown): SearchConfig | undefined { return record(value) as SearchConfig | undefined }
function warningsOf(value: unknown): string[] {
  const item = record(value)
  return Array.isArray(item?.warnings) ? item.warnings.map(String) : []
}
function hasDemoFlag(value: unknown) {
  const item = record(value)
  return item?.is_demo === true || record(item?.quality)?.is_demo === true || record(item?.metadata)?.is_demo === true
}
function isOptimizationJob(job: Job) {
  const config = record(job.payload?.config)
  return job.kind === 'optimization' || job.kind === 'optimize' || record(config?.base) != null
}
function protocolVersionOf(...values: unknown[]) {
  for (const value of values) {
    const item = record(value)
    const version = Number(item?.protocol_version ?? record(item?.config)?.protocol_version)
    if (version === 1 || version === 2 || version === 3 || version === 4) return version
  }
  return 1
}
function jobModeLabel(value: unknown, protocolVersion = 1) {
  if (protocolVersion === 4) return '等比网格 v4 · 最大邻域'
  if (protocolVersion === 3) return '统计驱动研究 v3'
  if (protocolVersion === 2) return '科学研究 v2'
  return value === 'historical' ? '旧版历史最优' : value === 'robust' ? '旧版稳健验证' : value === 'both' ? '旧版历史 + 稳健' : '旧版研究'
}
function rangeValue(key: string, value: unknown) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '—'
  return pctRange.has(key) ? percent(parsed, 3) : String(Number(parsed.toPrecision(6)))
}
function getMetrics(candidate: Candidate, outOfSample = false) {
  if (candidate.metrics && typeof candidate.metrics === 'object') return candidate.metrics
  if (outOfSample && candidate.validation_metrics && typeof candidate.validation_metrics === 'object') return candidate.validation_metrics as Record<string, number | null>
  if (candidate.train_metrics && typeof candidate.train_metrics === 'object') return candidate.train_metrics as Record<string, number | null>
  return candidate as Record<string, number | null>
}
function paramsOf(candidate: Candidate) { return record(candidate.params) ?? candidate }
function labelOf(key: string) { return candidateLabels[key] ?? labels[key] ?? key.replaceAll('_', ' ') }
function valueText(key: string, value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'string' || typeof value === 'number') {
    const numeric = typeof value === 'number' ? value : Number.NaN
    if (Number.isFinite(numeric) && /p[_ -]?value|pvalue/i.test(key)) return formatPValue(numeric)
    if (Number.isFinite(numeric) && /return|drawdown|ratio|rate|confidence|turnover/i.test(key) && Math.abs(numeric) <= 1.5) return percent(numeric)
    return String(value)
  }
  if (Array.isArray(value)) {
    if (value.some((item) => item != null && typeof item === 'object')) return `${value.length} 项`
    return value.length > 8 ? `${value.length} 项` : value.map((item) => valueText(key, item)).join('、')
  }
  return JSON.stringify(value)
}
function shortSummary(value: unknown): string {
  if (value == null) return '—'
  if (Array.isArray(value)) return value.length ? `${value.length} 项` : '—'
  const item = record(value)
  if (!item) return valueText('', value)
  const entries = Object.entries(item).filter(([, raw]) => raw == null || ['string', 'number', 'boolean'].includes(typeof raw)).slice(0, 4)
  return entries.length ? entries.map(([key, raw]) => `${labelOf(key)} ${valueText(key, raw)}`).join(' · ') : `${Object.keys(item).length} 项`
}
function metricRows(value: unknown, depth = 0, prefix = ''): Array<{ key: string; value: unknown }> {
  const item = record(value)
  if (!item) return []
  const rows: Array<{ key: string; value: unknown }> = []
  for (const [key, raw] of Object.entries(item)) {
    if (raw == null || ['string', 'number', 'boolean'].includes(typeof raw) || Array.isArray(raw)) {
      rows.push({ key: prefix ? `${prefix}.${key}` : key, value: raw })
    } else if (depth < 1 && record(raw)) rows.push(...metricRows(raw, depth + 1, prefix ? `${prefix}.${key}` : key))
  }
  return rows.slice(0, 18)
}
function displayKey(path: string) {
  return path.split('.').map(labelOf).join(' · ')
}
function reasonList(value: unknown): string[] {
  if (Array.isArray(value)) return value.flatMap((item) => typeof item === 'string' ? [item] : record(item) ? [shortSummary(item)] : [])
  if (typeof value === 'string') return [value]
  return record(value) ? [shortSummary(value)] : []
}
function runOf(value: unknown): TestRun | undefined {
  const item = record(value)
  return item && record(item.strategies) ? item as TestRun : undefined
}
function eligibleOf(primary: unknown, fallback?: unknown): boolean | undefined {
  const item = record(primary)
  const alt = record(fallback)
  const constraints = record(item?.constraints)
  const constraintFlags = constraints ? Object.values(constraints) : []
  const allConstraintFlags = constraintFlags.length > 0 && constraintFlags.every((value) => typeof value === 'boolean')
  const value = item?.constraint_eligible ?? item?.eligible ?? item?.constraints_passed ?? (allConstraintFlags ? constraintFlags.every(Boolean) : undefined) ?? alt?.constraint_eligible ?? alt?.eligible ?? alt?.constraints_passed
  return typeof value === 'boolean' ? value : undefined
}
function statusText(status: unknown) {
  const value = String(status ?? '')
  const names: Record<string, string> = { selected: '相对最佳候选', insufficient_history: '历史不足', insufficient_data: '数据不足', no_eligible_candidate: '没有约束合格候选', insufficient_evidence: '统计证据不足', no_qualified: '暂无合格推荐', completed: '已完成', running: '计算中', unavailable: '不可用' }
  return names[value] ?? (value || '未提供')
}
function firstString(item: Dict | undefined, keys: string[]) {
  for (const key of keys) if (typeof item?.[key] === 'string') return item[key] as string
  return undefined
}

function StrategyComparison({ run }: { run?: TestRun }) {
  const strategies = run?.strategies ?? {}
  const entries = [['grid', '网格策略'], ['same_account_passive', '同账户被动持有'], ['dca', '定投基准'], ['buy_hold', '全仓持有']] as const
  const metrics = [['total_return', '总收益'], ['cagr', '年化收益'], ['max_drawdown', '最大回撤'], ['sharpe', '夏普比率']] as const
  const available = entries.filter(([key]) => strategies[key])
  return <div className='robust-metric-block'><h4>策略对照</h4>{available.length ? <div className='table-wrap'><table><thead><tr><th>指标</th>{available.map(([key, label]) => <th key={key}>{label}</th>)}</tr></thead><tbody>{metrics.map(([key, label]) => <tr key={key}><td>{label}</td>{available.map(([strategyKey]) => { const raw = strategies[strategyKey].metrics?.[key]; const value = raw == null ? NaN : Number(raw); return <td key={strategyKey}>{Number.isFinite(value) ? key === 'sharpe' ? value.toFixed(2) : percent(value) : '—'}</td> })}</tr>)}</tbody></table></div> : <Empty title='策略对照暂不可用' />}</div>
}

function MetricStrip({ title, metrics }: { title: string; metrics?: Record<string, number> }) {
  if (!metrics) return <div className='robust-metric-block'><h4>{title}</h4><Empty title='暂无指标' /></div>
  return <div className='robust-metric-block'><h4>{title}</h4><div className='metric-chip-grid'>{[['总收益', metrics.total_return], ['年化收益', metrics.cagr], ['最大回撤', metrics.max_drawdown], ['夏普', metrics.sharpe]].map(([name, value]) => <div key={String(name)}><small>{String(name)}</small><strong>{value == null ? '—' : String(name).includes('夏普') ? n(value).toFixed(2) : percent(n(value))}</strong></div>)}</div></div>
}

function EvidenceFields({ value }: { value?: unknown }) {
  const rows = metricRows(value)
  return rows.length ? <div className='research-evidence-grid'>{rows.map(({ key, value: raw }) => <div key={key}><small>{displayKey(key)}</small><strong>{valueText(key.split('.').at(-1) ?? key, raw)}</strong></div>)}</div> : <Empty title='暂无可展示的统计字段' />
}

function Assessment({ recommendation, familyEvidence }: { recommendation?: Recommendation; familyEvidence?: Dict }) {
  if (!recommendation) return null
  const diagnostics = record(recommendation.diagnostics)
  const eligible = eligibleOf(diagnostics, recommendation)
  const bootstrapReview = reviewBootstrapEvidence(familyEvidence)
  const statBadge = bootstrapReview.state === 'positive' ? '校正下界为正' : bootstrapReview.state === 'not_positive' ? '尚无校正后正向证据' : '统计证据不足'
  return <div className='research-assessment'>
    <div><small>约束</small><Badge tone={eligible === true ? 'green' : eligible === false ? 'amber' : 'neutral'}>{eligible === true ? '满足' : eligible === false ? '未满足' : '未提供'}</Badge></div>
    <div><small>统计证据</small><Badge tone={bootstrapReview.state === 'positive' ? 'blue' : 'amber'}>{statBadge}</Badge></div>
    {recommendation.score != null && <div><small>选择得分</small><strong>{valueText('score', recommendation.score)}</strong></div>}
    <div className='research-assessment-reasons'><small>说明</small>{reasonList(recommendation.reasons ?? diagnostics?.constraints).length ? <ul>{reasonList(recommendation.reasons ?? diagnostics?.constraints).map((reason, index) => <li key={index}>{reason}</li>)}</ul> : <span>系统会返回当前候选中相对得分最高的参数，并单独标注约束与证据状态。</span>}</div>
  </div>
}

function sensitivityRows(value: unknown): Array<{ block: string; result: Dict }> {
  const root = record(value) ?? {}
  const bootstrap = record(root.bootstrap) ?? root
  const sensitivity = record(bootstrap.sensitivity)
  if (!sensitivity) return []
  return Object.entries(sensitivity)
    .map(([block, result]) => ({ block, result: record(result) ?? {} }))
    .sort((a, b) => n(a.block) - n(b.block))
}

function BootstrapSensitivityTable({ value }: { value: unknown }) {
  const rows = sensitivityRows(value)
  return rows.length ? <div className='table-wrap'><table><thead><tr><th>时间块长度</th><th>家族校正 p 值</th><th>显著性阈值 α</th><th>同时置信下界</th><th>完成重抽样</th></tr></thead><tbody>{rows.map(({ block, result }) => <tr key={block}><td>{block} 个交易日</td><td>{formatPValue(result.p_value)}</td><td>{result.alpha == null ? '—' : percent(n(result.alpha))}</td><td>{result.selected_simultaneous_lower_log_excess == null ? '—' : percent(n(result.selected_simultaneous_lower_log_excess), 4)}</td><td>{result.completed_reps == null ? '—' : `${result.completed_reps} / ${result.requested_reps ?? '—'}`}</td></tr>)}</tbody></table></div> : <Empty title='没有敏感性结果' />
}

function FamilyBootstrapPanel({ value }: { value?: Dict }) {
  if (!value) return <Empty title='当前候选族统计检验不可用' />
  const review = reviewBootstrapEvidence(value)
  const primary = record(record(value.bootstrap)?.primary) ?? {}
  const statusLabel = review.state === 'positive' ? '校正下界为正' : review.state === 'not_positive' ? '尚无校正后正向证据' : `证据不足（${statusText(value.status)}）`
  const tone = review.state === 'positive' ? 'blue' : 'amber'
  return <>
    <div className='research-legacy-banner'>这是用于择参的完整候选族内部验证，样本已经参与当前参数选择，因此不是独立样本外检验。族级 p 值描述整组候选的检验，不是入选参数的胜率或未来盈利概率。</div>
    <div className='result-meta'><Badge tone={tone}>{statusLabel}</Badge><span>{value.as_of ? `参数截止日：${String(value.as_of)}` : ''}</span><span>{value.independent_oos === false ? '非独立 OOS' : value.independent_oos === true ? '独立 OOS' : ''}</span></div>
    <div className='research-evidence-grid'>
      <div><small>候选族规模</small><strong>{value.family_count == null ? '—' : String(value.family_count)}</strong></div>
      <div><small>配对验证交易日</small><strong>{value.paired_days == null ? '—' : String(value.paired_days)}</strong></div>
      <div><small>验证用途</small><strong>用于本次择参</strong></div>
      <div><small>家族校正 p 值</small><strong>{formatPValue(review.pValue)}</strong></div>
      <div><small>显著性阈值 α</small><strong>{percent(review.alpha)}</strong></div>
      <div><small>同时置信下界（日均对数超额）</small><strong>{review.lowerBound == null ? '—' : percent(review.lowerBound, 4)}</strong></div>
    </div>
    <div className='param-block'><strong>主检验与敏感性</strong><BootstrapSensitivityTable value={value} /></div>
    {reasonList(value.reasons).length > 0 && <div className='warning-list'>{reasonList(value.reasons).map((reason, index) => <div key={index}><span className='warning-dot' />{reason}</div>)}</div>}
    {primary.interpretation && <p className='inline-warning'>{String(primary.interpretation)}</p>}
  </>
}

function Final126Evidence({ value }: { value?: Dict }) {
  if (!value) return <Empty title='核验信息暂不可用' detail='区间不足或本次结果没有提供末 126 日核验。' />
  const review = reviewBootstrapEvidence(value)
  const days = finite(value.paired_days) ? value.paired_days : finite(value.days) ? value.days : null
  const required = finite(value.required_days) ? value.required_days : 126
  const periodComplete = value.short_term_insufficient !== true && (days == null || days >= required)
  const verdict = !periodComplete
    ? `外层区间不足：${days ?? 0} / ${required} 个交易日`
    : review.state === 'positive' ? '家族校正后同时置信下界为正'
      : review.state === 'not_positive' ? '未得到家族校正后为正的证据'
        : `检验未完成（${statusText(value.status)}）`
  const tone = periodComplete && review.state === 'positive' ? 'blue' : 'amber'
  const bootstrap = record(value.bootstrap) ?? {}
  const procedures = record(value.procedures) ?? {}
  const family = Array.isArray(value.bootstrap_family) ? value.bootstrap_family.map(String).join('、') : String(value.bootstrap_family ?? '—')
  const procedureEntries = Object.entries(procedures).filter(([, raw]) => record(raw))
  return <>
    <div className='result-meta'><Badge tone={tone}>{verdict}</Badge><span>{String(value.paired_start ?? value.start ?? '—')} 至 {String(value.paired_end ?? value.end ?? '—')}</span><span>{days == null ? '—' : `${days} / ${required} 日`}</span></div>
    <div className='research-evidence-grid'>
      <div><small>核验状态</small><strong>{statusText(value.status)}</strong></div>
      <div><small>配对交易日</small><strong>{days == null ? '—' : String(days)}</strong></div>
      <div><small>比较流程族</small><strong>{family}</strong></div>
      <div><small>bootstrap 选中流程</small><strong>{String(value.bootstrap_selected_procedure ?? '—')}</strong></div>
      <div><small>家族校正 p 值</small><strong>{formatPValue(review.pValue)}</strong></div>
      <div><small>显著性阈值 α</small><strong>{percent(review.alpha)}</strong></div>
      <div><small>同时置信下界</small><strong>{review.lowerBound == null ? '—' : percent(review.lowerBound, 4)}</strong></div>
    </div>
    {procedureEntries.length > 0 && <div className='table-wrap'><table><thead><tr><th>连续样本外流程</th><th>状态</th><th>配对天数</th><th>总超额收益</th><th>日均对数超额</th></tr></thead><tbody>{procedureEntries.map(([name, raw]) => { const item = record(raw) ?? {}; return <tr key={name}><td>{name === 'fixed' ? '固定参数' : name === 'adaptive' ? '季度自适应' : name}</td><td>{statusText(item.status)}</td><td>{item.paired_days == null ? '—' : String(item.paired_days)}</td><td>{item.total_excess_return == null ? '—' : percent(n(item.total_excess_return))}</td><td>{item.mean_log_excess == null ? '—' : percent(n(item.mean_log_excess), 4)}</td></tr> })}</tbody></table></div>}
    <div className='param-block'><strong>10 / 20 / 40 日敏感性</strong><BootstrapSensitivityTable value={bootstrap} /></div>
    <p className='inline-warning'>该检验基于固定与自适应流程真实连续 OOS 轨迹并做家族校正；p 值不是流程胜率，也不是未来盈利概率。{value.source ? ` 数据来源：${String(value.source)}。` : ''}</p>
    {reasonList(value.reasons).length > 0 && <div className='warning-list'>{reasonList(value.reasons).map((reason, index) => <div key={index}><span className='warning-dot' />{reason}</div>)}</div>}
  </>
}

function ParameterBadges({ params, compact = false }: { params?: Dict; compact?: boolean }) {
  const entries = Object.entries(params ?? {})
  if (!entries.length) return <span className='muted'>暂无参数</span>
  return <div className={'research-parameter-badges' + (compact ? ' compact' : '')}>{entries.map(([key, value]) => <span className='research-parameter-badge' key={key}><small>{labelOf(key)}</small><strong>{pctRange.has(key) ? rangeValue(key, value) : record(value) ? shortSummary(value) : valueText(key, value)}</strong></span>)}</div>
}

function ResultCurve({ run }: { run?: TestRun }) {
  if (!run) return <Empty title='暂无样本外结果' detail='当前方案没有可绘制的连续样本外回测。' />
  const curveReady = Array.isArray(run.strategies?.grid?.equity) && run.strategies.grid.equity.length > 0
  return <>
    {curveReady ? <EquityCurve result={run as BacktestResult} height={350} /> : <Empty title='样本外曲线暂不可用' detail='本次结果未包含标准回测权益序列。' />}
    <StrategyComparison run={run} />
  </>
}

function constraintLabel(value: boolean | undefined) {
  return value === true ? '满足约束' : value === false ? '未满足约束' : '未提供约束状态'
}

export default function ResearchPage({ base, settings, onSettingsChange, onBaseChange, onLoadConfig, requestedJobId, onRequestHandled, onSelectedJobChange, onFeeDefaultsEdit, onDefaultsEdit, onRestore, onPrepareData }: Props) {
  const { protocol_version: protocolVersion, selection_target: selectionTarget, rebalance_days: rebalanceDays, bootstrap_reps: bootstrapReps, mode, n_trials: trials, local_search_budget: localSearchBudget, seed, top_candidates: topCandidates, max_drawdown: maxDrawdown, min_trades: minTrades, min_history_days: minHistoryDays, min_minute_ratio: minMinuteRatio, ranking_mode: rankingMode, search_method: searchMethod, training_budget: trainingBudget, validation_budget: validationBudget, recommendation_count: recommendationCount, max_shares_mode: maxSharesMode, slippage_mode: slippageMode, fixed_half_widths: fixedHalfWidths } = settings
  const tasks = useTasks()
  const researchJobs = useMemo(() => tasks.jobs.filter(isOptimizationJob), [tasks.jobs])
  const viewableJobs = useMemo(() => researchJobs.filter(job => Boolean(job.result_id)), [researchJobs])
  const ranges = researchRangesForSymbol(settings.ranges, base.symbol)
  const updateSettings = (patch: Partial<ResearchDefaults>) => onSettingsChange({ ...settings, ...patch })
  const [selectedJobId, setSelectedJobId] = useState(requestedJobId ?? '')
  const [fallbackJob, setFallbackJob] = useState<Job | null>(null)
  const [loadedResult, setLoadedResult] = useState<{ key: string; value: OptimizationResult } | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [searchPreview, setSearchPreview] = useState<Dict | null>(null)
  const [error, setError] = useState('')
  const [dataIssue, setDataIssue] = useState<DataPreparationRequest | null>(null)
  const [curveMode, setCurveMode] = useState<'fixed' | 'adaptive'>('fixed')
  const jobRequests = useRef(new Map<string, Promise<Job>>())
  const resultRequests = useRef(new Map<string, Promise<OptimizationResult>>())
  const err = useErrorMessage()
  useEffect(() => { setSearchPreview(null) }, [base, settings])
  const isStarBoard = base.symbol.startsWith('688') || base.symbol.startsWith('689')
  const selectedJob = researchJobs.find(job => job.id === selectedJobId) ?? (fallbackJob?.id === selectedJobId ? fallbackJob : null)
  const selectedResultKey = selectedJob?.result_id && ['completed', 'cancelled'].includes(selectedJob.status) ? selectedJob.id + ':' + selectedJob.result_id : ''
  const result = loadedResult?.key === selectedResultKey ? loadedResult.value : null

  const selectStudy = (id: string) => {
    setSelectedJobId(id)
    onSelectedJobChange?.(id)
  }
  useEffect(() => {
    if (!requestedJobId) return
    setSelectedJobId(requestedJobId)
    onRequestHandled?.()
  }, [requestedJobId, onRequestHandled])
  useEffect(() => {
    if (!selectedJobId && !requestedJobId && viewableJobs[0]) {
      setSelectedJobId(viewableJobs[0].id)
      onSelectedJobChange?.(viewableJobs[0].id)
    }
  }, [selectedJobId, requestedJobId, viewableJobs, onSelectedJobChange])

  // A result opened from outside this page may predate the task center's current list.
  // Resolve that selection once; the shared task center remains the only job poller.
  useEffect(() => {
    if (!selectedJobId || researchJobs.some(job => job.id === selectedJobId)) return
    let active = true
    let request = jobRequests.current.get(selectedJobId)
    if (!request) {
      request = get<Job>('/jobs/' + encodeURIComponent(selectedJobId))
      jobRequests.current.set(selectedJobId, request)
    }
    void request.then(job => {
      if (!active) return
      if (!isOptimizationJob(job)) { setError('所选任务不是参数研究。'); return }
      setFallbackJob(job)
    }).catch(cause => { if (active) setError(err(cause)) })
    return () => { active = false }
  }, [selectedJobId, researchJobs, err])

  // Cache by the immutable result id so job polling never refetches a completed result.
  useEffect(() => {
    if (!selectedResultKey) { setResultLoading(false); return }
    let active = true
    setResultLoading(true)
    let request = resultRequests.current.get(selectedResultKey)
    if (!request) {
      request = get<OptimizationResult>('/jobs/' + encodeURIComponent(selectedJobId) + '/result')
      resultRequests.current.set(selectedResultKey, request)
    }
    void request.then(value => { if (active) setLoadedResult({ key: selectedResultKey, value }) })
      .catch(cause => { if (active) setError(err(cause)) })
      .finally(() => { if (active) setResultLoading(false) })
    return () => { active = false }
  }, [selectedResultKey, selectedJobId, err])

  useEffect(() => {
    if (result) setCurveMode(runOf(result.fixed?.test_result) ? 'fixed' : 'adaptive')
  }, [result])

  const updateRange = (name: string, key: keyof Range, shown: number) => {
    updateSettings({ ranges: { ...settings.ranges, [name]: { ...settings.ranges[name], [key]: shown } } })
    setSearchPreview(null)
  }
  const submit = async () => {
    const baseError = configError(base)
    if (baseError) { setError(baseError); revealConfigError(baseError); return }
    if (!validResearchDefaults(settings)) { setError('请填写有效研究设置；搜索上限应不小于下限，步长须为正数，股数须以 100 为单位。'); revealDisclosure('research-search-ranges'); return }
    const minimumBuyShares = isStarBoard ? 200 : 100
    if (ranges.buy_shares.low < minimumBuyShares || ranges.buy_shares.high < minimumBuyShares) {
      setError('每次买入搜索范围下限应至少为 ' + minimumBuyShares + ' 股 / 份。')
      revealDisclosure('research-search-ranges')
      return
    }
    setBusy(true); setError(''); setDataIssue(null)
    try {
      const submittedRanges = Object.fromEntries(Object.entries(ranges).filter(([name]) => {
        if (protocolVersion === 4) return ['grid_step', 'buy_shares', 'sell_shares', 'min_shares', 'max_shares'].includes(name) && !(name === 'max_shares' && maxSharesMode === 'unlimited')
        if (protocolVersion >= 2 && name === 'initial_shares') return false
        if (protocolVersion >= 3 && base.max_shares === null && name === 'max_shares') return false
        return !['buy_step', 'sell_step', 'grid_step'].includes(name) || (base.grid_mode === 'trailing' ? name !== 'grid_step' : name === 'grid_step')
      }))
      const v4Base = { ...base, grid_mode: 'geometric' as const, max_shares: protocolVersion === 4 && maxSharesMode === 'unlimited' ? null : base.max_shares }
      const common = {
        base: configPayload(v4Base), protocol_version: protocolVersion, n_trials: trials, local_search_budget: localSearchBudget, seed, top_candidates: topCandidates,
        max_drawdown: maxDrawdown, min_trades: minTrades, min_minute_ratio: minMinuteRatio, ranges: submittedRanges,
        strategy: protocolVersion === 4 ? {
          search_method: searchMethod, ranking_mode: rankingMode, training_budget: trainingBudget,
          validation_budget: validationBudget, recommendation_count: recommendationCount,
          max_shares_mode: maxSharesMode, slippage_mode: slippageMode, fixed_half_widths: fixedHalfWidths,
        } : undefined,
      }
      const request = protocolVersion >= 2
        ? { ...common, selection_target: selectionTarget, rebalance_days: rebalanceDays, bootstrap_reps: bootstrapReps }
        : { ...common, mode, min_history_days: minHistoryDays }
      if (protocolVersion >= 3) {
        const preview = await post<Dict>('/optimizations/preview', request)
        setSearchPreview(preview)
        if (preview.status === 'budget_exceeded') {
          setError(protocolVersion === 4
            ? `完整枚举需要 ${String(preview.legal_combination_count ?? '—')} 个点，超过预算 ${String(preview.validation_budget ?? validationBudget)}；请调整范围或预算。`
            : `局部网格预计 ${String(preview.expected ?? '—')} 组，超过预算 ${String(preview.budget ?? localSearchBudget)}；请调整范围或预算。`)
          return
        }
      }
      const response = await tasks.run('参数研究', () => post<{ job_id: string }>('/optimizations', request))
      selectStudy(response.job_id)
    } catch (cause) {
      setError(err(cause))
      if (cause instanceof ApiError && cause.status === 409 && cause.detail && typeof cause.detail === 'object' && (cause.detail as Record<string, unknown>).code === 'DATA_NOT_READY') setDataIssue(cause.detail as DataPreparationRequest)
    } finally { setBusy(false) }
  }
  const loadLegacy = (which: 'historical' | 'robust') => {
    if (!result) return
    const params = which === 'historical' ? result.historical?.best_params : result.robust?.selected_params
    if (!params) { setError(which === 'historical' ? '旧版历史最优参数尚未生成。' : '旧版稳健参数尚未生成。'); return }
    const originalBase = result.config?.base ?? searchConfigOf(selectedJob?.payload?.config)?.base
    if (!originalBase) { setError('这条研究任务未保存原始股票与区间配置，无法安全载入。请重新运行参数研究。'); return }
    const { buy_ratio, sell_ratio, ...configParams } = params
    const patch = configParams as Partial<BacktestConfig>
    const config = { ...originalBase, ...patch, fees: { ...originalBase.fees, ...patch.fees } } as BacktestConfig
    const legacy = config as unknown as Record<string, unknown>
    if (typeof buy_ratio === 'number') legacy.buy_quantity = { mode: 'equity_ratio', value: buy_ratio }
    if (typeof sell_ratio === 'number') legacy.sell_quantity = { mode: 'equity_ratio', value: sell_ratio }
    void loadConfig(config).then(value => onLoadConfig(value.config)).catch(cause => setError(err(cause)))
  }
  const loadV2 = (params: Dict | undefined, selectionAsOf: string | undefined) => {
    if (!result || !params) { setError('当前研究结果没有可载入的参数。'); return }
    const originalBase = result.config?.base ?? searchConfigOf(selectedJob?.payload?.config)?.base
    if (!originalBase) { setError('这条研究任务未保存原始股票与区间配置，无法安全载入。请重新运行参数研究。'); return }
    const locked = record(result.locked_funding) ?? {}
    const { buy_ratio: _oldBuyRatio, sell_ratio: _oldSellRatio, ...configParams } = params
    const patch = configParams as Partial<BacktestConfig>
    const config: BacktestConfig = {
      ...originalBase,
      ...patch,
      grid_mode: Number(actualVersion) === 4 ? 'geometric' : patch.grid_mode ?? originalBase.grid_mode,
      initial_shares: finite(locked.initial_shares) ? locked.initial_shares : originalBase.initial_shares,
      selection_asof: selectionAsOf ?? originalBase.end_date ?? originalBase.start_date,
      fees: { ...originalBase.fees, ...patch.fees },
    }
    void loadConfig(config).then(value => onLoadConfig(value.config)).catch(cause => setError(err(cause)))
  }

  const historyCandidates = result?.historical?.candidates ?? []
  const robustCandidates = result?.robust?.candidates ?? []
  const robust = result?.robust
  const resultConfig = result?.config
  const taskConfig = searchConfigOf(selectedJob?.payload?.config)
  const originalBase = resultConfig?.base ?? taskConfig?.base
  const actualRanges = record(resultConfig?.ranges) ?? record(taskConfig?.ranges)
  const actualMode = result?.mode ?? resultConfig?.mode ?? taskConfig?.mode
  const actualTrials = result?.n_trials ?? resultConfig?.n_trials ?? taskConfig?.n_trials
  const actualSeed = result?.seed ?? resultConfig?.seed ?? taskConfig?.seed
  const actualVersion = result ? protocolVersionOf(result, resultConfig, taskConfig) : 1
  const isV2Result = Boolean(result && Number(actualVersion) >= 2 && (result.fixed || result.adaptive))
  const researchWarnings = Array.from(new Set(warningsOf(result)))
  const minuteCoverage = result?.diagnostics?.minute_ratio ?? result?.evidence?.minute_coverage ?? result?.quality?.minute_ratio
  const isDemoResearch = [result, result?.quality, result?.fixed?.test_result, result?.adaptive?.test_result, result?.historical?.best_result, robust?.test_result].some(hasDemoFlag) || researchWarnings.some(warning => /演示|demo/i.test(warning))
  const v2Candidates = useMemo(() => {
    if (!result?.evidence) return result?.fixed?.current?.candidates ?? []
    const candidates = [
      ...list(result.fixed?.current?.candidates),
      ...list(result.evidence.candidates),
      ...list(result.evidence.decisions).flatMap(decision => list(record(decision)?.candidates)),
    ].filter((value): value is Candidate => Boolean(record(value)))
    const seen = new Set<string>()
    return candidates.filter(candidate => {
      const signature = JSON.stringify({ id: candidate.id, params: paramsOf(candidate), score: candidate.selection_score ?? candidate.score })
      if (seen.has(signature)) return false
      seen.add(signature)
      return true
    })
  }, [result])
  const legacyChartOption: EChartsOption = useMemo(() => {
    const points: Array<{ value: [number, number]; name: string; params: Dict }> = []
    const sections: Array<[string, Candidate[]]> = [['旧版历史候选', historyCandidates], ['旧版稳健候选', robustCandidates]]
    for (const [section, candidates] of sections) for (const item of candidates) {
      const metrics = getMetrics(item, section === '旧版稳健候选')
      const ret = n(metrics.total_return ?? metrics.return)
      const dd = Math.abs(n(metrics.max_drawdown ?? metrics.drawdown))
      if (Number.isFinite(ret) && Number.isFinite(dd)) points.push({ value: [ret, dd], name: section, params: paramsOf(item) })
    }
    return {
      tooltip: { trigger: 'item', formatter: (value: unknown) => { const p = value as { data: { name: string; value: [number, number]; params: Dict } }; return p.data.name + '<br/>收益：' + percent(p.data.value[0]) + '<br/>回撤：' + percent(p.data.value[1]) + '<br/>' + Object.entries(p.data.params).map(([key, raw]) => labelOf(key) + '：' + (pctRange.has(key) ? rangeValue(key, raw) : valueText(key, raw))).join('<br/>') } },
      legend: { top: 0 }, grid: { left: 55, right: 24, top: 36, bottom: 48 },
      xAxis: { name: '总收益率', nameLocation: 'middle', nameGap: 30, axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#edf0f5' } } },
      yAxis: { name: '最大回撤', axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#edf0f5' } } },
      series: [
        { name: '旧版历史候选', type: 'scatter', data: points.filter(point => point.name === '旧版历史候选'), symbolSize: 10, itemStyle: { color: '#2864dc', opacity: 0.72 } },
        { name: '旧版稳健候选', type: 'scatter', data: points.filter(point => point.name === '旧版稳健候选'), symbolSize: 11, itemStyle: { color: '#e85c63', opacity: 0.8 } },
      ],
    }
  }, [historyCandidates, robustCandidates])

  const protocol = record(result?.protocol) ?? {}
  const fixed = result?.fixed
  const adaptive = result?.adaptive
  const current = fixed?.current
  const currentRecommended = Number(actualVersion) >= 3 ? current?.recommended_params ?? current?.params : current?.params
  const adaptiveRecommended = adaptive?.latest_params
  const fixedRun = runOf(fixed?.test_result)
  const adaptiveRun = runOf(adaptive?.test_result)
  const protocolTrain = record(protocol.training) ?? record(protocol.train) ?? {}
  const protocolEval = record(protocol.evaluation) ?? record(protocol.test) ?? {}
  const trainStart = firstString(protocol, ['train_start', 'training_start', 'training_start_date']) ?? firstString(protocolTrain, ['start', 'start_date', 'from'])
  const trainEnd = firstString(protocol, ['train_end', 'training_end', 'training_end_date']) ?? firstString(protocolTrain, ['end', 'end_date', 'to'])
  const evalStart = firstString(protocol, ['evaluation_start', 'eval_start', 'test_start']) ?? firstString(protocolEval, ['start', 'start_date', 'from']) ?? originalBase?.start_date
  const evalEnd = firstString(protocol, ['evaluation_end', 'eval_end', 'test_end']) ?? firstString(protocolEval, ['end', 'end_date', 'to']) ?? originalBase?.end_date ?? undefined
  const lockedFunding = record(result?.locked_funding) ?? {}
  const finalCheck = record(result?.evidence?.final_126_day_check)
  const currentFamilyBootstrap = record(result?.evidence?.current_candidate_family_bootstrap) ?? record(current?.diagnostics?.candidate_family_bootstrap)
  const localAudit = record(result?.evidence?.local_audit)
  const statisticalProfile = record(fixed?.diagnostics?.statistical_profile) ?? record(current?.diagnostics?.statistical_profile)
  const trainingDays = record(fixed?.diagnostics)?.selection_history_days ?? record(result?.evidence?.decisions?.[0]?.diagnostics)?.selection_history_days ?? current?.diagnostics?.selection_history_days
  const insufficientHistory = [fixed, adaptive, current].some(item => item?.status === 'insufficient_history') || finite(trainingDays) && trainingDays < 504
  const finalReview = reviewBootstrapEvidence(finalCheck)
  const finalDays = finite(finalCheck?.paired_days) ? finalCheck.paired_days : finite(finalCheck?.days) ? finalCheck.days : null
  const finalRequiredDays = finite(finalCheck?.required_days) ? finalCheck.required_days : 126
  const finalPeriodComplete = Boolean(finalCheck && finalCheck.short_term_insufficient !== true && (finalDays == null || finalDays >= finalRequiredDays))
  const finalEvidenceLabel = !finalPeriodComplete ? '样本不足' : finalReview.state === 'positive' ? '校正下界为正' : finalReview.state === 'not_positive' ? '尚无校正后正向证据' : '统计证据不足'
  const currentRecommendationStatus = String(current?.recommendation_status ?? (currentRecommended ? 'qualified' : 'no_qualified'))
  const adaptiveAdviceDiffers = Boolean(adaptiveRecommended && (!currentRecommended || JSON.stringify(adaptiveRecommended) !== JSON.stringify(currentRecommended) || adaptive?.latest_as_of !== current?.as_of))
  const curveRun = curveMode === 'fixed' ? fixedRun : adaptiveRun
  const rangeNamesToShow = protocolVersion === 4
    ? (['grid_step', 'sell_shares', 'buy_shares', 'min_shares', 'max_shares'] as string[]).filter(name => !(name === 'max_shares' && maxSharesMode === 'unlimited'))
    : knownRangeNames.filter(name => (protocolVersion === 1 || name !== 'initial_shares') && !(protocolVersion === 3 && base.max_shares === null && name === 'max_shares') && (!pctRange.has(name) || (base.grid_mode === 'trailing' ? name !== 'grid_step' : name === 'grid_step')))
  const schemeLabel = selectionTarget === 'both' ? '固定 + 自适应' : selectionTarget === 'fixed' ? '固定参数' : '每季自适应'
  const landscapeDecisions = [
    { value: 'current:0', label: '当前建议决策' },
    ...(adaptive?.schedule ?? []).map((item, index) => ({ value: `adaptive:${index}`, label: `自适应 ${item.effective_date ?? index + 1}` })),
    ...(fixed?.as_of ? [{ value: 'fixed:0', label: '固定方案决策' }] : []),
  ]
  const rangesSummary = rangeNamesToShow.map(name => labels[name] + ' ' + rangeValue(name, ranges[name].low) + '–' + rangeValue(name, ranges[name].high)).join(' · ')
  const savedRanges = actualRanges && Object.keys(actualRanges).length ? <div className='table-wrap'><table><thead><tr><th>参数</th><th>下限</th><th>上限</th><th>步长</th></tr></thead><tbody>{Object.entries(actualRanges).map(([key, raw]) => { const range = record(raw); return <tr key={key}><td>{labels[key] ?? key}</td><td>{range ? rangeValue(key, range.low) : '—'}</td><td>{range ? rangeValue(key, range.high) : '—'}</td><td>{range ? rangeValue(key, range.step) : '—'}</td></tr> })}</tbody></table></div> : <Empty title='任务中未保存搜索范围' />

  return <>
    <PageHeader eyebrow='PARAMETER RESEARCH' title='参数研究' description='用历史训练选择参数，在所选日期内检验固定与自适应方案。' action={<Button busy={busy} disabled={!base.symbol} onClick={() => void submit()}><FlaskConical size={15} />启动参数研究</Button>} />
    <div className='research-guide'>
      <p><strong>研究流程</strong><span>设置标的与评估期 → 选择方案并启动 → 查看样本外表现与证据 → 载入回测</span></p>
      <Disclosure id='research-method-help' title='如何理解日期、方案与证据' summary='评估日期 · 504 日训练 · 当前建议'>
        <div className='research-help-grid'>
          <p><strong>日期与训练</strong>所选开始、结束日期定义样本外评估期。科学研究 v2 自动向前读取开始日前的 504 个交易日训练；历史不足时仍可能返回相对最佳候选，并明确标记不足。</p>
          <p><strong>固定与自适应</strong>固定方案在评估开始前锁定一组参数；自适应方案每隔设定交易日重选一次，决策只使用当时可得的历史，账户连续运行。</p>
          <p><strong>历史参数与当前建议</strong>样本外曲线使用当时已选出的参数。评估结束时算出的下一期建议属于新的选择，不能追加到历史曲线或解释为历史已实现收益。</p>
          <p><strong>三个独立判断</strong>相对最佳表示候选中得分最高；满足约束表示通过回撤、成交等条件；统计证据需要另看家族校正和置信下界。p 值不是策略胜率或未来盈利概率。</p>
        </div>
      </Disclosure>
    </div>
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {dataIssue && <div className='notice'><span>本次研究缺少本地行情或质量依据。请在数据管理中显式准备所需周期和前置历史。</span><Button variant='secondary' onClick={() => onPrepareData?.(dataIssue)}>去准备数据</Button></div>}
    <div className='research-workspace'>
      <aside className='research-config-column'>
        <Card title='研究对象' subtitle='所选日期是样本外评估期，系统另取之前的历史训练。'>
          <ConfigForm value={base} onChange={onBaseChange} purpose='research' researchProtocol={protocolVersion} onFeeDefaultsEdit={onFeeDefaultsEdit} onDefaultsEdit={onDefaultsEdit} />
          <Button variant='quiet' disabled={!base.symbol} onClick={() => {
            get<{ first_date?: string }>('/data/coverage/' + base.symbol).then(value => {
              if (value.first_date) onBaseChange(previous => ({ ...previous, start_date: new Date(Date.parse(value.first_date! + 'T00:00:00Z') + 86400000).toISOString().slice(0, 10), reference_price_mode: 'auto', reference_price: null }))
            }).catch(cause => setError(err(cause)))
          }}>使用全部本地历史</Button>
        </Card>
        <Card title='研究设置' className='research-launch-card' action={onRestore && <Button variant='quiet' onClick={onRestore}><RotateCcw size={14} />恢复默认</Button>}>
          {protocolVersion >= 2 ? <>
            <div className='field-label'>参数方案</div>
            <div className='research-scheme-tabs' role='group' aria-label='参数方案'>
              {([['fixed', '固定参数'], ['adaptive', '每季自适应'], ['both', '两者对照']] as const).map(([value, label]) => <button type='button' className={selectionTarget === value ? 'selected' : ''} key={value} aria-pressed={selectionTarget === value} onClick={() => updateSettings({ selection_target: value })}>{label}</button>)}
            </div>
            <p className='field-hint'>{selectionTarget === 'fixed' ? '评估开始前锁定一组参数，整个评估期不变。' : selectionTarget === 'adaptive' ? '每隔 ' + rebalanceDays + ' 个交易日重选，账户连续运行。' : '同时比较一次锁定与每隔 ' + rebalanceDays + ' 个交易日重选。'}</p>
          </> : <div className='result-meta'><Badge tone='amber'>旧版研究 v1</Badge><span>{jobModeLabel(mode)}</span></div>}
          {protocolVersion === 4 && <>
            <div className='form-grid two'>
              <Field label='研究策略'><div className='result-meta'><Badge tone='blue'>等比网格正超额邻域 v1</Badge><span>五维整数刻度搜索</span></div></Field>
              <Field label='推荐模式'><SelectInput value={rankingMode} onChange={event => updateSettings({ ranking_mode: event.target.value as ResearchDefaults['ranking_mode'] })}><option value='max_volume'>最大正超额收益邻域</option><option value='fixed_mean'>固定邻域平均超额收益</option></SelectInput></Field>
              <Field label='搜索方法'><SelectInput value={searchMethod} onChange={event => updateSettings({ search_method: event.target.value as ResearchDefaults['search_method'] })}><option value='auto'>自动（小空间穷举）</option><option value='hierarchical'>分层搜索</option><option value='exhaustive'>完整枚举</option></SelectInput></Field>
              <NumField label='训练预算（唯一参数点）' min={1} max={1000000} step='1' value={trainingBudget} onChange={next => next != null && updateSettings({ training_budget: next })} />
              <NumField label='验证预算（唯一参数点）' min={1} max={2000000} step='1' value={validationBudget} onChange={next => next != null && updateSettings({ validation_budget: next })} />
              <NumField label='推荐参数个数' min={1} max={100} step='1' value={recommendationCount} onChange={next => next != null && updateSettings({ recommendation_count: next })} />
              <Field label='最高持股研究方式'><SelectInput value={maxSharesMode} onChange={event => updateSettings({ max_shares_mode: event.target.value as ResearchDefaults['max_shares_mode'] })}><option value='search'>搜索有限上限</option><option value='unlimited'>固定不限</option></SelectInput></Field>
              <Field label='滑点处理'><SelectInput value={slippageMode} onChange={event => updateSettings({ slippage_mode: event.target.value as ResearchDefaults['slippage_mode'] })}><option value='configured'>按当前配置</option><option value='zero'>不考虑滑点</option></SelectInput></Field>
            </div>
            <p className='field-hint'>等比网格最小间距 0.1 个百分点；交易与持股数量最小单位 100。研究只搜索这五维，基础配置中的对应值不参与搜索。</p>
            {rankingMode === 'fixed_mean' && <div className='form-grid two'>{Object.entries(fixedHalfWidths).filter(([name]) => !(name === 'max_shares' && maxSharesMode === 'unlimited')).map(([name, value]) => <NumField key={name} label={(labels[name] ?? name) + '半宽'} min={name === 'grid_step' ? .1 : 100} step={name === 'grid_step' ? '.1' : '100'} percentValue={name === 'grid_step'} suffix={name === 'grid_step' ? '%' : undefined} value={value} onChange={next => next != null && updateSettings({ fixed_half_widths: { ...fixedHalfWidths, [name]: next } })} hint={name === 'grid_step' ? '按 0.1 个百分点步进' : '按 100 股 / 份步进'} />)}</div>}
          </>}
          {protocolVersion === 3
            ? <NumField label='局部网格搜索预算' min={1} max={5000} step='1' value={localSearchBudget} onChange={next => next != null && updateSettings({ local_search_budget: next })} hint='统计中心各自展开中心及正负一步；超过预算时停止提交并提示调整范围。' />
            : protocolVersion !== 4 && <NumField label={protocolVersion === 2 ? '每个决策点基础试验数' : '旧版每种模式试验数'} min={1} max={5000} step='1' value={trials} onChange={next => next != null && updateSettings({ n_trials: next })} hint={protocolVersion === 2 ? '邻域压力测试会增加额外计算。' : '历史与稳健模式各自执行此次数。'} />}
          {protocolVersion === 4 && searchPreview && <p className={searchPreview.status === 'budget_exceeded' || searchPreview.minimum_neighborhood_possible === false ? 'inline-warning' : 'field-hint'}>合法组合 {String(searchPreview.legal_combination_count ?? '—')} 个 · {String(searchPreview.search_method ?? '—')} · 验证预算 {String(searchPreview.validation_budget ?? validationBudget)} · 最小邻域 {String(searchPreview.minimum_neighborhood_size ?? '—')} 点{searchPreview.full_504_day_history ? '' : ' · 前置历史不足'}{searchPreview.minimum_neighborhood_possible === false ? ' · 范围无法容纳完整对称邻域' : ''}</p>}
          {protocolVersion === 3 && searchPreview && <p className={searchPreview.status === 'budget_exceeded' ? 'inline-warning' : 'field-hint'}>局部网格预计 {String(searchPreview.expected ?? '—')} / {String(searchPreview.budget ?? localSearchBudget)} 组，统计中心 {String(searchPreview.statistical_center_count ?? '—')} 个{searchPreview.status === 'budget_exceeded' ? '；已超过预算，研究未提交。' : '。'}</p>}
          <div className='research-launch-action'><Button busy={busy} onClick={() => void submit()}><FlaskConical size={15} />启动参数研究</Button><small>{protocolVersion >= 2 ? schemeLabel + ' · 前置 504 日训练' : '旧版兼容协议'} · 设置自动记忆</small></div>
          <Disclosure id='research-search-ranges' title='可调搜索范围' summary={rangesSummary}>
            <p className='field-hint'>{protocolVersion === 4 ? '仅搜索等比网格间距、买卖数量及持股边界；最小单位固定为 0.1 个百分点和 100 股 / 份，范围步长不控制研究精度。' : protocolVersion === 2 ? '固定账户初始持股与现金，只搜索网格步长和交易边界。' : '股数以 100 股 / 份为单位。'}</p>
            <div className='range-grid'>{rangeNamesToShow.map(name => <div className='range-row' key={name}><div><strong>{labels[name]}</strong><small>{pctRange.has(name) ? '步长（%）' : '股 / 份'}</small></div>{(['low', 'high', 'step'] as const).map(key => {
              const minimum = name === 'buy_shares' ? isStarBoard ? 200 : 100 : name === 'sell_shares' ? 100 : 0
              if (protocolVersion === 4 && key === 'step') return <div key={key}><small>基础单位</small><strong>{pctRange.has(name) ? '0.1 个百分点' : '100 股 / 份'}</strong></div>
              return <NumField key={key} label={key === 'low' ? '下限' : key === 'high' ? '上限' : '步长'} value={ranges[name][key]} percentValue={pctRange.has(name)} step={protocolVersion === 4 && pctRange.has(name) ? '.1' : pctRange.has(name) ? 'any' : '100'} min={pctRange.has(name) ? protocolVersion === 4 ? .1 : .000001 : key === 'step' ? 100 : minimum} onChange={next => next != null && updateRange(name, key, next)} />
            })}</div>)}</div>
            <Button variant='quiet' onClick={() => updateSettings({ ranges: defaultSearchRanges() })}><RotateCcw size={14} />恢复默认范围</Button>
          </Disclosure>
          <Disclosure id='research-constraints' title='选择与样本约束' summary={'回撤 ≤ ' + percent(maxDrawdown) + ' · 成交 ≥ ' + minTrades + ' · 分钟覆盖 ≥ ' + percent(minMinuteRatio)}>
            <div className='form-grid two'>
              <NumField label='最大回撤约束' min={1} max={100} step='1' percentValue suffix='%' value={maxDrawdown} onChange={next => next != null && updateSettings({ max_drawdown: next })} />
              <NumField label='最少成交笔数' min={0} step='1' value={minTrades} onChange={next => next != null && updateSettings({ min_trades: next })} />
              <NumField label='最小分钟覆盖率' min={0} max={100} step='1' percentValue suffix='%' value={minMinuteRatio} onChange={next => next != null && updateSettings({ min_minute_ratio: next })} />
            </div>
            <p className='field-hint'>候选的相对排名与约束是否满足会分别展示。</p>
          </Disclosure>
          <Disclosure id='research-reproducibility' title='换参、统计与复现设置' summary={(protocolVersion >= 2 ? rebalanceDays + ' 日换参 · Bootstrap ' + bootstrapReps.toLocaleString() + ' · ' : '') + '种子 ' + seed + ' · 展示 ' + topCandidates + ' 组'}>
            <div className='form-grid two'>
              {protocolVersion >= 2 && <><NumField label='自适应换参周期（交易日）' min={1} max={504} step='1' value={rebalanceDays} onChange={next => next != null && updateSettings({ rebalance_days: next })} /><NumField label='Bootstrap 重抽样次数' min={100} max={100000} step='100' value={bootstrapReps} onChange={next => next != null && updateSettings({ bootstrap_reps: next })} /></>}
              <NumField label='随机种子' step='1' value={seed} onChange={next => next != null && updateSettings({ seed: next })} />
              <NumField label='结果候选展示上限' min={1} max={100} step='1' value={topCandidates} onChange={next => next != null && updateSettings({ top_candidates: next })} hint='只限制页面展示数量，不截断研究验证。' />
            </div>
          </Disclosure>
          <Disclosure id='research-legacy-protocol' title='研究协议与旧版兼容' summary={protocolVersion === 4 ? '等比网格 v4 · 最大邻域' : protocolVersion === 3 ? '统计驱动研究 v3' : protocolVersion === 2 ? '科学研究 v2' : '旧版 v1 · ' + jobModeLabel(mode)}>
            <Field label='研究协议'><SelectInput value={protocolVersion} onChange={event => updateSettings({ protocol_version: Number(event.target.value) as ResearchDefaults['protocol_version'] })}><option value='4'>等比网格 v4 · 最大邻域</option><option value='3'>统计驱动研究 v3</option></SelectInput></Field>
            {protocolVersion === 1 && <><Field label='旧版研究模式'><SelectInput value={mode} onChange={event => updateSettings({ mode: event.target.value as typeof mode })}><option value='both'>历史最优 + 稳健验证</option><option value='historical'>历史最优</option><option value='robust'>稳健验证</option></SelectInput></Field><NumField label='旧版最少历史交易日' min={20} step='1' value={minHistoryDays} onChange={next => next != null && updateSettings({ min_history_days: next })} /></>}
            <p className='field-hint'>旧版历史最优与单段切分结果仍可查看，但不等于科学研究 v2 的验证通过。</p>
          </Disclosure>
        </Card>
      </aside>
      <main className='research-main-column' aria-label='参数研究结果'>
        <div className='research-result-toolbar'>
          <h2>研究结果</h2>
          {viewableJobs.length > 0 && <Field label='选择研究结果'><SelectInput value={selectedJobId} onChange={event => selectStudy(event.target.value)}>
            {!viewableJobs.some(job => job.id === selectedJobId) && <option value={selectedJobId}>当前选择</option>}
            {viewableJobs.map(job => { const config = searchConfigOf(job.payload?.config); return <option key={job.id} value={job.id}>{config?.base?.symbol ?? '未识别标的'} · {config?.base?.start_date ?? job.created_at?.slice(0, 10) ?? ''} · {jobModeLabel(config?.mode, protocolVersionOf(config))}{job.status === 'cancelled' ? ' · 部分结果' : ''}</option> })}
          </SelectInput></Field>}
          <Button variant='quiet' onClick={() => void tasks.refresh()} aria-label='刷新研究结果列表'><RefreshCw size={14} />刷新</Button>
        </div>
        {resultLoading ? <Card><Loading label='正在读取研究结果…' /></Card> : !result ? <Card><Empty title='研究结果将在这里展示' detail='在左侧启动研究，或从任务中心打开已有研究结果。' /></Card> : <>
          <Card title={originalBase?.symbol ?? '研究样本'} subtitle={(evalStart ?? '开始日期不详') + ' 至 ' + (evalEnd ?? '最新可用日期') + ' · ' + (originalBase?.frequency ?? '频率不详')}>
            <div className='result-meta'><Badge tone={isV2Result ? 'blue' : 'amber'}>{jobModeLabel(actualMode, actualVersion)}</Badge>{isDemoResearch && <Badge tone='amber'>演示数据</Badge>}{selectedJob?.status === 'cancelled' && <Badge tone='amber'>已保留的部分结果</Badge>}<span>{isV2Result ? '所选日期为样本外评估；训练取开始日前 504 个交易日。' : '旧版历史最优或单段切分记录。'}</span></div>
            {isDemoResearch && <div className='demo-callout'>此研究使用演示行情生成，不能代表真实历史数据或投资结论。</div>}
            {isV2Result && <div className='research-summary-states'>
              <div><small>训练样本</small><Badge tone={insufficientHistory ? 'amber' : finite(trainingDays) && trainingDays >= 504 ? 'green' : 'neutral'}>{finite(trainingDays) ? trainingDays + ' / 504 个交易日' : insufficientHistory ? '历史不足' : '前置 504 个交易日'}</Badge></div>
              <div><small>当前推荐状态</small><Badge tone={currentRecommendationStatus === 'qualified' ? 'green' : 'amber'}>{currentRecommendationStatus === 'qualified' ? (Number(actualVersion) === 4 ? '完整正邻域已验证' : '已通过稳健性门槛') : '暂无合格推荐'}</Badge></div>
              <div><small>末 126 日样本外证据</small><Badge tone={finalPeriodComplete && finalReview.state === 'positive' ? 'blue' : 'amber'}>{finalEvidenceLabel}</Badge></div>
            </div>}
            {insufficientHistory && <p className='inline-warning'>可用训练历史不足。返回参数仅代表当前候选中的相对最佳选择，样本与证据不足会分别标记。</p>}
            {isV2Result && !finalPeriodComplete && <p className='inline-warning'>末段核验样本不足{finalDays == null ? '' : '（' + finalDays + ' / ' + finalRequiredDays + ' 个交易日）'}，不能据此得出显著性结论。</p>}
            {researchWarnings.length > 0 && <div className='warning-list'>{researchWarnings.map((warning, index) => <div key={index}><span className='warning-dot' />{warning}</div>)}</div>}
          </Card>
          {Number(actualVersion) >= 3 && selectedJob?.result_id && <ResearchLandscape resultId={selectedJob.result_id} defaultParams={currentRecommended ?? current?.params ?? undefined} decisionOptions={landscapeDecisions} regionOptions={(current?.recommendation_regions ?? []).map(region => ({ value: String(region.region_id ?? ''), label: '区域 ' + String(region.region_id ?? '') }))} />}
          {Number(actualVersion) === 4 && current && <Card title='最大正超额收益邻域' subtitle='邻域内全部合法离散组合必须在原成本下超额收益为正；双倍成本只作诊断。'>
            {(current.recommendation_regions ?? []).length === 0 ? <p className='inline-warning'>本次冻结参数域没有发现完整正超额邻域，正式推荐为空。</p> : <div className='table-wrap'><table><thead><tr><th>排名</th><th>中心参数</th><th>体积</th><th>组合数</th><th>平均超额</th><th>最低超额</th><th>峰值超额</th><th>状态</th><th>压力诊断</th><th></th></tr></thead><tbody>{(current.recommendation_regions ?? []).map((region, index) => <tr key={region.region_id ?? index}><td>{region.region_id ?? index + 1}</td><td><ParameterBadges params={region.center_params ?? {}} compact /></td><td>{percent(n(region.normalized_volume), 4)}</td><td>{valueText('组合数', region.completed_count)} / {valueText('组合数', region.combination_count)}</td><td>{percent(n(region.mean_excess_return), 4)}</td><td>{percent(n(region.min_excess_return), 4)}</td><td>{percent(n(region.center_excess_return), 4)}</td><td>{region.complete && region.maximum_search_complete ? '完整验证' : '未完成'}{region.global_exhaustive ? ' · 全局穷尽' : ' · 冻结域'}</td><td>{String(region.stress?.status ?? '未计算')}{region.stress?.mean_excess_return != null ? ' · 均值 ' + percent(n(region.stress.mean_excess_return), 2) : ''}</td><td><Button variant='quiet' onClick={() => loadV2(region.center_params, current.as_of)}>载入</Button></td></tr>)}</tbody></table></div>}
          </Card>}
          {Number(actualVersion) === 3 && statisticalProfile && <Card title='统计画像与稳健性门槛' subtitle='统计量只形成少数中心；推荐资格由原参数空间的真实邻点、双成本和跨窗表现共同决定。'>
            <div className='research-summary-states'>
              <div><small>生成样本</small><strong>{valueText('days', statisticalProfile.days)} 个交易日</strong></div>
              <div><small>趋势强度</small><strong>{valueText('trend_strength', statisticalProfile.trend_strength)}</strong></div>
              <div><small>下跌后 5 日修复</small><strong>{valueText('repair_ratio', statisticalProfile.repair_ratio)}</strong></div>
              <div><small>邻域门槛</small><strong>≥ 80% 邻点为正 · 第 25 百分位 &gt; 0</strong></div>
            </div>
            <details><summary>查看完整统计画像与中心</summary><pre className='json-preview'>{JSON.stringify(statisticalProfile, null, 2)}</pre></details>
          </Card>}
          {isV2Result ? <>
            <Card title='历史样本外方案对照' subtitle='固定方案先锁定参数，自适应方案按时点滚动选择；以下收益来自当时实际使用的参数。'>
              <div className='table-wrap research-comparison-table'><table><thead><tr><th>方案</th><th>参数使用方式</th><th>总收益</th><th>最大回撤</th><th>选择状态</th></tr></thead><tbody>
                {fixed && <tr><td><strong>固定参数</strong><small>锁定于 {fixed.as_of ?? '—'}</small></td><td><ParameterBadges params={fixed.params} compact /><Button variant='quiet' disabled={!fixed.params || !originalBase} onClick={() => loadV2(fixed.params, fixed.as_of)}>载入固定参数</Button></td><td>{fixedRun?.strategies?.grid?.metrics?.total_return == null ? '—' : percent(n(fixedRun.strategies.grid.metrics.total_return))}</td><td>{fixedRun?.strategies?.grid?.metrics?.max_drawdown == null ? '—' : percent(n(fixedRun.strategies.grid.metrics.max_drawdown))}</td><td><Badge tone={fixed.status === 'insufficient_history' ? 'amber' : 'neutral'}>{statusText(fixed.status)}</Badge><small>{constraintLabel(eligibleOf(fixed, fixed.diagnostics))}</small></td></tr>}
                {adaptive && <tr><td><strong>自适应参数</strong><small>每 {valueText('rebalance_days', protocol.rebalance_days ?? resultConfig?.rebalance_days ?? rebalanceDays)} 个交易日重选</small></td><td>账户连续运行，{adaptive.schedule?.length ?? 0} 次参数生效<small>各期参数见下方换参记录</small></td><td>{adaptiveRun?.strategies?.grid?.metrics?.total_return == null ? '—' : percent(n(adaptiveRun.strategies.grid.metrics.total_return))}</td><td>{adaptiveRun?.strategies?.grid?.metrics?.max_drawdown == null ? '—' : percent(n(adaptiveRun.strategies.grid.metrics.max_drawdown))}</td><td><Badge tone={adaptive.status === 'insufficient_history' ? 'amber' : 'neutral'}>{statusText(adaptive.status)}</Badge><small>约束按决策点分别判断</small></td></tr>}
              </tbody></table></div>
              <div className='research-locked-funding'>{Number(actualVersion) >= 3 ? <><span>初始持股 <strong>{valueText('initial_shares', lockedFunding.initial_shares ?? originalBase?.initial_shares)}</strong></span><span>资金模式 <strong>成功买入时补足差额</strong></span></> : <><span>固定初始持股 <strong>{valueText('initial_shares', lockedFunding.initial_shares ?? originalBase?.initial_shares)}</strong></span><span>固定现金 <strong>{valueText('reserve_cash', lockedFunding.reserve_cash ?? originalBase?.reserve_cash)}</strong></span></>}</div>
            </Card>
            {(current || adaptiveRecommended) && <Card title='下一期参考参数' className='research-recommendations' subtitle='参数只使用当时或之前可用的数据；通过邻域稳健性检查才列为合格推荐，相对最佳仅作研究对照。'>
              {current && <section className='research-next-period'>
                <div className='research-recommendation-head'><div><strong>{currentRecommended ? '当前合格推荐' : '暂无合格推荐'}</strong><small>决策日期 {current.as_of ?? '—'} · {statusText(current.status)}</small></div><Button variant='secondary' onClick={() => loadV2(currentRecommended ?? undefined, current.as_of)} disabled={!currentRecommended || !originalBase}>载入当前参数</Button></div>
                {currentRecommended ? <ParameterBadges params={currentRecommended} /> : <p className='inline-warning'>{Number(actualVersion) === 4 ? '冻结参数域内没有发现完整且全部为正的局部邻域。相对最佳参数仅在候选明细中保留。' : '没有参数同时通过双成本、80% 正超额邻点和邻域第 25 百分位门槛。相对最佳参数仅在候选明细中保留。'}</p>}
                <Assessment recommendation={current} familyEvidence={currentFamilyBootstrap} />
              </section>}
              {adaptiveAdviceDiffers && <section className='research-next-period'>
                <div className='research-recommendation-head'><div><strong>自适应参数最新建议</strong><small>决策日期 {adaptive?.latest_as_of ?? current?.as_of ?? '—'}</small></div><Button variant='secondary' onClick={() => loadV2(adaptiveRecommended ?? undefined, adaptive?.latest_as_of ?? current?.as_of ?? adaptive?.schedule?.at(-1)?.decision_date)} disabled={!adaptiveRecommended || !originalBase}>载入最新建议</Button></div>
                {adaptiveRecommended ? <ParameterBadges params={adaptiveRecommended} /> : <p className='inline-warning'>本期没有合格推荐，自适应网格保持暂停。</p>}
              </section>}
              {adaptiveRecommended && !adaptiveAdviceDiffers && <p className='field-hint'>自适应参数最新建议与当前参数相同。</p>}
            </Card>}
            <Card title='真实样本外曲线' subtitle='只绘制本次评估区间的连续回测；期末最新建议不并入曲线。'>
              <div className='research-curve-tabs' role='tablist' aria-label='样本外曲线方案'>
                <button type='button' role='tab' aria-selected={curveMode === 'fixed'} className={curveMode === 'fixed' ? 'active' : ''} disabled={!fixedRun} onClick={() => setCurveMode('fixed')}>固定参数</button>
                <button type='button' role='tab' aria-selected={curveMode === 'adaptive'} className={curveMode === 'adaptive' ? 'active' : ''} disabled={!adaptiveRun} onClick={() => setCurveMode('adaptive')}>自适应参数</button>
              </div>
              <div role='tabpanel' aria-label={curveMode === 'fixed' ? '固定参数样本外曲线' : '自适应参数样本外曲线'}><ResultCurve run={curveRun} /></div>
            </Card>
            <Disclosure id='research-rebalance-records' title='换参记录与滚动决策' summary={(adaptive?.schedule?.length ?? 0) + ' 次生效 · ' + (result.evidence?.decisions?.length ?? 0) + ' 个决策点'}>
              <p className='field-hint'>决策日表示何时重选参数；生效日表示参数开始作用的日期。各决策只使用当时已知历史。</p>
              {adaptive?.schedule?.length ? <div className='table-wrap scroll-table'><table><thead><tr><th>决策日期</th><th>生效日期</th><th>参数</th></tr></thead><tbody>{adaptive.schedule.map((item, index) => <tr key={(item.effective_date ?? '') + '-' + index}><td>{item.decision_date ?? '—'}</td><td>{item.effective_date ?? '—'}</td><td>{item.paused ? <Badge tone='amber'>暂停网格交易</Badge> : <ParameterBadges params={item.params} compact />}</td></tr>)}</tbody></table></div> : <Empty title='没有季度换参记录' detail={statusText(adaptive?.status)} />}
              {result.evidence?.decisions?.length ? <div className='table-wrap scroll-table'><table><thead><tr><th>决策日期</th><th>候选 / 实际评估</th><th>约束合格</th><th>选中参数</th><th>分窗得分</th><th>邻域得分</th></tr></thead><tbody>{result.evidence.decisions.map((raw, index) => { const item = record(raw) ?? {}; return <tr key={index}><td>{String(item.decision_date ?? item.as_of ?? '—')}</td><td>{valueText('candidate_count', item.candidate_count)} / {valueText('evaluated_count', item.evaluated_count)}</td><td>{valueText('eligible_count', item.eligible_count)}</td><td><ParameterBadges params={record(item.selected_params)} compact /></td><td>{shortSummary(item.windows ?? item.folds)}</td><td>{shortSummary(item.neighbor_scores ?? item.bootstrap)}</td></tr> })}</tbody></table></div> : <Empty title='暂无滚动决策明细' />}
            </Disclosure>
            <Disclosure id='research-statistical-evidence' title='统计证据与敏感性' summary={'末 126 日：' + finalEvidenceLabel + ' · p 值不是胜率'}>
              <Card title='当前完整候选族内部验证' subtitle='族级检验描述本次择参的不确定性；样本已经用于选择，不属于独立样本外检验。'><FamilyBootstrapPanel value={currentFamilyBootstrap} /></Card>
              <Card title='末 126 个交易日核验' subtitle='固定与自适应的真实连续样本外轨迹，对同现金被动持有做配对比较。'><Final126Evidence value={finalCheck} /></Card>
            </Disclosure>
            <Disclosure id='research-candidate-details' title='候选评分与稳健性' summary={v2Candidates.length + ' 组候选 · 展示最多 ' + topCandidates + ' 组'}>
              <div className='research-summary-states'>{v2Candidates.slice(0, 5).map((candidate, index) => <div key={String(candidate.candidate_id ?? index)}><small>候选 {String(candidate.candidate_id ?? index + 1)} · {candidate.is_center ? '中心' : '邻点'}</small><Badge tone={candidate.recommendation_eligible ? 'green' : candidate.island_suspect ? 'red' : 'amber'}>{candidate.recommendation_eligible ? '稳健性通过' : candidate.island_suspect ? '疑似收益孤岛' : candidate.robustness_status === 'failed' ? '邻域未通过' : '未评价'}</Badge><strong>正邻点 {percent(candidate.neighbor_positive_ratio as number | undefined)} · 第25百分位 {percent(candidate.neighbor_q25_excess as number | undefined)}</strong></div>)}</div>
              <p className='field-hint'>得分、分窗表现、邻域压力和成本共同展示；约束与统计证据分别判断。展示上限不减少实际验证数。</p>
              {v2Candidates.length ? <div className='table-wrap scroll-table'><table><thead><tr><th>综合得分</th><th>原始得分</th><th>约束</th><th>成本得分</th><th>各窗表现</th><th>邻域压力</th><th>参数</th></tr></thead><tbody>{v2Candidates.slice(0, topCandidates).map((candidate, index) => {
                const eligible = eligibleOf(candidate); const folds = candidate.folds ?? candidate.windows; const stress = candidate.stress_folds ?? candidate.neighbor_scores ?? candidate.neighbor_count; const costs = candidate.cost_scores ?? candidate.cost_score ?? candidate.cost_adjusted_score
                return <tr key={String(candidate.id ?? index)}><td>{valueText('selection_score', candidate.selection_score ?? candidate.score)}</td><td>{valueText('raw_score', candidate.raw_score)}</td><td><Badge tone={eligible === true ? 'green' : eligible === false ? 'amber' : 'neutral'}>{eligible === true ? '满足' : eligible === false ? '未满足' : '—'}</Badge><small>{reasonList(candidate.constraint_reasons).join('；')}</small></td><td><details><summary>{shortSummary(costs)}</summary><EvidenceFields value={costs} /></details></td><td><details><summary>{shortSummary(folds)}</summary><pre className='json-preview'>{JSON.stringify(folds ?? {}, null, 2)}</pre></details></td><td><details><summary>{shortSummary(stress)}</summary><pre className='json-preview'>{JSON.stringify(stress ?? {}, null, 2)}</pre></details></td><td><ParameterBadges params={paramsOf(candidate)} compact /></td></tr>
              })}</tbody></table></div> : <Empty title='暂无候选明细' />}
            </Disclosure>
          </> : <>
            <div className='research-legacy-banner'>这是旧版 {actualMode ?? '研究'} 方法的结果。旧版历史最优或切分验证可查看和载入，但不等于科学研究 v2 的跨窗与邻域验证通过。</div>
            <Card title='旧版历史最优候选' subtitle='搜索区间内得分最高的历史候选。' action={<Button variant='secondary' onClick={() => loadLegacy('historical')} disabled={!result.historical?.best_params || !originalBase}>载入旧版历史参数</Button>}>
              {!result.historical ? <Empty title='本次任务未运行旧版历史优化' /> : <><div className='result-meta'><Badge tone='amber'>{statusText(result.historical.status)}</Badge></div><ParameterBadges params={result.historical.best_params} /><MetricStrip title='历史网格策略' metrics={result.historical.best_result?.strategies?.grid?.metrics as Record<string, number> | undefined} /></>}
            </Card>
            <Card title='旧版稳健验证' subtitle='保留训练、验证、测试切分，不作为新协议的统计结论。' action={<Button variant='secondary' onClick={() => loadLegacy('robust')} disabled={!robust?.selected_params || !originalBase}>载入旧版验证参数</Button>}>
              {!robust ? <Empty title='本次任务未运行旧版稳健验证' /> : <><div className='result-meta'><Badge tone='amber'>{statusText(robust.status)}</Badge>{robust.reason && <span>{robust.reason}</span>}</div><ParameterBadges params={robust.selected_params} /><MetricStrip title='测试段网格策略' metrics={robust.test_metrics} /></>}
            </Card>
            <Disclosure id='research-legacy-result-details' title='旧版候选与切分明细' summary={'历史候选 ' + historyCandidates.length + ' · 稳健候选 ' + robustCandidates.length}>
              {robust && <><EvidenceFields value={robust.split} /><MetricStrip title='训练段' metrics={robust.train_metrics} /><MetricStrip title='验证段' metrics={robust.validation_metrics} /><StrategyComparison run={runOf(robust.test_result)} /></>}
              <ChartView option={legacyChartOption} height={350} />
              <div className='table-wrap scroll-table'><table><thead><tr><th>类别</th><th>总收益</th><th>最大回撤</th><th>夏普</th><th>参数</th></tr></thead><tbody>{[...historyCandidates.map(candidate => ['旧版历史', candidate] as const), ...robustCandidates.map(candidate => ['旧版稳健', candidate] as const)].map(([group, candidate], index) => { const metrics = getMetrics(candidate, group === '旧版稳健'); return <tr key={index}><td>{group}</td><td>{percent(n(metrics.total_return ?? metrics.return))}</td><td>{percent(n(metrics.max_drawdown ?? metrics.drawdown))}</td><td>{n(metrics.sharpe).toFixed(2)}</td><td><ParameterBadges params={paramsOf(candidate)} compact /></td></tr> })}</tbody></table></div>
            </Disclosure>
          </>}
          <Disclosure id='research-sample-data-quality' title='样本、数据质量与实际设置' summary={(finite(minuteCoverage) ? '准备区间分钟覆盖 ' + percent(minuteCoverage) + ' · ' : '') + '数据版本 ' + (result.data_version ?? '—')}>
            <div className='research-period-grid'><div><small>训练样本</small><strong>{trainStart && trainEnd ? trainStart + ' 至 ' + trainEnd : isV2Result ? '评估开始日前 504 个交易日' : '旧版切分'}</strong></div><div><small>样本外评估</small><strong>{evalStart ?? '—'} 至 {evalEnd ?? '最新可用日期'}</strong></div></div>
            <div className='result-meta'><span>基础试验数{isV2Result ? ' / 决策点' : ''}：{actualTrials ?? '—'}</span><span>随机种子：{actualSeed ?? '—'}</span>{isV2Result && <span>Bootstrap：{valueText('bootstrap_reps', protocol.bootstrap_reps ?? resultConfig?.bootstrap_reps ?? bootstrapReps)} 次</span>}</div>
            <DataDiagnosticsPanel diagnostics={result.diagnostics} sourceEvents={result.source_events} warnings={researchWarnings} syntheticBars={result.quality?.synthetic_bars} />
            <div className='param-block'><strong>本次实际搜索范围</strong>{savedRanges}</div>
            {isV2Result && <div className='param-block'><strong>锁定账户资金</strong><EvidenceFields value={lockedFunding} /></div>}
          </Disclosure>
          {isV2Result && <>
            <Disclosure id='research-computation-details' title='实际计算量' summary={shortSummary(result.computation)}><p className='field-hint'>记录真实评估次数和耗时，包括邻域压力测试的额外计算。</p><EvidenceFields value={result.computation} /></Disclosure>
            <Disclosure id='research-local-audit' title='重复研究记录与选择偏差' summary={localAudit ? '本地同窗重复 ' + valueText('exact_repeat_prior_studies', localAudit.exact_repeat_prior_studies) + ' 次' : '保留整次尝试记录'}>
              {localAudit && <EvidenceFields value={{ overlapping_prior_studies: localAudit.overlapping_prior_studies, exact_repeat_prior_studies: localAudit.exact_repeat_prior_studies }} />}
              <p className='research-repeat-note'>反复更改区间或参数后只保留最好一次，会增加选择偏差。请保留整次尝试记录；将方案用于新日期时，使用新的样本外区间重新核验。</p>
              <p className='inline-warning'>本地计数只覆盖已记录研究，不能证明研究者从未查看过这些行情。</p>
            </Disclosure>
          </>}
        </>}
      </main>
    </div>
  </>
}
