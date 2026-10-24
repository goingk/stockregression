import { researchQuantityUnit } from '../workbenchDefaults'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Download, FlaskConical, RefreshCw, RotateCcw } from 'lucide-react'
import { ApiError, download, get, post } from '../api'
import { LatestRequest, PromiseCache } from '../requestState'
import { resultsPageQuery } from '../resultsReview'
import ResearchLandscape from '../ResearchLandscape'
import DataDiagnosticsPanel from '../DataDiagnosticsPanel'
import Disclosure from '../Disclosure'
import { useTasks } from '../TaskCenter'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, Field, Loading, NumField, PageHeader, SelectInput, TextInput, configError, configPayload, loadConfig, percent, useErrorMessage } from '../components'
import { flushConfiguration } from '../useWorkbenchDefaults'
import type { BacktestConfig, BacktestResult, DataDiagnostics, DataPreparationRequest, DataSourceEvent, Job } from '../types'
import { defaultSearchRanges, percentageRanges as pctRange, researchConfigForSubmission, researchRangesForSymbol, validResearchDefaults } from '../workbenchDefaults'
import type { ResearchDefaults, SearchRange as Range } from '../workbenchDefaults'

type Dict = Record<string, unknown>
type SearchConfig = { base?: BacktestConfig; protocol_version?: number; seed?: number; ranges?: Dict; [key: string]: unknown }
type TestRun = Partial<BacktestResult> & { strategies?: Record<string, { metrics?: Record<string, number | null> }> }
type RecommendationRegion = { region_id?: number; center_params?: Dict; edge_length?: number; geometry?: string; scoring_basis?: string; normalized_volume?: number; combination_count?: number; completed_count?: number; center_excess_return?: number | null; mean_excess_return?: number | null; min_excess_return?: number | null; j_score?: number | null; downside_risk_tau?: number | null; pos_penalty?: number | null; actual_min_pos_ratio?: number | null; actual_max_pos_ratio?: number | null; neighborhood_max_eta?: number | null; pos_win_rate?: number | null; safe_win_rate?: number | null; stockout_count?: number | null; tplus1_block_count?: number | null; floor_episode_count?: number | null; floor_episode_grids?: number[]; floor_penalty?: number | null; floor_penalty_basis?: string; platform_size?: number | null; score_complete?: boolean; local_peak_verified?: boolean; complete?: boolean; maximum_search_complete?: boolean; global_exhaustive?: boolean; stress?: Dict }
type Recommendation = { params?: Dict | null; recommended_params?: Dict | null; diagnostic_params?: Dict | null; as_of?: string; score?: unknown; status?: string; reasons?: unknown; diagnostics?: Dict; recommendation_status?: string; recommendation_regions?: RecommendationRegion[] }
type CompletionScope = { revision?: number; decision?: string | null; decision_index?: number }
type CompletionState = { status?: string; latest_revision?: Dict | null; selected_revision?: number | null; revision?: Dict | null; revision_scope?: CompletionScope | null; endpoint?: string }
type OptimizationResult = {
  id?: string
  protocol_version?: number; result_schema_version?: number; seed?: number; data_version?: string; warnings?: string[]
  config?: SearchConfig; protocol?: Dict; is_demo?: boolean; quality?: Dict; diagnostics?: DataDiagnostics; source_events?: DataSourceEvent[]
  computation?: Dict; locked_funding?: Dict; configuration_snapshot?: Dict; execution?: Dict; performance?: Dict
  fixed?: { status?: string; params?: Dict; as_of?: string; test_result?: TestRun; current?: Recommendation; diagnostics?: Dict; oos_metrics?: Dict; [key: string]: unknown }
  adaptive?: { status?: string; schedule?: Array<Dict & { effective_date?: string; decision_date?: string; params?: Dict }>; test_result?: TestRun; latest_params?: Dict; latest_as_of?: string; oos_metrics?: Dict; [key: string]: unknown }
  evidence?: Dict & { decisions?: Array<Dict> }
  completion?: CompletionState
}
export type Props = {
  base: BacktestConfig
  settings: ResearchDefaults
  onSettingsChange: (settings: ResearchDefaults) => void
  onBaseChange: (config: BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)) => void
  onLoadConfig: (config: BacktestConfig) => void
  requestedResource?: string | null
  onSelectedResourceChange?: (resource: string) => void
  completionDecision?: string | null
  onCompletionDecisionChange?: (decision: string) => void
  onFeeDefaultsEdit?: (defaults: { commission_rate: number; min_commission: number }) => void
  onDefaultsEdit?: (group: 'grid' | 'position' | 'dca', config: BacktestConfig) => void
  onRestore?: () => void
  onPrepareData?: (request: DataPreparationRequest) => void
}

const robustFields = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'] as const
const labels: Record<string, string> = { buy_step: '买入间距', sell_step: '卖出间距', buy_shares: '每次买入股数', sell_shares: '每次卖出股数' }
const candidateLabels: Record<string, string> = {
  peak_principal_return: '峰值本金收益', principal_drawdown: '本金回撤', max_principal_drawdown: '本金回撤',
  matched_funding_hold_return: '同资金被动持有收益', matched_funding_hold_drawdown: '同资金被动持有回撤',
  excess_return: '差值超额', trade_count: '成交笔数', total_fees: '总手续费', turnover: '换手率',
  buy_count: '买入次数', sell_count: '卖出次数', total_invested: '累计投入', external_topups: '外部补款',
  total_profit: '累计盈利', j_score: '稳健 J', neighborhood_mean: '邻域均值', downside_risk_tau: '下行风险',
  pos_penalty: '持仓惩罚', neighborhood_size: '邻域点数', neighborhood_support_size: '邻域支持点数',
  pos_win_rate: '正收益占比', safe_win_rate: '安全垫占比', score_complete: '评分完整', constraint_eligible: '约束合格',
  constraint_reasons: '约束说明', actual_min_pos_ratio: '实际最低持仓', actual_max_pos_ratio: '实际最高持仓',
  floor_episode_count: '底仓段数', floor_episode_grids: '底仓各段上涨格数', floor_penalty: '底仓惩罚',
  floor_penalty_basis: '底仓惩罚口径', stockout_count: '库存不足次数', tplus1_block_count: 'T+1 限制次数',
  local_peak_verified: '峰值已核验', platform_size: '平台点数', training_excess_return: '训练超额收益',
  evaluation_status: '评价状态', status: '状态', reason: '说明', reasons: '说明', eligible: '约束合格',
  candidate_count: '候选数', eligible_count: '约束合格数', evaluated_count: '实际评估数', completed_count: '完成组合数',
  combination_count: '组合数', edge_length: '超正方体边长', hypercube_edge: '超正方体边长', decision_date: '决策日期',
  effective_date: '生效日期', as_of: '参数截止日', training_start: '训练开始', training_end: '训练结束',
  training_days: '训练交易日', paired_days: '配对交易日数', method: '方法', coverage_status: '覆盖状态',
  rebalance_days: '换参周期（日）', decision_points: '决策点数', unique_selection_calls: '独立选参调用',
  outer_backtest_runs: '外层回测次数', resamples: '重抽样次数', bootstrap_iterations: '自举抽样次数',
  base_trials_per_decision: '单决策基础试验数', trials_per_decision: '单决策试验数',
  total_evaluations: '总评估次数', unique_evaluations: '唯一参数评估数',
  neighborhood_evaluations: '邻域核验评估数', stress_evaluations: '压力测试评估数',
  elapsed_seconds: '耗时（秒）', wall_time_seconds: '耗时（秒）',
  cpu_budget_mode: 'CPU 预算模式', cpu_budget: 'CPU 线程上限', download_workers: '下载并发数',
  validation_workers: '校验进程数', blas_threads: '矩阵线程数', workspace_mb: '工作内存（MiB）',
  peak_invested: '峰值本金', initial_equity: '初始资产', reference_price: '参考价格',
  reference_date: '估值日期', valuation_price: '估值价格',
  buy_hold_initial_shares: '全仓初始持股', buy_hold_initial_equity: '全仓初始资产',
  dca_initial_equity: '定投初始资产', matched_funding_hold_invested: '同资金被动持有投入',
  configuration_revision: '配置修订号', data_version: '数据版本', protocol_version: '协议版本',
  strategy_version: '策略版本', ranking_mode: '排名模式', search_method: '搜索方法',
  training_min_days: '最短训练日数', training_max_days: '最长训练日数', training_window_mode: '训练窗口模式',
}

function revealDisclosure(id: string) {
  const element = document.querySelector<HTMLDetailsElement>(`details[data-disclosure="${id}"]`)
  if (!element) return
  element.open = true
  requestAnimationFrame(() => element.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}
function record(value: unknown): Dict | undefined { return value && typeof value === 'object' && !Array.isArray(value) ? value as Dict : undefined }
function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
function searchConfigOf(value: unknown): SearchConfig | undefined { return record(value) as SearchConfig | undefined }
function warningsOf(value: unknown): string[] {
  const item = record(value)
  return Array.isArray(item?.warnings) ? item.warnings.map(String) : []
}
function reasonsOf(value: unknown): string[] {
  const item = record(value)
  const diagnostics = record(item?.diagnostics)
  const values = item?.constraint_reasons ?? item?.reasons ?? diagnostics?.constraint_reasons ?? diagnostics?.reasons
  return Array.isArray(values) ? values.map(String).filter(Boolean) : []
}
function hasDemoFlag(value: unknown) {
  const item = record(value)
  return item?.is_demo === true || record(item?.quality)?.is_demo === true || record(item?.metadata)?.is_demo === true
}
function isOptimizationJob(job: Job) {
  const config = record(job.payload?.config)
  return job.kind === 'optimization' || job.kind === 'optimize' || record(config?.base) != null
}
function isCompletionJob(job: Job) { return job.kind === 'research_completion' }
function taskLabelForResearch(job: Job) {
  const config = searchConfigOf(job.payload?.config)
  return `${config?.base?.symbol ?? '研究任务'} · ${job.created_at?.slice(0, 10) ?? job.id.slice(0, 12)} · ${statusText(job.status)}`
}
function rangeValue(key: string, value: unknown) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '—'
  return pctRange.has(key) ? percent(parsed, 3) : String(Number(parsed.toPrecision(6)))
}
function labelOf(key: string) { return candidateLabels[key] ?? labels[key] ?? key.replaceAll('_', ' ') }
function valueText(key: string, value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'string' || typeof value === 'number') {
    const numeric = typeof value === 'number' ? value : Number.NaN
    if (Number.isFinite(numeric) && /return|drawdown|ratio|rate|excess/i.test(key) && Math.abs(numeric) <= 1.5) return percent(numeric)
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
function displayKey(path: string) { return path.split('.').map(labelOf).join(' · ') }
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
  const names: Record<string, string> = { selected: '相对最佳候选', insufficient_history: '历史不足', insufficient_data: '数据不足', no_eligible_candidate: '没有约束合格候选', no_qualified: '暂无合格推荐', completed: '已完成', running: '计算中', unavailable: '不可用', budget_exhausted: '预算已用尽', cancelled: '已取消', partial: '部分完成' }
  return names[value] ?? (value || '未提供')
}
function firstString(item: Dict | undefined, keys: string[]) {
  for (const key of keys) if (typeof item?.[key] === 'string') return item[key] as string
  return undefined
}
function schemeLabel(selectionTarget: string, rebalanceDays: number) {
  return selectionTarget === 'both' ? '固定 + 自适应' : selectionTarget === 'fixed' ? '固定参数' : `每 ${rebalanceDays} 日自适应`
}

function EvidenceFields({ value }: { value?: unknown }) {
  const rows = metricRows(value)
  return rows.length ? <div className='research-evidence-grid'>{rows.map(({ key, value: raw }) => <div key={key}><small>{displayKey(key)}</small><strong>{valueText(key.split('.').at(-1) ?? key, raw)}</strong></div>)}</div> : <Empty title='暂无可展示的字段' />
}

function ParameterBadges({ params, compact = false }: { params?: Dict; compact?: boolean }) {
  const entries = Object.entries(params ?? {})
  if (!entries.length) return <span className='muted'>暂无参数</span>
  return <div className={'research-parameter-badges' + (compact ? ' compact' : '')}>{entries.map(([key, value]) => <span className='research-parameter-badge' key={key}><small>{labelOf(key)}</small><strong>{pctRange.has(key) ? rangeValue(key, value) : record(value) ? shortSummary(value) : valueText(key, value)}</strong></span>)}</div>
}

function constraintLabel(value: boolean | undefined) {
  return value === true ? '满足约束' : value === false ? '未满足约束' : '未提供约束状态'
}

/** The v5 result only reports peak-principal returns; every other return definition is gone. */
function principalMetrics(value: unknown) {
  const metrics = record(value) ?? {}
  const peak = finite(metrics.peak_principal_return) ? metrics.peak_principal_return : null
  const matched = finite(metrics.matched_funding_hold_return) ? metrics.matched_funding_hold_return : null
  const drawdown = finite(metrics.principal_drawdown) ? metrics.principal_drawdown : finite(metrics.max_principal_drawdown) ? metrics.max_principal_drawdown : null
  const excess = peak != null && matched != null ? peak - matched : finite(metrics.excess_return) ? metrics.excess_return : null
  return { peak, matched, drawdown, excess, tradeCount: finite(metrics.trade_count) ? metrics.trade_count : null }
}

function PrincipalMetricRow({ label, metrics }: { label: string; metrics?: unknown }) {
  const values = principalMetrics(metrics)
  return <div className='research-summary-states'>
    <div><small>{label} · 峰值本金收益</small><strong className={values.peak != null ? (values.peak >= 0 ? 'gain' : 'loss') : ''}>{percent(values.peak)}</strong></div>
    <div><small>最大本金回撤</small><strong className={values.drawdown != null && values.drawdown > 0 ? 'loss' : ''}>{percent(values.drawdown)}</strong></div>
    <div><small>同资金被动持有收益</small><strong className={values.matched != null ? (values.matched >= 0 ? 'gain' : 'loss') : ''}>{percent(values.matched)}</strong></div>
    <div><small>差值超额</small><strong className={values.excess != null ? (values.excess >= 0 ? 'gain' : 'loss') : ''}>{percent(values.excess)}</strong></div>
    <div><small>成交笔数</small><strong>{values.tradeCount ?? '—'}</strong></div>
  </div>
}

type PreflightState = {
  status: 'idle' | 'loading' | 'ready' | 'blocked'
  preview: Dict | null
  message: string
}

export default function ResearchPage({ base, settings, onSettingsChange, onBaseChange, onLoadConfig, requestedResource, onSelectedResourceChange, completionDecision, onCompletionDecisionChange, onFeeDefaultsEdit, onDefaultsEdit, onRestore, onPrepareData }: Props) {
  const { protocol_version: protocolVersion, selection_target: selectionTarget, rebalance_days: rebalanceDays, seed, top_candidates: topCandidates, max_principal_drawdown: maxPrincipalDrawdown, min_trades: minTrades, min_history_days: minHistoryDays, min_minute_ratio: minMinuteRatio, strategy_version: strategyVersion, ranking_mode: rankingMode, hypercube_edge: hypercubeEdge, granularity, search_method: searchMethod, training_budget: trainingBudget, validation_budget: validationBudget, reporting_budget: reportingBudget, total_evaluation_budget: totalEvaluationBudget, max_runtime_seconds: maxRuntimeSeconds, recommendation_count: recommendationCount, slippage_mode: slippageMode, fixed_half_widths: fixedHalfWidths, sigma, alpha, beta: scoreBeta, tau, eta_limit: etaLimit, eta_hard_cap: etaHardCap, gamma_max: gammaMax, eta_floor: etaFloor, gamma_min: gammaMin, floor_k_ref: floorKRef, training_min_days: trainingMinDays, training_max_days: trainingMaxDays, training_window_mode: trainingWindowMode, peak_tolerance: peakTolerance, start_min: startMin, start_max: startMax, starts_per_candidate: startsPerCandidate, initial_strides: initialStrides, stride_multiplier: strideMultiplier, max_stride_fraction: maxStrideFraction, validation_reserve_fraction: validationReserveFraction, exhaustive_memory_limit_mb: exhaustiveMemoryLimitMb, estimated_bytes_per_point: estimatedBytesPerPoint, stress_cost_multiplier: stressCostMultiplier, stress_region_count: stressRegionCount } = settings
  const tasks = useTasks()
  const researchJobs = useMemo(() => tasks.jobs.filter(isOptimizationJob), [tasks.jobs])
  const ranges = useMemo(() => researchRangesForSymbol(settings.ranges, base.symbol), [base.symbol, settings.ranges])
  const updateSettings = (patch: Partial<ResearchDefaults>) => onSettingsChange({ ...settings, ...patch })
  const [selectedJobId, setSelectedJobId] = useState('')
  const [selectedResultId, setSelectedResultId] = useState('')
  const [researchResults, setResearchResults] = useState<Array<{ id: string; symbol: string; created_at: string; title: string; kind: string }>>([])
  const [researchResultsLoading, setResearchResultsLoading] = useState(true)
  const [researchResultsError, setResearchResultsError] = useState('')
  const [researchResultsMore, setResearchResultsMore] = useState(false)
  const [researchResultsOffset, setResearchResultsOffset] = useState(0)
  const [fallbackJob, setFallbackJob] = useState<Job | null>(null)
  const [loadedResult, setLoadedResult] = useState<{ key: string; value: OptimizationResult } | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [preflight, setPreflight] = useState<PreflightState>({ status: 'idle', preview: null, message: '' })
  const [preflightToken, setPreflightToken] = useState(0)
  const [error, setError] = useState('')
  const [retryable, setRetryable] = useState(false)
  const [retryToken, setRetryToken] = useState(0)
  const [dataIssue, setDataIssue] = useState<DataPreparationRequest | null>(null)
  const [completion, setCompletion] = useState<CompletionState | null>(null)
  const [completionBusy, setCompletionBusy] = useState(false)
  const [completionNote, setCompletionNote] = useState('')
  const [completionToken, setCompletionToken] = useState(0)
  const jobRequests = useRef(new PromiseCache<string, Job>())
  const resultRequests = useRef(new PromiseCache<string, OptimizationResult>())
  const resultsRequest = useRef(new LatestRequest<string>())
  const preflightRequest = useRef(new LatestRequest<string>())
  const err = useErrorMessage()
  // Errors that are not a failed job or result read cannot be retried by reloading the task.
  const reportError = useCallback((message: string) => { setRetryable(false); setError(message) }, [])
  useEffect(() => { setPreflight({ status: 'idle', preview: null, message: '' }); setError(''); setRetryable(false) }, [base, settings])
  const isStarBoard = base.symbol.startsWith('688') || base.symbol.startsWith('689')
  const selectedJob = researchJobs.find(job => job.id === selectedJobId) ?? (fallbackJob?.id === selectedJobId ? fallbackJob : null)
  const selectedResultKey = selectedResultId
    ? 'result:' + selectedResultId
    : selectedJob?.result_id && ['completed', 'cancelled', 'partial'].includes(selectedJob.status) ? 'job:' + selectedJob.id + ':' + selectedJob.result_id : ''
  const result = loadedResult?.key === selectedResultKey ? loadedResult.value : null

  const selectResult = (id: string) => {
    setSelectedResultId(id)
    setSelectedJobId('')
    setFallbackJob(null)
    onSelectedResourceChange?.(id)
  }
  const selectJob = (id: string) => {
    setSelectedJobId(id)
    setSelectedResultId('')
    onSelectedResourceChange?.(id)
  }

  const loadResearchResults = useCallback(async (offset = 0, append = false) => {
    const ticket = resultsRequest.current.begin(`optimization:${offset}:${append}`)
    setResearchResultsLoading(true); setResearchResultsError('')
    try {
      const page = await get<Array<{ id: string; symbol: string; created_at: string; title: string; kind: string }>>(
        resultsPageQuery(offset, 50, 'optimization'),
      )
      if (!resultsRequest.current.isCurrent(ticket)) return
      setResearchResults(current => append
        ? [...current, ...page.filter(item => !current.some(existing => existing.id === item.id))]
        : page)
      setResearchResultsOffset(offset)
      setResearchResultsMore(page.length >= 50)
      if (!requestedResource && !selectedResultId && !selectedJobId && page[0]) selectResult(page[0].id)
    } catch (cause) {
      if (resultsRequest.current.isCurrent(ticket)) setResearchResultsError(err(cause))
    } finally {
      if (resultsRequest.current.isCurrent(ticket)) setResearchResultsLoading(false)
    }
  }, [err, requestedResource, selectedJobId, selectedResultId])

  useEffect(() => { void loadResearchResults(0, false) }, [])
  const completedResearchJobs = tasks.jobs.filter(job => isOptimizationJob(job) && ['completed', 'partial', 'failed', 'cancelled'].includes(job.status)).map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => {
    if (completedResearchJobs) void loadResearchResults(0, false)
  }, [completedResearchJobs])

  // The URL may carry either a saved result id or a job id. Results are
  // immutable and preferred; a 404 then resolves the id as a task.
  useEffect(() => {
    if (!requestedResource) return
    if (requestedResource === selectedResultId || requestedResource === selectedJobId) return
    let active = true
    const resultRequest = get<OptimizationResult>('/results/' + encodeURIComponent(requestedResource))
    void resultRequest.then(value => {
      if (!active) return
      const key = 'result:' + requestedResource
      resultRequests.current.getOrCreate(key, () => Promise.resolve(value))
      setSelectedResultId(requestedResource); setSelectedJobId(''); setFallbackJob(null)
      setLoadedResult({ key, value }); setResultLoading(false); setError(''); setRetryable(false)
    }).catch(resultCause => {
      if (!active) return
      if (!(resultCause instanceof ApiError) || resultCause.status !== 404) {
        setRetryable(true); setError(err(resultCause)); return
      }
      const jobRequest = jobRequests.current.getOrCreate(
        requestedResource,
        () => get<Job>('/jobs/' + encodeURIComponent(requestedResource)),
      )
      void jobRequest.then(job => {
        if (!active) return
        if (!isOptimizationJob(job) && !isCompletionJob(job)) {
          setRetryable(false); setError('链接中的任务不是参数研究。')
          return
        }
        setSelectedJobId(requestedResource); setSelectedResultId(''); setFallbackJob(job); setError('')
      }).catch(jobCause => {
        jobRequests.current.forget(requestedResource, jobRequest)
        if (!active) return
        setRetryable(true)
        setError(jobCause instanceof ApiError && jobCause.status === 404
          ? '链接中的研究结果或任务已被清理，或不存在。'
          : err(jobCause))
      })
    })
    return () => { active = false }
  }, [requestedResource, selectedResultId, selectedJobId, err])

  // A result opened from outside this page may predate the task center's current list.
  useEffect(() => {
    if (!selectedJobId || researchJobs.some(job => job.id === selectedJobId)) return
    let active = true
    const request = jobRequests.current.getOrCreate(
      selectedJobId,
      () => get<Job>('/jobs/' + encodeURIComponent(selectedJobId)),
    )
    void request.then(job => {
      if (!active) return
      if (!isOptimizationJob(job) && !isCompletionJob(job)) { reportError('所选任务不是参数研究。'); return }
      setFallbackJob(job)
    }).catch(cause => {
      // A rejected request is dropped so a retry reads the task again. A newer request
      // for the same id keeps its own entry.
      jobRequests.current.forget(selectedJobId, request)
      if (active) { setRetryable(true); setError(err(cause)) }
    })
    return () => { active = false }
  }, [selectedJobId, researchJobs, err, retryToken, reportError])

  // Cache by the immutable result id so job polling never refetches a completed result.
  useEffect(() => {
    if (!selectedResultKey) { setResultLoading(false); return }
    let active = true
    setResultLoading(true)
    const request = resultRequests.current.getOrCreate(
      selectedResultKey,
      () => selectedResultId
        ? get<OptimizationResult>('/results/' + encodeURIComponent(selectedResultId))
        : get<OptimizationResult>('/jobs/' + encodeURIComponent(selectedJobId) + '/result'),
    )
    void request.then(value => { if (active) setLoadedResult({ key: selectedResultKey, value }) })
      .catch(cause => {
        resultRequests.current.forget(selectedResultKey, request)
        if (active) {
          setRetryable(true)
          setError(cause instanceof ApiError && cause.status === 404
            ? '这条研究结果已经被清理或不存在。'
            : err(cause))
        }
      })
      .finally(() => { if (active) setResultLoading(false) })
    return () => { active = false }
  }, [selectedResultKey, selectedJobId, selectedResultId, err, retryToken])

  const resultId = selectedResultId || selectedJob?.result_id || String(result?.id ?? '')
  const jobResultCompletion = result?.completion
  const resultProtocolVersion = Number(result?.protocol_version ?? record(result?.config)?.protocol_version ?? 0)
  const resultSchemaVersion = Number(result?.result_schema_version ?? 0)
  const isSupportedResult = resultProtocolVersion === 5 && resultSchemaVersion === 2

  // Completion revisions are only published on the result resource itself.
  useEffect(() => {
    if (!resultId || !isSupportedResult) { setCompletion(null); return }
    // A completion job owns one immutable revision.  The base result endpoint
    // correctly defaults to the newest revision, but using that default here
    // would make an older completion job silently display newer work.
    if (selectedJob && isCompletionJob(selectedJob) && jobResultCompletion) {
      setCompletion(jobResultCompletion)
      return
    }
    let active = true
    get<{ completion?: CompletionState }>('/results/' + encodeURIComponent(resultId))
      .then(value => { if (active) setCompletion(value.completion ?? null) })
      .catch(() => { if (active) setCompletion(null) })
    return () => { active = false }
  }, [resultId, isSupportedResult, selectedJob, jobResultCompletion, completionToken])

  const updateRange = (name: string, key: keyof Range, shown: number) => {
    updateSettings({ ranges: { ...settings.ranges, [name]: { ...settings.ranges[name], [key]: shown } } })
    setPreflight({ status: 'idle', preview: null, message: '' })
  }
  const researchValidationError = () => {
    const researchBase = researchConfigForSubmission(base)
    const baseError = configError(researchBase)
    if (baseError) return baseError
    if (!validResearchDefaults(settings)) return '请填写有效研究设置；搜索上限应不小于下限；研究细粒度须为参数最小单位（间距 0.01 个百分点、股数 100 股 / 份）的正整数倍；超正方体边长须为正奇数。'
    const minimumBuyShares = isStarBoard ? 200 : 100
    if (ranges.buy_shares.low < minimumBuyShares || ranges.buy_shares.high < minimumBuyShares) return '每次买入搜索范围下限应至少为 ' + minimumBuyShares + ' 股 / 份。'
    return ''
  }
  const buildResearchRequest = useCallback((configurationRevision?: string) => {
    const researchBase = researchConfigForSubmission(base)
    const submittedRanges = Object.fromEntries(Object.entries(ranges).map(([name, value]) => [name, { low: value.low, high: value.high }]))
    return {
      base: configPayload(researchBase),
      protocol_version: 5,
      selection_target: selectionTarget,
      rebalance_days: rebalanceDays,
      seed,
      top_candidates: topCandidates,
      max_principal_drawdown: maxPrincipalDrawdown,
      min_trades: minTrades,
      min_history_days: minHistoryDays,
      min_minute_ratio: minMinuteRatio,
      ranges: submittedRanges,
      strategy: {
        strategy_id: 'geometric_robust_j', strategy_version: 5, ranking_mode: 'robust_j',
        search_method: searchMethod, hypercube_edge: hypercubeEdge, granularity,
        training_budget: trainingBudget, validation_budget: validationBudget,
        reporting_budget: reportingBudget, total_evaluation_budget: totalEvaluationBudget,
        max_runtime_seconds: maxRuntimeSeconds, recommendation_count: recommendationCount,
        max_shares_mode: 'unlimited', slippage_mode: slippageMode,
        neighborhood_fields: [...robustFields], fixed_half_widths: fixedHalfWidths,
        sigma, alpha, beta: scoreBeta, tau,
        eta_limit: etaLimit, eta_hard_cap: etaHardCap, gamma_max: gammaMax,
        eta_floor: etaFloor, gamma_min: gammaMin, floor_k_ref: floorKRef,
        training_min_days: trainingMinDays, training_max_days: trainingMaxDays, training_window_mode: trainingWindowMode,
        peak_tolerance: peakTolerance, start_min: startMin, start_max: startMax, starts_per_candidate: startsPerCandidate,
        initial_strides: initialStrides, stride_multiplier: strideMultiplier, max_stride_fraction: maxStrideFraction,
        validation_reserve_fraction: validationReserveFraction, exhaustive_memory_limit_mb: exhaustiveMemoryLimitMb,
        estimated_bytes_per_point: estimatedBytesPerPoint,
        stress_cost_multiplier: stressCostMultiplier, stress_region_count: stressRegionCount,
      },
      ...(configurationRevision ? { configuration_revision: configurationRevision } : {}),
    }
  }, [base, settings, ranges])

  useEffect(() => {
    const validation = researchValidationError()
    if (!base.symbol) {
      setPreflight({ status: 'idle', preview: null, message: '选择标的并填写有效参数后自动预检。' })
      return
    }
    if (validation) {
      setPreflight({ status: 'blocked', preview: null, message: validation })
      return
    }
    const ticket = preflightRequest.current.begin(JSON.stringify(buildResearchRequest()))
    setPreflight({ status: 'loading', preview: null, message: '正在读取训练时段、预算与内存约束…' })
    const timer = window.setTimeout(() => {
      post<Dict>('/optimizations/preview', buildResearchRequest()).then(preview => {
        if (!preflightRequest.current.isCurrent(ticket)) return
        const reasons = Array.isArray(preview.reasons) ? preview.reasons.map(String).filter(Boolean) : []
        setPreflight({
          status: String(preview.status) === 'ready' ? 'ready' : 'blocked',
          preview,
          message: String(preview.status) === 'ready' ? '' : reasons.join('；') || '搜索预检未通过。',
        })
        if (String(preview.status) === 'ready') setDataIssue(null)
        if (String(preview.status) === 'ready') setDataIssue(null)
      }).catch(cause => {
        if (!preflightRequest.current.isCurrent(ticket)) return
        const detail = cause instanceof ApiError ? record(cause.detail) ?? {} : {}
        if (cause instanceof ApiError && cause.status === 409 && detail.code === 'DATA_NOT_READY') setDataIssue(detail as DataPreparationRequest)
        setPreflight({
          status: 'blocked', preview: Object.keys(detail).length ? detail : null,
          message: String(detail.message ?? err(cause)),
        })
      })
    }, 600)
    return () => { window.clearTimeout(timer); preflightRequest.current.invalidate(ticket.key) }
  }, [base, settings, ranges, buildResearchRequest, err, preflightToken])

  const submit = async () => {
    const validation = researchValidationError()
    if (validation) {
      reportError(validation); revealDisclosure('research-search-ranges'); return
    }
    if (preflight.status !== 'ready') { reportError(preflight.message || '请等待搜索预检完成并解决阻断项。'); return }
    setBusy(true); setError(''); setRetryable(false); setDataIssue(null)
    try {
      const configurationRevision = await flushConfiguration()
      const request = buildResearchRequest(configurationRevision)
      const response = await tasks.run('参数研究', () => post<{ job_id: string }>('/optimizations', request))
      selectJob(response.job_id)
      void loadResearchResults(0, false)
    } catch (cause) {
      reportError(err(cause))
      if (cause instanceof ApiError && cause.status === 409 && cause.detail && typeof cause.detail === 'object' && (cause.detail as Record<string, unknown>).code === 'DATA_NOT_READY') setDataIssue(cause.detail as DataPreparationRequest)
      if (cause instanceof ApiError && cause.status === 422) {
        const detail = record(cause.detail) ?? {}
        setPreflight({ status: 'blocked', preview: detail, message: String(detail.message ?? err(cause)) })
      }
    } finally { setBusy(false) }
  }

  const loadV2 = (params: Dict | undefined, selectionAsOf: string | undefined) => {
    if (!result || !params) { reportError('当前研究结果没有可载入的参数。'); return }
    const originalBase = result.config?.base ?? searchConfigOf(selectedJob?.payload?.config)?.base
    if (!originalBase) { reportError('这条研究任务未保存原始股票与区间配置，无法安全载入。请重新运行参数研究。'); return }
    const locked = record(result.locked_funding) ?? {}
    const frozenStrategy = record(result.config?.strategy) ?? record(record(selectedJob?.payload?.config)?.strategy)
    const { buy_ratio: _oldBuyRatio, sell_ratio: _oldSellRatio, ...configParams } = params
    const patch = configParams as Partial<BacktestConfig>
    const config: BacktestConfig = {
      ...originalBase,
      ...patch,
      grid_mode: 'geometric',
      initial_shares: finite(locked.initial_shares) ? locked.initial_shares : originalBase.initial_shares,
      selection_asof: selectionAsOf ?? originalBase.end_date ?? originalBase.start_date,
      floor_penalty_config: frozenStrategy && finite(frozenStrategy.eta_floor) && finite(frozenStrategy.gamma_min) && finite(frozenStrategy.floor_k_ref)
        ? { eta_floor: frozenStrategy.eta_floor, gamma_min: frozenStrategy.gamma_min, floor_k_ref: frozenStrategy.floor_k_ref }
        : null,
      fees: { ...originalBase.fees, ...patch.fees },
    }
    try { onLoadConfig(loadConfig(config).config) } catch (cause) { reportError(err(cause)) }
  }

  const submitCompletion = async () => {
    if (!resultId) return
    const rawDecision = completionDecision || 'current:0'
    const [decisionKind, decisionIndexText] = rawDecision.split(':')
    const decisionIndex = Math.max(0, Number(decisionIndexText || 0) || 0)
    if (!['current', 'fixed', 'adaptive'].includes(decisionKind)) {
      reportError('请选择要补算的研究决策。')
      return
    }
    setCompletionBusy(true); setCompletionNote(''); setError(''); setRetryable(false)
    try {
      const response = await post<{ job_id: string; status?: string; deduplicated?: boolean }>(
        '/results/' + encodeURIComponent(resultId) + '/completion',
        { decision: decisionKind, decision_index: decisionIndex, reporting_budget: reportingBudget, target: 'return_leader' },
      )
      setCompletionNote(response.deduplicated
        ? '相同补算已在队列中；已切换到现有任务，完成后刷新补算状态即可看到新的修订号。'
        : '补算任务已提交（' + String(response.job_id ?? '').slice(0, 12) + '），可在任务中心查看进度。')
      selectJob(response.job_id)
      if (response.deduplicated) tasks.showHistory()
      await tasks.refresh()
    } catch (cause) { reportError(err(cause)) } finally { setCompletionBusy(false) }
  }

  const exportFile = async (kind: 'json' | 'candidates' | 'equity' | 'trades') => {
    if (!resultId) return
    const query = new URLSearchParams({ kind })
    // The revision is a number: the API exposes the full revision record
    // under "latest_revision" and the selected revision number under
    // "selected_revision".
    const revision = completion?.selected_revision ?? null
    if (revision != null) query.set('revision', String(revision))
    setError(''); setRetryable(false)
    try {
      await download('/results/' + encodeURIComponent(resultId) + '/export?' + query.toString(), resultId + '-' + kind + (kind === 'json' ? '.json' : '.csv'))
    } catch (cause) { reportError(err(cause)) }
  }

  const resultConfig = result?.config
  const taskConfig = searchConfigOf(selectedJob?.payload?.config)
  const originalBase = resultConfig?.base ?? taskConfig?.base
  const actualRanges = record(resultConfig?.ranges) ?? record(taskConfig?.ranges)
  const actualSeed = result?.seed ?? resultConfig?.seed ?? taskConfig?.seed
  const researchWarnings = Array.from(new Set(warningsOf(result)))
  const minuteCoverage = result?.diagnostics?.minute_ratio ?? record(result?.evidence)?.minute_coverage ?? record(result?.quality)?.minute_ratio
  const isDemoResearch = [result, result?.quality, result?.fixed?.test_result, result?.adaptive?.test_result].some(hasDemoFlag) || researchWarnings.some(warning => /演示|demo/i.test(warning))
  const protocol = record(result?.protocol) ?? {}
  const protocolStatistics = record(protocol.statistics) ?? {}
  const protocolTraining = record(protocol.training) ?? {}
  const protocolOutOfSample = record(protocol.out_of_sample) ?? {}
  const fixed = result?.fixed
  const adaptive = result?.adaptive
  const current = fixed?.current
  const explicitlyUnqualified = current?.recommendation_status === 'no_qualified' || current?.status === 'no_qualified'
  const currentRecommended = current?.recommended_params ?? (explicitlyUnqualified ? undefined : current?.params ?? undefined)
  const relativeBest = current?.diagnostic_params ?? (currentRecommended ? undefined : current?.params ?? undefined)
  const lockedFunding = record(result?.locked_funding) ?? {}
  const trainStart = firstString(protocol, ['train_start', 'training_start', 'training_start_date']) ?? firstString(protocolTraining, ['start', 'start_date', 'from'])
  const trainEnd = firstString(protocol, ['train_end', 'training_end', 'training_end_date']) ?? firstString(protocolTraining, ['end', 'end_date', 'to'])
  const evalStart = firstString(protocol, ['evaluation_start', 'eval_start', 'test_start']) ?? originalBase?.start_date
  const evalEnd = firstString(protocol, ['evaluation_end', 'eval_end', 'test_end']) ?? originalBase?.end_date ?? undefined
  const currentRecommendationStatus = String(current?.recommendation_status ?? (currentRecommended ? 'qualified' : 'no_qualified'))
  const fixedMetrics = record(fixed?.oos_metrics) ?? record(fixed?.test_result?.strategies?.grid?.metrics)
  const adaptiveMetrics = record(adaptive?.oos_metrics) ?? record(adaptive?.test_result?.strategies?.grid?.metrics)
  const landscapeDecisions = [
    { value: 'current:0', label: '评估起点建议的训练候选' },
    ...(adaptive?.schedule ?? []).map((item, index) => ({ value: `adaptive:${index}`, label: `自适应 ${item.effective_date ?? index + 1} 的训练候选` })),
    ...(fixed?.as_of ? [{ value: 'fixed:0', label: '固定方案决策的训练候选' }] : []),
  ]
  const completionDecisionValue = landscapeDecisions.some(item => item.value === completionDecision)
    ? completionDecision || 'current:0'
    : landscapeDecisions[0]?.value ?? 'current:0'
  const [completionDecisionKind, completionDecisionIndexText] = completionDecisionValue.split(':')
  const completionDecisionIndex = Math.max(0, Number(completionDecisionIndexText || 0) || 0)
  const completionDecisionBranch = completionDecisionKind === 'adaptive'
    ? record(result?.evidence?.decisions?.[completionDecisionIndex])
    : completionDecisionKind === 'fixed' ? record(fixed) : record(current)
  const completionDecisionEligibility = eligibleOf(completionDecisionBranch, completionDecisionBranch?.diagnostics)
  const completionBlockedReasons = Array.from(new Set(reasonsOf(completionDecisionBranch)))
  const completionDecisionLabel = landscapeDecisions.find(item => item.value === completionDecisionValue)?.label ?? '评估起点建议'
  const previewTraining = record(preflight.preview?.training_window)
  const preflightReady = preflight.status === 'ready'
  const rangesSummary = robustFields.map(name => labels[name] + ' ' + rangeValue(name, ranges[name].low) + '–' + rangeValue(name, ranges[name].high)).join(' · ')
  const savedRanges = actualRanges && Object.keys(actualRanges).length ? <div className='table-wrap'><table><thead><tr><th>参数</th><th>下限</th><th>上限</th></tr></thead><tbody>{Object.entries(actualRanges).map(([key, raw]) => { const range = record(raw); return <tr key={key}><td>{labels[key] ?? key}</td><td>{range ? rangeValue(key, range.low) : '—'}</td><td>{range ? rangeValue(key, range.high) : '—'}</td></tr> })}</tbody></table></div> : <Empty title='任务中未保存搜索范围' />
  const revision = completion?.selected_revision ?? null
  const revisionScope = completion?.revision_scope ?? null
  const revisionScopeLabel = revisionScope
    ? String(revisionScope.decision ?? '—') + ' #' + String(revisionScope.decision_index ?? 0)
    : '—'
  const completionStatusLabel = completion?.status === 'complete' ? '候选评分已完整' : completion?.status === 'available' ? '存在未完成候选，可补算' : '暂无可补算内容'

  return <>
    <PageHeader eyebrow='PARAMETER RESEARCH' title='参数研究' description='用历史训练选择参数，在所选日期内检验固定与自适应方案。' action={<Button busy={busy} disabled={!base.symbol || !preflightReady} onClick={() => void submit()}><FlaskConical size={15} />启动参数研究</Button>} />
    <div className='research-guide'>
      <p><strong>研究流程</strong><span>设置标的与评估期 → 选择方案并启动 → 查看样本外表现与补算 → 载入回测</span></p>
      <Disclosure id='research-method-help' title='如何理解日期、方案与收益口径' summary={`评估日期 · ${trainingMinDays}–${trainingMaxDays} 日训练 · 峰值本金口径`}>
        <div className='research-help-grid'>
          <p><strong>日期与训练</strong>所选开始、结束日期定义样本外评估期。训练时段从评估日向前取 {trainingMinDays}–{trainingMaxDays} 个交易日，{trainingWindowMode === 'same_price' ? '起点优先取距离评估日最远、且当日最低价低于评估前最后收盘价、当日最高价高于该收盘价的同价日；没有这样的同价日时按最长天数回退。' : '固定按最长天数回退，不寻找同价日。'}训练时段整体用于稳健 J 评分，内部验证窗为 0，不再拆分内部验证。</p>
          <p><strong>训练候选与样本外</strong>参数收益分布里全部是训练候选，只用于发现入口与稳健 J 排名，不是样本外验证结果。样本外证据只有固定与自适应两条连续真实回测账本：账户在整个评估期连续运行，收益按峰值本金计算，并与同资金被动持有配对比较。</p>
          <p><strong>固定与自适应</strong>固定方案在评估开始前锁定一组参数；自适应方案每隔设定交易日重选一次，决策只使用当时可得的历史，账户连续运行。</p>
          <p><strong>收益口径</strong>收益按峰值本金计算：最终资产减外部投入本金，再除以峰值外部本金。风险指标是盈利回撤除以峰值外部本金。同资金被动持有是同账户、同资金节奏的对照基准，差值超额 = 峰值本金收益 − 同资金被动持有收益。</p>
          <p><strong>统计判断</strong>本版本不提供显著性检验，也不提供 p 值或置信下界；相对最佳、满足约束与样本外表现是相互独立的判断，请分别阅读。</p>
        </div>
      </Disclosure>
    </div>
    {error && <ErrorBox message={error} onClose={() => setError('')} onRetry={retryable ? () => { setError(''); setRetryToken(token => token + 1) } : undefined} />}
    {dataIssue && <div className='notice'><span>本次研究缺少本地行情或质量依据。请在数据管理中显式准备所需周期和前置历史。</span><Button variant='secondary' onClick={() => onPrepareData?.(dataIssue)}>去准备数据</Button></div>}
    <div className='research-workspace'>
      <aside className='research-config-column'>
        <Card title='研究对象' subtitle='所选日期是样本外评估期，系统另取之前的历史训练。'>
          <ConfigForm value={base} onChange={onBaseChange} purpose='research' researchProtocol={protocolVersion} onFeeDefaultsEdit={onFeeDefaultsEdit} onDefaultsEdit={onDefaultsEdit} />
          <Button variant='quiet' disabled={!base.symbol} onClick={() => {
            get<{ first_date?: string }>('/data/coverage/' + base.symbol).then(value => {
              if (value.first_date) onBaseChange(previous => ({ ...previous, start_date: new Date(Date.parse(value.first_date! + 'T00:00:00Z') + 86400000).toISOString().slice(0, 10), reference_price_mode: 'auto', reference_price: null }))
            }).catch(cause => reportError(err(cause)))
          }}>使用全部本地历史</Button>
        </Card>
        <Card title='研究设置' className='research-launch-card' action={onRestore && <Button variant='quiet' onClick={onRestore}><RotateCcw size={14} />恢复默认</Button>}>
          <div className='field-label'>参数方案</div>
          <div className='research-scheme-tabs' role='group' aria-label='参数方案'>
            {([['fixed', '固定参数'], ['adaptive', '每季自适应'], ['both', '两者对照']] as const).map(([value, label]) => <button type='button' className={selectionTarget === value ? 'selected' : ''} key={value} aria-pressed={selectionTarget === value} onClick={() => updateSettings({ selection_target: value })}>{label}</button>)}
          </div>
          <p className='field-hint'>{selectionTarget === 'fixed' ? '评估开始前锁定一组参数，整个评估期不变。' : selectionTarget === 'adaptive' ? '每隔 ' + rebalanceDays + ' 个交易日重选，账户连续运行。' : '同时比较一次锁定与每隔 ' + rebalanceDays + ' 个交易日重选。'}</p>
          <div className='form-grid two'>
            <NumField label='超正方体边长（研究刻度）' min={1} max={99} step='2' value={hypercubeEdge} onChange={next => next != null && updateSettings({ hypercube_edge: next })} hint='奇数边长使中心落在整数刻度；边长 3 表示每个维度取中心及两侧各一格。' />
            <Field label='搜索方法'><SelectInput value={searchMethod} onChange={event => updateSettings({ search_method: event.target.value as ResearchDefaults['search_method'] })}><option value='auto'>自动（小空间穷举）</option><option value='hierarchical'>分层搜索</option><option value='exhaustive'>完整枚举</option></SelectInput></Field>
            <NumField label='训练回测预算（唯一参数点）' min={1} max={1000000} step='1' value={trainingBudget} onChange={next => next != null && updateSettings({ training_budget: next })} />
            <NumField label='邻域与峰值核验预算（唯一参数点）' min={1} max={2000000} step='1' value={validationBudget} onChange={next => next != null && updateSettings({ validation_budget: next })} />
            <NumField label='展示补算预算（唯一参数点）' min={0} max={2000000} step='1' value={reportingBudget} onChange={next => next != null && updateSettings({ reporting_budget: next })} hint='用于对已归档候选做展示补算；不参与择参评分。' />
            <NumField label='推荐参数个数' min={1} max={100} step='1' value={recommendationCount} onChange={next => next != null && updateSettings({ recommendation_count: next })} />
            <Field label='滑点处理'><SelectInput value={slippageMode} onChange={event => updateSettings({ slippage_mode: event.target.value as ResearchDefaults['slippage_mode'] })}><option value='configured'>按当前配置</option><option value='zero'>不考虑滑点</option></SelectInput></Field>
          </div>
          <p className='field-hint'>本版本固定为等比网格 v5 · 四维稳健 J 排名（ranking_mode = robust_j）。只搜索买入间距、卖出间距、每次买入和每次卖出四轴；最低持仓固定为零、最高持仓自然放开。买卖比例仅由股数除以初始持股派生，持仓上限作为回测后的风险资格与惩罚，不会提前截断成交。</p>
          <Disclosure id='research-robust-j' title='稳健 J 目标与峰区评分' summary={'α ' + percent(alpha) + ' · β ' + scoreBeta.toFixed(2) + ' · σ ' + sigma.toFixed(2) + ' · τ ' + percent(tau)}>
            <div className='form-grid two'>
              <NumField label='中心超额权重 α' min={0} max={100} step='1' percentValue suffix='%' value={alpha} onChange={next => next != null && updateSettings({ alpha: next })} />
              <NumField label='下行风险系数 β' min={0} max={100} step='.1' value={scoreBeta} onChange={next => next != null && updateSettings({ beta: next })} />
              <NumField label='安全垫阈值 τ（训练期累计）' min={-100} max={1000} step='1' percentValue suffix='%' value={tau} onChange={next => next != null && updateSettings({ tau: next })} />
              <NumField label='高斯核宽度 σ（刻度）' min={.01} max={100} step='.1' value={sigma} onChange={next => next != null && updateSettings({ sigma: next })} />
              <NumField label='持仓免罚上限 η limit' min={0} max={1000} step='.1' value={etaLimit} onChange={next => next != null && updateSettings({ eta_limit: next })} />
              <NumField label='最大持仓淘汰阈值（0 表示关闭）' min={0} max={1000} step='.1' value={etaHardCap ?? 0} onChange={next => next != null && updateSettings({ eta_hard_cap: next > 0 ? next : null })} hint='仅判断中心是否合格，不提前停止回测或截断买入。' />
              <NumField label='上界二次惩罚 γ max' min={0} max={1000} step='.05' value={gammaMax} onChange={next => next != null && updateSettings({ gamma_max: next })} hint={'例如实际最大持仓 2.5 倍、免罚上限 2、γ=0.4 时扣 0.1。'} />
              <NumField label='底仓保护下限 η floor' min={0} max={100} step='1' percentValue suffix='%' value={etaFloor} onChange={next => next != null && updateSettings({ eta_floor: next })} />
              <NumField label='底仓整格惩罚 γ min' min={0} max={1000} step='.05' value={gammaMin} onChange={next => next != null && updateSettings({ gamma_min: next })} />
              <NumField label='底仓参考格数 K ref' min={.01} max={1000000} step='.1' value={floorKRef} onChange={next => next != null && updateSettings({ floor_k_ref: next })} hint='每段按该候选的卖出间距计整格；扣分为 γ min × Σ(k/K ref)²。' />
              <NumField label='峰值比较绝对容差' min={1e-15} max={.1} step='any' value={peakTolerance} onChange={next => next != null && updateSettings({ peak_tolerance: next })} />
            </div>
            <p className='field-hint'>J = αR + (1−α)μΩ − βσ⁻Ω − Cpos。底仓惩罚按每段触底后继续上涨的卖出整格数累计，首次触底和底仓附近震荡不扣分；原始 R 只作为中心收益与诊断。</p>
          </Disclosure>
          <Disclosure id='research-robust-search' title='多起点搜索与预算控制' summary={`起点 ${startMin}–${startMax} · 每候选 ${startsPerCandidate} · 核验预留 ${percent(validationReserveFraction)}`}>
            <div className='form-grid two'>
              <NumField label='最少起点数' min={1} max={1000} step='1' value={startMin} onChange={next => next != null && updateSettings({ start_min: next })} />
              <NumField label='最多起点数' min={1} max={1000} step='1' value={startMax} onChange={next => next != null && updateSettings({ start_max: next })} />
              <NumField label='每个推荐候选对应起点数' min={1} max={100} step='1' value={startsPerCandidate} onChange={next => next != null && updateSettings({ starts_per_candidate: next })} />
              <Field label='初始局部步长（刻度，逗号分隔）'><TextInput value={initialStrides.join(',')} onChange={event => { const values = event.target.value.split(',').map(value => Number(value.trim())).filter(value => Number.isSafeInteger(value) && value > 0); updateSettings({ initial_strides: values.length ? values : [1] }) }} /></Field>
              <NumField label='大步长倍率' min={2} max={16} step='1' value={strideMultiplier} onChange={next => next != null && updateSettings({ stride_multiplier: next })} />
              <NumField label='大步长最大跨度比例' min={1} max={100} step='1' percentValue suffix='%' value={maxStrideFraction} onChange={next => next != null && updateSettings({ max_stride_fraction: next })} />
              <NumField label='峰值核验预算预留' min={0} max={100} step='1' percentValue suffix='%' value={validationReserveFraction} onChange={next => next != null && updateSettings({ validation_reserve_fraction: next })} />
              <NumField label='穷举工作内存上限（MiB）' min={16} max={65536} step='16' value={exhaustiveMemoryLimitMb} onChange={next => next != null && updateSettings({ exhaustive_memory_limit_mb: next })} />
              <NumField label='单点预算内存估计（字节）' min={64} max={65536} step='64' value={estimatedBytesPerPoint} onChange={next => next != null && updateSettings({ estimated_bytes_per_point: next })} hint='用于自动搜索预检和穷举工作内存估算，不改变回测结果。' />
            </div>
            <p className='field-hint'>自动模式只在训练预算和工作内存均允许时完整枚举；否则使用打乱 Sobol 覆盖、统计启发式种子和多起点轮转局部搜索。所有输出峰区必须完成一格联合邻域核验。</p>
          </Disclosure>
          <Disclosure id='research-robust-window' title='训练窗口、预算与费用压力' summary={`${trainingMinDays}–${trainingMaxDays} 日训练 · ${trainingWindowMode === 'same_price' ? '同价日起点' : '固定天数'} · 费用 ×${stressCostMultiplier}`}>
            <div className='form-grid two'>
              <NumField label='训练回看最短交易日' min={20} max={5000} step='1' value={trainingMinDays} onChange={next => next != null && updateSettings({ training_min_days: next })} />
              <NumField label='训练回看最长交易日' min={20} max={5000} step='1' value={trainingMaxDays} onChange={next => next != null && updateSettings({ training_max_days: next })} />
              <Field label='训练窗口模式'><SelectInput value={trainingWindowMode} onChange={event => updateSettings({ training_window_mode: event.target.value as ResearchDefaults['training_window_mode'] })}><option value='same_price'>同价日起点</option><option value='fixed'>固定天数</option></SelectInput></Field>
              <NumField label='任务总评价预算（留空表示自动有效上限）' min={1} max={10000000} step='1' value={totalEvaluationBudget ?? null} onChange={next => updateSettings({ total_evaluation_budget: next })} hint='留空时由搜索上限、评估决策数与回放次数自动形成有效上限。' />
              <NumField label='最长运行时间（秒）' min={1} max={86400} step='1' value={maxRuntimeSeconds} onChange={next => next != null && updateSettings({ max_runtime_seconds: next })} />
              <NumField label='费用压力倍率' min={1.01} max={20} step='.1' value={stressCostMultiplier} onChange={next => next != null && updateSettings({ stress_cost_multiplier: next })} />
              <NumField label='费用压力推荐峰区数' min={0} max={50} step='1' value={stressRegionCount} onChange={next => next != null && updateSettings({ stress_region_count: next })} />
            </div>
          </Disclosure>
          <section className={`research-preflight ${preflightReady ? 'ready' : preflight.status === 'loading' ? 'loading' : 'blocked'}`} aria-live='polite'>
            <div className='research-preflight-head'><div><strong>只读搜索预检</strong><small>参数稳定后自动更新；编辑会立即作废旧结果</small></div><Badge tone={preflightReady ? 'green' : preflight.status === 'loading' ? 'blue' : 'amber'}>{preflightReady ? '可以启动' : preflight.status === 'loading' ? '检查中' : '存在阻断项'}</Badge></div>
            {preflight.status === 'loading' ? <Loading label={preflight.message} /> : preflightReady ? <div className='research-preflight-grid'>
              <div><small>训练时段</small><strong>{String(previewTraining?.start ?? '—')} 至 {String(previewTraining?.end ?? '—')}</strong><span>{String(preflight.preview?.training_days ?? '—')} 个交易日{previewTraining?.matched ? ' · 同价日起点' : ' · 固定天数'}</span></div>
              <div><small>搜索组合</small><strong>{Number(preflight.preview?.legal_combination_count ?? 0).toLocaleString()} 个</strong><span>{String(preflight.preview?.search_method ?? '—')} · 单峰支持 {String(preflight.preview?.estimated_peak_support_points ?? '—')} 点</span></div>
              <div><small>评价预算</small><strong>{Number(preflight.preview?.effective_total_evaluation_budget ?? 0).toLocaleString()} 点</strong><span>任务上限 {Number(preflight.preview?.task_evaluation_upper_bound ?? 0).toLocaleString()} · 邻域 {Number(preflight.preview?.neighborhood_budget ?? validationBudget).toLocaleString()}</span></div>
              <div><small>预计内存</small><strong>{Number(preflight.preview?.estimated_task_memory_mb ?? 0).toFixed(1)} MiB</strong><span>工作区上限 {Number(preflight.preview?.workspace_budget_mb ?? 0).toFixed(1)} MiB</span></div>
              <div><small>数据前置</small><strong>训练窗口可用</strong><span>提交时仍会重新核验冻结的数据版本</span></div>
            </div> : <div className='research-preflight-blocked'><p>{preflight.message || '预检尚未完成。'}</p>{dataIssue?.gaps?.length ? <ul>{dataIssue.gaps.map((gap, index) => <li key={index}>{String(gap.message ?? gap.kind ?? '数据缺口')}</li>)}</ul> : null}<Button variant='quiet' onClick={() => setPreflightToken(token => token + 1)}><RefreshCw size={14} />重新预检</Button></div>}
          </section>
          <div className='research-launch-action'><Button busy={busy} disabled={!preflightReady} onClick={() => void submit()}><FlaskConical size={15} />启动参数研究</Button><small>{schemeLabel(selectionTarget, rebalanceDays)} · {trainingMinDays}–{trainingMaxDays} 日训练时段 · 设置自动记忆</small></div>
          <Disclosure id='research-search-ranges' title='可调搜索范围' summary={rangesSummary}>
            <p className='field-hint'>四个轴各自只有一个步长来源。研究细粒度就是该轴整数刻度宽度，间距以 0.01 个百分点为最小单位，股数以 100 股 / 份为最小单位。</p>
            <div className='research-range-grid'>{robustFields.map(name => {
              const unit = researchQuantityUnit(name)
              const minimum = name === 'buy_shares' ? isStarBoard ? 200 : 100 : name === 'sell_shares' ? 100 : 0
              return <div className='research-range-card' key={name}>
                <div className='research-range-heading'>
                  <div className='research-range-title'><strong>{labels[name]}</strong></div>
                  <small>单位 {pctRange.has(name) ? '0.01 个百分点' : `${unit} 股 / 份`}</small>
                </div>
                <div className='research-range-inputs'>
                  <NumField key='low' label='下限' value={ranges[name].low} percentValue={pctRange.has(name)} step={pctRange.has(name) ? '.01' : String(unit)} min={pctRange.has(name) ? .01 : minimum} onChange={next => next != null && updateRange(name, 'low', next)} />
                  <NumField key='high' label='上限' value={ranges[name].high} percentValue={pctRange.has(name)} step={pctRange.has(name) ? '.01' : String(unit)} min={pctRange.has(name) ? .01 : minimum} onChange={next => next != null && updateRange(name, 'high', next)} />
                </div>
                <NumField label='研究细粒度' value={granularity[name]} percentValue={pctRange.has(name)} step={pctRange.has(name) ? '.01' : String(unit)} min={pctRange.has(name) ? .01 : unit} onChange={next => next != null && updateSettings({ granularity: { ...granularity, [name]: next } })} hint='仅控制搜索采样间隔' />
              </div>
            })}</div>
            <Button variant='quiet' onClick={() => updateSettings({ ranges: defaultSearchRanges() })}><RotateCcw size={14} />恢复默认范围</Button>
          </Disclosure>
          <Disclosure id='research-constraints' title='选择与样本约束' summary={'本金回撤 ≤ ' + percent(maxPrincipalDrawdown) + ' · 成交 ≥ ' + minTrades + ' · 分钟覆盖 ≥ ' + percent(minMinuteRatio)}>
            <div className='form-grid two'>
              <NumField label='最大本金回撤约束' min={1} max={100} step='1' percentValue suffix='%' value={maxPrincipalDrawdown} onChange={next => next != null && updateSettings({ max_principal_drawdown: next })} hint='按峰值外部本金口径的盈利回撤，与旧单位净值回撤不可直接比较。' />
              <NumField label='最少成交笔数' min={0} step='1' value={minTrades} onChange={next => next != null && updateSettings({ min_trades: next })} />
              <NumField label='最小分钟覆盖率' min={0} max={100} step='1' percentValue suffix='%' value={minMinuteRatio} onChange={next => next != null && updateSettings({ min_minute_ratio: next })} />
            </div>
            <p className='field-hint'>候选的相对排名与约束是否满足会分别展示。</p>
          </Disclosure>
          <Disclosure id='research-reproducibility' title='换参与复现设置' summary={`${rebalanceDays} 日换参 · 种子 ${seed} · 展示 ${topCandidates} 组`}>
            <div className='form-grid two'>
              <NumField label='自适应换参周期（交易日）' min={1} max={504} step='1' value={rebalanceDays} onChange={next => next != null && updateSettings({ rebalance_days: next })} />
              <NumField label='随机种子' step='1' value={seed} onChange={next => next != null && updateSettings({ seed: next })} />
              <NumField label='结果候选展示上限' min={1} max={100} step='1' value={topCandidates} onChange={next => next != null && updateSettings({ top_candidates: next })} hint='只限制页面展示数量，不截断研究评价。' />
            </div>
          </Disclosure>
        </Card>
      </aside>
      <main className='research-main-column' aria-label='参数研究结果'>
        <div className='research-result-toolbar'>
          <h2>研究结果</h2>
          {(researchResults.length > 0 || selectedJob) && <Field label='研究结果历史'><SelectInput value={selectedResultId ? 'result:' + selectedResultId : selectedJobId ? 'job:' + selectedJobId : ''} onChange={event => {
            const [kind, id] = event.target.value.split(':', 2)
            if (kind === 'result') selectResult(id)
            else if (kind === 'job') selectJob(id)
          }}>
            {selectedJob && <option value={'job:' + selectedJob.id}>{taskLabelForResearch(selectedJob)}</option>}
            {selectedResultId && !researchResults.some(item => item.id === selectedResultId) && <option value={'result:' + selectedResultId}>链接中的研究结果</option>}
            {researchResults.map(item => <option key={item.id} value={'result:' + item.id}>{item.symbol || '未识别标的'} · {item.created_at?.slice(0, 10) ?? ''} · {item.title || '等比网格 v5 · 稳健 J'}</option>)}
          </SelectInput></Field>}
          <Button variant='quiet' busy={researchResultsLoading} onClick={() => { void tasks.refresh(); void loadResearchResults(0, false) }} aria-label='刷新研究结果列表'><RefreshCw size={14} />刷新</Button>
        </div>
        {researchResultsError && <ErrorBox message={'研究结果历史读取失败：' + researchResultsError} onRetry={() => void loadResearchResults(0, false)} />}
        {researchResultsMore && <div className='button-row'><Button variant='quiet' busy={researchResultsLoading} onClick={() => void loadResearchResults(researchResultsOffset + 50, true)}>加载更早研究结果</Button></div>}
        {resultLoading && result && <p className='inline-warning'>正在刷新研究结果，当前显示上次成功读取的内容。</p>}
        {resultLoading && !result ? <Card><Loading label='正在读取研究结果…' /></Card> : !result ? <Card>{selectedJob && ['queued', 'running'].includes(selectedJob.status) ? <><div className='result-meta'><Badge tone='blue'>{statusText(selectedJob.status)}</Badge><span>{selectedJob.message || '任务正在执行'}</span></div><p className='field-hint'>任务完成并保存结果后，这里会自动刷新研究结果历史。</p></> : <Empty title='研究结果将在这里展示' detail='在左侧启动研究，或从任务中心打开已有研究结果。' />}</Card> : !isSupportedResult ? <Card title='不支持的历史版本' subtitle='这条结果产生于已移除的研究协议版本。'>
          <div className='research-legacy-banner'>该历史结果产生于协议版本 {resultProtocolVersion || '未知'}（结果结构版本 {resultSchemaVersion || '未知'}）。本工作台只支持协议 v5 的结果，旧版本不再提供详情、参数分布与导出适配。</div>
          <div className='result-meta'><Badge tone='amber'>不支持的历史版本</Badge><span>请重新运行一次 v5 参数研究以获得可查看、可补算的结果。</span></div>
        </Card> : <>
          <Card title={originalBase?.symbol ?? '研究样本'} subtitle={(evalStart ?? '开始日期不详') + ' 至 ' + (evalEnd ?? '最新可用日期') + ' · ' + (originalBase?.frequency ?? '频率不详')}>
            <div className='result-meta'><Badge tone='blue'>等比网格 v5 · 稳健 J</Badge>{isDemoResearch && <Badge tone='amber'>演示数据</Badge>}{selectedJob?.status === 'cancelled' && <Badge tone='amber'>已保留的部分结果</Badge>}<span>所选日期为样本外评估；训练时段为评估日前的单一区间，内部验证窗为 0。</span></div>
            {isDemoResearch && <div className='demo-callout'>此研究使用演示行情生成，不能代表真实历史数据或投资结论。</div>}
            <div className='research-summary-states'>
              <div><small>训练时段</small><strong>{trainStart && trainEnd ? trainStart + ' 至 ' + trainEnd : `评估日前 ${trainingMinDays}–${trainingMaxDays} 个交易日`}</strong></div>
              <div><small>评估起点推荐状态</small><Badge tone={currentRecommendationStatus === 'qualified' ? 'green' : 'amber'}>{currentRecommendationStatus === 'qualified' ? '按 J 排名推荐参数' : '暂无合格推荐'}</Badge></div>
              <div><small>协议统计状态</small><Badge tone='neutral'>暂不提供显著性判断</Badge></div>
            </div>
            {researchWarnings.length > 0 && <div className='warning-list'>{researchWarnings.map((warning, index) => <div key={index}><span className='warning-dot' />{warning}</div>)}</div>}
          </Card>
          <Card title='样本外收益对照' subtitle='固定与自适应都是连续真实回测账本：账户在整个评估期连续运行，收益按峰值本金计算，差值超额为与同资金被动持有的配对差值。'>
            {fixed && <div className='result-meta'><strong>固定参数</strong><span>锁定于 {fixed.as_of ?? '—'}</span><small>{constraintLabel(eligibleOf(fixed, fixed.diagnostics))}</small><Button variant='quiet' disabled={!fixed.params || !originalBase} onClick={() => loadV2(fixed.params, fixed.as_of)}>载入固定参数</Button></div>}
            {fixed && <PrincipalMetricRow label='固定参数' metrics={fixedMetrics} />}
            {fixed?.params && <ParameterBadges params={fixed.params} compact />}
            {adaptive && <div className='result-meta'><strong>自适应参数</strong><span>每 {rebalanceDays} 个交易日重选 · {adaptive.schedule?.length ?? 0} 次参数生效</span></div>}
            {adaptive && <PrincipalMetricRow label='自适应参数' metrics={adaptiveMetrics} />}
            {!fixedMetrics && !adaptiveMetrics && <Empty title='本次没有可比较的样本外收益' detail='任务可能只完成了参数选择，或评估区间没有可用行情。' />}
            {fixedMetrics && <p className='field-hint'>这里只展示连续回测账本的收益与配对超额；训练候选的分布与稳健 J 排名在下方训练候选参数分布中，训练候选不是样本外结果。</p>}
          </Card>
          <Card title='展示补算' subtitle='对已归档的候选邻域做补算，补算会生成新的修订号，参数分布与导出都会带上该修订。'>
            <div className='result-meta'><Badge tone={completion?.status === 'complete' ? 'green' : completion?.status === 'available' ? 'amber' : 'neutral'}>{completionStatusLabel}</Badge><span>当前修订号：{revision ?? '无'}</span><span>覆盖决策：{revisionScopeLabel}</span></div>
            <p className='field-hint'>补算只使用原结果的数据版本与计算来源；若代码或依赖已变化，后端会拒绝混合旧候选与当前引擎。</p>
            <div className='form-grid two research-completion-decision'><Field label='要补算的研究决策' hint='提交时绑定这里正在查看的决策，不再固定为 current #0。'><SelectInput value={completionDecisionValue} onChange={event => onCompletionDecisionChange?.(event.target.value)}>{landscapeDecisions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field><div className={`research-eligibility ${completionDecisionEligibility === true ? 'eligible' : completionDecisionEligibility === false ? 'blocked' : ''}`}><small>{completionDecisionLabel} · 约束资格</small><strong>{constraintLabel(completionDecisionEligibility)}</strong>{completionBlockedReasons.length ? <span>{completionBlockedReasons.join('；')}</span> : <span>未返回额外阻断原因</span>}</div></div>
            <div className='button-row'>
              <Button variant='secondary' busy={completionBusy} disabled={!resultId || !completionDecisionBranch} onClick={() => void submitCompletion()}>提交 {completionDecisionLabel}</Button>
              <Button variant='quiet' onClick={() => setCompletionToken(token => token + 1)}>刷新补算状态</Button>
              <Button variant='quiet' disabled={!resultId} onClick={() => void exportFile('json')}><Download size={14} />JSON</Button>
              <Button variant='quiet' disabled={!resultId} onClick={() => void exportFile('candidates')}><Download size={14} />候选 CSV</Button>
            </div>
            {completionNote && <p className='field-hint'>{completionNote}</p>}
          </Card>
          {resultId && <ResearchLandscape key={resultId + ':' + (revision ?? 0)} resultId={resultId} defaultParams={relativeBest} onLoad={(params, asOf) => loadV2(params, asOf ?? current?.as_of)} decisionOptions={landscapeDecisions} decision={completionDecisionValue} onDecisionChange={onCompletionDecisionChange} qualified={Boolean(currentRecommended)} revision={revision} revisionScope={revisionScope} />}
          <Disclosure id='research-rebalance-records' title='换参记录与滚动决策' summary={(adaptive?.schedule?.length ?? 0) + ' 次生效 · ' + (result?.evidence?.decisions?.length ?? 0) + ' 个决策点'}>
            <p className='field-hint'>决策日表示何时重选参数；生效日表示参数开始作用的日期。各决策只使用当时已知历史。</p>
            {adaptive?.schedule?.length ? <div className='table-wrap scroll-table'><table><thead><tr><th>决策日期</th><th>生效日期</th><th>参数</th></tr></thead><tbody>{adaptive.schedule.map((item, index) => <tr key={(item.effective_date ?? '') + '-' + index}><td>{item.decision_date ?? '—'}</td><td>{item.effective_date ?? '—'}</td><td>{item.paused ? <Badge tone='amber'>暂停网格交易</Badge> : <ParameterBadges params={item.params} compact />}</td></tr>)}</tbody></table></div> : <Empty title='没有换参记录' detail={statusText(adaptive?.status)} />}
            {result?.evidence?.decisions?.length ? <div className='table-wrap scroll-table'><table><thead><tr><th>决策日期</th><th>候选 / 实际评估</th><th>约束合格</th><th>选中参数</th><th>评分</th></tr></thead><tbody>{result.evidence.decisions.map((raw, index) => { const item = record(raw) ?? {}; return <tr key={index}><td>{String(item.decision_date ?? item.as_of ?? '—')}</td><td>{valueText('candidate_count', item.candidate_count)} / {valueText('evaluated_count', item.evaluated_count)}</td><td>{valueText('eligible_count', item.eligible_count)}</td><td><ParameterBadges params={record(item.selected_params)} compact /></td><td>{shortSummary(item.score ?? item.diagnostics)}</td></tr> })}</tbody></table></div> : <Empty title='暂无滚动决策明细' />}
          </Disclosure>
          <Disclosure id='research-protocol-status' title='协议与统计状态' summary='协议 v5 · 暂不提供显著性判断'>
            <div className='research-evidence-grid'>
              <div><small>协议版本</small><strong>v{String(protocol.protocol_version ?? resultProtocolVersion)}</strong></div>
              <div><small>策略版本</small><strong>v{String(protocol.strategy_version ?? strategyVersion)}</strong></div>
              <div><small>排名方式</small><strong>{String(protocol.ranking_mode ?? rankingMode)}</strong></div>
              <div><small>收益口径</small><strong>峰值本金</strong></div>
              <div><small>风险口径</small><strong>盈利回撤 ÷ 峰值本金</strong></div>
              <div><small>统计状态</small><strong>暂不提供显著性判断</strong></div>
            </div>
            <p className='field-hint'>{String(protocolStatistics.reason ?? '本版本未提供峰值本金超额收益的显著性检验。')}</p>
            <EvidenceFields value={{ training: protocolTraining, out_of_sample: protocolOutOfSample, budgets: protocol.budgets }} />
          </Disclosure>
          <Disclosure id='research-sample-data-quality' title='样本、数据质量与实际设置' summary={(finite(minuteCoverage) ? '准备区间分钟覆盖 ' + percent(minuteCoverage) + ' · ' : '') + '数据版本 ' + (result?.data_version ?? '—')}>
            <div className='research-period-grid'><div><small>训练时段</small><strong>{trainStart && trainEnd ? trainStart + ' 至 ' + trainEnd : `评估日前 ${trainingMinDays}–${trainingMaxDays} 个交易日`}</strong></div><div><small>样本外评估</small><strong>{evalStart ?? '—'} 至 {evalEnd ?? '最新可用日期'}</strong></div></div>
            <div className='result-meta'><span>随机种子：{actualSeed ?? '—'}</span><span>展示上限：{topCandidates} 组</span></div>
            <DataDiagnosticsPanel diagnostics={result?.diagnostics} sourceEvents={result?.source_events} warnings={researchWarnings} syntheticBars={record(result?.quality)?.synthetic_bars} />
            <div className='param-block'><strong>本次实际搜索范围</strong>{savedRanges}</div>
            <div className='param-block'><strong>锁定账户资金</strong><EvidenceFields value={lockedFunding} /></div>
          </Disclosure>
          {(result?.configuration_snapshot || result?.execution || result?.performance) && <Disclosure id='research-task-snapshot' title='任务配置、执行与资源快照' summary={'配置版本 ' + String(record(result?.configuration_snapshot)?.revision ?? result?.execution?.configuration_revision ?? '—') + ' · 数据版本 ' + String(result?.execution?.data_version ?? result?.data_version ?? '—')}>
            <p className='field-hint'>这里显示任务提交时冻结的配置与执行记录。之后修改当前项目配置不会改写这条历史结果；载入参数属于用户明确创建的新的回放草稿。</p>
            {result?.configuration_snapshot && <div className='param-block'><strong>提交时完整配置</strong><EvidenceFields value={result.configuration_snapshot} /></div>}
            {result?.execution && <div className='param-block'><strong>实际执行</strong><EvidenceFields value={result.execution} /></div>}
            {result?.performance && <div className='param-block'><strong>资源使用</strong><EvidenceFields value={result.performance} /></div>}
          </Disclosure>}
          {result?.computation && <Disclosure id='research-computation-details' title='实际计算量' summary={shortSummary(result.computation)}><p className='field-hint'>记录真实评估次数、预算消耗与耗时，包括邻域核验的额外计算。</p><EvidenceFields value={result.computation} /></Disclosure>}
        </>}
      </main>
    </div>
  </>
}
