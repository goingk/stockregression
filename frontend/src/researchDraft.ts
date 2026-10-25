import { normalizeFeeConfig } from './etfResearch.ts'
import type { EtfCandidate, EtfResearchResult, EtfRow } from './etfResearch.ts'
import type { BacktestConfig } from './types.ts'
import type { ResearchDefaults } from './workbenchDefaults.ts'

export type ResearchPriorPolicy = 'off' | 'daily_state_v1'

/** The immutable ETF result used as the entry point for one research draft. */
export interface ResearchSourceEtf {
  result_id: string
  candidate_ids?: number[]
}

/** Display-only metadata. It is never sent back as source or target identity. */
export interface ResearchSourceSummary {
  result_id: string
  symbol: string
  symbol_name: string
  as_of: string
  data_version: string | null
  status: string
  method_version: string
  candidate_count: number
  qualified_count: number
  recommendation_id: string | number | null
  gss_score: number | null
  score_cohort_size: number
  data_status: string
  eligible: boolean
  validation_status: string
  validation_reason: string | null
  archived?: boolean
}

export interface ResearchDraftContextState {
  source_etf: ResearchSourceEtf | null
  source_summary: ResearchSourceSummary | null
  prior_policy: ResearchPriorPolicy
}

export interface ResearchDraftBundle {
  config: BacktestConfig
  settings: ResearchDefaults
  context: ResearchDraftContextState
}

export const RESEARCH_PRIOR_POLICIES: readonly ResearchPriorPolicy[] = ['off', 'daily_state_v1']

export function defaultResearchDraftContext(): ResearchDraftContextState {
  return { source_etf: null, source_summary: null, prior_policy: 'off' }
}

/** A source draft starts from the daily-state prior; ordinary drafts keep the off default. */
export function sourceResearchDraftContext(source: ResearchSourceEtf): ResearchDraftContextState {
  return { source_etf: normalizeSourceEtf(source), source_summary: null, prior_policy: 'daily_state_v1' }
}

export function withResearchDraftSource(
  source: ResearchSourceEtf,
  summary: ResearchSourceSummary,
): ResearchDraftContextState {
  return { source_etf: normalizeSourceEtf(source), source_summary: summary, prior_policy: 'daily_state_v1' }
}

/** Removing the source must not touch the numeric research draft. */
export function clearResearchDraftSource(): ResearchDraftContextState {
  return defaultResearchDraftContext()
}

export function setResearchPriorPolicy(
  context: ResearchDraftContextState,
  policy: ResearchPriorPolicy,
): ResearchDraftContextState {
  return { ...context, prior_policy: RESEARCH_PRIOR_POLICIES.includes(policy) ? policy : 'off' }
}

export function normalizeSourceEtf(source: ResearchSourceEtf): ResearchSourceEtf {
  const candidateIds = Array.isArray(source.candidate_ids)
    ? source.candidate_ids.map(value => typeof value === 'number' ? value : Number(String(value).trim()))
      .filter(value => Number.isSafeInteger(value) && value >= 0)
      .filter((value, index, values) => values.indexOf(value) === index)
    : []
  return candidateIds.length
    ? { result_id: source.result_id, candidate_ids: candidateIds }
    : { result_id: source.result_id }
}

export function researchPriorPolicyLabel(policy: ResearchPriorPolicy): string {
  return policy === 'daily_state_v1' ? '使用日线市场状态先验（daily_state_v1）' : '关闭研究先验（off）'
}

/** Top-level research fields. An independent draft may still enable the state prior. */
export function researchSourceRequestFields(context: ResearchDraftContextState): {
  source_etf: ResearchSourceEtf | null
  prior_policy: ResearchPriorPolicy
} {
  return {
    source_etf: context.source_etf ? normalizeSourceEtf(context.source_etf) : null,
    prior_policy: context.prior_policy,
  }
}

/** Calendar-year subtraction with the same leap-day clamp used by the default config. */
export function dateYearsAgo(value: string, years = 1): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim())
  if (!match || !Number.isSafeInteger(years) || years < 1) return ''
  const year = Number(match[1]) - years
  const month = Number(match[2])
  const requestedDay = Number(match[3])
  if (year < 1 || month < 1 || month > 12 || requestedDay < 1) return ''
  const day = Math.min(requestedDay, new Date(Date.UTC(year, month, 0)).getUTCDate())
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
}

export function dateNextDay(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim())
  if (!match) return ''
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  if (month < 1 || month > 12 || day < 1 || day > 31) return ''
  const stamp = new Date(Date.UTC(year, month - 1, day))
  if (!Number.isFinite(stamp.getTime())
    || stamp.getUTCFullYear() !== year || stamp.getUTCMonth() !== month - 1 || stamp.getUTCDate() !== day) return ''
  stamp.setUTCDate(stamp.getUTCDate() + 1)
  return stamp.toISOString().slice(0, 10)
}

/** Calendar date in Asia/Shanghai, independent of the browser's local timezone. */
export function shanghaiDateToday(now = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(now)
  const value = (type: Intl.DateTimeFormatPartTypes) => parts.find(part => part.type === type)?.value ?? ''
  const year = value('year')
  const month = value('month')
  const day = value('day')
  return /^\d{4}$/.test(year) && /^\d{2}$/.test(month) && /^\d{2}$/.test(day)
    ? `${year}-${month}-${day}`
    : ''
}

export function isValidResearchSymbol(value: string): boolean {
  return /^\d{6}\.(?:SH|SZ)$/i.test(value.trim())
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function candidateList(row: EtfRow): EtfCandidate[] {
  return Array.isArray(row.validation?.candidates) ? row.validation.candidates : []
}

export function etfSourceCandidateIds(row: EtfRow): number[] {
  const candidates = candidateList(row)
  const recommendationId = row.validation?.recommendation?.id
  const qualified = candidates.filter(candidate => candidate.qualified === true)
  const ordered = recommendationId == null
    ? qualified
    : [
        ...qualified.filter(candidate => String(candidate.id) === String(recommendationId)),
        ...qualified.filter(candidate => String(candidate.id) !== String(recommendationId)),
      ]
  return normalizeSourceEtf({
    result_id: '',
    candidate_ids: ordered.slice(0, 3).map(candidate => {
      const value = typeof candidate.id === 'number' ? candidate.id : Number(String(candidate.id).trim())
      return Number.isSafeInteger(value) ? value : Number.NaN
    }).filter(Number.isSafeInteger),
  }).candidate_ids ?? []
}

export function etfSourceSummary(result: EtfResearchResult, row: EtfRow): ResearchSourceSummary {
  const candidates = candidateList(row)
  return {
    result_id: result.id,
    symbol: row.symbol,
    symbol_name: row.symbol_name || row.symbol,
    as_of: row.window_end || result.as_of,
    data_version: row.version || null,
    status: result.status,
    method_version: result.method_version,
    candidate_count: candidates.length,
    qualified_count: candidates.filter(candidate => candidate.qualified === true).length,
    recommendation_id: row.validation?.recommendation?.id ?? null,
    gss_score: finite(row.gss_score) ? row.gss_score : null,
    score_cohort_size: Number.isFinite(row.score_cohort_size) ? row.score_cohort_size : 0,
    data_status: row.data_status || row.status || '',
    eligible: row.eligible === true,
    validation_status: row.validation?.status || 'not_selected',
    validation_reason: row.validation?.reason || null,
  }
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

/**
 * Restore the frozen ETF source from a saved research result. The top-level
 * origin is authoritative; config.research_origin keeps older archived results
 * readable when their public payload only exposed the nested copy.
 */
export function archivedResearchSource(result: unknown): ResearchDraftContextState | null {
  const resultValue = objectValue(result)
  if (!resultValue) return null
  const config = objectValue(resultValue.config)
  const topOrigin = objectValue(resultValue.research_origin)
  const nestedOrigin = objectValue(config?.research_origin)
  const origin = [topOrigin, nestedOrigin].find(value =>
    typeof value?.etf_result_id === 'string' && value.etf_result_id.trim() !== '')
  if (!origin) return null

  const resultId = String(origin.etf_result_id).trim()
  const seeds = Array.isArray(origin.seeds) ? origin.seeds : []
  const candidateIds = seeds
    .map(seed => objectValue(seed)?.candidate_id)
    .map(value => typeof value === 'number' ? value : typeof value === 'string' ? Number(value.trim()) : Number.NaN)
    .filter(Number.isSafeInteger)
  const sourceEtf = normalizeSourceEtf({ result_id: resultId, candidate_ids: candidateIds })
  const candidateCount = sourceEtf.candidate_ids?.length ?? 0
  const symbol = typeof origin.symbol === 'string' ? origin.symbol.trim() : ''
  const sourceAsOf = typeof origin.source_as_of === 'string' ? origin.source_as_of.trim() : ''
  const sourceDataVersion = typeof origin.source_data_version === 'string'
    ? origin.source_data_version.trim() || null
    : null
  const methodVersion = typeof origin.method_version === 'string' ? origin.method_version.trim() : ''
  const gssScore = finite(origin.gss_score) ? origin.gss_score : null
  const priorPolicy: ResearchPriorPolicy = config?.prior_policy === 'off'
    ? 'off'
    : config?.prior_policy === 'daily_state_v1'
      ? 'daily_state_v1'
      : 'daily_state_v1'
  return {
    source_etf: sourceEtf,
    source_summary: {
      result_id: resultId,
      symbol,
      symbol_name: symbol,
      as_of: sourceAsOf,
      data_version: sourceDataVersion,
      status: 'archived',
      method_version: methodVersion,
      candidate_count: candidateCount,
      qualified_count: candidateCount,
      recommendation_id: null,
      gss_score: gssScore,
      score_cohort_size: 0,
      data_status: '',
      eligible: false,
      validation_status: candidateCount ? 'completed' : 'no_qualified',
      validation_reason: candidateCount
        ? '归档快照只保存入选候选，原始逐候选核验明细未随来源快照保存。'
        : '归档来源快照没有入选候选。',
      archived: true,
    },
    prior_policy: priorPolicy,
  }
}

/**
 * Create a source research draft from any valid ETF row. Data-limited, filtered
 * and no-qualified rows remain valid entry points; only the source prior differs.
 */
export function researchDraftFromEtfSource(input: {
  result: EtfResearchResult
  row: EtfRow
  current: BacktestConfig
  settings: ResearchDefaults
  range_mode?: 'prior_year' | 'after_cutoff'
  shanghai_today?: string
}): { draft?: ResearchDraftBundle; error?: string } {
  const { result, row, current, settings } = input
  if (!isValidResearchSymbol(row.symbol)) return { error: 'ETF 行没有有效的证券代码，无法建立参数研究草稿。' }
  const cutoff = (row.window_end || result.as_of || '').trim()
  const afterCutoff = input.range_mode === 'after_cutoff'
  const start = afterCutoff ? dateNextDay(cutoff) : dateYearsAgo(cutoff)
  if (!start) return { error: afterCutoff ? 'ETF 结果没有有效的截止日，无法确定截止日后的研究起点。' : 'ETF 结果没有有效的截止日，无法确定默认研究区间。' }
  const shanghaiToday = input.shanghai_today ?? shanghaiDateToday()
  const end = afterCutoff
    ? (/^\d{4}-\d{2}-\d{2}$/.test(shanghaiToday) && shanghaiToday > start ? shanghaiToday : start)
    : cutoff
  const execution = result.request?.execution
  const sourceFees = execution?.fees
  const fees = normalizeFeeConfig({ ...current.fees, ...(sourceFees ?? {}) })
  const sourceInitialShares = execution?.initial_shares
  const initialShares = finite(sourceInitialShares) && Number.isSafeInteger(sourceInitialShares) && sourceInitialShares >= 0
    ? sourceInitialShares
    : current.initial_shares
  const sourceMinTrades = result.request?.min_trades
  const sourceMaxDrawdown = result.request?.max_principal_drawdown
  const minTrades = finite(sourceMinTrades) && Number.isSafeInteger(sourceMinTrades) && sourceMinTrades >= 0
    ? sourceMinTrades
    : settings.min_trades
  const maxDrawdown = finite(sourceMaxDrawdown) && sourceMaxDrawdown > 0 && sourceMaxDrawdown <= 1
    ? sourceMaxDrawdown
    : settings.max_principal_drawdown
  const config: BacktestConfig = {
    ...structuredClone(current),
    symbol: row.symbol,
    start_date: start,
    end_date: end,
    initial_shares: initialShares,
    reference_price_mode: 'auto',
    reference_price: null,
    lower_price: null,
    upper_price: null,
    selection_asof: null,
    fees,
  }
  const nextSettings: ResearchDefaults = {
    ...structuredClone(settings),
    min_trades: minTrades,
    max_principal_drawdown: maxDrawdown,
  }
  const source = normalizeSourceEtf({ result_id: result.id, candidate_ids: etfSourceCandidateIds(row) })
  return {
    draft: {
      config,
      settings: nextSettings,
      context: withResearchDraftSource(source, etfSourceSummary(result, row)),
    },
  }
}

/** Submit the data version locked by the latest preview; source versions are display-only. */
export function requestedDataVersionFromPreview(preview: unknown): string | undefined {
  if (!preview || typeof preview !== 'object' || Array.isArray(preview)) return undefined
  const value = (preview as Record<string, unknown>).data_version
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function sourceText(value: unknown): string {
  try { return JSON.stringify(value ?? '') } catch { return '' }
}

/** Detect a missing or unreadable source without treating every 404 as one. */
export function isMissingResearchSourceError(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const item = value as { status?: unknown; detail?: unknown; message?: unknown; code?: unknown }
  const detail = item.detail && typeof item.detail === 'object' && !Array.isArray(item.detail)
    ? item.detail as Record<string, unknown>
    : null
  const code = String(item.code ?? detail?.code ?? '')
  const text = [item.message, sourceText(item.detail), sourceText(item.code)].filter(Boolean).join(' ')
  if (/SOURCE_ETF_(?:NOT_FOUND|MISSING|UNAVAILABLE)|RESEARCH_SOURCE_(?:NOT_FOUND|INVALID)|ETF_SOURCE_NOT_FOUND/i.test(code + ' ' + text)) return true
  return /source[_ ]etf|来源\s*ETF|ETF\s*来源|来源研究/i.test(text)
    && /not found|不存在|已清理|缺失|不可用|不可读|无法读取|不属于|不在来源|invalid/i.test(text)
}

export function researchSourceErrorFromPreview(preview: unknown): string {
  if (!preview || typeof preview !== 'object' || Array.isArray(preview)) return ''
  const item = preview as Record<string, unknown>
  for (const key of ['source_error', 'source_etf_error', 'source_not_found_reason']) {
    const value = item[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  const source = item.source_etf
  if (source && typeof source === 'object' && !Array.isArray(source)) {
    const sourceItem = source as Record<string, unknown>
    if (sourceItem.exists === false) return '来源 ETF 研究结果不存在或已被清理。'
    const error = sourceItem.error
    if (typeof error === 'string' && error.trim()) return error.trim()
  }
  const reasons = Array.isArray(item.reasons) ? item.reasons.map(String).filter(Boolean) : []
  const sourceReason = reasons.find(reason =>
    /source[_ ]etf|来源\s*ETF|ETF\s*来源|来源研究/i.test(reason)
    && /不存在|已清理|缺失|不可用|不可读|无法读取|不属于|不在来源|not found|invalid/i.test(reason))
  if (sourceReason) return sourceReason
  return ''
}
