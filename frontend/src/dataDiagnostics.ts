import type { DataDiagnostics, DataSourceEvent, DataSourceSpan } from './types'

const providerNames: Record<string, string> = {
  hf: 'Hugging Face',
  huggingface: 'Hugging Face',
  tdx: '通达信',
  sina: '新浪',
  import: '本地导入',
}

export function sourceSpanLabel(span: DataSourceSpan): string {
  const kind = (span.source_kind ?? '').trim().toLowerCase()
  const source = (span.source ?? '').trim()
  const provider = providerNames[source.toLowerCase()]

  if (kind === 'import') return '本地文件导入'
  if (kind === 'api' || kind === 'provider') return provider ? `在线接口 · ${provider}` : source ? `在线接口 · ${source}` : '在线行情接口'
  if (kind === 'unknown' || !kind) return source ? `来源类型未识别 · ${source}` : '来源类型未识别'

  const knownKinds: Record<string, string> = {
    local: '本地行情数据',
    cache: '本地缓存数据',
    database: '本地数据库',
  }
  const label = knownKinds[kind] ?? `其他来源 · ${span.source_kind}`
  return provider ? `${label} · ${provider}` : label
}

export function sourceEventLabel(event: DataSourceEvent): string {
  const source = (event.source ?? '').trim()
  const provider = providerNames[source.toLowerCase()] ?? source
  const kind = (event.kind ?? '').toLowerCase()
  const action = kind.includes('fallback') || kind.includes('回退') ? '数据源回退'
    : kind.includes('request') || kind.includes('fetch') || kind.includes('请求') ? '来源请求'
      : kind ? event.kind : '来源事件'
  return provider ? `${String(action)} · ${provider}` : String(action)
}

export function uniqueCurrentWarnings(...groups: Array<unknown>): string[] {
  return Array.from(new Set(groups.flatMap((group) => Array.isArray(group) ? group.map(String).filter(Boolean) : [])))
}

export function uniqueSourceEvents(...groups: Array<unknown>): DataSourceEvent[] {
  const seen = new Set<string>()
  const events: DataSourceEvent[] = []
  for (const group of groups) {
    if (!Array.isArray(group)) continue
    for (const item of group) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) continue
      const event = item as DataSourceEvent
      const key = [event.kind, event.source, event.status, event.at, event.message].map((value) => value ?? '').join('|')
      if (seen.has(key)) continue
      seen.add(key)
      events.push(event)
    }
  }
  return events
}

export function diagnosticsStatusLabel(status?: DataDiagnostics['status']): string {
  if (status === 'complete') return '已完整核验'
  if (status === 'partial') return '部分核验'
  if (status === 'unavailable') return '暂无可核验数据'
  return '核验状态未提供'
}

export function actualRepairMinutes(syntheticBars: unknown, auctionPlaceholders: unknown): number | null {
  if (typeof syntheticBars !== 'number' || !Number.isFinite(syntheticBars) || typeof auctionPlaceholders !== 'number' || !Number.isFinite(auctionPlaceholders)) return null
  return Math.max(0, Math.floor(syntheticBars) - Math.floor(auctionPlaceholders))
}

export function effectiveMinuteCompletenessLabel(diagnostics: Pick<DataDiagnostics, 'raw_complete' | 'missing_raw_minutes' | 'pending_calibration_days' | 'daily_days'>): string {
  if (diagnostics.raw_complete === true) return '已完整'
  if (diagnostics.raw_complete !== false) return '状态未提供'
  const issues: string[] = []
  if ((diagnostics.missing_raw_minutes ?? 0) > 0) issues.push('存在原始缺口')
  if ((diagnostics.pending_calibration_days ?? 0) > 0) issues.push('待校准')
  if ((diagnostics.daily_days ?? 0) > 0) issues.push('有日线降级')
  return issues.length ? issues.join(' · ') : '待完善'
}

export type CoverageDay = {
  mode?: string | null
  missing_minutes?: number | null
  missing_data_minutes?: number | null
  auction_placeholders?: number | null
  reason?: string | null
}

function isAuctionOnlyReason(reason: string): boolean {
  const text = reason.replace(/\s+/g, '')
  const mentionsAuction = /收盘集合竞价|竞价占位|14[:：]58.*14[:：]59/.test(text)
  const mentionsError = /失败|异常|错误|校验不通过|数据损坏|待校准/.test(text)
  return mentionsAuction && !mentionsError
}

function isLegacyMissingCountReason(reason: string): boolean {
  const text = reason.replace(/\s+/g, '')
  return /^(?:缺失|缺少|缺口)\d+(?:根|条)?分钟(?:数据)?$/.test(text)
}

export function coverageDayHasIssue(day: CoverageDay): boolean {
  const mode = (day.mode ?? '').trim().toLowerCase()
  if (mode !== 'minute' && mode !== 'complete') return true

  const hasExplicitMissingCount = typeof day.missing_data_minutes === 'number' && Number.isFinite(day.missing_data_minutes)
  const missingDataMinutes = hasExplicitMissingCount ? day.missing_data_minutes! : day.missing_minutes ?? 0
  if (missingDataMinutes > 0) return true

  const reason = (day.reason ?? '').trim()
  if (!reason || reason === '完整') return false

  const hasAuctionBreakdown = typeof day.auction_placeholders === 'number' && Number.isFinite(day.auction_placeholders) && day.auction_placeholders > 0
  if (hasExplicitMissingCount && missingDataMinutes === 0 && hasAuctionBreakdown && (isAuctionOnlyReason(reason) || isLegacyMissingCountReason(reason))) return false
  return true
}
