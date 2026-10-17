import { Badge } from './components'
import { actualRepairMinutes, diagnosticsStatusLabel, effectiveMinuteCompletenessLabel, sourceEventLabel, sourceSpanLabel, uniqueCurrentWarnings, uniqueSourceEvents } from './dataDiagnostics'
import type { DataDiagnostics, DataSourceEvent } from './types'

function count(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : '—'
}

function ratio(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—'
}

function verificationLabel(verified?: boolean | null): string {
  if (verified === true) return '已验证'
  if (verified === false) return '待验证'
  return '验证状态未提供'
}

function eventStatusLabel(status?: DataSourceEvent['status']): string {
  if (status === 'resolved') return '已解决'
  if (status === 'historical') return '历史记录'
  return status || '记录'
}

export default function DataDiagnosticsPanel({
  diagnostics,
  sourceEvents,
  warnings,
  syntheticBars,
}: {
  diagnostics?: DataDiagnostics | null
  sourceEvents?: DataSourceEvent[] | null
  warnings?: unknown
  syntheticBars?: unknown
}) {
  const currentWarnings = uniqueCurrentWarnings(diagnostics?.current_warnings, warnings)
  const events = uniqueSourceEvents(sourceEvents)
  if (!diagnostics && currentWarnings.length === 0 && events.length === 0) return null

  const minuteApplicable = diagnostics?.minute_quality_applicable !== false
  const statusTone = diagnostics?.status === 'complete' ? 'green' : diagnostics?.status === 'partial' ? 'amber' : 'neutral'
  const spans = diagnostics?.source_spans ?? []
  const repairMinutes = minuteApplicable ? actualRepairMinutes(syntheticBars, diagnostics?.auction_placeholders) : null
  const checkedRange = diagnostics?.checked_start || diagnostics?.checked_end
    ? `${diagnostics.checked_start ?? '—'} 至 ${diagnostics.checked_end ?? '—'}`
    : '核验区间未提供'

  return <section className="data-diagnostics" aria-label="本地数据诊断">
    {diagnostics && <>
      <div className="data-diagnostics-heading"><strong>本地数据当前核验</strong><Badge tone={statusTone}>{diagnosticsStatusLabel(diagnostics.status)}</Badge><span>{checkedRange}</span></div>
      <div className="data-diagnostics-metrics">
        <span><small>分钟覆盖</small><strong>{minuteApplicable ? `${count(diagnostics.minute_days)} 日` : '本次未校验'}</strong></span>
        <span><small>{minuteApplicable ? '日线降级' : '所选日线模式'}</small><strong>{count(diagnostics.daily_days)} 日</strong></span>
        <span><small>交易日总数</small><strong>{count(diagnostics.total_days)} 日</strong></span>
        <span><small>分钟覆盖率</small><strong>{minuteApplicable ? ratio(diagnostics.minute_ratio) : '本次未校验'}</strong></span>
        <span><small>待校准分钟日</small><strong>{minuteApplicable ? count(diagnostics.pending_calibration_days) : '本次未校验'}</strong></span>
        <span><small>数据截至</small><strong>{minuteApplicable ? `分钟 ${diagnostics.minute_through ?? '—'} · 日线 ${diagnostics.daily_through ?? '—'}` : `日线 ${diagnostics.daily_through ?? '—'}`}</strong></span>
        <span className={minuteApplicable && diagnostics.raw_complete === false ? 'data-raw-incomplete' : ''}><small>有效交易分钟完整性</small><strong>{minuteApplicable ? effectiveMinuteCompletenessLabel(diagnostics) : '本次未校验'}</strong></span>
        <span><small>原始分钟缺失</small><strong>{minuteApplicable ? `${count(diagnostics.missing_raw_minutes)} 根` : '本次未校验'}</strong></span>
        <span><small>实际因果修复</small><strong>{minuteApplicable ? repairMinutes == null ? '—' : `${repairMinutes} 根` : '本次未校验'}</strong></span>
        <span><small>修复涉及日期</small><strong>{minuteApplicable ? `${count(diagnostics.repaired_days)} 日` : '本次未校验'}</strong></span>
        <span><small>收盘集合竞价占位</small><strong>{minuteApplicable ? `${count(diagnostics.auction_placeholders)} 根` : '本次未校验'}</strong></span>
      </div>
      {minuteApplicable && (diagnostics.auction_placeholders ?? 0) > 0 && <p className="data-auction-note">14:58、14:59 收盘集合竞价零量占位按非交易分钟处理，不计为原始缺失或实际修复。</p>}
      {spans.length > 0 && <div className="data-source-spans">
        <div className="data-diagnostics-subtitle">{minuteApplicable ? '本地分钟来源覆盖' : '本地数据来源覆盖'}</div>
        {spans.map((span, index) => <div className="data-source-span" key={`${span.source_kind ?? ''}-${span.source ?? ''}-${span.start ?? ''}-${span.end ?? ''}-${index}`}>
          <strong>{sourceSpanLabel(span)}</strong>
          <span>{span.start ?? '—'} 至 {span.end ?? '—'} · {count(span.days)} 日</span>
          <Badge tone={span.verified === true ? 'green' : span.verified === false ? 'amber' : 'neutral'}>{verificationLabel(span.verified)}</Badge>
        </div>)}
      </div>}
    </>}
    {currentWarnings.length > 0 && <div className="warning-list" aria-label="当前有效数据问题">{currentWarnings.map((warning, index) => <div key={`${warning}-${index}`}><span className="warning-dot" />{warning}</div>)}</div>}
    {repairMinutes != null && repairMinutes > 0 && <p className="data-repair-note">当前数据包含 {count(diagnostics?.missing_raw_minutes)} 根原始分钟缺失；扣除收盘集合竞价占位后，已因果修复 {repairMinutes} 根，涉及 {count(diagnostics?.repaired_days)} 个交易日。</p>}
    {events.length > 0 && <details className="data-source-events">
      <summary>来源请求与回退记录（{events.length}）</summary>
      <p>这些记录说明历史来源请求或回退过程；当前数据状态以本地核验结果为准。</p>
      <div className="data-source-event-list">{events.map((event, index) => <div className="data-source-event" key={`${event.kind ?? ''}-${event.source ?? ''}-${event.at ?? ''}-${index}`}>
        <div><strong>{sourceEventLabel(event)}</strong><Badge tone="neutral">{eventStatusLabel(event.status)}</Badge></div>
        <span>{event.at ?? '时间未提供'}</span>
        {event.message && <p>{event.message}</p>}
      </div>)}</div>
    </details>}
  </section>
}
