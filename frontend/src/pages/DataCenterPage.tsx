import TableViewport from '../TableViewport'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Database, RefreshCw, Search, Settings2, Trash2 } from 'lucide-react'
import { del, get, post } from '../api'
import { LatestRequest } from '../requestState'
import { Badge, Button, Card, Empty, ErrorBox, Loading, PageHeader, Stat, TextInput, compact, dateText, percent, useConfirm, useErrorMessage } from '../components'
import DataDiagnosticsPanel from '../DataDiagnosticsPanel'
import { actualRepairMinutes, coverageDayHasIssue, uniqueCurrentWarnings } from '../dataDiagnostics'
import MarketPanel from '../MarketPanel'
import { useTasks } from '../TaskCenter'
import { useWorkspaces } from '../WorkbenchState'
import type { Coverage, DataDiagnostics, Instrument, WatchItem } from '../types'

const DATA_JOB_KINDS = new Set([
  'instruments', 'market_sync', 'market_import', 'market_import_batch',
  'market_import_package', 'market_index_package', 'market_validate',
  'market_aggregate', 'market_prepare', 'market_archive',
])
const TERMINAL = new Set(['completed', 'partial', 'failed', 'cancelled'])

function prettyBytes(value?: number) {
  if (value == null) return '—'
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(2)} GB`
}

function CoverageQualityStats({ coverage, diagnostics }: { coverage: Coverage; diagnostics?: DataDiagnostics }) {
  const minuteDays = diagnostics?.minute_days ?? coverage.minute_days
  const totalDays = diagnostics?.total_days ?? coverage.total_days
  const dailyDays = diagnostics?.daily_days ?? coverage.daily_days
  const minuteRatio = diagnostics?.minute_ratio ?? coverage.minute_ratio
  const auctionPlaceholders = diagnostics?.auction_placeholders
  const repairMinutes = actualRepairMinutes(coverage.synthetic_bars, auctionPlaceholders)
  return <div className="stat-grid four quality-stats">
    <Stat label="分钟数据保留率" value={percent(minuteRatio)} note={String(minuteDays ?? 0) + " 分钟覆盖日 / " + String(totalDays ?? 0) + " 个交易日"} tone="blue" />
    <Stat label="确认停牌日" value={coverage.suspended_days ?? 0} note="来自交易状态与成交信息" />
    <Stat label={repairMinutes == null ? "合成分钟条" : "实际修复分钟条"} value={repairMinutes ?? coverage.synthetic_bars ?? 0} note={repairMinutes == null ? "旧诊断未区分收盘竞价占位" : `${diagnostics?.repaired_days ?? 0} 个交易日含因果修复`} tone={repairMinutes == null ? "neutral" : repairMinutes > 0 ? "amber" : "green"} />
    <Stat label="收盘竞价占位" value={auctionPlaceholders ?? '—'} note="零量不可交易，不计缺失" />
    <Stat label="日线降级日" value={dailyDays ?? 0} note="不还原盘中价格路径" tone={dailyDays ? "amber" : "green"} />
  </div>
}

export default function DataCenterPage({ requestedReport, onReportHandled, onOpenDataManagement }: {
  requestedReport?: string | null
  onReportHandled?: () => void
  onOpenDataManagement?: () => void
}) {
  const tasks = useTasks()
  const workspace = useWorkspaces()
  const ui = (workspace.configuration.ui ?? {}) as Record<string, unknown>
  const refreshIntervalMs = Math.max(1000, Number(ui.data_refresh_interval_ms) || 5000)
  const qualityPageSize = Math.max(10, Number(ui.data_health_page_size) || 100)
  const [query, setQuery] = useState('')
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [watchlist, setWatchlist] = useState<WatchItem[]>([])
  const [activeSymbol, setActiveSymbol] = useState('')
  const [coverage, setCoverage] = useState<Coverage | null>(null)
  const [watchLoading, setWatchLoading] = useState(true)
  const [searchLoading, setSearchLoading] = useState(false)
  const [coverageLoading, setCoverageLoading] = useState(false)
  const [watchError, setWatchError] = useState('')
  const [searchError, setSearchError] = useState('')
  const [coverageError, setCoverageError] = useState('')
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState('')
  const [qualityFilter, setQualityFilter] = useState<'all' | 'issues'>('all')
  const [qualityPage, setQualityPage] = useState(1)
  const coverageRequest = useRef(new LatestRequest<string>())
  const searchRequest = useRef(new LatestRequest<string>())
  const activeSymbolRef = useRef(activeSymbol)
  activeSymbolRef.current = activeSymbol
  const err = useErrorMessage()
  const { confirm, node: confirmNode } = useConfirm()

  const loadWatchlist = useCallback(async () => {
    setWatchLoading(true); setWatchError('')
    try {
      const items = await get<WatchItem[]>('/watchlist')
      setWatchlist(items)
      setActiveSymbol(current => current || items.find(item => item.has_data !== false)?.symbol || items[0]?.symbol || '')
    } catch (cause) { setWatchError(err(cause)) } finally { setWatchLoading(false) }
  }, [err])
  const loadCoverage = useCallback(async (symbol: string) => {
    const ticket = coverageRequest.current.begin(symbol)
    if (!symbol) { setCoverage(null); setCoverageLoading(false); setCoverageError(''); return }
    setCoverageLoading(true); setCoverageError('')
    try {
      const value = await get<Coverage>(`/data/coverage/${encodeURIComponent(symbol)}`)
      if (!coverageRequest.current.isCurrent(ticket) || activeSymbolRef.current !== symbol) return
      if (String(value?.symbol ?? symbol).toUpperCase() !== symbol.toUpperCase()) return
      setCoverage(value)
    } catch (cause) {
      if (coverageRequest.current.isCurrent(ticket) && activeSymbolRef.current === symbol) setCoverageError(err(cause))
    } finally {
      if (coverageRequest.current.isCurrent(ticket)) setCoverageLoading(false)
    }
  }, [err])
  useEffect(() => { void loadWatchlist() }, [loadWatchlist])
  useEffect(() => {
    void loadCoverage(activeSymbol)
    return () => coverageRequest.current.invalidate(activeSymbol)
  }, [activeSymbol, loadCoverage])
  useEffect(() => {
    const timer = window.setInterval(() => { void loadWatchlist() }, refreshIntervalMs)
    return () => window.clearInterval(timer)
  }, [loadWatchlist, refreshIntervalMs])
  useEffect(() => { setQualityPage(1) }, [activeSymbol, qualityFilter])

  const completedDataJobs = tasks.jobs
    .filter(job => DATA_JOB_KINDS.has(job.kind) && TERMINAL.has(job.status))
    .map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => {
    if (!completedDataJobs) return
    void loadWatchlist()
    if (activeSymbol) void loadCoverage(activeSymbol)
  }, [activeSymbol, completedDataJobs, loadCoverage, loadWatchlist])

  const search = async () => {
    const value = query.trim()
    const ticket = searchRequest.current.begin(value)
    setSearchLoading(true); setSearchError('')
    try {
      const list = await get<Instrument[]>(`/instruments?q=${encodeURIComponent(value)}`)
      if (searchRequest.current.isCurrent(ticket)) setInstruments(list)
    } catch (cause) {
      if (searchRequest.current.isCurrent(ticket)) setSearchError(err(cause))
    } finally {
      if (searchRequest.current.isCurrent(ticket)) setSearchLoading(false)
    }
  }
  const addWatch = async (item: Instrument) => {
    setBusy('watch'); setWatchError(''); setNotice('')
    try {
      await post('/watchlist', { symbol: item.symbol, name: item.name })
      setActiveSymbol(item.symbol)
      setNotice(`${item.name}（${item.symbol}）已加入自选。`)
      await loadWatchlist()
    } catch (cause) { setWatchError(err(cause)) } finally { setBusy('') }
  }
  const removeWatch = async (symbol: string) => {
    if (!await confirm(`从自选中移除 ${symbol}？`)) return
    setBusy('remove'); setWatchError(''); setNotice('')
    try {
      await del(`/watchlist/${encodeURIComponent(symbol)}`)
      if (activeSymbol === symbol) { setActiveSymbol(''); setCoverage(null) }
      setNotice(`${symbol} 已从自选中移除。`)
      await loadWatchlist()
    } catch (cause) { setWatchError(err(cause)) } finally { setBusy('') }
  }

  const profiles = useMemo(() => {
    const value = coverage?.profiles
    if (Array.isArray(value)) return value.map((profile, index) => ({ source: String((profile as Record<string, unknown>)?.source ?? (profile as Record<string, unknown>)?.name ?? `来源 ${index + 1}`), profile }))
    if (value && typeof value === 'object') return Object.entries(value).map(([source, profile]) => ({ source, profile }))
    return []
  }, [coverage?.profiles])
  const filteredDays = useMemo(() => (qualityFilter === 'issues' ? (coverage?.days ?? []).filter(coverageDayHasIssue) : coverage?.days ?? []).slice().reverse(), [coverage?.days, qualityFilter])
  const pageCount = Math.max(1, Math.ceil(filteredDays.length / qualityPageSize))
  const currentQualityPage = Math.min(qualityPage, pageCount)
  const visibleDays = filteredDays.slice((currentQualityPage - 1) * qualityPageSize, currentQualityPage * qualityPageSize)
  const warningCount = coverage ? uniqueCurrentWarnings(coverage.diagnostics?.current_warnings, coverage.warnings).length : 0

  return <>
    <PageHeader eyebrow="MARKET" title="行情" description="浏览本地行情、维护自选并查看覆盖质量；完整同步、导入、来源与运行设置集中在数据管理。" action={<Button variant="secondary" onClick={onOpenDataManagement}><Settings2 size={15} />打开数据管理</Button>} />
    {watchError && <ErrorBox message={'自选或覆盖状态读取失败：' + watchError} onRetry={() => void loadWatchlist()} />}
    {notice && <div className="notice success"><span>{notice}</span><button className="text-link" onClick={() => setNotice('')}>收起</button></div>}
    <MarketPanel onSelect={setActiveSymbol} requestedReport={requestedReport} onReportHandled={onReportHandled} />
    <div className="data-overview market-shortcuts">
      <Card title="证券检索" subtitle="搜索沪深股票、ETF 并维护自选列表。">
        <div className="search-row"><TextInput value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => event.key === 'Enter' && void search()} placeholder="输入代码或名称，例如 600000 / 浦发银行" /><Button busy={searchLoading} onClick={() => void search()}><Search size={15} />搜索</Button></div>
        {searchError && <ErrorBox message={searchError} onRetry={() => void search()} />}
        {instruments.length > 0 && <div className="instrument-results">{instruments.map(item => <div className="instrument-row" key={item.symbol}><div><strong>{item.name}</strong><small>{item.symbol} · {item.exchange} · {item.board}</small></div>{watchlist.some(watch => watch.symbol === item.symbol) ? <Badge tone="green">已自选</Badge> : <Button variant="secondary" busy={busy === 'watch'} onClick={() => void addWatch(item)}>加入自选</Button>}</div>)}</div>}
        {instruments.length === 0 && !searchError && <div className="search-hint"><span>支持证券代码或本地清单名称搜索</span><span>{watchlist.length} 只自选</span></div>}
      </Card>
      <Card title="自选证券" subtitle="选择标的查看覆盖；维护操作统一在数据管理页完成。" action={<Badge tone="blue">{watchlist.length} 只</Badge>}>
        {watchLoading && !watchlist.length ? <Loading /> : watchlist.length === 0 ? <Empty title="自选列表为空" detail="搜索股票或 ETF 后添加自选，再准备本地行情。" /> : <div className="watch-list">{watchlist.map(item => <div key={item.symbol} className={`watch-row ${activeSymbol === item.symbol ? 'selected' : ''}`}><button className="watch-main" onClick={() => setActiveSymbol(item.symbol)}><strong>{item.name}</strong><small>{item.symbol} · {item.exchange}</small></button><button className="icon-button danger-icon" aria-label="移除自选" disabled={busy === 'remove'} onClick={() => void removeWatch(item.symbol)}><Trash2 size={15} /></button></div>)}</div>}
      </Card>
    </div>
    <Card title="数据覆盖与质量" subtitle={activeSymbol ? `当前标的：${activeSymbol}` : '选择自选股票以读取覆盖日历。'} action={activeSymbol && <Button variant="quiet" busy={coverageLoading} onClick={() => void loadCoverage(activeSymbol)}><RefreshCw size={14} />刷新</Button>}>
      {!activeSymbol ? <Empty title="请选择一只自选股票" detail="覆盖日历和来源校准状态会在此展示。" /> : coverageError ? <ErrorBox message={'覆盖情况读取失败：' + coverageError} onRetry={() => void loadCoverage(activeSymbol)} /> : !coverage && coverageLoading ? <Loading label="读取覆盖情况…" /> : coverage ? <>
        {coverageLoading && <p className="inline-warning">正在刷新 {activeSymbol} 的覆盖情况…</p>}
        <CoverageQualityStats coverage={coverage} diagnostics={coverage.diagnostics} />
        <div className="stat-grid four"><Stat label="数据行数" value={compact(coverage.rows)} note={`版本 ${coverage.version ? (String(coverage.version).length > 12 ? String(coverage.version).slice(0, 12) + '…' : coverage.version) : '—'}`} tone="blue" /><Stat label="首末日期" value={<span className="stat-date-range">{dateText(coverage.first_date)}<br />至 {dateText(coverage.last_date)}</span>} note="日历区间" /><Stat label="分钟覆盖日" value={coverage.diagnostics?.minute_days ?? coverage.minute_days ?? '—'} note={`日线降级 ${coverage.diagnostics?.daily_days ?? coverage.daily_days ?? '—'} 日`} /><Stat label="数据占用" value={prettyBytes(coverage.bytes)} note={`${warningCount} 条当前质量提示`} tone={warningCount ? 'red' : 'green'} /></div>
        {Number(coverage.price_anomaly_count ?? 0) > 0 && <p className="inline-warning">行情区间已隔离 {String(coverage.price_anomaly_count)} 条非正、空值或高低价关系错误的记录；受影响缺口不参与回测、研究或均线计算。</p>}
        <DataDiagnosticsPanel diagnostics={coverage.diagnostics} sourceEvents={coverage.source_events} warnings={coverage.warnings} syntheticBars={coverage.synthetic_bars} />
        {profiles.length > 0 && <div className="profile-section"><div className="coverage-title"><Database size={16} /><strong>来源校准状态</strong><span>{profiles.length} 个档案</span></div><div className="profile-grid">{profiles.map(({ source, profile }) => { const item = profile && typeof profile === 'object' ? profile as Record<string, unknown> : {}; const verified = item.verified === true || item.calibrated === true; const convention = item.convention ?? item.declared_convention; const sourceName: Record<string, string> = { hf: 'Hugging Face 分钟源', tdx: '通达信近期分钟源', sina: '新浪近期分钟备源', import: '本地导入分钟源' }; return <div className="profile-card" key={source}><div><strong>{sourceName[source] ?? source}</strong><Badge tone={verified ? 'green' : 'amber'}>{verified ? '已校准' : '待校准'}</Badge></div><div className="profile-summary"><span>时间标签：{convention === 'end' ? '结束标签' : convention === 'start' ? '开始标签' : '未确认'}</span><span>成交量倍率：{item.volume_multiplier == null ? '—' : `${String(item.volume_multiplier)}×`}</span><span>核验交易日：{item.verified_days == null ? '—' : String(item.verified_days)}</span>{item.match_score != null && <span>交叉匹配：{percent(Number(item.match_score))}</span>}</div>{item.reason != null && <p className="profile-reason">{String(item.reason)}</p>}<details className="profile-raw"><summary>查看原始档案</summary><pre>{JSON.stringify(profile, null, 2)}</pre></details></div> })}</div></div>}
        {coverage.days?.length ? <details className="advanced-data" open><summary>逐日覆盖（{filteredDays.length} 日）</summary><div className="button-row"><div className="segmented"><button className={qualityFilter === 'all' ? 'selected' : ''} onClick={() => setQualityFilter('all')}>全部</button><button className={qualityFilter === 'issues' ? 'selected' : ''} onClick={() => setQualityFilter('issues')}>仅异常</button></div></div><TableViewport className="table-wrap compact-coverage" owner='local' stickyKeyColumn={1}><table><thead><tr><th>日期</th><th>状态</th><th>分钟 / 缺口</th><th>说明</th></tr></thead><tbody>{visibleDays.map(day => <tr key={day.date}><td>{day.date}</td><td>{coverageDayHasIssue(day) ? <Badge tone="amber">需关注</Badge> : <Badge tone="green">正常</Badge>}</td><td>{String(day.missing_minutes ?? '—')} / {String(day.missing_data_minutes ?? '—')}</td><td>{String(day.reason ?? '—')}</td></tr>)}</tbody></table></TableViewport><div className="pagination-row"><span>第 {currentQualityPage} / {pageCount} 页</span><div><Button variant="quiet" disabled={qualityPage <= 1} onClick={() => setQualityPage(page => Math.max(1, page - 1))}>上一页</Button><Button variant="quiet" disabled={currentQualityPage >= pageCount} onClick={() => setQualityPage(page => page + 1)}>下一页</Button></div></div></details> : null}
      </> : <Empty title="没有覆盖详情" detail="刷新覆盖或前往数据管理检查该标的的数据包。" />}
    </Card>
    {confirmNode}
  </>
}
