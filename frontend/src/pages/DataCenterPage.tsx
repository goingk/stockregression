import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CalendarClock, Database, Download, FileUp, RefreshCw, Search, Settings2, Trash2 } from 'lucide-react'
import { del, get, post } from '../api'
import { Badge, Button, Card, Empty, ErrorBox, Field, Loading, PageHeader, SelectInput, Stat, TextInput, compact, dateText, percent, useConfirm, useErrorMessage } from '../components'
import DataDiagnosticsPanel from '../DataDiagnosticsPanel'
import { actualRepairMinutes, coverageDayHasIssue, uniqueCurrentWarnings } from '../dataDiagnostics'
import MarketPanel from '../MarketPanel'
import Disclosure from '../Disclosure'
import { useTasks } from '../TaskCenter'
import { useWorkspaces } from '../WorkbenchState'
import type { Coverage, DataDiagnostics, Instrument, WatchItem } from '../types'

type PerformanceSettings = { cpu_budget_mode: 'auto' | 'custom'; cpu_budget?: number | null; download_workers: number; validation_workers: number; blas_threads: number; workspace_mb: number }
type SettingsResponse = { data_dir?: string; active_data_dir?: string; active_package_dir?: string; package_dir?: string | null; restart_required?: boolean; market_cache_mb?: number; market_cache?: { used_bytes?: number; budget_bytes?: number; hits?: number; misses?: number }; schedule?: { enabled?: boolean; daily_time?: string; weekly_day?: string; weekly_time?: string }; performance?: PerformanceSettings; [key: string]: unknown }
type ScheduleDraft = { daily_time: string; weekly_day: string; weekly_time: string }
const JOB_TERMINAL = new Set(['completed', 'partial', 'failed', 'cancelled'])

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

export default function DataCenterPage({ requestedReport, onReportHandled }: { requestedReport?: string | null; onReportHandled?: () => void }) {
  const tasks = useTasks()
  const workspace = useWorkspaces()
  const ui = (workspace.configuration.ui ?? {}) as Record<string, unknown>
  const refreshIntervalMs = Math.max(1000, Number(ui.data_refresh_interval_ms) || 5000)
  const jobs = tasks.jobs
  const [query, setQuery] = useState('')
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [watchlist, setWatchlist] = useState<WatchItem[]>([])
  const [activeSymbol, setActiveSymbol] = useState('')
  const [coverage, setCoverage] = useState<Coverage | null>(null)
  const [qualityFilter, setQualityFilter] = useState<'all' | 'issues'>('all')
  const [qualityPage, setQualityPage] = useState(1)
  const [settings, setSettings] = useState<SettingsResponse>({ schedule: { enabled: false, daily_time: '18:30', weekly_day: 'SAT', weekly_time: '10:00' } })
  const [scheduleDraft, setScheduleDraft] = useState<Partial<ScheduleDraft>>({})
  const [dataDirDraft, setDataDirDraft] = useState<string | null>(null)
  const [cacheMbDraft, setCacheMbDraft] = useState<number | null>(null)
  const [performanceDraft, setPerformanceDraft] = useState<PerformanceSettings | null>(null)
  const dataDirDraftEdited = useRef(false)
  const [syncMode, setSyncMode] = useState<'history' | 'daily' | 'repair'>('daily')
  const [syncTimeframe, setSyncTimeframe] = useState('1day')
  const [syncRange, setSyncRange] = useState({ start_date: '', end_date: '' })
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([])
  const [importForm, setImportForm] = useState({ symbol: '', minute_path: '', daily_path: '', actions_path: '', calendar_path: '', label_convention: 'auto', etf_category: '' })
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const err = useErrorMessage()
  const { confirm, node: confirmNode } = useConfirm()

  const refresh = useCallback(async () => {
    try {
      const [items, config] = await Promise.all([get<WatchItem[]>('/watchlist'), get<SettingsResponse>('/settings')])
      setWatchlist(items); setSettings(config)
      if (!dataDirDraftEdited.current) setDataDirDraft(config.data_dir ?? '')
      setCacheMbDraft(current => current ?? Number(config.market_cache_mb ?? 256))
      setPerformanceDraft(current => current ?? config.performance ?? { cpu_budget_mode: 'auto', cpu_budget: null, download_workers: 4, validation_workers: 4, blas_threads: 8, workspace_mb: 8192 })
      setActiveSymbol((old) => old || items.find(item => item.has_data !== false)?.symbol || items[0]?.symbol || '')
    } catch (e) { setError(err(e)) } finally { setLoading(false) }
  }, [err])

  useEffect(() => { void refresh() }, [refresh])
  const completedDataJobs = jobs.filter(job => ['instruments', 'market_sync', 'market_import', 'market_import_batch', 'market_index_package', 'market_validate', 'market_aggregate'].includes(job.kind) && JOB_TERMINAL.has(job.status)).map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => { void refresh() }, [refresh, completedDataJobs])
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh() }, refreshIntervalMs)
    return () => window.clearInterval(timer)
  }, [refresh, refreshIntervalMs])
  useEffect(() => {
    if (!activeSymbol) { setCoverage(null); return }
    let cancelled = false
    get<Coverage>(`/data/coverage/${encodeURIComponent(activeSymbol)}`).then((value) => { if (!cancelled) setCoverage(value) }).catch((e) => { if (!cancelled) setError(err(e)) })
    return () => { cancelled = true }
  }, [activeSymbol, err, completedDataJobs])
  useEffect(() => { setQualityPage(1) }, [activeSymbol, qualityFilter])

  const runAction = async (label: string, action: () => Promise<unknown>, success: (result: unknown) => string) => {
    setBusy(label); setError(''); setNotice('')
    try {
      const taskName = ({ refresh: '更新证券清单', sync: '行情同步', import: '本地文件导入', schedule: '更新本机计划' } as Record<string, string>)[label]
      const result = taskName ? await tasks.run(taskName, action) : await action()
      if (!taskName) setNotice(success(result))
      await refresh()
    }
    catch (e) { setError(err(e)) } finally { setBusy('') }
  }

  const search = async () => {
    setBusy('search'); setError('')
    try { const list = await get<Instrument[]>(`/instruments?q=${encodeURIComponent(query.trim())}`); setInstruments(list) }
    catch (e) { setError(err(e)) } finally { setBusy('') }
  }
  const addWatch = async (item: Instrument) => {
    await runAction('watch', () => post('/watchlist', { symbol: item.symbol, name: item.name }), () => `${item.name}（${item.symbol}）已加入自选。`)
    setActiveSymbol(item.symbol)
  }
  const removeWatch = async (symbol: string) => {
    if (!await confirm(`从自选中移除 ${symbol}？`)) return
    await runAction('remove', () => del(`/watchlist/${encodeURIComponent(symbol)}`), () => `${symbol} 已从自选中移除。`)
  }
  const toggleSelection = (symbol: string, checked: boolean) => setSelectedSymbols((list) => checked ? [...new Set([...list, symbol])] : list.filter((item) => item !== symbol))
  const allSelected = watchlist.length > 0 && watchlist.every((item) => selectedSymbols.includes(item.symbol))
  const symbolsToSync = selectedSymbols.length ? selectedSymbols : activeSymbol ? [activeSymbol] : []

  const syncData = () => {
    if (!symbolsToSync.length) { setError('请先选择一只股票，或在自选列表中勾选要同步的股票。'); return }
    if (syncMode !== 'daily' && !syncRange.start_date) { setError('历史导入和缺口修补需要填写开始日期。'); return }
    void runAction('sync', () => post<{ job_id?: string }>('/data/sync', { symbols: symbolsToSync, timeframe: syncTimeframe, mode: syncMode === 'daily' ? 'incremental' : syncMode, resume: true, start_date: syncRange.start_date || null, end_date: syncRange.end_date || null }), () => `已提交 ${syncTimeframe} ` + modeLabel(syncMode) + '任务。')
  }
  const importData = () => {
    if (!importForm.symbol.trim()) { setError('导入前请填写股票代码。'); return }
    if (!importForm.daily_path.trim() && !importForm.minute_path.trim()) { setError('请至少填写日线或分钟线文件路径。'); return }
    const common = {
      symbol: importForm.symbol.trim().toUpperCase(),
      ...(importForm.etf_category ? { etf_category: importForm.etf_category } : {}),
      ...(importForm.calendar_path.trim() ? { calendar_path: importForm.calendar_path.trim() } : {}),
    }
    void runAction('import', async () => {
      const jobs = []
      if (importForm.daily_path.trim()) jobs.push(await post('/data/import', { ...common, timeframe: '1day', path: importForm.daily_path.trim(), label_convention: 'end', ...(importForm.actions_path.trim() ? { actions_path: importForm.actions_path.trim() } : {}) }))
      if (importForm.minute_path.trim()) jobs.push(await post('/data/import', { ...common, timeframe: '1min', path: importForm.minute_path.trim(), label_convention: importForm.label_convention }))
      return { job_ids: jobs.map((job) => (job as { job_id: string }).job_id) }
    }, () => '日线和分钟线已作为独立周期任务提交，不会自动下载其他周期。')
  }
  const schedule = settings.schedule ?? {}
  const scheduleToSave = () => ({
    enabled: schedule.enabled ?? false,
    daily_time: scheduleDraft.daily_time ?? schedule.daily_time ?? '18:30',
    weekly_day: scheduleDraft.weekly_day ?? schedule.weekly_day ?? 'SAT',
    weekly_time: scheduleDraft.weekly_time ?? schedule.weekly_time ?? '10:00',
  })
  const setScheduleValue = (key: keyof ScheduleDraft, value: string) => setScheduleDraft((draft) => ({ ...draft, [key]: value }))
  const installSchedule = (install: boolean) => void runAction('schedule', async () => {
    if (install) {
      await workspace.saveConfiguration({ schedule: scheduleToSave() })
      setScheduleDraft({})
    }
    return post<{ message?: string }>(install ? '/schedule/install' : '/schedule/remove')
  }, (result) => (result as { message?: string }).message ?? (install ? '已请求安装本机定时任务。' : '已请求移除本机定时任务。'))
  const saveSchedule = () => void runAction('schedule', async () => {
    const savedSchedule = scheduleToSave()
    await workspace.saveConfiguration({ schedule: savedSchedule })
    setScheduleDraft({})
    return savedSchedule.enabled ? post<{ message?: string }>('/schedule/install') : { message: '更新计划配置已保存。' }
  }, (result) => (result as { message?: string }).message ?? '更新计划配置已保存。')
  const saveDataDirectory = () => {
    const path = (dataDirDraft ?? settings.data_dir ?? '').trim()
    if (!path) { setError('请填写下次启动的数据目录。'); return }
    void runAction('data-dir', async () => {
      await workspace.saveConfiguration({ paths: { data_dir: path } })
      const saved = await get<SettingsResponse>('/settings')
      setSettings(saved)
      dataDirDraftEdited.current = false
      setDataDirDraft(saved.data_dir ?? path)
      return saved
    }, () => '数据目录设置已保存。重启后生效，旧数据不会自动迁移。')
  }
  const saveCacheBudget = () => {
    const value = Math.max(0, Math.round(Number(cacheMbDraft ?? settings.market_cache_mb ?? 256)))
    void runAction('cache', async () => {
      await workspace.saveConfiguration({ market_data: { cache_mb: value } })
      const saved = await get<SettingsResponse>('/settings'); setSettings(saved); return saved
    }, () => `行情与预览缓存上限已设为 ${value} MiB。`)
  }
  const savePerformance = () => {
    const value = performanceDraft ?? { cpu_budget_mode: 'auto' as const, cpu_budget: null, download_workers: 4, validation_workers: 4, blas_threads: 8, workspace_mb: 8192 }
    void runAction('performance', async () => {
      await workspace.saveConfiguration({ runtime: value })
      const saved = await get<SettingsResponse>('/settings'); setSettings(saved); return saved
    }, () => '计算资源设置已保存，将在后续任务中生效。')
  }
  const setPerformanceValue = (key: keyof PerformanceSettings, value: string | number | null) => {
    setPerformanceDraft(current => ({ ...(current ?? { cpu_budget_mode: 'auto', cpu_budget: null, download_workers: 4, validation_workers: 4, blas_threads: 8, workspace_mb: 8192 }), [key]: value }))
  }
  const profiles = useMemo(() => {
    const value = coverage?.profiles
    if (Array.isArray(value)) return value.map((profile, index) => ({ source: String((profile as Record<string, unknown>)?.source ?? (profile as Record<string, unknown>)?.name ?? `来源 ${index + 1}`), profile }))
    if (value && typeof value === 'object') return Object.entries(value).map(([source, profile]) => ({ source, profile }))
    return []
  }, [coverage?.profiles])
  const filteredDays = useMemo(() => {
    const days = coverage?.days ?? []
    const filtered = qualityFilter === 'issues'
      ? days.filter(coverageDayHasIssue)
      : days
    return filtered.slice().reverse()
  }, [coverage?.days, qualityFilter])
  const pageSize = Math.max(10, Number(ui.data_health_page_size) || 100)
  const pageCount = Math.max(1, Math.ceil(filteredDays.length / pageSize))
  const currentQualityPage = Math.min(qualityPage, pageCount)
  const visibleDays = filteredDays.slice((currentQualityPage - 1) * pageSize, currentQualityPage * pageSize)
  const warningCount = coverage ? uniqueCurrentWarnings(coverage.diagnostics?.current_warnings, coverage.warnings).length : 0

  return <>
    <PageHeader eyebrow="MARKET" title="行情" description="浏览沪深股票与 ETF 的本地行情，维护自选并持续更新数据库。" action={<Button variant="secondary" busy={busy === 'refresh'} onClick={() => void runAction('refresh', () => post('/instruments/refresh'), () => '股票及 ETF 清单更新任务已提交。')}><RefreshCw size={15} />更新证券清单</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}{notice && <div className="notice success">{notice}</div>}
    <MarketPanel onSelect={setActiveSymbol} requestedReport={requestedReport} onReportHandled={onReportHandled} /><Disclosure id="market-admin" title="数据维护与本机设置" summary="覆盖质量 · 历史同步 · 文件导入 · 更新计划" className="market-admin"><div className="data-overview">
      <Card title="证券检索" subtitle="搜索沪深股票、ETF 并维护自选列表。">
        <div className="search-row"><TextInput value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && void search()} placeholder="输入代码或名称，例如 600000 / 浦发银行" /><Button busy={busy === 'search'} onClick={() => void search()}><Search size={15} />搜索</Button></div>
        {instruments.length > 0 && <div className="instrument-results">{instruments.map((item) => <div className="instrument-row" key={item.symbol}><div><strong>{item.name}</strong><small>{item.symbol} · {item.exchange} · {item.board}</small></div>{watchlist.some((w) => w.symbol === item.symbol) ? <Badge tone="green">已自选</Badge> : <Button variant="secondary" onClick={() => void addWatch(item)}>加入自选</Button>}</div>)}</div>}
        {instruments.length === 0 && <div className="search-hint"><span>支持证券代码或本地清单名称搜索</span><span>{watchlist.length} 只自选</span></div>}
      </Card>
      <Card title="自选证券" subtitle="选择标的查看覆盖；勾选后可批量同步。" action={<Badge tone="blue">{watchlist.length} 只</Badge>}>
        {loading ? <Loading /> : watchlist.length === 0 ? <Empty title="自选列表为空" detail="搜索股票或 ETF 后添加自选，再准备本地行情。" /> : <div className="watch-list">
          <div className="watch-select-all"><label className="check-label"><input type="checkbox" checked={allSelected} onChange={(e) => setSelectedSymbols(e.target.checked ? watchlist.map((x) => x.symbol) : [])} /><span>全选用于同步</span></label></div>
          {watchlist.map((item) => <div key={item.symbol} className={`watch-row ${activeSymbol === item.symbol ? 'selected' : ''}`}><input aria-label={`选择 ${item.symbol} 同步`} type="checkbox" checked={selectedSymbols.includes(item.symbol)} onChange={(e) => toggleSelection(item.symbol, e.target.checked)} /><button className="watch-main" onClick={() => setActiveSymbol(item.symbol)}><strong>{item.name}</strong><small>{item.symbol} · {item.exchange}</small></button><button className="icon-button danger-icon" aria-label="移除自选" onClick={() => void removeWatch(item.symbol)}><Trash2 size={15} /></button></div>)}
        </div>}
      </Card>
    </div>

    <div className="data-columns">
      <div className="main-column">
        <Card title="数据覆盖与质量" subtitle={activeSymbol ? `当前标的：${activeSymbol}` : '选择自选股票以读取覆盖日历。'} action={activeSymbol && <Button variant="quiet" onClick={() => { setError(''); setCoverage(null); void get<Coverage>(`/data/coverage/${encodeURIComponent(activeSymbol)}`).then(setCoverage).catch((e) => setError(err(e))) }}><RefreshCw size={14} />刷新</Button>}>
          {!activeSymbol ? <Empty title="请选择一只自选股票" detail="覆盖日历和来源校准状态会在此展示。" /> : !coverage ? <Loading label="读取覆盖情况…" /> : <>
            <CoverageQualityStats coverage={coverage} diagnostics={coverage.diagnostics} /><div className="stat-grid four"><Stat label="数据行数" value={compact(coverage.rows)} note={`版本 ${coverage.version ?? '—'}`} tone="blue" /><Stat label="首末日期" value={<>{dateText(coverage.first_date)}<br />{dateText(coverage.last_date)}</>} note="日历区间" /><Stat label="分钟覆盖日" value={coverage.diagnostics?.minute_days ?? coverage.minute_days ?? '—'} note={`日线降级 ${coverage.diagnostics?.daily_days ?? coverage.daily_days ?? '—'} 日`} /><Stat label="数据占用" value={prettyBytes(coverage.bytes)} note={`${warningCount} 条当前质量提示`} tone={warningCount ? 'red' : 'green'} /></div>
            {Number(coverage.price_anomaly_count ?? 0) > 0 && <p className="inline-warning">行情区间已隔离 {String(coverage.price_anomaly_count)} 条非正、空值或高低价关系错误的记录；受影响缺口不参与回测、研究或均线计算。</p>}
            <DataDiagnosticsPanel diagnostics={coverage.diagnostics} sourceEvents={coverage.source_events} warnings={coverage.warnings} syntheticBars={coverage.synthetic_bars} />
            <div className="coverage-summary"><div className="coverage-title"><CalendarClock size={16} /><strong>交易日覆盖日历</strong><span>{coverage.days?.length ?? 0} 日 · 默认最近 100 日</span></div>
              {coverage.days?.length ? <>
                <div className="coverage-pagination"><Field label="日历范围"><SelectInput value={qualityFilter} onChange={(e) => setQualityFilter(e.target.value as typeof qualityFilter)}><option value="all">全部交易日</option><option value="issues">仅问题日（降级、缺口、停牌）</option></SelectInput></Field><span className="coverage-page-count">{filteredDays.length} 日 · 每页 {pageSize} 日</span><div className="coverage-page-controls"><Button variant="quiet" disabled={currentQualityPage <= 1} onClick={() => setQualityPage((page) => Math.max(1, page - 1))}>上一页</Button><Field label="跳转页"><TextInput type="number" min={1} max={pageCount} value={currentQualityPage} onChange={(e) => setQualityPage(Math.max(1, Math.min(pageCount, Number(e.target.value) || 1)))} /></Field><span>/ {pageCount}</span><Button variant="quiet" disabled={currentQualityPage >= pageCount} onClick={() => setQualityPage((page) => Math.min(pageCount, page + 1))}>下一页</Button></div></div>
                {visibleDays.length ? <div className="table-wrap coverage-table"><table><thead><tr><th>交易日</th><th>数据模式</th><th>有效分钟缺失</th><th>竞价占位</th><th>说明</th></tr></thead><tbody>{visibleDays.map((day) => { const mode = day.mode ?? ''; const label = mode === 'minute' || mode === 'complete' ? '分钟数据' : mode === 'daily' ? '日线降级' : mode === 'suspended' ? '确认停牌' : mode || '未知'; return <tr key={day.date}><td>{day.date}</td><td><Badge tone={mode === 'minute' || mode === 'complete' ? 'green' : mode === 'daily' ? 'amber' : 'neutral'}>{label}</Badge></td><td>{day.missing_data_minutes ?? day.missing_minutes ?? 0}</td><td>{day.auction_placeholders ?? 0}</td><td className="muted-cell">{day.reason || '—'}</td></tr> })}</tbody></table></div> : <Empty title="当前筛选条件下没有交易日" />}
              </> : <Empty title="暂无逐日覆盖明细" />}
            </div>
            <div className="profile-section"><div className="coverage-title"><Settings2 size={16} /><strong>来源校准状态</strong><span>{profiles.length} 个档案</span></div>{profiles.length === 0 ? <p className="muted-copy">暂无来源校准档案。同步或导入数据后，校准状态会显示在这里。</p> : <div className="profile-grid">{profiles.map(({ source, profile }) => { const item = profile && typeof profile === 'object' ? profile as Record<string, unknown> : {}; const verified = item.verified === true || item.calibrated === true; const convention = item.convention ?? item.declared_convention; const sourceName: Record<string, string> = { hf: 'Hugging Face 分钟源', tdx: '通达信近期分钟源', sina: '新浪近期分钟备源', import: '本地导入分钟源' }; return <div className="profile-card" key={source}><div><strong>{sourceName[source] ?? source}</strong><Badge tone={verified ? 'green' : 'amber'}>{verified ? '已校准' : '待校准'}</Badge></div><div className="profile-summary"><span>时间标签：{convention === 'end' ? '结束标签' : convention === 'start' ? '开始标签' : '未确认'}</span><span>成交量倍率：{item.volume_multiplier == null ? '—' : `${String(item.volume_multiplier)}×`}</span><span>核验交易日：{item.verified_days == null ? '—' : String(item.verified_days)}</span>{item.match_score != null && <span>交叉匹配：{percent(Number(item.match_score))}</span>}</div>{item.reason != null && <p className="profile-reason">{String(item.reason)}</p>}<details className="profile-raw"><summary>查看原始档案</summary><pre>{JSON.stringify(profile, null, 2)}</pre></details></div> })}</div>}</div>
          </>}
        </Card>
        <Card title="数据同步" subtitle="按需历史导入、日常增量更新或窗口缺口修补。" className="sync-card">
          <div className="sync-controls"><Field label="数据周期"><SelectInput value={syncTimeframe} onChange={(e) => setSyncTimeframe(e.target.value)}><option value="1day">日线</option><option value="1min">1 分钟</option><option value="5min">5 分钟</option><option value="15min">15 分钟</option><option value="30min">30 分钟</option><option value="60min">60 分钟</option><option value="1week">周线</option><option value="1month">月线</option></SelectInput></Field><Field label="任务类型"><SelectInput value={syncMode} onChange={(e) => setSyncMode(e.target.value as typeof syncMode)}><option value="history">历史数据导入</option><option value="daily">日常增量更新</option><option value="repair">窗口缺口修补</option></SelectInput></Field><Field label="开始日期"><TextInput type="date" value={syncRange.start_date} disabled={syncMode === 'daily'} onChange={(e) => setSyncRange({ ...syncRange, start_date: e.target.value })} /></Field><Field label="结束日期"><TextInput type="date" value={syncRange.end_date} disabled={syncMode === 'daily'} onChange={(e) => setSyncRange({ ...syncRange, end_date: e.target.value })} /></Field><div className="sync-target"><span>同步目标</span><strong>{symbolsToSync.length ? symbolsToSync.join('、') : '未选择股票'}</strong><small>本次只处理 {syncTimeframe}</small></div><Button busy={busy === 'sync'} onClick={syncData}><Download size={15} />提交同步任务</Button></div>
          <div className="inline-note"><span className="info-dot">i</span><span>任务进度显示在右下角，历史和报告可从顶部任务入口查看。</span></div>
        </Card>
      </div>
      <aside className="side-column">
        <Card title="本地文件导入" subtitle="读取用户本机 CSV 或 Parquet 文件路径。" action={<Badge>手动</Badge>}>
          <div className="form-grid one"><Field label="证券代码"><TextInput value={importForm.symbol} onChange={(e) => setImportForm({ ...importForm, symbol: e.target.value.toUpperCase() })} placeholder="600000.SH" /></Field><Field label="ETF 类型（股票可忽略）"><SelectInput value={importForm.etf_category} onChange={event => setImportForm({ ...importForm, etf_category: event.target.value })}><option value="">按证券清单识别</option><option value="stock">股票 ETF（T+1）</option><option value="bond">债券 ETF（T+0）</option><option value="crossborder">跨境 ETF（T+0）</option><option value="commodity">黄金 / 商品 ETF（T+0）</option><option value="money">货币 ETF（T+0）</option></SelectInput></Field><Field label="1 分钟文件路径（可选）"><TextInput value={importForm.minute_path} onChange={(e) => setImportForm({ ...importForm, minute_path: e.target.value })} placeholder="D:\\market\\600000_1m.parquet" /></Field><Field label="日线文件路径（可选）" hint="日线与 1 分钟会作为两个独立任务提交；更粗周期请在“数据管理”中明确选择并导入。严格回测需 preclose、tradestatus，股票另需 is_st/isST"><TextInput value={importForm.daily_path} onChange={(e) => setImportForm({ ...importForm, daily_path: e.target.value })} placeholder="CSV / Parquet 路径" /></Field><Field label="分红送转文件路径"><TextInput value={importForm.actions_path} onChange={(e) => setImportForm({ ...importForm, actions_path: e.target.value })} placeholder="可选公司行动文件" /></Field><Field label="交易日历文件路径" hint="需为真实交易日；留空时数据仍可入库，但会标记质量待核验"><TextInput value={importForm.calendar_path} onChange={(e) => setImportForm({ ...importForm, calendar_path: e.target.value })} placeholder="真实交易日列表，列名 date" /></Field><Field label="分钟线时间标签" hint="不确定时保持自动；数据先入库并标记待核验，之后可在数据管理页确认标签。"><SelectInput value={importForm.label_convention} onChange={(e) => setImportForm({ ...importForm, label_convention: e.target.value })}><option value="auto">自动 / 先标记待核验</option><option value="end">我确认：结束时间标签</option><option value="start">我确认：开始时间标签</option></SelectInput></Field></div>
          <div className="card-bottom"><Button busy={busy === 'import'} onClick={importData}><FileUp size={15} />提交导入</Button></div>
        </Card>
        <Card title="本机存储与计划" subtitle="存储和定时更新配置保存在本地。" action={<Badge tone="blue">本机</Badge>}>
          <div className="storage-path"><Database size={16} /><div><small>当前活动数据包</small><strong title={settings.active_package_dir}>{settings.active_package_dir ?? '读取中或未配置'}</strong><small>行情、辅助资料和历史版本的唯一正式来源</small></div></div>
          <div className="storage-path"><Database size={16} /><div><small>应用数据目录</small><strong title={settings.active_data_dir}>{settings.active_data_dir ?? '读取中或未配置'}</strong><small>任务、配置、结果和可重建索引</small></div></div>
          <div className="schedule-box">
            <Field label="下次启动的数据目录" hint={settings.restart_required ? '已设置的新目录需要重启后生效；旧数据不会自动迁移。' : '保存不同的数据目录后需重启生效；旧数据不会自动迁移。'}><TextInput value={dataDirDraft ?? settings.data_dir ?? ''} onChange={(e) => { dataDirDraftEdited.current = true; setDataDirDraft(e.target.value) }} placeholder="例如 D:\\stockregression\\data" /></Field>
            <div className="card-bottom"><Button variant="secondary" busy={busy === 'data-dir'} onClick={saveDataDirectory}>保存数据目录</Button></div>
            <Field label="行情与预览缓存上限（MiB）" hint="按需使用，不会启动时预分配；0 表示关闭跨请求缓存。回测任务工作内存单独计算。"><TextInput type="number" min={0} max={16384} value={cacheMbDraft ?? settings.market_cache_mb ?? 256} onChange={e => setCacheMbDraft(Number(e.target.value))} /></Field>
            <div className="card-bottom"><Button variant="quiet" busy={busy === 'cache'} onClick={saveCacheBudget}>保存缓存设置</Button><span className="muted-copy">{settings.market_cache ? `已用 ${prettyBytes(settings.market_cache.used_bytes)} / ${prettyBytes(settings.market_cache.budget_bytes)}` : ''}</span></div>
            <div className="schedule-head"><div><strong>计算资源</strong><small>默认自动使用本机 75% 逻辑核心，为网页和日常操作保留余量</small></div><Badge tone="blue">本机</Badge></div>
            <div className="schedule-grid">
              <Field label="CPU 预算"><SelectInput value={performanceDraft?.cpu_budget_mode ?? 'auto'} onChange={e => setPerformanceValue('cpu_budget_mode', e.target.value)}><option value="auto">自动</option><option value="custom">指定线程</option></SelectInput></Field>
              <Field label="CPU 线程上限"><TextInput type="number" min={1} max={128} disabled={(performanceDraft?.cpu_budget_mode ?? 'auto') === 'auto'} value={performanceDraft?.cpu_budget ?? ''} onChange={e => setPerformanceValue('cpu_budget', e.target.value ? Number(e.target.value) : null)} /></Field>
              <Field label="下载并发"><TextInput type="number" min={1} max={8} value={performanceDraft?.download_workers ?? 4} onChange={e => setPerformanceValue('download_workers', Number(e.target.value))} /></Field>
              <Field label="校验进程"><TextInput type="number" min={1} max={4} value={performanceDraft?.validation_workers ?? 4} onChange={e => setPerformanceValue('validation_workers', Number(e.target.value))} /></Field>
              <Field label="矩阵线程"><TextInput type="number" min={1} max={16} value={performanceDraft?.blas_threads ?? 8} onChange={e => setPerformanceValue('blas_threads', Number(e.target.value))} /></Field>
              <Field label="任务工作集（MiB）"><TextInput type="number" min={256} max={65536} value={performanceDraft?.workspace_mb ?? 8192} onChange={e => setPerformanceValue('workspace_mb', Number(e.target.value))} /></Field>
            </div>
            <div className="card-bottom"><Button variant="quiet" busy={busy === 'performance'} onClick={savePerformance}>保存计算资源设置</Button></div>
          </div>
          <div className="schedule-box"><div className="schedule-head"><div><strong>自动更新计划</strong><small>{schedule.enabled ? '当前已启用' : '当前已停用'}</small></div><Badge tone={schedule.enabled ? 'green' : 'neutral'}>{schedule.enabled ? '启用' : '停用'}</Badge></div>
            <div className="schedule-grid"><Field label="每日时间"><TextInput type="time" value={scheduleDraft.daily_time ?? schedule.daily_time ?? '18:30'} onInput={(e) => setScheduleValue('daily_time', e.currentTarget.value)} onChange={(e) => setScheduleValue('daily_time', e.target.value)} /></Field><Field label="每周星期"><SelectInput value={scheduleDraft.weekly_day ?? schedule.weekly_day ?? 'SAT'} onChange={(e) => setScheduleValue('weekly_day', e.target.value)}>{[['MON', '周一'], ['TUE', '周二'], ['WED', '周三'], ['THU', '周四'], ['FRI', '周五'], ['SAT', '周六'], ['SUN', '周日']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}</SelectInput></Field><Field label="每周时间"><TextInput type="time" value={scheduleDraft.weekly_time ?? schedule.weekly_time ?? '10:00'} onInput={(e) => setScheduleValue('weekly_time', e.currentTarget.value)} onChange={(e) => setScheduleValue('weekly_time', e.target.value)} /></Field></div>
            <div className="button-row schedule-actions"><Button variant="secondary" busy={busy === 'schedule'} onClick={saveSchedule}>保存设置</Button>{schedule.enabled ? <Button variant="quiet" busy={busy === 'schedule'} onClick={() => installSchedule(false)}>移除系统计划</Button> : <Button busy={busy === 'schedule'} onClick={() => installSchedule(true)}><CalendarClock size={14} />安装系统计划</Button>}</div>
          </div>
        </Card>
      </aside>
    </div>

    </Disclosure>{confirmNode}
  </>
}

function modeLabel(mode: string) { return mode === 'history' ? '历史导入' : mode === 'daily' ? '日常更新' : '窗口修补' }
