import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CalendarClock, Database, Download, FileUp, RefreshCw, Search, Settings2, Sparkles, Trash2 } from 'lucide-react'
import { del, get, post, put } from '../api'
import { Badge, Button, Card, Empty, ErrorBox, Field, JobProgress, Loading, PageHeader, SelectInput, Stat, TextInput, compact, dateText, percent, useConfirm, useErrorMessage } from '../components'
import MarketPanel from '../MarketPanel'
import type { Coverage, Instrument, Job, WatchItem } from '../types'

type SettingsResponse = { data_dir?: string; active_data_dir?: string; restart_required?: boolean; schedule?: { enabled?: boolean; daily_time?: string; weekly_day?: string; weekly_time?: string }; [key: string]: unknown }
type ScheduleDraft = { daily_time: string; weekly_day: string; weekly_time: string }
const JOB_TERMINAL = new Set(['completed', 'failed', 'cancelled'])

function prettyBytes(value?: number) {
  if (value == null) return '—'
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(2)} GB`
}

function CoverageQualityStats({ coverage }: { coverage: Coverage }) {
  return <div className="stat-grid four quality-stats">
    <Stat label="分钟数据保留率" value={percent(coverage.minute_ratio)} note={String(coverage.minute_days ?? 0) + " 分钟覆盖日 / " + String(coverage.total_days ?? 0) + " 个交易日"} tone="blue" />
    <Stat label="确认停牌日" value={coverage.suspended_days ?? 0} note="来自交易状态与成交信息" />
    <Stat label="合成分钟条" value={coverage.synthetic_bars ?? 0} note="少量缺口的因果填补" />
    <Stat label="日线降级日" value={coverage.daily_days ?? 0} note="不还原盘中价格路径" tone={coverage.daily_days ? "amber" : "green"} />
  </div>
}

export default function DataCenterPage() {
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
  const dataDirDraftEdited = useRef(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const [syncMode, setSyncMode] = useState<'history' | 'daily' | 'repair'>('history')
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([])
  const [importForm, setImportForm] = useState({ symbol: '', minute_path: '', daily_path: '', actions_path: '', calendar_path: '', label_convention: 'auto', etf_category: '' })
  const [demoForm, setDemoForm] = useState({ symbol: '000001.SZ', days: 780 })
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const err = useErrorMessage()
  const { confirm, node: confirmNode } = useConfirm()

  const refresh = useCallback(async () => {
    try {
      const [items, config, allJobs] = await Promise.all([get<WatchItem[]>('/watchlist'), get<SettingsResponse>('/settings'), get<Job[]>('/jobs')])
      setWatchlist(items); setSettings(config); setJobs(allJobs.slice(0, 12))
      if (!dataDirDraftEdited.current) setDataDirDraft(config.data_dir ?? '')
      setActiveSymbol((old) => old || items[0]?.symbol || '')
    } catch (e) { setError(err(e)) } finally { setLoading(false) }
  }, [err])

  useEffect(() => { void refresh() }, [refresh])
  const completedDataJobs = jobs.filter(job => ['sync', 'instruments', 'import', 'demo'].includes(job.kind) && JOB_TERMINAL.has(job.status)).map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh() }, 3000)
    return () => window.clearInterval(timer)
  }, [refresh])
  useEffect(() => {
    if (!activeSymbol) { setCoverage(null); return }
    let cancelled = false
    get<Coverage>(`/data/coverage/${encodeURIComponent(activeSymbol)}`).then((value) => { if (!cancelled) setCoverage(value) }).catch((e) => { if (!cancelled) setError(err(e)) })
    return () => { cancelled = true }
  }, [activeSymbol, err, completedDataJobs])
  useEffect(() => { setQualityPage(1) }, [activeSymbol, qualityFilter])

  const runAction = async (label: string, action: () => Promise<unknown>, success: (result: unknown) => string) => {
    setBusy(label); setError(''); setNotice('')
    try { const result = await action(); setNotice(success(result)); await refresh() }
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
    void runAction('sync', () => post<{ job_id?: string }>('/data/sync', { symbols: symbolsToSync, mode: syncMode }), () => '已提交' + modeLabel(syncMode) + '任务。')
  }
  const importData = () => {
    if (!importForm.symbol.trim()) { setError('导入前请填写股票代码。'); return }
    const body: Record<string, string> = { symbol: importForm.symbol.trim().toUpperCase(), label_convention: importForm.label_convention }
    if (importForm.etf_category) body.etf_category = importForm.etf_category
    if (importForm.minute_path.trim()) body.minute_path = importForm.minute_path.trim()
    if (importForm.daily_path.trim()) body.daily_path = importForm.daily_path.trim()
    if (importForm.actions_path.trim()) body.actions_path = importForm.actions_path.trim()
    if (importForm.calendar_path.trim()) body.calendar_path = importForm.calendar_path.trim()
    if (!body.minute_path && !body.daily_path && !body.actions_path) { setError('请至少填写一个本地 CSV 或 Parquet 文件路径。'); return }
    void runAction('import', () => post('/data/import', body), () => '本地文件导入任务已提交。')
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
      await put('/settings', { schedule: scheduleToSave() })
      setScheduleDraft({})
    }
    return post<{ message?: string }>(install ? '/schedule/install' : '/schedule/remove')
  }, (result) => (result as { message?: string }).message ?? (install ? '已请求安装本机定时任务。' : '已请求移除本机定时任务。'))
  const saveSchedule = () => void runAction('schedule', async () => {
    const savedSchedule = scheduleToSave()
    await put('/settings', { schedule: savedSchedule })
    setScheduleDraft({})
    return savedSchedule.enabled ? post<{ message?: string }>('/schedule/install') : { message: '更新计划配置已保存。' }
  }, (result) => (result as { message?: string }).message ?? '更新计划配置已保存。')
  const saveDataDirectory = () => {
    const path = (dataDirDraft ?? settings.data_dir ?? '').trim()
    if (!path) { setError('请填写下次启动的数据目录。'); return }
    void runAction('data-dir', async () => {
      const saved = await put<SettingsResponse>('/settings', { data_dir: path })
      dataDirDraftEdited.current = false
      setDataDirDraft(saved.data_dir ?? path)
      return saved
    }, () => '数据目录设置已保存。重启后生效，旧数据不会自动迁移。')
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
      ? days.filter((day) => (day.mode !== 'minute' && day.mode !== 'complete') || Boolean(day.reason && day.reason !== '完整') || (day.missing_minutes ?? 0) > 0)
      : days
    return filtered.slice().reverse()
  }, [coverage?.days, qualityFilter])
  const pageSize = 100
  const pageCount = Math.max(1, Math.ceil(filteredDays.length / pageSize))
  const currentQualityPage = Math.min(qualityPage, pageCount)
  const visibleDays = filteredDays.slice((currentQualityPage - 1) * pageSize, currentQualityPage * pageSize)
  const warningCount = coverage?.warnings?.length ?? 0

  return <>
    <PageHeader eyebrow="MARKET" title="行情" description="浏览沪深股票与 ETF 的本地行情，维护自选并持续更新数据库。" action={<Button variant="secondary" busy={busy === 'refresh'} onClick={() => void runAction('refresh', () => post('/instruments/refresh'), () => '股票及 ETF 清单更新任务已提交。')}><RefreshCw size={15} />更新证券清单</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}{notice && <div className="notice success">{notice}</div>}
    <MarketPanel onSelect={setActiveSymbol} /><details className="market-admin"><summary>数据覆盖、历史同步、导入与本机更新计划</summary><div className="data-overview">
      <Card title="证券检索" subtitle="搜索沪深股票、ETF 并维护自选列表。">
        <div className="search-row"><TextInput value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && void search()} placeholder="输入代码或名称，例如 600000 / 浦发银行" /><Button busy={busy === 'search'} onClick={() => void search()}><Search size={15} />搜索</Button></div>
        {instruments.length > 0 && <div className="instrument-results">{instruments.map((item) => <div className="instrument-row" key={item.symbol}><div><strong>{item.name}</strong><small>{item.symbol} · {item.exchange} · {item.board}</small></div>{watchlist.some((w) => w.symbol === item.symbol) ? <Badge tone="green">已自选</Badge> : <Button variant="secondary" onClick={() => void addWatch(item)}>加入自选</Button>}</div>)}</div>}
        {instruments.length === 0 && <div className="search-hint"><span>支持证券代码或本地清单名称搜索</span><span>{watchlist.length} 只自选</span></div>}
      </Card>
      <Card title="自选证券" subtitle="选择标的查看覆盖；勾选后可批量同步。" action={<Badge tone="blue">{watchlist.length} 只</Badge>}>
        {loading ? <Loading /> : watchlist.length === 0 ? <Empty title="自选列表为空" detail="搜索股票后添加，或先生成演示数据。" /> : <div className="watch-list">
          <div className="watch-select-all"><label className="check-label"><input type="checkbox" checked={allSelected} onChange={(e) => setSelectedSymbols(e.target.checked ? watchlist.map((x) => x.symbol) : [])} /><span>全选用于同步</span></label></div>
          {watchlist.map((item) => <div key={item.symbol} className={`watch-row ${activeSymbol === item.symbol ? 'selected' : ''}`}><input aria-label={`选择 ${item.symbol} 同步`} type="checkbox" checked={selectedSymbols.includes(item.symbol)} onChange={(e) => toggleSelection(item.symbol, e.target.checked)} /><button className="watch-main" onClick={() => setActiveSymbol(item.symbol)}><strong>{item.name}</strong><small>{item.symbol} · {item.exchange}</small></button><button className="icon-button danger-icon" aria-label="移除自选" onClick={() => void removeWatch(item.symbol)}><Trash2 size={15} /></button></div>)}
        </div>}
      </Card>
    </div>

    <div className="data-columns">
      <div className="main-column">
        <Card title="数据覆盖与质量" subtitle={activeSymbol ? `当前标的：${activeSymbol}` : '选择自选股票以读取覆盖日历。'} action={activeSymbol && <Button variant="quiet" onClick={() => { setError(''); setCoverage(null); void get<Coverage>(`/data/coverage/${encodeURIComponent(activeSymbol)}`).then(setCoverage).catch((e) => setError(err(e))) }}><RefreshCw size={14} />刷新</Button>}>
          {!activeSymbol ? <Empty title="请选择一只自选股票" detail="覆盖日历和来源校准状态会在此展示。" /> : !coverage ? <Loading label="读取覆盖情况…" /> : <>
            {coverage.is_demo && <div className="demo-callout"><Sparkles size={16} /><span>演示数据由本机生成，不代表真实行情。</span></div>}<CoverageQualityStats coverage={coverage} /><div className="stat-grid four"><Stat label="数据行数" value={compact(coverage.rows)} note={`版本 ${coverage.version ?? '—'}`} tone="blue" /><Stat label="首末日期" value={<>{dateText(coverage.first_date)}<br />{dateText(coverage.last_date)}</>} note="日历区间" /><Stat label="分钟覆盖日" value={coverage.minute_days ?? '—'} note={`日线覆盖 ${coverage.daily_days ?? '—'} 日`} /><Stat label="数据占用" value={prettyBytes(coverage.bytes)} note={`${warningCount} 条质量提示`} tone={warningCount ? 'red' : 'green'} /></div>
            <div className="coverage-summary"><div className="coverage-title"><CalendarClock size={16} /><strong>交易日覆盖日历</strong><span>{coverage.days?.length ?? 0} 日 · 默认最近 100 日</span></div>
              {(coverage.warnings?.length ?? 0) > 0 && <div className="warning-list">{coverage.warnings?.map((warning, i) => <div key={i}><span className="warning-dot" />{warning}</div>)}</div>}
              {coverage.days?.length ? <>
                <div className="coverage-pagination"><Field label="日历范围"><SelectInput value={qualityFilter} onChange={(e) => setQualityFilter(e.target.value as typeof qualityFilter)}><option value="all">全部交易日</option><option value="issues">仅问题日（降级、缺口、停牌）</option></SelectInput></Field><span className="coverage-page-count">{filteredDays.length} 日 · 每页 {pageSize} 日</span><div className="coverage-page-controls"><Button variant="quiet" disabled={currentQualityPage <= 1} onClick={() => setQualityPage((page) => Math.max(1, page - 1))}>上一页</Button><Field label="跳转页"><TextInput type="number" min={1} max={pageCount} value={currentQualityPage} onChange={(e) => setQualityPage(Math.max(1, Math.min(pageCount, Number(e.target.value) || 1)))} /></Field><span>/ {pageCount}</span><Button variant="quiet" disabled={currentQualityPage >= pageCount} onClick={() => setQualityPage((page) => Math.min(pageCount, page + 1))}>下一页</Button></div></div>
                {visibleDays.length ? <div className="table-wrap coverage-table"><table><thead><tr><th>交易日</th><th>数据模式</th><th>缺失分钟</th><th>说明</th></tr></thead><tbody>{visibleDays.map((day) => { const mode = day.mode ?? ''; const label = mode === 'minute' || mode === 'complete' ? '分钟数据' : mode === 'daily' ? '日线降级' : mode === 'suspended' ? '确认停牌' : mode || '未知'; return <tr key={day.date}><td>{day.date}</td><td><Badge tone={mode === 'minute' || mode === 'complete' ? 'green' : mode === 'daily' ? 'amber' : 'neutral'}>{label}</Badge></td><td>{day.missing_minutes ?? 0}</td><td className="muted-cell">{day.reason || '—'}</td></tr> })}</tbody></table></div> : <Empty title="当前筛选条件下没有交易日" />}
              </> : <Empty title="暂无逐日覆盖明细" />}
            </div>
            <div className="profile-section"><div className="coverage-title"><Settings2 size={16} /><strong>来源校准状态</strong><span>{profiles.length} 个档案</span></div>{profiles.length === 0 ? <p className="muted-copy">暂无来源校准档案。同步或导入数据后，校准状态会显示在这里。</p> : <div className="profile-grid">{profiles.map(({ source, profile }) => { const item = profile && typeof profile === 'object' ? profile as Record<string, unknown> : {}; const verified = item.verified === true || item.calibrated === true; const convention = item.convention ?? item.declared_convention; const sourceName: Record<string, string> = { hf: 'Hugging Face 分钟源', tdx: '通达信近期分钟源', sina: '新浪近期分钟备源', import: '本地导入分钟源' }; return <div className="profile-card" key={source}><div><strong>{sourceName[source] ?? source}</strong><Badge tone={verified ? 'green' : 'amber'}>{verified ? '已校准' : '待校准'}</Badge></div><div className="profile-summary"><span>时间标签：{convention === 'end' ? '结束标签' : convention === 'start' ? '开始标签' : '未确认'}</span><span>成交量倍率：{item.volume_multiplier == null ? '—' : `${String(item.volume_multiplier)}×`}</span><span>核验交易日：{item.verified_days == null ? '—' : String(item.verified_days)}</span>{item.match_score != null && <span>交叉匹配：{percent(Number(item.match_score))}</span>}</div>{item.reason != null && <p className="profile-reason">{String(item.reason)}</p>}<details className="profile-raw"><summary>查看原始档案</summary><pre>{JSON.stringify(profile, null, 2)}</pre></details></div> })}</div>}</div>
          </>}
        </Card>
        <Card title="数据同步" subtitle="按需历史导入、日常增量更新或窗口缺口修补。" className="sync-card">
          <div className="sync-controls"><Field label="任务类型"><SelectInput value={syncMode} onChange={(e) => setSyncMode(e.target.value as typeof syncMode)}><option value="history">历史数据导入</option><option value="daily">日常增量更新</option><option value="repair">窗口缺口修补</option></SelectInput></Field><div className="sync-target"><span>同步目标</span><strong>{symbolsToSync.length ? symbolsToSync.join('、') : '未选择股票'}</strong><small>未勾选列表时，使用当前选中标的</small></div><Button busy={busy === 'sync'} onClick={syncData}><Download size={15} />提交同步任务</Button></div>
          <div className="inline-note"><span className="info-dot">i</span><span>任务完成后可在下方查看状态、进度和错误说明。</span></div>
        </Card>
      </div>
      <aside className="side-column">
        <Card title="本地文件导入" subtitle="读取用户本机 CSV 或 Parquet 文件路径。" action={<Badge>手动</Badge>}>
          <div className="form-grid one"><Field label="证券代码"><TextInput value={importForm.symbol} onChange={(e) => setImportForm({ ...importForm, symbol: e.target.value.toUpperCase() })} placeholder="600000.SH" /></Field><Field label="ETF 类型（股票可忽略）"><SelectInput value={importForm.etf_category} onChange={event => setImportForm({ ...importForm, etf_category: event.target.value })}><option value="">按证券清单识别</option><option value="stock">股票 ETF（T+1）</option><option value="bond">债券 ETF（T+0）</option><option value="crossborder">跨境 ETF（T+0）</option><option value="commodity">黄金 / 商品 ETF（T+0）</option><option value="money">货币 ETF（T+0）</option></SelectInput></Field><Field label="分钟线文件路径"><TextInput value={importForm.minute_path} onChange={(e) => setImportForm({ ...importForm, minute_path: e.target.value })} placeholder="D:\\market\\600000_1m.parquet" /></Field><Field label="日线文件路径" hint="首次导入必需；股票需 preclose、tradestatus、is_st/isST；ETF 无需 ST 字段"><TextInput value={importForm.daily_path} onChange={(e) => setImportForm({ ...importForm, daily_path: e.target.value })} placeholder="CSV / Parquet 路径" /></Field><Field label="分红送转文件路径"><TextInput value={importForm.actions_path} onChange={(e) => setImportForm({ ...importForm, actions_path: e.target.value })} placeholder="可选公司行动文件" /></Field><Field label="交易日历文件路径" hint="需为真实交易日；未填时将尝试在线获取"><TextInput value={importForm.calendar_path} onChange={(e) => setImportForm({ ...importForm, calendar_path: e.target.value })} placeholder="真实交易日列表，列名 date" /></Field><Field label="分钟线时间标签" hint="不确定时保持自动；如已核对数据源，再选择开始或结束时间标签。"><SelectInput value={importForm.label_convention} onChange={(e) => setImportForm({ ...importForm, label_convention: e.target.value })}><option value="auto">自动核验（无法确认时按日线）</option><option value="end">我确认：结束时间标签</option><option value="start">我确认：开始时间标签</option></SelectInput></Field></div>
          <div className="card-bottom"><Button busy={busy === 'import'} onClick={importData}><FileUp size={15} />提交导入</Button></div>
        </Card>
        <Card title="生成演示数据" subtitle="用于首次体验流程，演示数据由本机生成。" action={<Badge tone="amber">演示</Badge>}>
          <div className="demo-callout"><Sparkles size={17} /><span>所有演示数据会明确标记，不能代表真实行情。</span></div><div className="form-grid two"><Field label="证券代码"><TextInput value={demoForm.symbol} onChange={(e) => setDemoForm({ ...demoForm, symbol: e.target.value.toUpperCase() })} /></Field><Field label="交易日数量"><TextInput type="number" min={20} max={1600} value={demoForm.days} onChange={(e) => setDemoForm({ ...demoForm, days: Number(e.target.value) })} /></Field></div><div className="card-bottom"><Button variant="secondary" busy={busy === 'demo'} onClick={() => void runAction('demo', () => post('/data/demo', { symbol: demoForm.symbol || undefined, days: demoForm.days }), () => `已提交 ${demoForm.symbol} 的演示数据任务。`)}><Sparkles size={15} />生成演示数据</Button></div>
        </Card>
        <Card title="本机存储与计划" subtitle="存储和定时更新配置保存在本地。" action={<Badge tone="blue">本机</Badge>}>
          <div className="storage-path"><Database size={16} /><div><small>当前活动数据目录</small><strong title={settings.active_data_dir}>{settings.active_data_dir ?? '读取中或未配置'}</strong><small>本次运行正在使用的目录</small></div></div>
          <div className="schedule-box">
            <Field label="下次启动的数据目录" hint={settings.restart_required ? '已设置的新目录需要重启后生效；旧数据不会自动迁移。' : '保存不同的数据目录后需重启生效；旧数据不会自动迁移。'}><TextInput value={dataDirDraft ?? settings.data_dir ?? ''} onChange={(e) => { dataDirDraftEdited.current = true; setDataDirDraft(e.target.value) }} placeholder="例如 D:\\stockregression\\data" /></Field>
            <div className="card-bottom"><Button variant="secondary" busy={busy === 'data-dir'} onClick={saveDataDirectory}>保存数据目录</Button></div>
          </div>
          <div className="schedule-box"><div className="schedule-head"><div><strong>自动更新计划</strong><small>{schedule.enabled ? '当前已启用' : '当前已停用'}</small></div><Badge tone={schedule.enabled ? 'green' : 'neutral'}>{schedule.enabled ? '启用' : '停用'}</Badge></div>
            <div className="schedule-grid"><Field label="每日时间"><TextInput type="time" value={scheduleDraft.daily_time ?? schedule.daily_time ?? '18:30'} onInput={(e) => setScheduleValue('daily_time', e.currentTarget.value)} onChange={(e) => setScheduleValue('daily_time', e.target.value)} /></Field><Field label="每周星期"><SelectInput value={scheduleDraft.weekly_day ?? schedule.weekly_day ?? 'SAT'} onChange={(e) => setScheduleValue('weekly_day', e.target.value)}>{[['MON', '周一'], ['TUE', '周二'], ['WED', '周三'], ['THU', '周四'], ['FRI', '周五'], ['SAT', '周六'], ['SUN', '周日']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}</SelectInput></Field><Field label="每周时间"><TextInput type="time" value={scheduleDraft.weekly_time ?? schedule.weekly_time ?? '10:00'} onInput={(e) => setScheduleValue('weekly_time', e.currentTarget.value)} onChange={(e) => setScheduleValue('weekly_time', e.target.value)} /></Field></div>
            <div className="button-row schedule-actions"><Button variant="secondary" busy={busy === 'schedule'} onClick={saveSchedule}>保存设置</Button>{schedule.enabled ? <Button variant="quiet" busy={busy === 'schedule'} onClick={() => installSchedule(false)}>移除系统计划</Button> : <Button busy={busy === 'schedule'} onClick={() => installSchedule(true)}><CalendarClock size={14} />安装系统计划</Button>}</div>
          </div>
        </Card>
      </aside>
    </div>

    <Card title="任务日志" subtitle="任务会自动更新进度和状态；可在这里查看说明或取消运行中的任务。" action={<Button variant="quiet" onClick={() => { setError(''); void refresh() }}><RefreshCw size={14} />刷新</Button>}>
      {loading ? <Loading /> : jobs.length === 0 ? <Empty title="暂无任务日志" detail="执行同步、导入、定时安装或演示生成后会显示任务。" /> : <div className="table-wrap"><table><thead><tr><th>任务</th><th>创建时间</th><th>状态</th><th>进度</th><th>说明</th><th>操作</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.id}><td><strong>{job.kind}</strong><small className="table-sub">{job.id}</small></td><td>{dateText(job.created_at)}</td><td><Badge tone={job.status === 'completed' ? 'green' : job.status === 'failed' ? 'red' : JOB_TERMINAL.has(job.status) ? 'neutral' : 'blue'}>{job.status}</Badge></td><td><JobProgress progress={job.progress} status={job.status} /></td><td><span>{job.message || '—'}</span>{job.error != null && <small className="error-detail error-detail-cell">{typeof job.error === 'string' ? job.error : JSON.stringify(job.error, null, 2)}</small>}</td><td>{!JOB_TERMINAL.has(job.status) && <button className="text-link danger-text" onClick={() => void runAction('cancel', () => post(`/jobs/${encodeURIComponent(job.id)}/cancel`), () => '已请求取消任务。')}>取消</button>}</td></tr>)}</tbody></table></div>}
    </Card>
    </details>{confirmNode}
  </>
}

function modeLabel(mode: string) { return mode === 'history' ? '历史导入' : mode === 'daily' ? '日常更新' : '窗口修补' }
