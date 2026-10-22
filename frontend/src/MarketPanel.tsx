import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, Star, X } from 'lucide-react'
import { del, get, post } from './api'
import ChartView from './ChartView'
import { Badge, Button, Card, Empty, ErrorBox, Field, SelectInput, TextInput, compact } from './components'
import type { Frequency, Instrument, WatchItem } from './types'
import Disclosure, { usePersistentFlag } from './Disclosure'
import { useTasks } from './TaskCenter'
import { chartValue, priceText, validPrice, validVolume } from './marketValues'
import { useWorkbenchDefaults } from './useWorkbenchDefaults'

type Bar = { timestamp: string; open: number; close: number; high: number; low: number; volume: number; [key: string]: unknown }
type MarketFrequency = Frequency | '1week' | '1month'
type MarketData = { quote?: { close: number; preclose: number }; symbol: string; version?: string; bars: Bar[]; has_more?: boolean; next_before?: string; warnings?: string[]; quality_flags?: string[]; is_demo?: boolean; instrument?: Instrument }

export default function MarketPanel({ onSelect, requestedReport, onReportHandled }: { onSelect?: (symbol: string) => void; requestedReport?: string | null; onReportHandled?: () => void }) {
  const tasks = useTasks()
  const configuration = useWorkbenchDefaults()
  const ui = (configuration.document.ui ?? {}) as Record<string, unknown>
  const refreshIntervalMs = Math.max(1000, Number(ui.data_refresh_interval_ms) || 5000)
  const lookupLimit = Math.max(5, Number(ui.instrument_lookup_limit) || 40)
  const lookupDebounceMs = Math.max(50, Number(ui.instrument_lookup_debounce_ms) || 250)
  const chartPoints = Math.max(100, Number(ui.chart_points) || 2000)
  const configuredMa = Array.isArray(ui.default_ma) ? (ui.default_ma as unknown[]).filter(value => Number.isInteger(value) && Number(value) > 0).map(Number) : [5, 20, 60]
  const jobs = tasks.jobs.filter(task => ['instruments', 'market_sync', 'market_import', 'market_import_batch', 'market_index_package'].includes(task.kind))
  const [sidebarCollapsed, setSidebarCollapsed] = usePersistentFlag('market-sidebar-collapsed')
  const [watchlist, setWatchlist] = useState<WatchItem[]>([])
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState('')
  const [found, setFound] = useState<Instrument[]>([])
  const [active, setActive] = useState<Instrument | null>(null)
  const [frequency, setFrequency] = useState<MarketFrequency>('1d')
  const [maDraft, setMaDraft] = useState(() => localStorage.getItem('market-ma') ?? configuredMa.join(','))
  const [periods, setPeriods] = useState(() => {
    const values = [...new Set(maDraft.split(/[,，\s]+/).filter(Boolean).map(Number))]
    return values.every(value => Number.isInteger(value) && value > 0 && value <= 500) ? values : configuredMa
  })
  const [market, setMarket] = useState<MarketData | null>(null)
  const [report, setReport] = useState<{ message?: string; warnings?: string[]; symbols?: Array<{ symbol: string; last_date?: string; minute_through?: string; warnings?: string[] }>; failures?: Array<{ symbol: string; reason: string }> } | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [refreshToken, setRefreshToken] = useState(0)
  const chartRevision = useRef(0)
  const refresh = useCallback(async () => {
    const items = await get<WatchItem[]>('/watchlist')
    setWatchlist(items)
    setActive(old => old ?? items.find(item => item.has_data !== false) ?? items[0] ?? null)
  }, [])
  const finished = jobs.filter(job => ['completed', 'partial', 'failed', 'cancelled'].includes(job.status)).map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => {
    void refresh().catch(error => setError(String(error)))
    const timer = window.setInterval(() => { void refresh().catch(() => undefined) }, refreshIntervalMs)
    return () => window.clearInterval(timer)
  }, [refresh, finished, refreshIntervalMs])
  useEffect(() => {
    if (!requestedReport) return
    let active = true
    get<NonNullable<typeof report>>('/results/' + encodeURIComponent(requestedReport)).then(value => {
      if (active) { setReport(value); onReportHandled?.() }
    }).catch(error => { if (active) { setError(String(error)); onReportHandled?.() } })
    return () => { active = false }
  }, [requestedReport, onReportHandled])
  useEffect(() => {
    if (!report) return
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setReport(null) }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [report])

  useEffect(() => { if (active) onSelect?.(active.symbol) }, [active?.symbol, onSelect])
  useEffect(() => {
    let valid = true
    const timer = setTimeout(() => { get<Instrument[]>(`/instruments?q=${encodeURIComponent(query)}${kind ? `&asset_type=${kind}` : ''}&limit=${lookupLimit}`).then(items => { if (valid) { setFound(items); setActive(old => old ?? items[0] ?? null) } }).catch(error => { if (valid) setError(String(error)) }) }, lookupDebounceMs)
    return () => { valid = false; clearTimeout(timer) }
  }, [query, kind, finished, lookupLimit, lookupDebounceMs])
  useEffect(() => {
    if (!active) { setMarket(null); return }
    let valid = true
    chartRevision.current += 1
    setLoading(true); setError(''); setMarket(null)
    get<MarketData>(`/market/bars/${encodeURIComponent(active.symbol)}?frequency=${frequency}&limit=${chartPoints}&ma=${periods.join(',')}`).then(value => { if (valid) setMarket(value) }).catch(error => { if (valid) { setMarket(null); setError(String(error)) } }).finally(() => { if (valid) setLoading(false) })
    return () => { valid = false }
  }, [active?.symbol, frequency, periods, finished, refreshToken, chartPoints])
  const update = async (scope: string) => {
    setBusy(true); setError('')
    try { await tasks.run(scope === 'market' ? '全市场行情更新' : scope === 'watchlist' ? '自选行情更新' : '当前标的更新', () => post('/data/sync', { scope, mode: 'incremental', frequency, resume: true, symbols: scope === 'selected' && active ? [active.symbol] : [] })); await refresh() }
    catch (error) { setError(String(error)) } finally { setBusy(false) }
  }
  const toggle = async (item: Instrument) => {
    setError('')
    try { if (watchlist.some(watch => watch.symbol === item.symbol)) await del(`/watchlist/${item.symbol}`); else await post('/watchlist', { symbol: item.symbol, name: item.name }); await refresh() }
    catch (error) { setError(String(error)) }
  }
  const earlier = async () => {
    if (!active || !market?.next_before) return
    const revision = chartRevision.current
    setLoading(true)
    try {
      const previous = await get<MarketData>(`/market/bars/${active.symbol}?frequency=${frequency}&ma=${periods.join(',')}&before=${encodeURIComponent(market.next_before)}`)
      if (revision === chartRevision.current) setMarket(current => current?.symbol === previous.symbol ? { ...previous, quote: current.quote, bars: [...previous.bars, ...current.bars] } : current)
    } catch (error) { if (revision === chartRevision.current) setError(String(error)) } finally { if (revision === chartRevision.current) setLoading(false) }
  }
  const bars = market?.bars ?? []
  const latest = [...bars].reverse().find(bar => [bar.open, bar.high, bar.low, bar.close].every(value => validPrice(value) != null))
  const latestIndex = latest ? bars.lastIndexOf(latest) : -1
  const previous = latestIndex > 0 ? [...bars.slice(0, latestIndex)].reverse().find(bar => validPrice(bar.close) != null) : undefined
  const quoteClose = validPrice(market?.quote?.close), quotePreclose = validPrice(market?.quote?.preclose)
  const change = quoteClose != null && quotePreclose ? (quoteClose / quotePreclose - 1) * 100 : validPrice(latest?.close) != null && validPrice(previous?.close) != null ? (validPrice(latest!.close)! / validPrice(previous!.close)! - 1) * 100 : null
  const chart: EChartsOption = useMemo(() => {
    const labels = bars.map(bar => bar.timestamp.replace('T', ' ').slice(0, frequency === '1d' ? 10 : 16))
    return {
      animation: false, legend: { top: 0, right: 0 }, tooltip: {
        trigger: 'axis', axisPointer: { type: 'cross' },
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;',
        formatter: value => {
        const item = Array.isArray(value) ? value[0] : value
        const bar = bars[item?.dataIndex]
        if (!bar) return ''
        const digits = market?.instrument?.asset_type === 'etf' ? 3 : 2
        return `${labels[item.dataIndex]}<br/>开 ${priceText(bar.open, digits)}　收 ${priceText(bar.close, digits)}<br/>高 ${priceText(bar.high, digits)}　低 ${priceText(bar.low, digits)}<br/>成交量 ${validVolume(bar.volume) == null ? '—' : compact(validVolume(bar.volume))}${periods.map(period => `<br/>MA${period} ${bar[`ma${period}`] == null ? '—' : priceText(bar[`ma${period}`], digits)}`).join('')}${bar.quality_note ? `<br/>${bar.quality_note}` : ''}`
      } }, axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
      grid: [{ left: 60, right: 24, top: 45, height: '59%' }, { left: 60, right: 24, top: '77%', height: '12%' }],
      xAxis: [0, 1].map(index => ({ type: 'category', data: labels, gridIndex: index, boundaryGap: false, axisLabel: { show: index === 1, hideOverlap: true } })),
      yAxis: [{ scale: true, splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } } }, { scale: true, gridIndex: 1, splitNumber: 2, axisLabel: { formatter: (value: number) => compact(value) }, splitLine: { show: false } }],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1], start: 65, end: 100 },
        { type: 'slider', xAxisIndex: [0, 1], height: 18, bottom: 0, start: 65, end: 100, fillerColor: 'rgba(37, 99, 235, 0.10)', borderColor: '#e2e8f0' },
      ],
      series: [{ name: 'K线', type: 'candlestick', data: bars.map(bar => [chartValue(bar.open), chartValue(bar.close), chartValue(bar.low), chartValue(bar.high)]), itemStyle: { color: '#e11d48', color0: '#0d9488', borderColor: '#e11d48', borderColor0: '#0d9488' } }, ...periods.map(period => ({ name: `MA${period}`, type: 'line' as const, showSymbol: false, connectNulls: false, data: bars.map(bar => validPrice(bar[`ma${period}`])), lineStyle: { width: 1 } })), { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: bars.map(bar => ({ value: validVolume(bar.volume), itemStyle: { color: (validPrice(bar.close) ?? 0) >= (validPrice(bar.open) ?? 0) ? '#fda4af' : '#5eead4' } })) }],
    }
  }, [market, frequency, periods])
  const items = query.trim() || kind ? found : watchlist.length ? watchlist : found
  return <>
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    <div className={`market-layout ${sidebarCollapsed ? 'market-sidebar-collapsed' : ''}`}>{!sidebarCollapsed && <Card title={query.trim() || kind ? '搜索证券' : '自选与证券'} className="market-sidebar" action={<Badge tone="blue">{watchlist.length} 自选</Badge>}>
      <TextInput aria-label="搜索股票或ETF" placeholder="代码 / 名称" value={query} onChange={event => setQuery(event.target.value)} />
      <div className="market-filter"><SelectInput value={kind} onChange={event => setKind(event.target.value)}><option value="">全部类型</option><option value="stock">股票</option><option value="etf">ETF</option></SelectInput></div>
      {!items.length ? <Empty title="暂无证券" detail="先更新证券清单，或在数据管理中导入行情。" /> : <div className="market-symbol-list">{items.map(item => <div key={item.symbol} className={item.symbol === active?.symbol ? 'active' : ''}><button className="market-symbol" onClick={() => setActive(item)}><strong>{item.name}</strong><small>{item.symbol} · {item.asset_type === 'etf' ? 'ETF' : '股票'}</small></button><button className="icon-button" aria-label={watchlist.some(watch => watch.symbol === item.symbol) ? '移除自选' : '加入自选'} onClick={() => void toggle(item)}>{watchlist.some(watch => watch.symbol === item.symbol) ? <Star size={15} fill="#d6a13a" color="#d6a13a" /> : <Plus size={15} />}</button></div>)}</div>}
    </Card>}<Card title={active ? `${active.name} · ${active.symbol}` : '本地行情'} subtitle="已收盘行情 · 股票和 ETF · 红涨绿跌" className="market-chart-card" action={<div className="button-row"><Button variant="quiet" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-expanded={!sidebarCollapsed}>{sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}自选列表</Button><Button busy={busy} onClick={() => void update('market')}><RefreshCw size={14} />更新数据包已有证券</Button></div>}>
      <div className="market-toolbar"><Field label="周期"><SelectInput value={frequency} onChange={event => setFrequency(event.target.value as Frequency)}>{['1month', '1week', '1d', '1m', '5m', '15m', '30m', '60m'].map(value => <option key={value}>{value}</option>)}</SelectInput></Field><Field label="均线周期（逗号分隔，留空关闭）"><TextInput value={maDraft} onChange={event => setMaDraft(event.target.value)} onBlur={() => { const next = [...new Set(maDraft.split(/[,，\s]+/).filter(Boolean).map(Number))]; if (next.every(value => Number.isInteger(value) && value > 0 && value <= 500)) { setPeriods(next); localStorage.setItem('market-ma', next.join(',')) } else setError('均线周期须为 1–500 的整数') }} /></Field><Button variant="quiet" disabled={!active || busy} onClick={() => void update('selected')}>更新当前</Button><Button variant="quiet" disabled={!watchlist.length || busy} onClick={() => void update('watchlist')}>更新自选</Button></div>
      {latest && <div className="quote-strip"><strong className={change != null && change < 0 ? 'loss' : 'gain'}>{priceText(latest.close, market?.instrument?.asset_type === 'etf' ? 3 : 2)}</strong><span className={change != null ? (change < 0 ? 'quote-change loss' : 'quote-change gain') : 'quote-change'}>{change == null ? '—' : `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`}</span><span>开 {priceText(latest.open, market?.instrument?.asset_type === 'etf' ? 3 : 2)}</span><span>高 {priceText(latest.high, market?.instrument?.asset_type === 'etf' ? 3 : 2)}</span><span>低 {priceText(latest.low, market?.instrument?.asset_type === 'etf' ? 3 : 2)}</span><span>量 {validVolume(latest.volume) == null ? '—' : compact(validVolume(latest.volume))}</span><span>{latest.timestamp.slice(0, 16).replace('T', ' ')}</span>{market?.is_demo && <Badge tone="amber">历史结果来源含演示数据</Badge>}</div>}
      {loading && <div className="market-loading">正在读取行情…</div>}
      {bars.length ? <ChartView option={chart} height={370} stretch /> : <Empty title={loading ? '正在读取行情' : '该标的尚无所选周期的行情'} detail="可更新当前标的，或在数据管理中执行历史同步 / 导入。" />}
      <div className="market-chart-footer"><Button variant="quiet" disabled={!market?.has_more || loading} onClick={() => void earlier()}>加载更早行情</Button><span>{bars.length} 根真实周期 K 线 · {market?.version == null ? '无版本' : String(market.version).slice(0, 10)}</span></div>
      {market?.quality_flags?.map((flag, index) => <p className="inline-warning" key={index}>{flag}</p>)}
      {market?.warnings?.length ? <Disclosure id="market-sources" title="来源与覆盖说明" summary={`${market.warnings.length} 条`}>{market.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</Disclosure> : null}
    </Card></div>
    {report && <div className="task-drawer-overlay" onMouseDown={event => { if (event.target === event.currentTarget) setReport(null) }}><section className="task-drawer report-drawer" role="dialog" aria-modal="true" aria-label="数据任务报告"><Card title="数据任务报告" subtitle={report.message} action={<Button variant="quiet" onClick={() => setReport(null)}><X size={14} />关闭报告</Button>}>{report.warnings?.map((warning, index) => <p key={index} className="inline-warning">{warning}</p>)}<div className="table-wrap scroll-table"><table><thead><tr><th>标的</th><th>日线截止</th><th>分钟末条</th><th>覆盖 / 来源说明</th></tr></thead><tbody>{report.symbols?.map(item => <tr key={item.symbol}><td>{item.symbol}</td><td>{item.last_date ?? '—'}</td><td>{item.minute_through ?? '无分钟数据'}</td><td>{item.warnings?.join('；') || '请结合来源校准及逐日覆盖检查'}</td></tr>)}{report.failures?.map(item => <tr key={item.symbol}><td>{item.symbol}</td><td colSpan={3} className="loss">失败：{item.reason}</td></tr>)}</tbody></table></div>{!report.symbols && !report.failures && <pre className="json-preview">{JSON.stringify(report, null, 2)}</pre>}</Card></section></div>}
    <button className="text-link market-refresh" onClick={() => setRefreshToken(value => value + 1)}>刷新本地视图</button>
  </>
}
