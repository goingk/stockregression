import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { Download, RefreshCw } from 'lucide-react'
import { download, get } from '../api'
import ChartView from '../ChartView'
import { Badge, Button, Card, Empty, ErrorBox, Loading, PageHeader, SelectInput, money, percent, useErrorMessage } from '../components'
import type { BacktestResult, EquityPoint, ResultSummary, StrategyResult, Trade } from '../types'

type Props = { requestedResult: string | null; clearRequestedResult: () => void }
const names: Record<string, string> = { grid: '网格策略', dca: '定期定额', buy_hold: '买入并持有' }
const colors: Record<string, string> = { grid: '#2864dc', dca: '#e85c63', buy_hold: '#8795aa' }
const reasonNames: Record<string, string> = {
  initial_position: '初始建仓', dca: '定期购买', buy_hold: '初始全仓', grid_buy: '网格买入', grid_sell: '网格卖出',
  dca_period_skipped: '该期无可成交时点，未投入资金', non_tradable_bar: '不可交易行情', suspended_day: '当日停牌', invalid_open: '开盘价无效', zero_volume: '无成交量',
  limit_up_locked: '开盘封涨停', limit_down_locked: '开盘封跌停', invalid_execution_price: '成交价格无效',
  below_lot: '不足最小申报数量', position_limit: '仓位限制', position_limit_or_below_lot: '仓位限制或不足最小数量',
  insufficient_cash: '现金不足', t_plus_1: '股份尚未满足 T+1', insufficient_sellable_shares: '可卖股份不足',
}
const reasonText = (value: unknown) => reasonNames[String(value)] ?? String(value ?? '—')
const sideText = (value: unknown) => value === 'buy' ? '买入' : value === 'sell' ? '卖出' : String(value ?? '—')
const metricLabels: Record<string, string> = { total_return: '总收益率', cagr: '年化收益', max_drawdown: '最大回撤', sharpe: '夏普比率', calmar: '卡玛比率', trade_count: '成交笔数', total_fees: '总手续费', turnover: '换手率', buy_count: '买入次数', sell_count: '卖出次数' }
type Strategy = { key: string; name: string; value: StrategyResult }
const normalized = (points: EquityPoint[], base: number) => points.map((p) => [p.timestamp, p.normalized_equity !== undefined ? p.normalized_equity : base ? p.equity / base : p.equity])
const tone = (key: string, value: number) => ['total_return', 'cagr'].includes(key) ? value >= 0 ? 'gain' : 'loss' : key === 'max_drawdown' ? 'loss' : ''
const timestampMs = (value: unknown) => {
  const text = String(value ?? '').trim().replace(' ', 'T')
  const parsed = Date.parse(text)
  return Number.isFinite(parsed) ? parsed : Number.NaN
}

function bucketIndex(timestamp: unknown, bucketTimes: number[], bucketLabels: string[], frequency: string) {
  if (!bucketTimes.length) return -1
  const target = timestampMs(timestamp)
  if (!Number.isFinite(target)) return -1
  const tradeDay = String(timestamp ?? '').slice(0, 10)
  if (frequency === '1d') {
    const sameDay = bucketLabels.findIndex((label) => label.slice(0, 10) === tradeDay)
    if (sameDay >= 0) return sameDay
  }
  let low = 0
  let high = bucketTimes.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (bucketTimes[middle] < target) low = middle + 1
    else high = middle
  }
  return Math.min(low, bucketTimes.length - 1)
}

function periodRows(strategies: Strategy[], period: 'year' | 'month') {
  const all: Record<string, Record<string, number>> = {}
  for (const strategy of strategies) {
    if (strategy.value.return_basis === 'cost') continue
    const groups: Record<string, number[]> = {}
    for (const point of strategy.value.period_returns ?? []) {
      const timestamp = String(point.timestamp ?? '')
      if (!timestamp) continue
      const day = timestamp.slice(0, 10)
      const key = period === 'year' ? day.slice(0, 4) : day.slice(0, 7)
      const dailyReturn = Number(point.return)
      if (Number.isFinite(dailyReturn)) (groups[key] ??= []).push(dailyReturn)
    }
    if (Object.keys(groups).length) {
      for (const [key, returns] of Object.entries(groups)) {
        ;(all[key] ??= {})[strategy.key] = returns.reduce((growth, value) => growth * (1 + value), 1) - 1
      }
    } else {
      const fallback: Record<string, EquityPoint[]> = {}
      for (const point of strategy.value.equity ?? []) {
        const day = String(point.timestamp).slice(0, 10)
        const key = period === 'year' ? day.slice(0, 4) : day.slice(0, 7)
        ;(fallback[key] ??= []).push(point)
      }
      for (const [key, points] of Object.entries(fallback)) {
        const first = points[0]?.equity ?? 0
        const last = points[points.length - 1]?.equity ?? 0
        ;(all[key] ??= {})[strategy.key] = first ? last / first - 1 : 0
      }
    }
  }
  return Object.entries(all).sort(([a], [b]) => b.localeCompare(a))
}

export default function ResultsPage({ requestedResult, clearRequestedResult }: Props) {
  const [results, setResults] = useState<ResultSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [bars, setBars] = useState<Array<Record<string, unknown>>>([])
  const [loading, setLoading] = useState(true)
  const [barsLoading, setBarsLoading] = useState(false)
  const [unit, setUnit] = useState<'year' | 'month'>('year')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const err = useErrorMessage()
  const strategies: Strategy[] = useMemo(() => result ? Object.entries(result.strategies ?? {}).map(([key, value]) => ({ key, name: value.return_basis === 'cost' ? '定投（成本收益）' : names[key] ?? key, value })) : [], [result])
  const [tradeStrategy, setTradeStrategy] = useState('grid')
  const reference = strategies.find((x) => x.key === tradeStrategy) ?? strategies[0]
  const isDemoResult = result?.is_demo === true || result?.quality?.is_demo === true || Boolean(result?.warnings?.some((warning) => warning.includes('演示数据')))
  const equityFrequency = result?.equity_frequency ?? Object.values(result?.strategies ?? {}).find((strategy) => strategy.equity_frequency)?.equity_frequency
  const curvePeriod = equityFrequency === '1d' ? '每日收盘' : '回测时间点'
  const shareUnit = /^[15]/.test(result?.symbol ?? '') ? '份' : '股'
  const hasPositionRatio = strategies.length > 0 && strategies.every((strategy) => (strategy.value.equity ?? []).every((point) => typeof point.position_ratio === 'number'))

  useEffect(() => {
    let active = true
    get<ResultSummary[]>('/results').then((list) => {
      if (!active) return
      const backtests = list.filter((item) => item.kind === 'backtest')
      setResults(backtests)
      setSelectedId(requestedResult && backtests.some((x) => x.id === requestedResult) ? requestedResult : backtests[0]?.id ?? '')
    }).catch((e) => { if (active) setError(err(e)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [requestedResult, err])
  useEffect(() => { if (requestedResult) clearRequestedResult() }, [requestedResult, clearRequestedResult])

  useEffect(() => {
    if (!selectedId) { setResult(null); return }
    let active = true
    setLoading(true); setResult(null); setError('')
    get<BacktestResult>('/results/' + encodeURIComponent(selectedId)).then((value) => { if (active) setResult(value) }).catch((e) => { if (active) setError(err(e)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [selectedId, err])
  useEffect(() => {
    if (!result) { setBars([]); return }
    let active = true
    setBarsLoading(true)
    const params = new URLSearchParams({ start_date: result.config.start_date, frequency: result.config.frequency, limit: '1500' })
    if (result.data_version) params.set('version', result.data_version)
    if (result.effective_end_date ?? result.config.end_date) params.set('end_date', (result.effective_end_date ?? result.config.end_date)!)
    get<Array<Record<string, unknown>>>('/data/bars/' + encodeURIComponent(result.symbol) + '?' + params).then((data) => { if (active) setBars(data) }).catch((e) => { if (active) setError('K 线读取失败：' + err(e)) }).finally(() => { if (active) setBarsLoading(false) })
    return () => { active = false }
  }, [result, err])

  const exportFile = async (kind: 'json' | 'equity' | 'trades') => {
    if (!selectedId) return
    setBusy(kind); setError('')
    try { await download('/results/' + encodeURIComponent(selectedId) + '/export?kind=' + kind, (result?.symbol ?? 'result') + '-' + kind + (kind === 'json' ? '.json' : '.csv')) }
    catch (e) { setError(err(e)) } finally { setBusy('') }
  }

  const equityOption: EChartsOption = useMemo(() => ({
    tooltip: { trigger: 'axis', valueFormatter: (v) => Number(v).toFixed(3) + '×' }, legend: { top: 2, right: 4 },
    grid: { left: 55, right: 20, top: 50, bottom: 36 }, xAxis: { type: 'time', axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', axisLabel: { formatter: (v: number) => v.toFixed(1) + 'x' }, splitLine: { lineStyle: { color: '#eef1f5' } } },
    dataZoom: [{ type: 'inside' }], series: strategies.filter(s => s.value.return_basis !== 'cost').map((s) => ({ name: s.name, type: 'line' as const, showSymbol: false, data: normalized(s.value.equity ?? [], result?.config.initial_cash ?? 1), itemStyle: { color: colors[s.key] ?? '#2864dc' }, lineStyle: { color: colors[s.key] ?? '#2864dc', width: s.key === 'grid' ? 2.4 : 1.7 } })),
  }), [strategies])
  const drawdownOption: EChartsOption = useMemo(() => ({
    tooltip: { trigger: 'axis', valueFormatter: (v) => percent(Number(v)) }, legend: { top: 2, right: 4 }, grid: { left: 55, right: 20, top: 50, bottom: 30 }, xAxis: { type: 'time', axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#eef1f5' } } },
    series: strategies.filter(s => s.value.return_basis !== 'cost').map((s) => ({ name: s.name, type: 'line' as const, showSymbol: false, data: (s.value.equity ?? []).map((p) => [p.timestamp, p.drawdown ?? 0]), itemStyle: { color: colors[s.key] ?? '#2864dc' }, lineStyle: { color: colors[s.key] ?? '#2864dc' }, areaStyle: s.key === 'grid' ? { opacity: 0.08 } : undefined })),
  }), [strategies])
  const positionOption: EChartsOption = useMemo(() => ({
    tooltip: { trigger: 'axis', valueFormatter: (v) => hasPositionRatio ? percent(Number(v)) : Number(v).toLocaleString('zh-CN') + ` ${shareUnit}` }, legend: { top: 2, right: 4 }, grid: { left: 55, right: 20, top: 50, bottom: 30 }, xAxis: { type: 'time', axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', axisLabel: { formatter: (v: number) => hasPositionRatio ? (v * 100).toFixed(0) + '%' : v.toLocaleString('zh-CN') }, splitLine: { lineStyle: { color: '#eef1f5' } } },
    series: strategies.map((s) => ({ name: s.name, type: 'line' as const, showSymbol: false, data: (s.value.equity ?? []).map((p) => [p.timestamp, hasPositionRatio ? p.position_ratio ?? 0 : p.position ?? 0]), itemStyle: { color: colors[s.key] ?? '#2864dc' }, lineStyle: { color: colors[s.key] ?? '#2864dc' } })),
  }), [strategies, hasPositionRatio])
  const candleOption: EChartsOption = useMemo(() => {
    const rows = bars.map((b) => [String(b.timestamp ?? '').replace('T', ' ').slice(0, 16), Number(b.open ?? 0), Number(b.close ?? 0), Number(b.low ?? 0), Number(b.high ?? 0)])
    const labels = rows.map((row) => String(row[0]))
    const bucketTimes = bars.map((bar) => timestampMs(bar.timestamp))
    const trades = (reference?.value.trades ?? []) as Trade[]
    const markers = trades.flatMap((trade) => {
      const index = bucketIndex(trade.timestamp, bucketTimes, labels, result?.config.frequency ?? '')
      if (index < 0) return []
      const buy = String(trade.side).toLowerCase().includes('buy') || String(trade.side).includes('买')
      return [{ name: trade.side, xAxis: index, yAxis: trade.price, value: buy ? '买' : '卖', itemStyle: { color: buy ? '#d95160' : '#168b78' }, label: { color: '#fff', fontSize: 9 } }]
    })
    return {
      animation: false, grid: { left: 55, right: 20, top: 18, bottom: 50 }, tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      xAxis: { type: 'category', data: labels, boundaryGap: false, axisLabel: { hideOverlap: true } }, yAxis: { scale: true, splitLine: { lineStyle: { color: '#eef1f5' } } },
      dataZoom: [{ type: 'inside', start: 35, end: 100 }, { type: 'slider', height: 18, bottom: 5 }],
      series: [{ name: 'K 线', type: 'candlestick', data: rows.map((x) => x.slice(1)), itemStyle: { color: '#d95160', color0: '#168b78', borderColor: '#d95160', borderColor0: '#168b78' }, markPoint: { symbol: 'circle', symbolSize: 8, data: markers } }],
    }
  }, [bars, reference, result?.config.frequency])

  return <>
    <PageHeader eyebrow='PERFORMANCE REVIEW' title='结果分析' description='比较三类策略净值、回撤和持仓，沿 K 线核对交易并导出明细。' action={<div className='button-row export-row'><Button variant='secondary' busy={busy === 'json'} disabled={!result} onClick={() => void exportFile('json')}><Download size={14} />JSON</Button><Button variant='secondary' busy={busy === 'equity'} disabled={!result} onClick={() => void exportFile('equity')}>净值 CSV</Button><Button variant='secondary' busy={busy === 'trades'} disabled={!result} onClick={() => void exportFile('trades')}>成交 CSV</Button></div>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    <Card title='回测结果' subtitle='查看策略回测结果；参数研究详情可在参数研究页查看。' action={<div className='result-select'><SelectInput value={selectedId} onChange={(e) => setSelectedId(e.target.value)}><option value=''>选择结果</option>{results.map((x) => <option key={x.id} value={x.id}>{x.symbol} · {x.title} · {new Date(x.created_at).toLocaleString('zh-CN', { hour12: false })}</option>)}</SelectInput><Button variant='quiet' onClick={() => { setError(''); setLoading(true); get<ResultSummary[]>('/results').then((list) => setResults(list.filter((item) => item.kind === 'backtest'))).catch((e) => setError(err(e))).finally(() => setLoading(false)) }}><RefreshCw size={14} /></Button></div>}>
      {loading ? <Loading /> : !results.length ? <Empty title='尚无回测结果' detail='在回测工作台运行真实回测后，结果会在这里出现。' /> : !result ? <Loading label='正在读取结果…' /> : <>
        <div className='result-meta'><div><strong>{result.symbol}</strong><span>{result.config.frequency} · {result.config.start_date} 至 {result.config.end_date ?? '最新'}</span></div><div>{isDemoResult && <Badge tone='amber'>演示数据</Badge>}<Badge tone='blue'>数据版本 {result.data_version ?? '未知'}</Badge></div></div>
        {isDemoResult && <div className='demo-callout'>此结果基于本机生成的演示行情，不代表真实历史数据或投资结论。</div>}
        {result.warnings?.map((warning, i) => <div className='warning-list' key={i}><span className='warning-dot' />{warning}</div>)}
        <div className='strategy-metrics'>{strategies.map((s) => <div className={'strategy-card ' + s.key} key={s.key}><div className='strategy-top'><div><span className='strategy-swatch' style={{ background: colors[s.key] ?? '#2864dc' }} /><strong>{s.name}</strong></div><Badge tone={Number(s.value.metrics?.total_return ?? 0) >= 0 ? 'red' : 'green'}>{percent(s.value.metrics?.total_return)}</Badge></div><div className='strategy-detail-grid'>{s.value.return_basis === 'cost' ? <><div><small>累计投入</small><strong>{money(s.value.metrics.total_invested)} 元</strong></div><div><small>总手续费</small><strong>{money(s.value.metrics.total_fees)} 元</strong></div><div><small>跳过期数</small><strong>{s.value.metrics.skipped_periods ?? '—'}</strong></div><div><small>成交笔数</small><strong>{s.value.metrics.trade_count ?? '—'}</strong></div></> : <><div><small>年化收益</small><strong>{percent(s.value.metrics?.cagr)}</strong></div><div><small>最大回撤</small><strong className='loss'>{percent(s.value.metrics?.max_drawdown)}</strong></div><div><small>夏普比率</small><strong>{s.value.metrics?.sharpe == null ? '—' : Number(s.value.metrics.sharpe).toFixed(2)}</strong></div><div><small>成交笔数</small><strong>{s.value.metrics?.trade_count ?? '—'}</strong></div></>}</div></div>)}</div>
      </>}
    </Card>
    {result && <>
      {result.funding && <Card title='股数回测口径'><p>估值日 {result.funding.reference_date} · 起始总资产 {money(result.funding.initial_equity)} 元 · 预留现金 {money(result.funding.reserve_cash)} 元 · 实际截至 {result.effective_end_date}</p><p>定投累计投入 {money(result.strategies.dca.metrics.total_invested)} 元，获利 {percent(result.strategies.dca.metrics.total_return)}；定投新增资金不计作盈利，不与网格共用本金。</p><ChartView height={220} option={{ tooltip: { trigger: 'axis' }, grid: { left: 55, right: 20, top: 15, bottom: 30 }, xAxis: { type: 'time' }, yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } }, series: [{ type: 'line', name: '定投获利百分比', showSymbol: false, data: result.strategies.dca.equity.map(point => [point.timestamp, point.normalized_equity == null ? null : (point.normalized_equity - 1) * 100]) }] }} /></Card>}
      <div className='chart-grid'><Card title='策略净值' subtitle={`${curvePeriod}净值 · 以起始总资产为 1.0${equityFrequency === '1d' ? '；净值 CSV 保留完整曲线' : ''}`}><ChartView option={equityOption} height={290} /></Card><Card title='回撤曲线' subtitle={`${curvePeriod}回撤`}><ChartView option={drawdownOption} height={290} /></Card></div>
      <div className='chart-grid'><Card title={hasPositionRatio ? '仓位变化' : '持股数量'} subtitle={hasPositionRatio ? `${curvePeriod}股票市值占净资产比例` : `${curvePeriod}持股数量 · 单位：${shareUnit}`}><ChartView option={positionOption} height={245} /></Card><Card title='收益指标'><div className='table-wrap'><table><thead><tr><th>指标</th>{strategies.map((s) => <th key={s.key}>{s.name}</th>)}</tr></thead><tbody>{Object.entries(metricLabels).map(([key, label]) => <tr key={key}><td>{label}</td>{strategies.map((s) => { const value = s.value.metrics?.[key]; const shown = ['total_return', 'cagr', 'max_drawdown'].includes(key) ? percent(value) : key.includes('count') ? value ?? '—' : key === 'total_fees' ? money(value) : value == null ? '—' : Number(value).toFixed(2); return <td key={s.key} className={tone(key, Number(value))}>{shown}</td> })}</tr>)}{((result.schema_version === 2 ? ['buy_hold'] : ['dca', 'buy_hold']) as Array<'dca' | 'buy_hold'>).map((baseline) => <tr key={baseline}><td>{baseline === 'dca' ? '相对定投超额' : '相对全仓超额'}（百分点）</td>{strategies.map((strategy) => { const excess = Number(strategy.value.metrics?.total_return ?? 0) - Number(result.strategies[baseline]?.metrics?.total_return ?? 0); return <td key={strategy.key} className={excess >= 0 ? 'gain' : 'loss'}>{strategy.value.return_basis === 'cost' ? '—' : (excess * 100).toFixed(2)}</td> })}</tr>)}</tbody></table></div></Card></div>
      <Card title={result.symbol + ' K 线与网格交易'} subtitle={'K线按时间汇总，买卖点映射到包含成交时刻的区间；标记取自 ' + (reference?.name ?? '策略') + ' 成交记录。'} action={<Badge tone='blue'>{result.config.frequency}</Badge>}>
        {barsLoading ? <Loading label='读取 K 线…' /> : bars.length ? <ChartView option={candleOption} height={410} /> : <Empty title='当前区间没有可用 K 线' detail='成交和净值仍可查看；请确认该标的存在对应频率行情。' />}
      </Card>
      <Card title='年 / 月收益' subtitle={result.schema_version === 2 ? '网格和持有按每日收益连乘；定投采用累计成本收益，不列月年收益。' : '由每日收益连乘计算，包含区间首日收益。'} action={<div className='segmented'><button className={unit === 'year' ? 'selected' : ''} onClick={() => setUnit('year')}>年度</button><button className={unit === 'month' ? 'selected' : ''} onClick={() => setUnit('month')}>月度</button></div>}>
        <div className='table-wrap'><table><thead><tr><th>区间</th>{strategies.map((s) => <th key={s.key}>{s.name}</th>)}</tr></thead><tbody>{periodRows(strategies, unit).map(([period, values]) => <tr key={period}><td>{period}</td>{strategies.map((s) => <td key={s.key} className={Number(values[s.key]) >= 0 ? 'gain' : 'loss'}>{percent(values[s.key])}</td>)}</tr>)}</tbody></table></div>
      </Card>
      <div className='chart-grid'><Card title='成交记录' action={<SelectInput aria-label='查看策略成交' value={tradeStrategy} onChange={event => setTradeStrategy(event.target.value)}>{strategies.map(strategy => <option value={strategy.key} key={strategy.key}>{strategy.name}</option>)}</SelectInput>} subtitle={(reference?.value.trades?.length ?? 0) + ' 笔 · ' + (reference?.name ?? '策略')}><div className='table-wrap scroll-table'>{reference?.value.trades?.length ? <table><thead><tr><th>时间</th><th>方向</th><th>数量</th><th>价格</th><th>费用</th><th>现金</th><th>持股</th><th>原因</th></tr></thead><tbody>{(reference.value.trades as Trade[]).slice().reverse().map((t, i) => <tr key={i}><td>{String(t.timestamp).replace('T', ' ').slice(0, 19)}</td><td><Badge tone={t.side.toLowerCase().includes('buy') ? 'red' : 'green'}>{sideText(t.side)}</Badge></td><td>{money(t.quantity, 0)}</td><td>{money(t.price, 3)}</td><td>{money(t.fees)}</td><td>{money(t.cash)}</td><td>{money(t.shares, 0)}</td><td>{reasonText(t.reason)}</td></tr>)}</tbody></table> : <Empty title='没有成交记录' />}</div></Card>
        <Card title='拒单记录' subtitle={(reference?.value.rejections?.length ?? 0) + ' 条 · 未成交订单及原因'}><div className='table-wrap scroll-table'>{reference?.value.rejections?.length ? <table><thead><tr><th>时间</th><th>方向</th><th>拒单原因</th><th>申请数量</th></tr></thead><tbody>{reference.value.rejections.map((row, i) => <tr key={i}><td>{String(row.timestamp ?? '').replace('T', ' ')}</td><td>{sideText(row.side)}</td><td>{reasonText(row.reason)}</td><td>{String(row.requested_quantity ?? '—')}</td></tr>)}</tbody></table> : <Empty title='无拒单记录' />}</div></Card></div>
      {result.quality && <Card title='数据质量说明'><pre className='json-preview'>{JSON.stringify(result.quality, null, 2)}</pre></Card>}
    </>}
  </>
}

