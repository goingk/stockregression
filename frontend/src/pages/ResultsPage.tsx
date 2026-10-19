import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { Download, RefreshCw } from 'lucide-react'
import { download, get } from '../api'
import ChartView from '../ChartView'
import Disclosure from '../Disclosure'
import ResultsCurves from '../ResultsCurves'
import { resultShareUnit, savedGridParameters } from '../resultPresentation'
import { Badge, Button, Card, Empty, ErrorBox, Loading, PageHeader, SelectInput, money, percent, useErrorMessage } from '../components'
import { chartValue } from '../marketValues'
import type { BacktestResult, EquityPoint, ResultSummary, StrategyResult, Trade } from '../types'

type Props = { requestedResult: string | null; clearRequestedResult: () => void }
const names: Record<string, string> = { grid: '网格策略', dca: '定期定额', buy_hold: '买入并持有', same_account_passive: '同账户被动持有' }
const colors: Record<string, string> = { grid: '#2864dc', dca: '#e29136', buy_hold: '#8795aa', same_account_passive: '#13957b' }
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
  const strategies: Strategy[] = useMemo(() => result ? Object.entries(result.strategies ?? {}).filter(([, value]) => value != null).map(([key, value]) => ({ key, name: value.return_basis === 'cost' ? '定投（成本收益）' : names[key] ?? key, value })) : [], [result])
  const [tradeStrategy, setTradeStrategy] = useState('grid')
  const reference = strategies.find((x) => x.key === tradeStrategy) ?? strategies[0]
  const isDemoResult = result?.is_demo === true || result?.quality?.is_demo === true || Boolean(result?.warnings?.some((warning) => warning.includes('演示数据')))
  const equityFrequency = result?.equity_frequency ?? Object.values(result?.strategies ?? {}).find((strategy) => strategy?.equity_frequency)?.equity_frequency
  const curvePeriod = equityFrequency === '1d' ? '每日收盘' : '回测时间点'
  const shareUnit = resultShareUnit(result?.symbol)
  const parameters = useMemo(() => savedGridParameters(result), [result])

  useEffect(() => {
    let active = true
    get<ResultSummary[]>('/results').then((list) => {
      if (!active) return
      const backtests = list.filter((item) => item.kind === 'backtest')
      setResults(backtests)
      setSelectedId((previous) => backtests.some((item) => item.id === previous) ? previous : backtests[0]?.id ?? '')
    }).catch((e) => { if (active) setError(err(e)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [err])
  useEffect(() => {
    if (requestedResult) { setSelectedId(requestedResult); clearRequestedResult() }
  }, [requestedResult, clearRequestedResult])

  useEffect(() => {
    if (!selectedId) { setResult(null); return }
    let active = true
    setLoading(true); setResult(null); setError('')
    get<BacktestResult>('/results/' + encodeURIComponent(selectedId)).then((value) => { if (active) setResult(value) }).catch((e) => { if (active) setError(err(e)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [selectedId, err])
  useEffect(() => {
    if (!result?.symbol || !result.config?.start_date || !result.config?.frequency) { setBars([]); setBarsLoading(false); return }
    if (result.data_version_available === false) { setBars([]); setBarsLoading(false); return }
    let active = true
    setBars([])
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

  const candleOption: EChartsOption = useMemo(() => {
    const rows = bars.map((b) => [String(b.timestamp ?? '').replace('T', ' ').slice(0, 16), chartValue(b.open), chartValue(b.close), chartValue(b.low), chartValue(b.high)])
    const labels = rows.map((row) => String(row[0]))
    const bucketTimes = bars.map((bar) => timestampMs(bar.timestamp))
    const trades = (reference?.value.trades ?? []) as Trade[]
    const markers = trades.flatMap((trade) => {
      const index = bucketIndex(trade.timestamp, bucketTimes, labels, result?.config?.frequency ?? '')
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
  }, [bars, reference, result?.config?.frequency])

  return <div className='results-page'>
    <PageHeader eyebrow='PERFORMANCE REVIEW' title='结果分析' description='比较策略收益、回撤和网格持仓，按同一日期核对表现。' action={<div className='button-row export-row'><Button variant='secondary' busy={busy === 'json'} disabled={!result} onClick={() => void exportFile('json')}><Download size={14} />JSON</Button><Button variant='secondary' busy={busy === 'equity'} disabled={!result} onClick={() => void exportFile('equity')}>净值 CSV</Button><Button variant='secondary' busy={busy === 'trades'} disabled={!result} onClick={() => void exportFile('trades')}>成交 CSV</Button></div>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    <div className='results-primary'>
      <Card title='回测结果' className='results-overview' action={<div className='result-select'><SelectInput aria-label='选择回测结果' value={selectedId} onChange={(event) => setSelectedId(event.target.value)}><option value=''>选择结果</option>{results.map((item) => <option key={item.id} value={item.id}>{item.symbol} · {item.title} · {new Date(item.created_at).toLocaleString('zh-CN', { hour12: false })}</option>)}</SelectInput><Button variant='quiet' aria-label='刷新回测结果' onClick={() => {
        setError(''); setLoading(true)
        get<ResultSummary[]>('/results').then((list) => {
          const backtests = list.filter((item) => item.kind === 'backtest')
          setResults(backtests)
          setSelectedId((previous) => backtests.some((item) => item.id === previous) ? previous : backtests[0]?.id ?? '')
        }).catch((error) => setError(err(error))).finally(() => setLoading(false))
      }}><RefreshCw size={14} /></Button></div>}>
        {loading ? <Loading /> : !results.length ? <Empty title='尚无回测结果' detail='在“回测与研究”中运行回测后，结果会在这里出现。' /> : !result ? <Loading label='正在读取结果…' /> : <>
          <div className='result-meta'><div><strong>{result.symbol}</strong><span>{result.config?.frequency ?? '—'} · {result.config?.start_date ?? '—'} 至 {result.effective_end_date ?? result.config?.end_date ?? '最新'}</span></div><div>{isDemoResult && <Badge tone='amber'>演示数据</Badge>}<Badge tone={result.data_version_available === false ? 'red' : 'blue'}>数据版本 {result.data_version ?? '未知'}</Badge></div></div>
          {result.data_version_warning && <p className='inline-warning'>{result.data_version_warning}</p>}
          {isDemoResult && <div className='demo-callout'>此结果基于本机生成的演示行情，不代表真实历史数据或投资结论。</div>}
          <div className='strategy-metrics results-metrics'>{strategies.map((strategy) => {
            const metrics = strategy.value.metrics
            const totalReturn = metrics?.total_return
            return <div className={'strategy-card ' + strategy.key} key={strategy.key}>
              <div className='strategy-top'><div><span className='strategy-swatch' style={{ background: colors[strategy.key] ?? '#69758b' }} /><strong>{strategy.name}</strong></div><Badge tone={totalReturn == null || !Number.isFinite(totalReturn) ? 'neutral' : totalReturn >= 0 ? 'red' : 'green'}>{percent(totalReturn)}</Badge></div>
              <div className='strategy-detail-grid'>{strategy.value.return_basis === 'cost' ? <>
                <div><small>累计投入</small><strong>{money(metrics?.total_invested)} 元</strong></div><div><small>总手续费</small><strong>{money(metrics?.total_fees)} 元</strong></div><div><small>跳过期数</small><strong>{metrics?.skipped_periods ?? '—'}</strong></div><div><small>成交笔数</small><strong>{metrics?.trade_count ?? '—'}</strong></div>
              </> : <>
                <div><small>{result.schema_version === 3 ? '时间加权年化' : '年化收益'}</small><strong>{percent(metrics?.cagr)}</strong></div><div><small>{result.schema_version === 3 ? '资金流调整回撤' : '最大回撤'}</small><strong className='loss'>{percent(metrics?.max_drawdown)}</strong></div><div><small>夏普比率</small><strong>{metrics?.sharpe == null || !Number.isFinite(metrics.sharpe) ? '—' : metrics.sharpe.toFixed(2)}</strong></div><div><small>成交笔数</small><strong>{metrics?.trade_count ?? '—'}</strong></div>
              </>}</div>
            </div>
          })}</div>
        </>}
      </Card>
      {result && <>
        <Card title='本次网格参数' className='result-parameters' subtitle='来自此结果保存的配置与资金口径'>
          <dl className='result-parameter-list'>{parameters.map((parameter) => <div className='result-parameter' key={parameter.key}><dt>{parameter.label}</dt><dd>{parameter.value}</dd>{parameter.note && <small className='result-parameter-note'>{parameter.note}</small>}</div>)}</dl>
        </Card>
        <Card title='收益、回撤与网格持仓' className='results-curves-card' subtitle={(result.schema_version === 3 ? '收益率 = 累计盈利 ÷ 周期峰值净投入 · 风险指标使用资金流调整后的单位净值 · ' : '收益率 = 资产总值 ÷ 累计实际投入 − 1 · ') + curvePeriod + ' · 三图共享日期、缩放和悬浮值'} action={<Badge tone={result.accounting_basis ? 'blue' : 'amber'}>{result.schema_version === 3 ? '峰值本金' : result.accounting_basis ? '独立投入' : '历史口径'}</Badge>}>
          <ResultsCurves result={result} />
        </Card>
      </>}
    </div>
    {result && <div className='result-disclosures'>
      <Disclosure id='results-funding' title='资金口径与完整配置' summary={result.funding ? '估值、初始资产、定投与费用参数' : '查看此结果保存的参数'}>
        <div className='result-details-grid'>
          <Card title={result.funding ? '股数回测口径' : '历史回测口径'}>
            {result.funding ? <>
              <p>估值日 {result.funding.reference_date ?? '—'} · 网格起始资产 {money(result.funding.initial_equity)} 元 · {result.schema_version === 3 ? `峰值净投入 ${money(result.funding.peak_invested ?? result.funding.initial_equity)} 元 · 累计补款 ${money(result.funding.topups)} 元` : `网格预留现金 ${money(result.funding.reserve_cash)} 元`} · 实际截至 {result.effective_end_date ?? '—'}</p>
              <p>期初持有：网格 {result.config?.initial_shares ?? '—'} {shareUnit} · 全仓 {result.funding.buy_hold_initial_shares ?? '—'} {shareUnit} · 定投 {result.config?.dca?.initial_shares ?? '—'} {shareUnit}。初始资产：全仓 {money(result.funding.buy_hold_initial_equity)} 元 · 定投 {money(result.funding.dca_initial_equity)} 元；定投累计实际投入 {money(result.strategies?.dca?.metrics?.total_invested)} 元，各策略使用独立资产账户。</p>
              <pre className='json-preview'>{JSON.stringify(result.funding, null, 2)}</pre>
            </> : <p>本结果按保存记录中的历史资金口径展示。缺失的参数与投入数据以“—”表示。</p>}
          </Card>
          <Card title='保存的完整配置'><pre className='json-preview'>{JSON.stringify(result.config ?? {}, null, 2)}</pre></Card>
        </div>
      </Disclosure>
      <Disclosure id='results-metrics' title='完整收益指标' summary={strategies.length + ' 类策略 · 收益、风险、费用与超额'}>
        <div className='table-wrap'><table><thead><tr><th>指标</th>{strategies.map((strategy) => <th key={strategy.key}>{strategy.name}</th>)}</tr></thead><tbody>
          {Object.entries(metricLabels).map(([key, label]) => <tr key={key}><td>{label}</td>{strategies.map((strategy) => {
            const value = strategy.value.metrics?.[key]
            const shown = ['total_return', 'cagr', 'max_drawdown'].includes(key) ? percent(value) : key.includes('count') ? value ?? '—' : key === 'total_fees' ? money(value) : value == null || !Number.isFinite(value) ? '—' : value.toFixed(2)
            return <td key={strategy.key} className={tone(key, Number(value))}>{shown}</td>
          })}</tr>)}
          {((result.schema_version === 2 ? ['buy_hold'] : ['dca', 'buy_hold']) as Array<'dca' | 'buy_hold'>).map((baseline) => <tr key={baseline}><td>{baseline === 'dca' ? '相对定投超额' : '相对全仓超额'}（百分点）</td>{strategies.map((strategy) => {
            const strategyReturn = strategy.value.metrics?.total_return
            const baselineReturn = result.strategies?.[baseline]?.metrics?.total_return
            const excess = strategyReturn != null && baselineReturn != null && Number.isFinite(strategyReturn) && Number.isFinite(baselineReturn) ? strategyReturn - baselineReturn : null
            return <td key={strategy.key} className={excess == null ? '' : excess >= 0 ? 'gain' : 'loss'}>{strategy.value.return_basis === 'cost' || excess == null ? '—' : (excess * 100).toFixed(2)}</td>
          })}</tr>)}
        </tbody></table></div>
      </Disclosure>
      <Disclosure id='results-candles' title='K 线与成交标记' summary={(result.config?.frequency ?? '频率未知') + ' · ' + (reference?.name ?? '策略')}>
        <Card title={result.symbol + ' K 线与交易'} subtitle={'K 线按时间汇总，买卖点映射到包含成交时刻的区间；标记取自 ' + (reference?.name ?? '策略') + ' 成交记录。'}>
          {result.data_version_available === false ? <Empty title='原行情数据版本已清理' detail='历史结果、成交和净值仍可查看和导出；原 K 线与按原版本重算已停用。' /> : barsLoading ? <Loading label='读取 K 线…' /> : bars.length ? <ChartView option={candleOption} height={410} /> : <Empty title='当前区间没有可用 K 线' detail='成交和净值仍可查看；请确认该标的存在对应频率行情。' />}
        </Card>
      </Disclosure>
      <Disclosure id='results-period-returns' title='年 / 月收益' summary={unit === 'year' ? '年度分解' : '月度分解'}>
        <Card subtitle={result.schema_version === 2 ? '网格和持有按每日收益连乘；定投采用累计成本收益，不列月年收益。' : '由每日收益连乘计算，包含区间首日收益。'} action={<div className='segmented'><button className={unit === 'year' ? 'selected' : ''} onClick={() => setUnit('year')}>年度</button><button className={unit === 'month' ? 'selected' : ''} onClick={() => setUnit('month')}>月度</button></div>}>
          <div className='table-wrap'><table><thead><tr><th>区间</th>{strategies.map((strategy) => <th key={strategy.key}>{strategy.name}</th>)}</tr></thead><tbody>{periodRows(strategies, unit).map(([period, values]) => <tr key={period}><td>{period}</td>{strategies.map((strategy) => <td key={strategy.key} className={Number(values[strategy.key]) >= 0 ? 'gain' : 'loss'}>{percent(values[strategy.key])}</td>)}</tr>)}</tbody></table></div>
        </Card>
      </Disclosure>
      <Disclosure id='results-trades' title='成交记录' summary={(reference?.value.trades?.length ?? 0) + ' 笔 · ' + (reference?.name ?? '策略')}>
        <Card action={<SelectInput aria-label='查看策略成交' value={reference?.key ?? ''} onChange={(event) => setTradeStrategy(event.target.value)}>{strategies.map((strategy) => <option value={strategy.key} key={strategy.key}>{strategy.name}</option>)}</SelectInput>}>
          <div className='table-wrap scroll-table'>{reference?.value.trades?.length ? <table><thead><tr><th>时间</th><th>方向</th><th>数量</th><th>价格</th><th>费用</th><th>现金</th><th>持股</th><th>原因</th></tr></thead><tbody>{reference.value.trades.slice().reverse().map((trade, index) => <tr key={index}><td>{String(trade.timestamp).replace('T', ' ').slice(0, 19)}</td><td><Badge tone={String(trade.side).toLowerCase().includes('buy') ? 'red' : 'green'}>{sideText(trade.side)}</Badge></td><td>{money(trade.quantity, 0)}</td><td>{money(trade.price, 3)}</td><td>{money(trade.fees)}</td><td>{money(trade.cash)}</td><td>{money(trade.shares, 0)}</td><td>{reasonText(trade.reason)}</td></tr>)}</tbody></table> : <Empty title='没有成交记录' />}</div>
        </Card>
      </Disclosure>
      <Disclosure id='results-rejections' title='拒单记录' summary={(reference?.value.rejections?.length ?? 0) + ' 条 · ' + (reference?.name ?? '策略')}>
        <Card action={<SelectInput aria-label='查看策略拒单' value={reference?.key ?? ''} onChange={(event) => setTradeStrategy(event.target.value)}>{strategies.map((strategy) => <option value={strategy.key} key={strategy.key}>{strategy.name}</option>)}</SelectInput>}>
          <div className='table-wrap scroll-table'>{reference?.value.rejections?.length ? <table><thead><tr><th>时间</th><th>方向</th><th>拒单原因</th><th>申请数量</th></tr></thead><tbody>{reference.value.rejections.map((row, index) => <tr key={index}><td>{String(row.timestamp ?? '').replace('T', ' ')}</td><td>{sideText(row.side)}</td><td>{reasonText(row.reason)}</td><td>{String(row.requested_quantity ?? '—')}</td></tr>)}</tbody></table> : <Empty title='无拒单记录' />}</div>
        </Card>
      </Disclosure>
      <Disclosure id='results-data-quality' title='数据质量与运行提示' summary={(result.warnings?.length ?? 0) + ' 条提示 · 数据版本 ' + (result.data_version ?? '未知')}>
        {result.data_version_warning && <div className='warning-list'><span className='warning-dot' />{result.data_version_warning}</div>}
        {result.warnings?.map((warning, index) => <div className='warning-list' key={index}><span className='warning-dot' />{warning}</div>)}
        {result.quality || result.diagnostics || result.source_events?.length ? <pre className='json-preview'>{JSON.stringify({ quality: result.quality, diagnostics: result.diagnostics, source_events: result.source_events }, null, 2)}</pre> : <Empty title='本结果没有保存数据质量明细' />}
      </Disclosure>
    </div>}
  </div>
}

