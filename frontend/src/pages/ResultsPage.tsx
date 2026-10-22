import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { Download, RefreshCw } from 'lucide-react'
import { ApiError, download, get } from '../api'
import ChartView from '../ChartView'
import Disclosure from '../Disclosure'
import ResultsCurves from '../ResultsCurves'
import { RESULT_PAGE_SIZE, barTimestampRange, bucketTradeMarkers, earliestBarTimestamp, groupTradeMarkers, mapTradesToBars, markerLabel, mergeBars, naiveStamp, periodReturnBaseline, periodReturns, resultsPageQuery } from '../resultsReview'
import { resultShareUnit, savedGridParameters } from '../resultPresentation'
import { Badge, Button, Card, Empty, ErrorBox, Loading, PageHeader, SelectInput, money, percent, useErrorMessage } from '../components'
import { chartValue } from '../marketValues'
import type { BacktestResult, ResultSummary, StrategyResult, Trade } from '../types'
import { LatestRequest } from '../requestState'
import type { ResultView } from '../navigation'

type Props = {
  selectedResultId: string | null
  selectedCompareId: string | null
  view: ResultView
  onSelectionChange: (result: string | null, compare: string | null) => void
  onViewChange: (view: ResultView) => void
}
const names: Record<string, string> = { grid: '网格策略', dca: '定期定额', buy_hold: '买入并持有', same_account_passive: '同账户被动持有', matched_funding_hold: '同资金被动持有' }
const colors: Record<string, string> = { grid: '#2563eb', dca: '#d97706', buy_hold: '#64748b', same_account_passive: '#0d9488', matched_funding_hold: '#7c3aed' }
const reasonNames: Record<string, string> = {
  initial_position: '初始建仓', dca: '定期购买', buy_hold: '初始全仓', grid_buy: '网格买入', grid_sell: '网格卖出',
  dca_period_skipped: '该期无可成交时点，未投入资金', non_tradable_bar: '不可交易行情', suspended_day: '当日停牌', invalid_open: '开盘价无效', zero_volume: '无成交量',
  limit_up_locked: '开盘封涨停', limit_down_locked: '开盘封跌停', invalid_execution_price: '成交价格无效',
  below_lot: '不足最小申报数量', position_limit: '仓位限制', position_limit_or_below_lot: '仓位限制或不足最小数量',
  insufficient_cash: '现金不足', t_plus_1: '股份尚未满足 T+1', insufficient_sellable_shares: '可卖股份不足',
}
const reasonText = (value: unknown) => reasonNames[String(value)] ?? String(value ?? '—')
const sideText = (value: unknown) => value === 'buy' ? '买入' : value === 'sell' ? '卖出' : String(value ?? '—')
const metricLabels: Record<string, string> = { peak_principal_return: '峰值本金收益', max_principal_drawdown: '最大本金回撤', matched_funding_hold_return: '同资金被动持有', trade_count: '成交笔数', total_fees: '总手续费', buy_count: '买入次数', sell_count: '卖出次数', turnover: '换手率' }
const percentMetrics = ['peak_principal_return', 'max_principal_drawdown', 'matched_funding_hold_return']
const countMetrics = ['trade_count', 'buy_count', 'sell_count']
type Strategy = { key: string; name: string; value: StrategyResult }
const BAR_PAGE_LIMIT = 1500
const MAX_MARKERS = 240
const tone = (key: string, value: number | null | undefined) => {
  if (value == null || !Number.isFinite(value)) return ''
  if (key === 'max_principal_drawdown') return 'loss'
  return percentMetrics.includes(key) ? value >= 0 ? 'gain' : 'loss' : ''
}
const summaryLabel = (item: ResultSummary) => {
  const created = item.created_at ? new Date(item.created_at).toLocaleString('zh-CN', { hour12: false }) : '时间未知'
  return item.symbol + ' · ' + (item.title || '未命名结果') + ' · ' + created
}
const stampText = (value: string) => value.replace('T', ' ').slice(0, 16)

export default function ResultsPage({ selectedResultId, selectedCompareId, view, onSelectionChange, onViewChange }: Props) {
  const [results, setResults] = useState<ResultSummary[]>([])
  const [resultsOffset, setResultsOffset] = useState(0)
  const [resultsHasMore, setResultsHasMore] = useState(false)
  const [resultsLoading, setResultsLoading] = useState(true)
  const [resultsLoadingMore, setResultsLoadingMore] = useState(false)
  const selectedId = selectedResultId ?? ''
  const compareId = selectedCompareId ?? ''
  const setSelectedId = useCallback((value: string) => {
    onSelectionChange(value || null, compareId || null)
  }, [compareId, onSelectionChange])
  const setCompareId = useCallback((value: string) => {
    onSelectionChange(selectedId || null, value || null)
  }, [onSelectionChange, selectedId])
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [resultError, setResultError] = useState('')
  const [resultRetryToken, setResultRetryToken] = useState(0)
  const [compareResult, setCompareResult] = useState<BacktestResult | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState('')
  const [compareRetryToken, setCompareRetryToken] = useState(0)
  const [bars, setBars] = useState<Array<Record<string, unknown>>>([])
  const [barsLoading, setBarsLoading] = useState(false)
  const [barsError, setBarsError] = useState('')
  const [barsNotice, setBarsNotice] = useState('')
  const [earlierBefore, setEarlierBefore] = useState('')
  const [earlierLoading, setEarlierLoading] = useState(false)
  const [busy, setBusy] = useState('')
  const [resultsError, setResultsError] = useState('')
  const [actionError, setActionError] = useState('')
  const err = useErrorMessage()
  const resultsRequest = useRef(new LatestRequest<string>())
  const resultRequest = useRef(new LatestRequest<string>())
  const compareRequest = useRef(new LatestRequest<string>())
  const resultSelection = useRef('')
  const compareSelection = useRef('')
  const autoSelected = useRef(false)
  const barsRequest = useRef(0)
  const loadResults = useCallback(async (offset: number, mode: 'replace' | 'append') => {
    const ticket = resultsRequest.current.begin(`${offset}:${mode}`)
    if (mode === 'replace') { setResultsLoading(true); setResultsOffset(0); setResultsError('') }
    else setResultsLoadingMore(true)
    try {
      const list = await get<ResultSummary[]>(resultsPageQuery(offset))
      if (!resultsRequest.current.isCurrent(ticket)) return
      const backtests = (Array.isArray(list) ? list : []).filter((item) => item.kind === 'backtest')
      setResults((current) => mode === 'replace'
        ? backtests
        : [...current, ...backtests.filter((item) => !current.some((existing) => existing.id === item.id))])
      setResultsOffset(offset + backtests.length)
      setResultsHasMore(backtests.length >= RESULT_PAGE_SIZE)
      // An explicit URL selection is never replaced by the first history row.
      if (!selectedResultId && !autoSelected.current && backtests[0]) {
        autoSelected.current = true
        onSelectionChange(backtests[0].id, selectedCompareId)
      }
    } catch (e) {
      if (resultsRequest.current.isCurrent(ticket)) setResultsError(err(e))
    } finally {
      if (resultsRequest.current.isCurrent(ticket)) { setResultsLoading(false); setResultsLoadingMore(false) }
    }
  }, [err, onSelectionChange, selectedCompareId, selectedResultId])
  const barsPath = useCallback((before?: string) => {
    if (!result?.symbol || !result.config?.frequency) return ''
    if (result.data_version_available === false) return ''
    const params = new URLSearchParams({ frequency: result.config.frequency, limit: String(BAR_PAGE_LIMIT), latest: 'true' })
    if (result.config.start_date) params.set('start_date', result.config.start_date)
    const end = result.effective_end_date ?? result.config.end_date
    if (end) params.set('end_date', end)
    if (result.data_version) params.set('version', result.data_version)
    if (before) params.set('before', before)
    return '/data/bars/' + encodeURIComponent(result.symbol) + '?' + params.toString()
  }, [result])
  const strategies: Strategy[] = useMemo(() => result ? Object.entries(result.strategies ?? {}).filter(([, value]) => value != null).map(([key, value]) => ({ key, name: value.return_basis === 'cost' ? '定投（成本收益）' : names[key] ?? key, value })) : [], [result])
  const [tradeStrategy, setTradeStrategy] = useState('grid')
  const reference = strategies.find((x) => x.key === tradeStrategy) ?? strategies[0]
  const isDemoResult = result?.is_demo === true || result?.quality?.is_demo === true || Boolean(result?.warnings?.some((warning) => warning.includes('演示数据')))
  const equityFrequency = result?.equity_frequency ?? Object.values(result?.strategies ?? {}).find((strategy) => strategy?.equity_frequency)?.equity_frequency
  const curvePeriod = equityFrequency === '1d' ? '每日收盘' : '回测时间点'
  const shareUnit = resultShareUnit(result?.symbol)
  const parameters = useMemo(() => savedGridParameters(result), [result])
  const compareOptions = useMemo(() => results.filter(item => item.id !== selectedId), [results, selectedId])
  const comparisonDifferences = useMemo(() => {
    if (!result || !compareResult) return []
    const left = result
    const right = compareResult
    const checks: Array<[string, unknown, unknown]> = [
      ['标的', left.symbol, right.symbol],
      ['频率', left.config?.frequency, right.config?.frequency],
      ['区间', `${left.config?.start_date ?? ''}|${left.effective_end_date ?? left.config?.end_date ?? ''}`, `${right.config?.start_date ?? ''}|${right.effective_end_date ?? right.config?.end_date ?? ''}`],
      ['佣金费率', left.config?.fees?.commission_rate, right.config?.fees?.commission_rate],
      ['最低佣金', left.config?.fees?.min_commission, right.config?.fees?.min_commission],
      ['滑点', left.config?.fees?.slippage_bps, right.config?.fees?.slippage_bps],
      ['数据版本', left.data_version, right.data_version],
      ['收益口径', left.accounting_basis ?? 'historical', right.accounting_basis ?? 'historical'],
    ]
    return checks.filter(([, a, b]) => JSON.stringify(a ?? null) !== JSON.stringify(b ?? null)).map(([label]) => label)
  }, [result, compareResult])
  const comparisonRows = useMemo(() => {
    if (!result || !compareResult) return []
    const format = (value: unknown) => value == null ? '—' : typeof value === 'number'
      ? (Number.isInteger(value) ? String(value) : value.toFixed(6).replace(/0+$/, '').replace(/\.$/, ''))
      : String(value)
    const leftFees = result.config?.fees ?? {}
    const rightFees = compareResult.config?.fees ?? {}
    return [
      ['标的', result.symbol, compareResult.symbol],
      ['频率', result.config?.frequency, compareResult.config?.frequency],
      ['开始日期', result.config?.start_date, compareResult.config?.start_date],
      ['结束日期', result.effective_end_date ?? result.config?.end_date, compareResult.effective_end_date ?? compareResult.config?.end_date],
      ['数据版本', result.data_version, compareResult.data_version],
      ['收益口径', result.accounting_basis ?? '历史口径', compareResult.accounting_basis ?? '历史口径'],
      ['佣金费率', leftFees.commission_rate, rightFees.commission_rate],
      ['最低佣金', leftFees.min_commission, rightFees.min_commission],
      ['滑点（bp）', leftFees.slippage_bps, rightFees.slippage_bps],
      ['初始持股', result.config?.initial_shares, compareResult.config?.initial_shares],
      ['每次买入', result.config?.buy_shares, compareResult.config?.buy_shares],
      ['每次卖出', result.config?.sell_shares, compareResult.config?.sell_shares],
    ].map(([label, left, right]) => ({ label: String(label), left: format(left), right: format(right) }))
  }, [result, compareResult])
  const peakPrincipalBasis = result?.accounting_basis === 'peak_principal_v2' || (result?.accounting_basis == null && (result?.result_schema_version === 2 || result?.schema_version === 3))
  const externalTopups = result?.funding?.external_topups ?? result?.strategies?.grid?.metrics?.external_topups ?? result?.funding?.topups
  const matchedHoldInvested = result?.funding?.matched_funding_hold_invested ?? result?.strategies?.grid?.metrics?.matched_funding_hold_invested ?? result?.strategies?.matched_funding_hold?.metrics?.total_invested
  const periodBaseline = useMemo(() => periodReturnBaseline(result), [result])
  const unit = view
  const periodData = useMemo(() => periodReturns(strategies.map((strategy) => ({ key: strategy.key, equity: strategy.value.equity ?? [] })), unit, { baseline: periodBaseline }), [strategies, unit, periodBaseline])
  const barsRange = useMemo(() => {
    const range = barTimestampRange(bars)
    if (!range.count) return ''
    return '显示 ' + stampText(range.first) + ' 至 ' + stampText(range.last) + ' · ' + range.count + ' 根'
  }, [bars])
  const resultOptions = useMemo(() => {
    const options = [...results]
    if (selectedId && !options.some((item) => item.id === selectedId)) {
      // The selected result may sit outside the loaded pages, for example when it was
      // requested from the task center. It still belongs to the selector.
      options.push({ id: selectedId, kind: 'backtest', symbol: result?.symbol ?? '', created_at: '', title: '来自链接的结果' })
    }
    return options
  }, [results, selectedId, result])

  useEffect(() => {
    void loadResults(0, 'replace')
  }, [loadResults])
  useEffect(() => {
    resultRequest.current.invalidate()
    if (!selectedId) { setResult(null); setResultLoading(false); setResultError(''); return }
    if (resultSelection.current !== selectedId) { resultSelection.current = selectedId; setResult(null) }
    const ticket = resultRequest.current.begin(selectedId)
    setResultLoading(true); setResultError('')
    get<BacktestResult>('/results/' + encodeURIComponent(selectedId)).then((value) => {
      if (resultRequest.current.isCurrent(ticket)) setResult(value)
    }).catch((cause) => {
      if (!resultRequest.current.isCurrent(ticket)) return
      setResult(null)
      setResultError(cause instanceof ApiError && cause.status === 404
        ? '这条结果已经被清理或不存在，请从结果历史中重新选择。'
        : '结果详情读取失败：' + err(cause))
    }).finally(() => {
      if (resultRequest.current.isCurrent(ticket)) setResultLoading(false)
    })
  }, [selectedId, err, resultRetryToken])
  useEffect(() => {
    compareRequest.current.invalidate()
    if (!compareId || compareId === selectedId) { setCompareResult(null); setCompareLoading(false); setCompareError(''); return }
    if (compareSelection.current !== compareId) { compareSelection.current = compareId; setCompareResult(null) }
    const ticket = compareRequest.current.begin(compareId)
    setCompareLoading(true); setCompareError('')
    get<BacktestResult>('/results/' + encodeURIComponent(compareId)).then((value) => {
      if (compareRequest.current.isCurrent(ticket)) setCompareResult(value)
    }).catch((cause) => {
      if (!compareRequest.current.isCurrent(ticket)) return
      setCompareResult(null)
      setCompareError(cause instanceof ApiError && cause.status === 404
        ? '对照结果已经被清理或不存在，请重新选择。'
        : '对照结果读取失败：' + err(cause))
    }).finally(() => {
      if (compareRequest.current.isCurrent(ticket)) setCompareLoading(false)
    })
  }, [compareId, selectedId, err, compareRetryToken])
  useEffect(() => {
    const path = barsPath()
    if (!path) { setBars([]); setBarsLoading(false); setEarlierBefore(''); return }
    const seq = ++barsRequest.current
    let active = true
    setBars([])
    setBarsLoading(true)
    setBarsError('')
    setBarsNotice('')
    setEarlierBefore('')
    get<Array<Record<string, unknown>>>(path).then((data) => {
      if (!active || seq !== barsRequest.current) return
      const rows = Array.isArray(data) ? data : []
      setBars(rows)
      setEarlierBefore(earliestBarTimestamp(rows))
      const end = result?.effective_end_date ?? result?.config?.end_date
      if (rows.length >= BAR_PAGE_LIMIT && end && barTimestampRange(rows).last.slice(0, 10) < end.slice(0, 10)) {
        setBarsNotice('当前后端还没有支持 latest 参数，这里显示的是区间起始的 ' + rows.length + ' 根行情；后端支持后会自动改为实际结束之前最后 ' + BAR_PAGE_LIMIT + ' 根。')
      }
    }).catch((e) => { if (active && seq === barsRequest.current) setBarsError('K 线读取失败：' + err(e)) }).finally(() => { if (active && seq === barsRequest.current) setBarsLoading(false) })
    return () => { active = false }
  }, [barsPath, err])

  const loadEarlier = () => {
    const path = barsPath(earlierBefore)
    if (!path || !earlierBefore) return
    const seq = ++barsRequest.current
    setEarlierLoading(true)
    setBarsError('')
    setBarsNotice('')
    get<Array<Record<string, unknown>>>(path).then((data) => {
      if (seq !== barsRequest.current) return
      const rows = Array.isArray(data) ? data : []
      if (!rows.length) { setBarsNotice('这个区间之前没有更多行情了。'); return }
      const earliest = earliestBarTimestamp(rows)
      const bound = naiveStamp(earlierBefore)?.value ?? Number.NaN
      const reached = naiveStamp(earliest)?.value ?? Number.NaN
      if (!Number.isFinite(reached) || !Number.isFinite(bound) || reached >= bound) {
        setBarsNotice('当前后端还没有支持 before / latest 翻页参数，无法加载更早的行情。')
        return
      }
      setBars((current) => mergeBars(current, rows))
      setEarlierBefore(earliest)
    }).catch((e) => { if (seq === barsRequest.current) setBarsError('更早行情读取失败：' + err(e)) }).finally(() => { if (seq === barsRequest.current) setEarlierLoading(false) })
  }

  const exportFile = async (kind: 'json' | 'equity' | 'trades') => {
    if (!selectedId) return
    setBusy(kind); setActionError('')
    try { await download('/results/' + encodeURIComponent(selectedId) + '/export?kind=' + kind, (result?.symbol ?? 'result') + '-' + kind + (kind === 'json' ? '.json' : '.csv')) }
    catch (e) { setActionError(err(e)) } finally { setBusy('') }
  }

  const candleOption: EChartsOption = useMemo(() => {
    const rows = bars.map((b) => [String(b.timestamp ?? '').replace('T', ' ').slice(0, 16), chartValue(b.open), chartValue(b.close), chartValue(b.low), chartValue(b.high)])
    const labels = rows.map((row) => String(row[0]))
    const trades = (reference?.value.trades ?? []) as Trade[]
    // A trade belongs to the bar that opened at its own time; trades outside the loaded
    // window or inside a data gap are dropped instead of being clamped onto a bar.
    const mapped = mapTradesToBars(trades, bars, result?.config?.frequency ?? '')
    const markers = bucketTradeMarkers(groupTradeMarkers(mapped), MAX_MARKERS)
      .filter((marker) => Number.isFinite(marker.price))
      .map((marker) => ({
        name: marker.side,
        xAxis: marker.barIndex,
        yAxis: marker.price,
        value: markerLabel(marker),
        itemStyle: { color: marker.side === 'buy' ? '#e11d48' : '#0d9488' },
        label: { color: '#fff', fontSize: 10, fontWeight: 600 },
      }))
    return {
      animation: false,
      grid: { left: 58, right: 22, top: 20, bottom: 54 },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;',
      },
      xAxis: { type: 'category', data: labels, boundaryGap: false, axisLabel: { hideOverlap: true, color: '#64748b', fontSize: 11 } },
      yAxis: { scale: true, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } } },
      dataZoom: [
        { type: 'inside', start: 35, end: 100 },
        { type: 'slider', height: 18, bottom: 6, fillerColor: 'rgba(37, 99, 235, 0.10)', borderColor: '#e2e8f0', backgroundColor: '#f8fafc' },
      ],
      series: [{ name: 'K 线', type: 'candlestick', data: rows.map((x) => x.slice(1)), itemStyle: { color: '#e11d48', color0: '#0d9488', borderColor: '#e11d48', borderColor0: '#0d9488' }, markPoint: { symbol: 'circle', symbolSize: 10, data: markers } }],
    }
  }, [bars, reference, result?.config?.frequency])

  return <div className='results-page'>
    <PageHeader eyebrow='PERFORMANCE REVIEW' title='结果分析' description='比较各策略的峰值本金收益、回撤和网格持仓，按同一日期核对表现。' action={<div className='button-row export-row'><Button variant='secondary' busy={busy === 'json'} disabled={!result} onClick={() => void exportFile('json')}><Download size={14} />JSON</Button><Button variant='secondary' busy={busy === 'equity'} disabled={!result} onClick={() => void exportFile('equity')}>净值 CSV</Button><Button variant='secondary' busy={busy === 'trades'} disabled={!result} onClick={() => void exportFile('trades')}>成交 CSV</Button></div>} />
    {actionError && <ErrorBox message={actionError} onClose={() => setActionError('')} />}
    <div className='results-primary'>
      <Card title='回测结果' className='results-overview' action={<div className='result-select'><SelectInput aria-label='选择回测结果' value={selectedId} onChange={(event) => setSelectedId(event.target.value)}><option value=''>选择结果</option>{resultOptions.map((item) => <option key={item.id} value={item.id}>{summaryLabel(item)}</option>)}</SelectInput><Button variant='quiet' aria-label='刷新回测结果' busy={resultsLoading} onClick={() => { setResultsError(''); void loadResults(0, 'replace') }}><RefreshCw size={14} /></Button></div>}>
        {resultsError && <ErrorBox message={'结果历史读取失败：' + resultsError} onClose={() => setResultsError('')} onRetry={() => void loadResults(0, 'replace')} />}
        {resultsLoading && !results.length ? <Loading /> : !results.length && !selectedId ? <Empty title='尚无回测结果' detail='运行回测后，结果会在这里出现。' /> : resultError ? <ErrorBox message={resultError} onRetry={() => setResultRetryToken(token => token + 1)} /> : resultLoading && !result ? <Loading label='正在读取结果…' /> : !result ? <Empty title='请选择回测结果' detail='从上方历史中选择一条结果。' /> : <>
          {resultLoading && <p className='inline-warning'>正在刷新结果详情，当前显示上次成功读取的内容。</p>}
          <div className='result-meta'><div><strong>{result.symbol}</strong><span>{result.config?.frequency ?? '—'} · {result.config?.start_date ?? '—'} 至 {result.effective_end_date ?? result.config?.end_date ?? '最新'}</span></div><div>{isDemoResult && <Badge tone='amber'>演示数据</Badge>}<Badge tone={result.data_version_available === false ? 'red' : 'blue'}>数据版本 {result.data_version ?? '未知'}</Badge></div></div>
          {result.data_version_warning && <p className='inline-warning'>{result.data_version_warning}</p>}
          {isDemoResult && <div className='demo-callout'>此结果基于本机生成的演示行情，不代表真实历史数据或投资结论。</div>}
          <div className='strategy-metrics results-metrics'>{strategies.map((strategy) => {
            const metrics = strategy.value.metrics
            const peakReturn = metrics?.peak_principal_return
            const matchedHoldReturn = metrics?.matched_funding_hold_return
            const excess = peakReturn != null && matchedHoldReturn != null && Number.isFinite(peakReturn) && Number.isFinite(matchedHoldReturn) ? peakReturn - matchedHoldReturn : null
            return <div className={'strategy-card ' + strategy.key} key={strategy.key}>
              <div className='strategy-top'><div><span className='strategy-swatch' style={{ background: colors[strategy.key] ?? '#69758b' }} /><strong>{strategy.name}</strong></div><Badge tone={peakReturn == null || !Number.isFinite(peakReturn) ? 'neutral' : peakReturn >= 0 ? 'red' : 'green'}>{percent(peakReturn)}</Badge></div>
              <div className='strategy-detail-grid'>
                <div><small>最大本金回撤</small><strong className='loss'>{percent(metrics?.max_principal_drawdown)}</strong></div>
                <div><small>成交笔数</small><strong>{metrics?.trade_count ?? '—'}</strong></div>
                <div><small>总手续费</small><strong>{money(metrics?.total_fees)} 元</strong></div>
                {strategy.key === 'grid' && <>
                  <div><small>同资金被动持有</small><strong>{percent(matchedHoldReturn)}</strong></div>
                  <div><small>超额（同资金被动持有）</small><strong className={excess == null ? '' : excess >= 0 ? 'gain' : 'loss'}>{excess == null ? '—' : percent(excess)}</strong></div>
                </>}
                {strategy.key === 'dca' && <>
                  <div><small>累计投入</small><strong>{money(metrics?.total_invested)} 元</strong></div>
                  <div><small>跳过期数</small><strong>{metrics?.skipped_periods ?? '—'}</strong></div>
                </>}
              </div>
              {strategy.value.floor_penalty_basis && <p className='field-hint'>底仓惩罚 {percent(strategy.value.floor_penalty, 4)} · {strategy.value.floor_episode_count ?? 0} 段 · 各段上涨格数 {strategy.value.floor_episode_grids?.length ? strategy.value.floor_episode_grids.join(' / ') : '无'}</p>}
            </div>
          })}</div>
        </>}
        {resultsHasMore && <div className='button-row'><span className='field-hint'>已加载 {results.length} 条 · 每页 {RESULT_PAGE_SIZE} 条</span><Button variant='secondary' busy={resultsLoadingMore} onClick={() => void loadResults(resultsOffset, 'append')}>加载更多历史结果</Button></div>}
      </Card>
      {result && <Card title='A/B 已保存结果对照' className='results-comparison' subtitle='两侧独立读取；只比较已保存的表格数据，不叠加曲线，也不在日期、费用或数据版本不一致时给出名次。' action={<div className='result-select'><SelectInput aria-label='选择对照结果' value={compareId} onChange={event => setCompareId(event.target.value)}><option value=''>选择对照结果</option>{compareOptions.map(item => <option key={item.id} value={item.id}>{summaryLabel(item)}</option>)}</SelectInput>{compareId && <Button variant='quiet' onClick={() => setCompareId('')}>移除对照</Button>}</div>}>
        {!compareId ? <p className='field-hint'>在右上角选择另一条已保存回测结果。对照读取失败不会影响当前结果。</p> : compareId === selectedId ? <p className='inline-warning'>A/B 对照需要两条不同的结果。</p> : compareError ? <ErrorBox message={compareError} onRetry={() => setCompareRetryToken(token => token + 1)} /> : compareLoading && !compareResult ? <Loading label='独立读取对照结果…' /> : compareResult ? <>
          {compareLoading && <p className='inline-warning'>正在刷新对照结果，当前显示上次成功读取的内容。</p>}
          <p className={comparisonDifferences.length ? 'inline-warning' : 'field-hint'}>{comparisonDifferences.length ? '两侧以下口径不同：' + comparisonDifferences.join('、') + '。数值并列展示，不据此排名。' : '两侧标的、区间、费用口径与数据版本一致，可按同一口径逐项核对。'}</p>
          <div className='table-wrap'><table><thead><tr><th>项目</th><th>A · 当前</th><th>B · 对照</th></tr></thead><tbody>{comparisonRows.map(row => <tr key={row.label}><td>{row.label}</td><td>{row.left}</td><td>{row.right}</td></tr>)}</tbody></table></div>
          <div className='table-wrap results-compare-metrics'><table><thead><tr><th>网格指标</th><th>A</th><th>B</th><th>差值（A − B）</th></tr></thead><tbody>{Object.entries(metricLabels).map(([key, label]) => {
            const left = result.strategies?.grid?.metrics?.[key]
            const right = compareResult.strategies?.grid?.metrics?.[key]
            const difference = left == null || right == null || !Number.isFinite(left) || !Number.isFinite(right) ? null : left - right
            const display = (value: number | null | undefined) => percentMetrics.includes(key) ? percent(value) : key === 'total_fees' ? money(value) : value == null || !Number.isFinite(value) ? '—' : String(value)
            return <tr key={key}><td>{label}</td><td>{display(left)}</td><td>{display(right)}</td><td className={difference == null ? '' : difference >= 0 ? 'gain' : 'loss'}>{percentMetrics.includes(key) ? percent(difference) : difference == null ? '—' : String(difference)}</td></tr>
          })}</tbody></table></div>
          {comparisonDifferences.length > 0 && <p className='field-hint'>差值仅供核对，不表示更优：口径不一致时，收益率、回撤和费用不能直接排名。</p>}
        </> : <Empty title='对照结果不可用' detail='可移除对照后重新选择。' />}
      </Card>}
      {result && <>
        <Card title='本次网格参数' className='result-parameters' subtitle='来自此结果保存的配置与资金口径'>
          <dl className='result-parameter-list'>{parameters.map((parameter) => <div className='result-parameter' key={parameter.key}><dt>{parameter.label}</dt><dd>{parameter.value}</dd>{parameter.note && <small className='result-parameter-note'>{parameter.note}</small>}</div>)}</dl>
        </Card>
        <Card title='收益、回撤与网格持仓' className='results-curves-card' subtitle={(peakPrincipalBasis ? '收益率 = (资产总值 − 累计投入) ÷ 峰值本金 · 回撤 = 本金利润曲线回撤 · ' : result.accounting_basis ? '收益率 = 资产总值 ÷ 累计实际投入 − 1 · ' : '收益率 = 资产总值 ÷ 初始资产 − 1 · ') + curvePeriod + ' · 三图共享日期、缩放和悬浮值'} action={<Badge tone={result.accounting_basis ? 'blue' : 'amber'}>{peakPrincipalBasis ? '峰值本金' : result.accounting_basis ? '独立投入' : '历史口径'}</Badge>}>
          <ResultsCurves result={result} />
        </Card>
      </>}
    </div>
    {result && <div className='result-disclosures'>
      <Disclosure id='results-funding' title='资金口径与完整配置' summary={result.funding ? '峰值本金、外部补款与费用参数' : '查看此结果保存的参数'}>
        <div className='result-details-grid'>
          <Card title={result.funding ? '股数回测口径' : '历史回测口径'}>
            {result.funding ? <>
              <p>估值日 {result.funding.reference_date ?? '—'} · 峰值本金 {money(result.funding.peak_invested)} 元 · 外部补款 {money(externalTopups)} 元 · 同资金被动持有投入 {money(matchedHoldInvested)} 元 · 实际截至 {result.effective_end_date ?? '—'}</p>
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
            const shown = percentMetrics.includes(key) ? percent(value) : countMetrics.includes(key) ? value == null || !Number.isFinite(value) ? '—' : String(value) : key === 'total_fees' ? money(value) : value == null || !Number.isFinite(value) ? '—' : value.toFixed(2)
            return <td key={strategy.key} className={tone(key, value)}>{shown}</td>
          })}</tr>)}
          <tr><td>相对同资金被动持有超额（百分点）</td>{strategies.map((strategy) => {
            const strategyReturn = strategy.value.metrics?.peak_principal_return
            const baselineReturn = result.strategies?.matched_funding_hold?.metrics?.peak_principal_return
            const excess = strategy.key === 'matched_funding_hold' || strategyReturn == null || baselineReturn == null || !Number.isFinite(strategyReturn) || !Number.isFinite(baselineReturn) ? null : strategyReturn - baselineReturn
            return <td key={strategy.key} className={excess == null ? '' : excess >= 0 ? 'gain' : 'loss'}>{excess == null ? '—' : (excess * 100).toFixed(2)}</td>
          })}</tr>
        </tbody></table></div>
      </Disclosure>
      <Disclosure id='results-candles' title='K 线与成交标记' summary={(result.config?.frequency ?? '频率未知') + ' · ' + (reference?.name ?? '策略')}>
        <Card title={result.symbol + ' K 线与交易'} subtitle={'每根 K 线覆盖自身开盘到下一根 bar 开盘；窗口外或缺口内的成交不画标记。标记取自 ' + (reference?.name ?? '策略') + ' 成交记录。'} action={<div className='button-row'><span className='field-hint'>{barsRange}</span><Button variant='quiet' busy={earlierLoading} disabled={!earlierBefore || barsLoading} onClick={loadEarlier}>加载更早</Button></div>}>
          {result.data_version_available === false ? <Empty title='原行情数据版本已清理' detail='历史结果、成交和净值仍可查看和导出；原 K 线与按原版本重算已停用。' /> : barsError ? <ErrorBox message={barsError} onClose={() => setBarsError('')} /> : barsLoading ? <Loading label='读取 K 线…' /> : bars.length ? <><ChartView option={candleOption} height={410} />{barsNotice && <p className='inline-warning'>{barsNotice}</p>}</> : <Empty title='当前区间没有可用 K 线' detail='成交和净值仍可查看；请确认该标的存在对应频率行情。' />}
        </Card>
      </Disclosure>
      <Disclosure id='results-period-returns' title='年 / 月收益' summary={unit === 'year' ? '年度分解' : '月度分解'}>
        <Card subtitle={periodData.length ? '区间收益 = 期末峰值本金净值 − 期初峰值本金净值：本期期初取上一期期末净值，首期取运行基准；各期相加等于整段收益。' : '当前结果没有峰值本金净值，无法计算区间收益。'} action={<div className='segmented'><button className={unit === 'year' ? 'selected' : ''} onClick={() => onViewChange('year')}>年度</button><button className={unit === 'month' ? 'selected' : ''} onClick={() => onViewChange('month')}>月度</button></div>}>
          {periodData.length ? <div className='table-wrap'><table><thead><tr><th>区间</th>{strategies.map((strategy) => <th key={strategy.key}>{strategy.name}</th>)}</tr></thead><tbody>{periodData.map((row) => <tr key={row.period}><td>{row.period}</td>{strategies.map((strategy) => {
            const value = row.values[strategy.key]
            if (value === undefined) return <td key={strategy.key}>—</td>
            if (value === null) return <td key={strategy.key}>不可计算</td>
            return <td key={strategy.key} className={value >= 0 ? 'gain' : 'loss'}>{percent(value)}</td>
          })}</tr>)}</tbody></table></div> : <Empty title='后端未提供区间收益' detail='此结果的净值点没有峰值本金口径数据，无法计算年 / 月收益。' />}
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

