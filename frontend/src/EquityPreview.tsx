import { useEffect, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { del, get, post } from './api'
import ChartView from './ChartView'
import { Badge, Card, Empty, configError, configPayload, money, percent } from './components'
import type { BacktestConfig, BacktestResult } from './types'

type Preview = { revision: number; status: string; error?: string; result?: BacktestResult }

export default function EquityPreview({ config }: { config: BacktestConfig }) {
  const session = useRef(`preview_${Math.random().toString(36).slice(2)}`)
  const revision = useRef(0)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [status, setStatus] = useState('等待参数')
  const [error, setError] = useState('')
  const payload = JSON.stringify(configPayload(config))
  useEffect(() => {
    const current = ++revision.current
    let active = true
    let poll: number | undefined
    const validation = configError(config)
    setError(validation); setStatus(validation ? '参数待完善' : '等待计算')
    const accept = (value: Preview) => {
      if (!active || current !== revision.current || value.revision !== current) return
      if (value.status === 'completed' && value.result) { setResult(value.result); setError(''); setStatus('已更新'); return }
      if (value.status === 'failed') { setError(value.error ?? '预览计算失败'); setStatus('计算失败'); return }
      setStatus('计算中')
      poll = window.setTimeout(() => { get<Preview>(`/backtests/preview/${session.current}`).then(accept).catch(fail) }, 400)
    }
    const fail = (value: unknown) => { if (active) { setStatus('无法预览'); setError(value instanceof Error ? value.message : String(value)) } }
    const timer = validation ? undefined : window.setTimeout(() => {
      setStatus('计算中')
      post<Preview>('/backtests/preview', { session_id: session.current, revision: current, config: JSON.parse(payload) }).then(accept).catch(fail)
    }, 600)
    return () => { active = false; clearTimeout(timer); clearTimeout(poll) }
  }, [payload])
  useEffect(() => () => { void del(`/backtests/preview/${session.current}`).catch(() => undefined) }, [])
  const option: EChartsOption = {
    animation: false, tooltip: { trigger: 'axis' }, legend: { top: 0, right: 0 }, grid: { left: 52, right: 15, top: 38, bottom: 35 },
    xAxis: { type: 'time' }, yAxis: { type: 'value', scale: true, axisLabel: { formatter: (value: number) => value.toFixed(2) }, splitLine: { lineStyle: { color: '#eef1f5' } } },
    series: (['grid', 'buy_hold'] as const).map((name, i) => ({ type: 'line', name: name === 'grid' ? '网格净值' : '买入持有', showSymbol: false, data: (result?.strategies[name]?.equity ?? []).map(point => [point.timestamp, point.normalized_equity]), lineStyle: { width: i === 0 ? 2 : 1.5, color: i === 0 ? '#2864dc' : '#8a98af' }, itemStyle: { color: i === 0 ? '#2864dc' : '#8a98af' } })),
  }
  const dcaOption: EChartsOption = {
    animation: false, grid: { left: 52, right: 15, top: 15, bottom: 28 }, tooltip: { trigger: 'axis', valueFormatter: value => `${Number(value).toFixed(2)}%` }, xAxis: { type: 'time' }, yAxis: { type: 'value', scale: true, axisLabel: { formatter: '{value}%' } },
    series: [{ type: 'line', showSymbol: false, name: '定投获利', data: (result?.strategies.dca?.equity ?? []).map(point => [point.timestamp, point.normalized_equity == null ? null : (point.normalized_equity - 1) * 100]), lineStyle: { color: '#e29136', width: 1.5 }, itemStyle: { color: '#e29136' } }],
  }
  return <Card title="净值曲线" subtitle="修改参数后自动重算 · 预览不保存任务或结果" className="preview-card" action={<Badge tone={status === '已更新' ? 'green' : 'amber'}>{status}</Badge>}>
    {result?.funding && <div className="preview-funding"><span>预留现金 <strong>{money(result.funding.reserve_cash)} 元</strong></span><span>起始总资产 <strong>{money(result.funding.initial_equity)} 元</strong></span><span>估值日 {result.funding.reference_date}</span><span>实际截至 {result.effective_end_date}</span></div>}
    {error && <p className="inline-warning" role="status">{error}</p>}
    <div className={result && status !== '已更新' ? 'preview-stale' : ''}>{result ? <ChartView option={option} height={285} /> : <Empty title="填写参数后显示净值预览" detail="请先在行情页同步或导入目标标的的数据。" />}</div>
    {result && <><div className="preview-metrics"><span>网格收益 <strong>{percent(result.strategies.grid.metrics.total_return)}</strong></span><span>最大回撤 <strong>{percent(result.strategies.grid.metrics.max_drawdown)}</strong></span><span>成交 <strong>{result.strategies.grid.metrics.trade_count ?? 0} 笔</strong></span></div><div className="dca-preview-heading"><strong>定投获利百分比</strong><span>累计盈利 ÷ 累计投入 · {percent(result.strategies.dca.metrics.total_return)}</span></div><ChartView option={dcaOption} height={130} />{Boolean(result.warnings?.length) && <details className="preview-warnings"><summary>数据说明与警告（{result.warnings!.length}）</summary>{result.warnings!.map((warning, index) => <p key={index}>{warning}</p>)}</details>}</>}
  </Card>
}
