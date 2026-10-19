import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { get } from './api'
import ChartView from './ChartView'
import { Badge, Card, Empty, Field, Loading, SelectInput } from './components'

type Dict = Record<string, unknown>
type LandscapePoint = {
  candidate_id: number
  params: Dict
  is_center: boolean
  is_baseline: boolean
  status: string
  excess_return: number | null
  grid_return: number | null
  baseline_return: number | null
  drawdown: number | null
  trade_count: number | null
  peak_invested: number | null
  neighbor_candidate_ids: number[]
  neighbor_of_candidate_ids: number[]
  robustness_status: string | null
  neighbor_positive_ratio: number | null
  neighbor_q25_excess: number | null
  island_suspect: boolean
  pca?: [number, number]
  folds: Dict[]
}
type LandscapeResponse = {
  parameters: string[]
  points: LandscapePoint[]
  bounds: { min: number; max: number } | null
  pca: { parameters: string[]; explained_variance: number[]; loadings: number[][] } | null
}

const labels: Record<string, string> = {
  buy_step: '买入步长', sell_step: '卖出步长', grid_step: '网格间距',
  min_shares: '最低持股', max_shares: '最高持股', buy_shares: '每次买入', sell_shares: '每次卖出',
}

function number(value: number, digits = 2) {
  return value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
}
function ratio(value: number | null) {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
}
function same(left: unknown, right: unknown) {
  if (left == null && right == null) return true
  const a = Number(left), b = Number(right)
  return Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) < 1e-9 : left === right
}

export default function ResearchLandscape({ resultId, defaultParams, decisionOptions, regionOptions = [] }: {
  resultId: string
  defaultParams?: Dict
  decisionOptions: Array<{ value: string; label: string }>
  regionOptions?: Array<{ value: string; label: string }>
}) {
  const [decision, setDecision] = useState(decisionOptions[0]?.value ?? 'current')
  const [sample, setSample] = useState<'training' | 'validation'>('validation')
  const [cost, setCost] = useState<'original' | 'stress'>('original')
  const [fold, setFold] = useState('total')
  const [metric, setMetric] = useState<'excess' | 'grid'>('excess')
  const [region, setRegion] = useState('all')
  const [xName, setXName] = useState('buy_step')
  const [yName, setYName] = useState('sell_step')
  const [data, setData] = useState<LandscapeResponse | null>(null)
  const [selected, setSelected] = useState<LandscapePoint | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const [decisionKind, decisionIndexText] = decision.split(':')
  const decisionIndex = Number(decisionIndexText || 0)
  useEffect(() => {
    let active = true
    setLoading(true); setError('')
    const query = new URLSearchParams({ decision: decisionKind, decision_index: String(decisionIndex), sample, cost })
    if (fold !== 'total') query.set('fold', fold)
    if (region !== 'all') query.set('region_id', region)
    void get<LandscapeResponse>(`/results/${encodeURIComponent(resultId)}/landscape?${query}`)
      .then(value => { if (active) setData(value) })
      .catch(cause => { if (active) setError(cause instanceof Error ? cause.message : String(cause)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [resultId, decisionKind, decisionIndex, sample, cost, fold, region])

  const parameters = data?.parameters ?? []
  useEffect(() => {
    if (!parameters.length) return
    if (!parameters.includes(xName)) setXName(parameters[0])
    if (!parameters.includes(yName) || parameters[1] === yName) setYName(parameters.find(name => name !== xName) ?? parameters[0])
  }, [parameters, xName, yName])

  const center = useMemo(() => {
    if (defaultParams && data?.points.some(point => Object.keys(defaultParams).every(name => same(point.params[name], defaultParams[name])))) return defaultParams
    return data?.points.find(point => point.is_center && !point.is_baseline)?.params ?? data?.points[0]?.params ?? {}
  }, [data, defaultParams])

  const slice = useMemo(() => (data?.points ?? []).filter(point => Object.keys(center).every(name => name === xName || name === yName || name === 'initial_shares' || same(point.params[name], center[name]))), [data, center, xName, yName])
  const valueOf = (point: LandscapePoint) => metric === 'grid' ? point.grid_return : point.excess_return
  const metricValues = (data?.points ?? []).map(valueOf).filter((value): value is number => value != null && Number.isFinite(value))
  const min = metricValues.length ? Math.min(...metricValues) : -0.1
  const max = metricValues.length ? Math.max(...metricValues) : 0.1
  const span = Math.max(Math.abs(min), Math.abs(max), 0.001)
  const xs = [...new Set(slice.map(point => Number(point.params[xName])).filter(Number.isFinite))].sort((a, b) => a - b)
  const ys = [...new Set(slice.map(point => Number(point.params[yName])).filter(Number.isFinite))].sort((a, b) => a - b)

  const heatmapOption = useMemo(() => ({
    animation: false,
    tooltip: { formatter: (raw: unknown) => {
      const params = (raw as { data?: unknown }).data as [number, number, number, LandscapePoint] | undefined
      if (!params) return ''
      const point = params[3]
      return `${point.is_center ? '<b>中心候选</b><br/>' : ''}${Object.entries(point.params).filter(([key, value]) => key !== 'initial_shares' && value != null).map(([key, value]) => `${labels[key] ?? key}: ${number(Number(value), key.includes('step') ? 4 : 0)}`).join('<br/>')}<br/>超额收益: ${ratio(point.excess_return)}<br/>网格收益: ${ratio(point.grid_return)}<br/>状态: ${point.status}${point.pca ? '' : ''}`
    } },
    grid: { left: 70, right: 25, top: 22, bottom: 55 },
    xAxis: { type: 'category', name: labels[xName] ?? xName, data: xs.map(value => number(value, xName.includes('step') ? 4 : 0)), splitArea: { show: true } },
    yAxis: { type: 'category', name: labels[yName] ?? yName, data: ys.map(value => number(value, yName.includes('step') ? 4 : 0)), splitArea: { show: true } },
    visualMap: { min: -span, max: span, calculable: true, orient: 'horizontal', left: 'center', bottom: 0, inRange: { color: ['#b42318', '#f3f4f6', '#067647'] }, formatter: (value: unknown) => `${(Number(value) * 100).toFixed(1)}%` },
    series: [{
      type: 'heatmap',
      data: slice.map(point => [xs.indexOf(Number(point.params[xName])), ys.indexOf(Number(point.params[yName])), valueOf(point) ?? null, point]),
      label: { show: true, formatter: (raw: unknown) => {
        const value = ((raw as { data?: [number, number, number] }).data ?? [])[2]
        return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '未完成'
      } },
      emphasis: { itemStyle: { borderColor: '#111827', borderWidth: 2 } },
    }],
  }) as unknown as EChartsOption, [slice, xs, ys, xName, yName, min, max, span, metric])

  const pcaOption = useMemo(() => ({
    animation: false,
    tooltip: { formatter: (raw: unknown) => {
      const point = ((raw as { data?: unknown }).data as [number, number, number, LandscapePoint] | undefined)?.[3]
      return point ? `${point.is_center ? '中心候选<br/>' : ''}超额收益 ${ratio(point.excess_return)}<br/>${Object.entries(point.params).filter(([key]) => key !== 'initial_shares').map(([key, value]) => `${labels[key] ?? key} ${String(value)}`).join('<br/>')}` : ''
    } },
    grid: { left: 55, right: 25, top: 25, bottom: 45 },
    xAxis: { type: 'value', name: `PC1 ${((data?.pca?.explained_variance[0] ?? 0) * 100).toFixed(1)}%`, splitLine: { lineStyle: { color: '#e6e8ee' } } },
    yAxis: { type: 'value', name: `PC2 ${((data?.pca?.explained_variance[1] ?? 0) * 100).toFixed(1)}%`, splitLine: { lineStyle: { color: '#e6e8ee' } } },
    visualMap: { min: -span, max: span, show: false, inRange: { color: ['#b42318', '#f3f4f6', '#067647'] } },
    series: [{ type: 'scatter', symbolSize: (value: unknown) => ((value as [number, number, number, LandscapePoint])[3].is_center ? 13 : 7), data: (data?.points ?? []).filter(point => point.pca).map(point => [point.pca![0], point.pca![1], valueOf(point) ?? null, point]) }],
  }) as unknown as EChartsOption, [data, span, metric])

  if (loading && !data) return <Loading label='读取参数收益分布' />
  if (error) return <Empty title='参数收益分布不可用' detail={error} />
  if (!data || !data.points.length) return <Empty title='暂无参数收益分布' detail='本次研究没有可展示的训练或验证候选。' />
  return <Card title='参数收益分布' subtitle='二维切片固定其余参数；PCA 仅作全局浏览，邻域判断始终使用原参数空间。' action={<Badge tone='blue'>{data.points.length} 个点</Badge>}>
    <div className='form-grid four research-landscape-controls'>
      <Field label='研究决策'><SelectInput value={decision} onChange={event => setDecision(event.target.value)}>{decisionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>
      <Field label='样本用途'><SelectInput value={sample} onChange={event => setSample(event.target.value as 'training' | 'validation')}><option value='validation'>内部验证</option><option value='training'>训练搜索</option></SelectInput></Field>
      <Field label='成本情景'><SelectInput value={cost} onChange={event => setCost(event.target.value as 'original' | 'stress')}><option value='original'>原始成本</option><option value='stress'>双倍压力</option></SelectInput></Field>
      <Field label='验证窗口'><SelectInput value={fold} onChange={event => setFold(event.target.value)} disabled={sample === 'training'}><option value='total'>整个验证段</option>{[0, 1, 2, 3].map(index => <option key={index} value={index}>窗口 {index + 1}</option>)}</SelectInput></Field>
      <Field label='着色指标'><SelectInput value={metric} onChange={event => setMetric(event.target.value as 'excess' | 'grid')}><option value='excess'>相对全仓超额</option><option value='grid'>网格收益</option></SelectInput></Field>
      {regionOptions.length > 1 && <Field label='推荐区域'><SelectInput value={region} onChange={event => setRegion(event.target.value)}><option value='all'>全部候选</option>{regionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>}
      <Field label='横轴'><SelectInput value={xName} onChange={event => setXName(event.target.value)}>{parameters.map(name => <option key={name} value={name}>{labels[name] ?? name}</option>)}</SelectInput></Field>
      <Field label='纵轴'><SelectInput value={yName} onChange={event => setYName(event.target.value)}>{parameters.map(name => <option key={name} value={name}>{labels[name] ?? name}</option>)}</SelectInput></Field>
    </div>
    <div className='research-landscape-grid'>
      <div><h4>参数切片</h4><ChartView option={heatmapOption} height={390} onEvents={{ click: raw => { const point = (((raw as { data?: unknown }).data as [number, number, number, LandscapePoint] | undefined)?.[3]); if (point) setSelected(point) } }} /></div>
      {data.pca && <div><h4>PCA 概览</h4><ChartView option={pcaOption} height={390} onEvents={{ click: raw => { const point = (((raw as { data?: unknown }).data as [number, number, number, LandscapePoint] | undefined)?.[3]); if (point) setSelected(point) } }} /><p className='field-hint'>解释方差 {((data.pca.explained_variance[0] + data.pca.explained_variance[1]) * 100).toFixed(1)}%；图上距离不能替代真实邻点。</p></div>}
    </div>
    <p className='field-hint'>当前切片固定于 {Object.entries(center).filter(([key, value]) => key !== xName && key !== yName && key !== 'initial_shares' && value != null).map(([key, value]) => `${labels[key] ?? key}=${number(Number(value), key.includes('step') ? 4 : 0)}`).join('，') || '中心参数'}。</p>
    {selected && <details open><summary>候选 {selected.candidate_id} 明细</summary><pre className='json-preview'>{JSON.stringify({ params: selected.params, status: selected.status, excess_return: selected.excess_return, grid_return: selected.grid_return, drawdown: selected.drawdown, trade_count: selected.trade_count, peak_invested: selected.peak_invested, robustness_status: selected.robustness_status, neighbor_positive_ratio: selected.neighbor_positive_ratio, neighbor_q25_excess: selected.neighbor_q25_excess, island_suspect: selected.island_suspect, neighbor_candidate_ids: selected.neighbor_candidate_ids, folds: selected.folds }, null, 2)}</pre></details>}
  </Card>
}
