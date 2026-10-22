import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { get } from './api'
import ChartView from './ChartView'
import { Badge, Card, Empty, Field, Loading, SelectInput } from './components'
import { RobustJRanking, RobustPointSummary } from './RobustJResults'
import {
  chooseLandscapeAxes,
  formatLandscapeAxis,
  formatLandscapeNumber,
  formatLandscapeParameter,
  landscapeLabels,
  landscapeSliceKey,
  landscapeValue,
  revisionMatchesScope,
  type LandscapePoint,
  type LandscapeResponse,
  type LandscapeRevisionScope,
} from './landscapeData'

type Dict = Record<string, unknown>

function number(value: number, digits = 2) {
  return formatLandscapeNumber(value, digits)
}

function ratio(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
}

function same(left: unknown, right: unknown) {
  if (left == null && right == null) return true
  const a = Number(left), b = Number(right)
  return Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) < 1e-9 : left === right
}

function ParameterChips({ point }: { point: LandscapePoint }) {
  return <div className='research-parameter-badges research-landscape-params'>
    {Object.entries(point.params).filter(([, value]) => value != null).map(([name, value]) =>
      <span key={name}><small>{landscapeLabels[name] ?? name}</small><strong>{formatLandscapeParameter(name, value)}</strong></span>)}
  </div>
}

function pointFromEvent(raw: unknown): LandscapePoint | undefined {
  return ((raw as { data?: { item?: LandscapePoint } }).data ?? {}).item
}

export default function ResearchLandscape({ resultId, defaultParams, decisionOptions, decision: controlledDecision, onDecisionChange, qualified = false, revision = null, revisionScope = null, onLoad }: {
  resultId: string
  defaultParams?: Dict
  decisionOptions: Array<{ value: string; label: string }>
  decision?: string
  onDecisionChange?: (decision: string) => void
  qualified?: boolean
  revision?: number | null
  revisionScope?: LandscapeRevisionScope | null
  onLoad?: (params: Dict, asOf?: string) => void
}) {
  const [internalDecision, setInternalDecision] = useState(decisionOptions[0]?.value ?? 'current:0')
  const decision = controlledDecision ?? internalDecision
  useEffect(() => {
    if (!decisionOptions.some(item => item.value === decision) && decisionOptions[0]) {
      setInternalDecision(decisionOptions[0].value)
      onDecisionChange?.(decisionOptions[0].value)
    }
  }, [decision, decisionOptions, onDecisionChange])
  const [metric, setMetric] = useState<'excess' | 'grid'>('excess')
  const [xName, setXName] = useState('')
  const [yName, setYName] = useState('')
  const [data, setData] = useState<LandscapeResponse | null>(null)
  const [selected, setSelected] = useState<LandscapePoint | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const [decisionKind, decisionIndexText] = decision.split(':')
  const decisionIndex = Number(decisionIndexText || 0)
  // A completion revision only covers one decision branch.  Sending it for
  // another branch would show the pre-completion archive as if it had been
  // revised, so the revision is only attached when the scope matches.
  const revisionApplies = revisionMatchesScope(revisionScope, decisionKind, decisionIndex)
  useEffect(() => {
    let active = true
    setLoading(true); setError('')
    // Protocol v5 has a single training window and no internal validation
    // folds, so the candidate archive is always the training archive.
    const query = new URLSearchParams({ decision: decisionKind, decision_index: String(decisionIndex), sample: 'training', cost: 'original', limit: '1000' })
    if (revision != null && revisionApplies) query.set('revision', String(revision))
    void get<LandscapeResponse>(`/results/${encodeURIComponent(resultId)}/landscape?${query}`)
      .then(value => {
        if (!active) return
        setData(value)
        setSelected(null)
      })
      .catch(cause => { if (active) setError(cause instanceof Error ? cause.message : String(cause)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [resultId, decisionKind, decisionIndex, revision, revisionApplies])

  const parameters = data?.parameters ?? []
  const axisSignature = parameters.join('|')
  useEffect(() => {
    if (!parameters.length) return
    const [nextX, nextY] = chooseLandscapeAxes(parameters, data?.points ?? [], xName, yName)
    if (nextX !== xName) setXName(nextX)
    if (nextY !== yName) setYName(nextY)
  }, [axisSignature, data, xName, yName])

  const center = useMemo(() => {
    if (data?.focus?.params) return data.focus.params
    if (defaultParams && data?.points.some(point => Object.keys(defaultParams).every(name => same(point.params[name], defaultParams[name])))) return defaultParams
    return data?.points.find(point => point.is_center && !point.is_baseline)?.params ?? data?.points[0]?.params ?? {}
  }, [data, defaultParams])

  const validPair = Boolean(xName && yName && xName !== yName)
  const fixedNames = useMemo(() => parameters.filter(name => name !== xName && name !== yName), [parameters, xName, yName])
  const activeSlice = data?.slices?.[landscapeSliceKey(xName, yName)]
  const slice = useMemo(() => {
    if (!validPair) return []
    if (activeSlice) return activeSlice.points
    return (data?.points ?? []).filter(point => fixedNames.every(name => same(point.params[name], center[name])))
  }, [data, center, fixedNames, validPair, activeSlice])
  const valueOf = (point: LandscapePoint) => landscapeValue(point, metric)
  const metricValues = (data?.points ?? []).map(valueOf).filter((value): value is number => value != null && Number.isFinite(value))
  const rawMin = metricValues.length ? Math.min(...metricValues) : -0.1
  const rawMax = metricValues.length ? Math.max(...metricValues) : 0.1
  const padding = rawMin === rawMax ? Math.max(Math.abs(rawMin) * 0.1, 0.001) : 0
  const minValue = rawMin - padding
  const maxValue = rawMax + padding
  const xs = [...new Set(slice.map(point => Number(point.params[xName])).filter(Number.isFinite))].sort((a, b) => a - b)
  const ys = [...new Set(slice.map(point => Number(point.params[yName])).filter(Number.isFinite))].sort((a, b) => a - b)
  const showCellLabels = xs.length * ys.length <= 120

  const heatmapOption = useMemo(() => {
    const xIndex = new Map(xs.map((value, index) => [value, index]))
    const yIndex = new Map(ys.map((value, index) => [value, index]))
    const focus = data?.focus
    const focusX = focus ? xIndex.get(Number(focus.params[xName])) : undefined
    const focusY = focus ? yIndex.get(Number(focus.params[yName])) : undefined
    return {
      animation: false,
      tooltip: { formatter: (raw: unknown) => {
        const point = pointFromEvent(raw)
        if (!point) return ''
        return `${point.is_focus ? '<b>相对最佳单点</b><br/>' : ''}${point.is_center ? '<b>推荐区域中心</b><br/>' : ''}${Object.entries(point.params).filter(([, value]) => value != null).map(([key, value]) => `${landscapeLabels[key] ?? key}: ${formatLandscapeParameter(key, value)}`).join('<br/>')}<br/>差值超额: ${ratio(point.excess_return)}<br/>峰值本金收益: ${ratio(point.grid_return)}<br/>最大本金回撤: ${ratio(point.drawdown)}<br/>成交笔数: ${point.trade_count == null ? '—' : number(Number(point.trade_count), 0)}`
      }, backgroundColor: 'rgba(15, 23, 42, 0.92)', borderColor: 'rgba(255, 255, 255, 0.12)', borderWidth: 1, textStyle: { color: '#f8fafc', fontSize: 12 }, extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;' },
      grid: { left: 78, right: 34, top: 42, bottom: 92 },
      xAxis: {
        type: 'category', name: landscapeLabels[xName] ?? xName, nameLocation: 'middle', nameGap: 40,
        data: xs.map(value => formatLandscapeAxis(xName, value)),
        axisLabel: { rotate: xs.length > 10 ? 35 : 0, hideOverlap: true, fontSize: 10, margin: 11, color: '#475569' },
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category', name: landscapeLabels[yName] ?? yName, nameLocation: 'middle', nameGap: 48,
        data: ys.map(value => formatLandscapeAxis(yName, value)),
        axisLabel: { fontSize: 10, color: '#475569' }, splitArea: { show: true },
      },
      visualMap: {
        type: 'continuous', dimension: 2, seriesIndex: 0, min: minValue, max: maxValue,
        calculable: true, orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 12, itemHeight: 190,
        text: ['较高', '较低'], textGap: 8,
        inRange: { color: ['#e11d48', '#f8fafc', '#0d9488'] },
        formatter: (value: unknown) => `${(Number(value) * 100).toFixed(1)}%`,
      },
      series: [
        {
          type: 'heatmap',
          data: slice.map(point => ({
            value: [xIndex.get(Number(point.params[xName])), yIndex.get(Number(point.params[yName])), valueOf(point) ?? null],
            item: point,
          })),
          label: {
            show: showCellLabels,
            formatter: (raw: unknown) => {
              const value = ((raw as { data?: { value?: [number, number, number] } }).data?.value ?? [])[2]
              return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '未完成'
            },
            fontSize: 9,
          },
          itemStyle: { borderColor: '#ffffff', borderWidth: 1 },
          emphasis: { itemStyle: { borderColor: '#18212f', borderWidth: 2 } },
        },
        {
          type: 'scatter', z: 5, symbol: 'pin', symbolSize: 26,
          data: focus && focusX != null && focusY != null ? [{ value: [focusX, focusY], item: focus }] : [],
          itemStyle: { color: '#1d4ed8', borderColor: '#ffffff', borderWidth: 1.5 },
          label: { show: true, formatter: '相对最佳', position: 'top', color: '#1d4ed8', fontSize: 10, fontWeight: 600 },
        },
      ],
    } as unknown as EChartsOption
  }, [slice, xs, ys, xName, yName, metric, minValue, maxValue, data?.focus, showCellLabels])

  if (loading && !data) return <Loading label='读取训练候选分布' />
  if (error) return <Empty title='训练候选分布不可用' detail={error} />
  if (!data || !data.points.length) return <Empty title='暂无训练候选分布' detail='本次研究没有可展示的训练候选。' />

  const focusValue = data.focus?.excess_return ?? null
  const loadPoint = onLoad && !loading ? (params: Dict) => onLoad(params, data.decision_as_of ?? undefined) : undefined
  return <Card
    title='训练候选参数分布'
    subtitle='协议 v5 只有单段训练时段，内部验证窗为 0：这里全部是训练候选，超额收益相对同资金被动持有，收益与回撤都按峰值本金口径。样本外表现请看上方固定 / 自适应的连续真实回测账本。'
    action={<Badge tone='blue'>{loading ? '读取中…' : `${data.total.toLocaleString()} 个候选 · 当前切片 ${slice.length.toLocaleString()}`}</Badge>}
  >
    <div className='form-grid four research-landscape-controls'>
      {decisionOptions.length > 1 && <Field label='研究决策'><SelectInput value={decision} onChange={event => { setInternalDecision(event.target.value); onDecisionChange?.(event.target.value) }}>{decisionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>}
      <Field label='着色指标'><SelectInput value={metric} onChange={event => setMetric(event.target.value as 'excess' | 'grid')}><option value='excess'>差值超额</option><option value='grid'>峰值本金收益</option></SelectInput></Field>
      <Field label='横轴参数'><SelectInput value={xName} onChange={event => {
        const next = event.target.value
        if (next === yName) setYName(xName)
        setXName(next)
      }}>{parameters.map(name => <option key={name} value={name}>{landscapeLabels[name] ?? name}</option>)}</SelectInput></Field>
      <Field label='纵轴参数'><SelectInput value={yName} onChange={event => {
        const next = event.target.value
        if (next === xName) setXName(yName)
        setYName(next)
      }}>{parameters.map(name => <option key={name} value={name}>{landscapeLabels[name] ?? name}</option>)}</SelectInput></Field>
    </div>

    <div className='research-representatives'>
      <RobustPointSummary point={data.focus} kind='excess' onLoad={loadPoint} />
      <RobustPointSummary point={data.j_focus ?? null} kind='j' onLoad={loadPoint} />
    </div>

    {data.focus && <div className={`research-landscape-focus ${qualified ? 'verified' : 'diagnostic'}`}>
      <div className='research-landscape-focus-copy'>
        <small>训练候选中的相对最佳单点</small>
        <strong>差值超额 {ratio(focusValue)}</strong>
        <span>从全部 {data.total.toLocaleString()} 个候选中按当前筛选选出；候选 #{data.focus.candidate_id ?? '—'}。{qualified
          ? '它是当前 J 排名中的推荐参数之一，正式推荐仍以完整评分为准。'
          : '它没有通过当前策略的资格核验，不能当作合格推荐。'}</span>
      </div>
      <ParameterChips point={data.focus} />
    </div>}

    {data.completion?.status === 'available' && <p className='inline-warning'>本次归档仍有未完成评分的候选。提交展示补算后会生成新的修订号，参数分布与导出都会改用该修订。</p>}

    {revision != null && !revisionApplies && <p className='inline-warning'>补算修订 #{revision} 只覆盖决策 {revisionScope?.decision ?? '—'} #{revisionScope?.decision_index ?? 0}，当前选择的决策没有这一版补算，下面显示的是补算前的候选。</p>}

    <div className='research-chart-guide'>
      <span><b>每个格子</b>是一组参数</span>
      <span><b>绿色</b>表示当前筛选内相对更高，红色更低；数值仍可能为负</span>
      <span><b>空白</b>表示该组合不合法、未计算或超出切片显示上限，不做插值</span>
    </div>

    <p className='field-hint research-landscape-sampling'>{activeSlice ? `参数切片读取固定其余参数后的完整平面，共 ${activeSlice.total.toLocaleString()} 个候选${activeSlice.sampled ? `；当前超过显示上限，图中等距展示 ${activeSlice.points.length.toLocaleString()} 个` : '，邻域判断不依赖全空间抽样'}。` : '当前组合没有可读取的二维平面，已回退到全空间抽样点。'}</p>

    {validPair ? <div>
      <div>
        <h4>参数切片 · {landscapeLabels[xName] ?? xName} × {landscapeLabels[yName] ?? yName}</h4>
        {slice.length ? <ChartView option={heatmapOption} height={420} onEvents={{ click: raw => { const point = pointFromEvent(raw); if (point) setSelected(point) } }} /> : <div className='research-chart-empty'><Empty title='当前切片没有候选' detail='尝试切换横纵轴、样本或研究决策。' /></div>}
      </div>
    </div> : <p className='inline-warning'>可用参数不足两个，无法形成二维切片。相对最佳参数仍显示在上方。</p>}

    <p className='field-hint'>当前切片固定于 {fixedNames.filter(name => center[name] != null).map(name => `${landscapeLabels[name] ?? name}=${formatLandscapeParameter(name, center[name])}`).join('，') || '相对最佳单点的其余参数'}。</p>

    <RobustJRanking points={data.j_ranking ?? []} total={data.j_candidate_count ?? 0} count={data.recommendation_count ?? 10} coverage={data.coverage_status} onLoad={loadPoint} />

    {selected && <section className='research-selected-candidate'>
      <div className='research-selected-head'>
        <div><strong>候选 #{selected.candidate_id ?? '—'}</strong><span>{selected.is_focus ? '相对最佳单点' : selected.is_center ? '推荐区域中心' : selected.is_baseline ? '对照策略' : '普通候选'} · {selected.status === 'completed' ? '已计算' : selected.status}</span></div>
        <button type='button' className='text-link' onClick={() => setSelected(null)}>收起</button>
      </div>
      <ParameterChips point={selected} />
      <div className='research-selected-metrics'>
        <span><small>差值超额</small><strong>{ratio(selected.excess_return)}</strong></span>
        <span><small>峰值本金收益</small><strong>{ratio(selected.grid_return)}</strong></span>
        <span><small>最大本金回撤</small><strong>{ratio(selected.drawdown)}</strong></span>
        <span><small>成交笔数</small><strong>{selected.trade_count == null ? '—' : number(Number(selected.trade_count), 0)}</strong></span>
        <span><small>邻点正收益占比</small><strong>{ratio(selected.neighbor_positive_ratio)}</strong></span>
        <span><small>邻点第 25 百分位</small><strong>{ratio(selected.neighbor_q25_excess)}</strong></span>
      </div>
    </section>}
  </Card>
}
