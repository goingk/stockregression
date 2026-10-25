import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { get } from './api'
import ChartView from './ChartView'
import { Badge, Card, Empty, Field, Loading, SelectInput } from './components'
import { RobustJRanking, RobustPointSummary } from './RobustJResults'
import {
  chooseLandscapeAxes,
  canonicalLandscapePoint,
  canonicalParameter,
  parameterSpaceAxes,
  formatLandscapeAxis,
  formatLandscapeNumber,
  formatLandscapeParameter,
  landscapeLabels,
  landscapeParameterLabel,
  landscapeValue,
  revisionMatchesScope,
  type LandscapePoint,
  type LandscapeResponse,
  type LandscapeRevisionScope,
} from './landscapeData'

const Landscape3D = lazy(() => import('./Landscape3D'))
const landscapeColorRamp = ['#16a34a', '#94a3b8', '#dc2626']
const landscapeMissingColor = '#d1d5db'

type Dict = Record<string, unknown>

function number(value: number, digits = 2) {
  return formatLandscapeNumber(value, digits)
}

function ratio(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
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

function exportLandscapeCsv(points: LandscapePoint[]) {
  const fields = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares']
  const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`
  const rows = [['candidate_id', ...fields, 'excess_return'], ...points.map(point => [point.candidate_id, ...fields.map(field => point.params[field]), point.excess_return])]
  const blob = new Blob(['\ufeff' + rows.map(row => row.map(quote).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'landscape-candidates.csv'; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}

export default function ResearchLandscape({ resultId, decisionOptions, decision: controlledDecision, onDecisionChange, qualified = false, revision = null, revisionScope = null, onLoad }: {
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
  const [view, setView] = useState<'2d' | '3d'>('2d')
  const [zName, setZName] = useState('')
  const [fixedAxis, setFixedAxis] = useState('')
  const [fixedTick, setFixedTick] = useState<number | null>(null)
  const [webglFailed, setWebglFailed] = useState(false)
  const [spatialData, setSpatialData] = useState<LandscapeResponse | null>(null)
  const initialViewResult = useRef('')

  const [decisionKind, decisionIndexText] = decision.split(':')
  const decisionIndex = Number(decisionIndexText || 0)
  // A completion revision only covers one decision branch.  Sending it for
  // another branch would show the pre-completion archive as if it had been
  // revised, so the revision is only attached when the scope matches.
  const revisionApplies = revisionMatchesScope(revisionScope, decisionKind, decisionIndex)
  useEffect(() => {
    let active = true
    const controller = new AbortController()
    setLoading(true); setError('')
    // Protocol v5 has a single training window and no internal validation
    // folds, so the candidate archive is always the training archive.
    const query = new URLSearchParams({ decision: decisionKind, decision_index: String(decisionIndex), sample: 'training', cost: 'original', view: 'space', limit: '5000' })
    if (view === '3d') {
      const axes = parameterSpaceAxes(data?.parameters ?? [], data?.parameter_space)
      const axes3 = (zName ? [xName, yName, zName] : [xName, yName, ...axes]).filter((axis, index, all) => axis && axis !== fixedAxis && axes.includes(axis) && all.indexOf(axis) === index).slice(0, 3)
      if (axes.length > 3 && fixedAxis) {
        query.set('fixed_axis', fixedAxis)
        if (fixedTick != null) query.set('fixed_tick', String(fixedTick))
      }
      if (axes3[0]) query.set('x', axes3[0])
      if (axes3[1]) query.set('y', axes3[1])
      if (axes3[2]) query.set('z', axes3[2])
      if (selected?.candidate_id) query.set('selected_candidate_id', String(selected.candidate_id))
    } else {
      if (xName) query.set('x', xName)
      if (yName) query.set('y', yName)
    }
    if (revision != null && revisionApplies) query.set('revision', String(revision))
    void get<LandscapeResponse>(`/results/${encodeURIComponent(resultId)}/landscape?${query}`, { signal: controller.signal })
      .then(value => {
        if (!active) return
        setData(value)
        setSpatialData(value)
      })
      .catch(cause => { if (active) setError(cause instanceof Error ? cause.message : String(cause)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false; controller.abort() }
  }, [resultId, decisionKind, decisionIndex, revision, revisionApplies, view, xName, yName, zName, fixedAxis, fixedTick, view === '3d' ? selected?.candidate_id : null])

  const parameterSpace = data?.parameter_space
  const parameters = data?.parameters ?? []
  const activeAxes = parameterSpaceAxes(parameters, parameterSpace)
  useEffect(() => {
    const hasAxisMetadata = Boolean(data)
    if (!hasAxisMetadata || initialViewResult.current === resultId) return
    initialViewResult.current = resultId
    setView(activeAxes.length >= 3 ? '3d' : '2d')
  }, [data, resultId, activeAxes.length])
  const axisSignature = parameters.join('|')
  useEffect(() => {
    if (!parameters.length) return
    const [nextX, nextY] = chooseLandscapeAxes(activeAxes, data?.points ?? [], xName, yName)
    if (nextX !== xName) setXName(nextX)
    if (nextY !== yName) setYName(nextY)
  }, [axisSignature, data, xName, yName])
  useEffect(() => {
    if (!activeAxes.length) return
    const responseFixedAxis = data?.fixed_axis && activeAxes.includes(data.fixed_axis) ? data.fixed_axis : ''
    const nextFixedAxis = activeAxes.length > 3 ? (activeAxes.includes(fixedAxis) ? fixedAxis : responseFixedAxis || (activeAxes.includes('sell_shares') ? 'sell_shares' : activeAxes[0])) : ''
    if (nextFixedAxis !== fixedAxis) {
      setFixedAxis(nextFixedAxis)
      setFixedTick(data?.fixed_axis === nextFixedAxis ? data.fixed_tick ?? null : null)
    }
    const usable = activeAxes.filter(axis => axis !== nextFixedAxis)
    if (!usable.includes(xName)) setXName(usable[0] ?? '')
    if (!usable.includes(yName) || yName === xName) setYName(usable.find(axis => axis !== xName) ?? '')
    if (!usable.includes(zName) || zName === xName || zName === yName) setZName(usable.find(axis => axis !== xName && axis !== yName) ?? '')
    const ticks = data?.axis_ticks?.[nextFixedAxis] ?? []
    if (ticks.length && fixedTick == null && data?.fixed_axis === nextFixedAxis && data.fixed_tick != null) setFixedTick(data.fixed_tick)
  }, [axisSignature, activeAxes.join('|'), fixedAxis, fixedTick, zName, xName, yName, data?.axis_ticks])

  const validPair = Boolean(xName && yName && xName !== yName)
  const slice = useMemo(() => {
    if (!validPair) return []
    return data?.points ?? []
  }, [data, validPair])
  const valueOf = (point: LandscapePoint) => landscapeValue(point, metric)
  const metricValues = (data?.points ?? []).map(valueOf).filter((value): value is number => value != null && Number.isFinite(value))
  const rawMin = metricValues.length ? Math.min(...metricValues) : -0.1
  const rawMax = metricValues.length ? Math.max(...metricValues) : 0.1
  const metricBounds2d = data?.metric_bounds?.[metric]
  const colorLimit2d = Math.max(Math.abs(metricBounds2d?.min ?? rawMin), Math.abs(metricBounds2d?.max ?? rawMax), 0.001)
  const minValue = -colorLimit2d
  const maxValue = colorLimit2d
  const xs = [...new Set(slice.map(point => Number(point.params[xName])).filter(Number.isFinite))].sort((a, b) => a - b)
  const ys = [...new Set(slice.map(point => Number(point.params[yName])).filter(Number.isFinite))].sort((a, b) => a - b)
  const showCellLabels = xs.length * ys.length <= 120

  const renderData = spatialData ?? data
  const renderAxes = renderData ? parameterSpaceAxes(renderData.parameters, renderData.parameter_space) : activeAxes
  const axesFor3d = (view === '3d' ? [xName, yName, zName] : renderAxes).filter((axis, index, all) => axis && axis !== fixedAxis && axis !== renderData?.fixed_axis && renderAxes.includes(axis) && all.indexOf(axis) === index).slice(0, 3)
  const points3d = (renderData?.points ?? []).map(point => canonicalLandscapePoint(point, renderData?.parameter_space))
  const colorBounds = renderData?.metric_bounds?.[metric]
  const colorAbs = Math.max(Math.abs(colorBounds?.min ?? rawMin), Math.abs(colorBounds?.max ?? rawMax), 0.001)
  const boundsForAxis = (axis: string | undefined) => axis ? renderData?.axis_bounds?.[axis] : undefined
  const scatter3dOption = useMemo(() => ({
    animation: false,
    legend: { top: 4, textStyle: { fontSize: 11 }, data: ['已评估候选', '超额最高点', 'J 最高点'], selectedMode: false },
    tooltip: { formatter: (raw: unknown) => {
      const point = pointFromEvent(raw)
      if (!point) return ''
      return `${point.is_focus ? '<b>相对最佳单点</b><br/>' : ''}${Object.entries(point.params).filter(([, value]) => value != null).map(([key, value]) => `${landscapeParameterLabel(key, renderData?.parameter_space)}: ${formatLandscapeParameter(key, value)}`).join('<br/>')}<br/>差值超额: ${ratio(point.excess_return)}<br/>峰值本金收益: ${ratio(point.grid_return)}<br/>最大本金回撤: ${ratio(point.drawdown)}<br/>成交笔数: ${point.trade_count == null ? '—' : number(Number(point.trade_count), 0)}`
    } },
    visualMap: { dimension: 3, min: -colorAbs, max: colorAbs, inRange: { color: landscapeColorRamp }, outOfRange: { color: landscapeMissingColor }, text: ['高', '低'], formatter: (value: number) => ratio(value), calculable: true, orient: 'horizontal', left: 'center', bottom: 10 },
    xAxis3D: { type: 'value', splitNumber: 3, nameGap: 15, nameTextStyle: { fontSize: 12 }, name: landscapeParameterLabel(axesFor3d[0] ?? '', renderData?.parameter_space), axisLabel: { formatter: (value: number) => formatLandscapeAxis(axesFor3d[0] ?? '', value) }, ...boundsForAxis(axesFor3d[0]) },
    yAxis3D: { type: 'value', splitNumber: 3, nameGap: 15, nameTextStyle: { fontSize: 12 }, name: landscapeParameterLabel(axesFor3d[1] ?? '', renderData?.parameter_space), axisLabel: { formatter: (value: number) => formatLandscapeAxis(axesFor3d[1] ?? '', value) }, ...boundsForAxis(axesFor3d[1]) },
    zAxis3D: { type: 'value', splitNumber: 3, nameGap: 15, nameTextStyle: { fontSize: 12 }, name: landscapeParameterLabel(axesFor3d[2] ?? '', renderData?.parameter_space), axisLabel: { formatter: (value: number) => formatLandscapeAxis(axesFor3d[2] ?? '', value) }, ...boundsForAxis(axesFor3d[2]) },
    grid3D: { left: '4%', top: 32, width: '92%', height: '75%', axisPointer: { show: false }, boxWidth: 120, boxHeight: 90, boxDepth: 100, axisLabel: { textStyle: { fontSize: 10 } }, axisLine: { lineStyle: { color: '#94a3b8', width: 1 } }, viewControl: { projection: 'orthographic', alpha: 25, beta: 40, rotateSensitivity: 1, zoomSensitivity: 1, panMouseButton: 'middle' } },
    series: [{ id: 'candidates', name: '已评估候选', type: 'scatter3D', symbolSize: 8, data: points3d.filter(point => axesFor3d.length === 3 && axesFor3d.every(axis => Number.isFinite(Number(point.params[axis])))).map(point => ({ value: [...axesFor3d.map(axis => Number(point.params[axis])), landscapeValue(point, metric) ?? null], item: point, symbolSize: point.candidate_id === selected?.candidate_id ? 15 : 8, itemStyle: { color: landscapeValue(point, metric) == null ? '#a8b0bc' : undefined, opacity: 0.82, borderColor: point.candidate_id === selected?.candidate_id ? '#1d4ed8' : undefined, borderWidth: point.candidate_id === selected?.candidate_id ? 2 : 0 } })) }, ...[
      { id: 'return-leader', name: '超额最高点', symbol: 'diamond', candidate: renderData?.focus?.candidate_id },
      { id: 'j-leader', name: 'J 最高点', symbol: 'triangle', candidate: renderData?.j_focus?.candidate_id },
    ].map(marker => ({ id: marker.id, name: marker.name, type: 'scatter3D', symbol: marker.symbol, symbolSize: 17,
      data: points3d.filter(point => point.candidate_id === marker.candidate && axesFor3d.length === 3).map(point => ({
        value: [...axesFor3d.map(axis => Number(point.params[axis])), landscapeValue(point, metric) ?? null], item: point,
      })), itemStyle: { opacity: 1 },
    }))],
  }) as unknown as EChartsOption, [points3d, axesFor3d.join('|'), metric, colorAbs, selected?.candidate_id, renderData?.axis_bounds, renderData?.parameter_space, renderData?.focus?.candidate_id, renderData?.j_focus?.candidate_id])

  const heatmapOption = useMemo(() => {
    const xIndex = new Map(xs.map((value, index) => [value, index]))
    const yIndex = new Map(ys.map((value, index) => [value, index]))
    const focus = data?.focus && slice.some(point => point.candidate_id === data.focus?.candidate_id) ? data.focus : null
    const focusX = focus ? xIndex.get(Number(focus.params[xName])) : undefined
    const focusY = focus ? yIndex.get(Number(focus.params[yName])) : undefined
    return {
      animation: false,
      tooltip: { formatter: (raw: unknown) => {
        const point = pointFromEvent(raw)
        if (!point) return ''
        return `${point.is_focus ? '<b>相对最佳单点</b><br/>' : ''}${point.is_center ? '<b>推荐区域中心</b><br/>' : ''}${Object.entries(point.params).filter(([, value]) => value != null).map(([key, value]) => `${landscapeParameterLabel(key, parameterSpace)}: ${formatLandscapeParameter(key, value)}`).join('<br/>')}<br/>差值超额: ${ratio(point.excess_return)}<br/>峰值本金收益: ${ratio(point.grid_return)}<br/>最大本金回撤: ${ratio(point.drawdown)}<br/>成交笔数: ${point.trade_count == null ? '—' : number(Number(point.trade_count), 0)}`
      }, backgroundColor: 'rgba(15, 23, 42, 0.92)', borderColor: 'rgba(255, 255, 255, 0.12)', borderWidth: 1, textStyle: { color: '#f8fafc', fontSize: 12 }, extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;' },
      grid: { left: 78, right: 58, top: 42, bottom: 110 },
      xAxis: {
        type: 'category', name: landscapeParameterLabel(xName, parameterSpace), nameLocation: 'middle', nameGap: 40,
        data: xs.map(value => formatLandscapeAxis(xName, value)),
        axisLabel: { rotate: xs.length > 10 ? 35 : 0, hideOverlap: true, fontSize: 10, margin: 11, color: '#475569' },
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category', name: landscapeParameterLabel(yName, parameterSpace), nameLocation: 'middle', nameGap: 48,
        data: ys.map(value => formatLandscapeAxis(yName, value)),
        axisLabel: { fontSize: 10, color: '#475569' }, splitArea: { show: true },
      },
      visualMap: {
        type: 'continuous', dimension: 2, seriesIndex: 0, min: minValue, max: maxValue,
        calculable: true, orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 12, itemHeight: 190,
        text: ['较高', '较低'], textGap: 8,
        inRange: { color: landscapeColorRamp }, outOfRange: { color: landscapeMissingColor },
        formatter: (value: unknown) => `${(Number(value) * 100).toFixed(1)}%`,
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        { type: 'inside', yAxisIndex: 0, filterMode: 'none' },
        { type: 'slider', xAxisIndex: 0, bottom: 38, height: 14, filterMode: 'none' },
        { type: 'slider', yAxisIndex: 0, right: 10, width: 14, filterMode: 'none' },
      ],
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
  }, [slice, xs, ys, xName, yName, metric, minValue, maxValue, data?.focus, showCellLabels, parameterSpace])

  if (loading && !data) return <Loading label='读取训练候选分布' />
  if (error) return <Empty title='训练候选分布不可用' detail={error} />
  if (!data || !data.points.length) return <Empty title='暂无训练候选分布' detail='本次研究没有可展示的训练候选。' />

  const focusValue = data.focus?.excess_return ?? null
  const conditionAxes = view === '3d' ? axesFor3d : [xName, yName]
  const fixedConditions = Object.entries(data.fixed_parameters ?? {}).filter(([name]) => !conditionAxes.includes(canonicalParameter(name, parameterSpace)))
    .reduce<Array<[string, unknown]>>((items, [name, value]) => {
      const canonical = canonicalParameter(name, parameterSpace)
      if (!items.some(([existing]) => existing === canonical)) items.push([canonical, value])
      return items
    }, [])
  const loadPoint = onLoad && !loading ? (params: Dict) => onLoad(params, data.decision_as_of ?? undefined) : undefined
  return <Card
    title='训练候选参数分布'
    subtitle='协议 v5 只有单段训练时段，内部验证窗为 0：这里全部是训练候选，超额收益相对同资金被动持有，收益与回撤都按峰值本金口径。样本外表现请看上方固定 / 自适应的连续真实回测账本。'
    action={<Badge tone='blue'>{loading ? '读取中…' : `${data.total.toLocaleString()} 个候选 · 当前切片 ${slice.length.toLocaleString()}`}</Badge>}
  >
    <div className='form-grid four research-landscape-controls'>
      {decisionOptions.length > 1 && <Field label='研究决策'><SelectInput value={decision} onChange={event => { setInternalDecision(event.target.value); onDecisionChange?.(event.target.value) }}>{decisionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>}
      <Field label='着色指标'><SelectInput value={metric} onChange={event => setMetric(event.target.value as 'excess' | 'grid')}><option value='excess'>差值超额</option><option value='grid'>峰值本金收益</option></SelectInput></Field>
      <Field label='视图'><SelectInput value={view} onChange={event => setView(event.target.value as '2d' | '3d')}><option value='2d'>2D 热图</option><option value='3d' disabled={activeAxes.length < 3}>3D 散点 / 切片</option></SelectInput></Field>
      <Field label='横轴参数'><SelectInput value={xName} onChange={event => {
        const next = event.target.value
        if (next === yName) setYName(xName)
        setXName(next)
      }}>{activeAxes.map(name => <option key={name} value={name}>{landscapeParameterLabel(name, parameterSpace)}</option>)}</SelectInput></Field>
      <Field label='纵轴参数'><SelectInput value={yName} onChange={event => {
        const next = event.target.value
        if (next === xName) setXName(yName)
        setYName(next)
      }}>{activeAxes.map(name => <option key={name} value={name}>{landscapeParameterLabel(name, parameterSpace)}</option>)}</SelectInput></Field>
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
      <span><b>{view === '2d' ? '每个格子' : '每个点'}</b>是一组参数</span>
      <span><b>红色 / 绿色</b>表示相对全局零点偏高 / 偏低；灰色为零，浅灰色为缺失</span>
      <span>指标：{metric === 'excess' ? '差值超额' : '峰值本金收益'}；{data.sampled ? `显示 ${data.returned_count ?? data.points.length} / ${data.total}（服务端空间分桶抽样）` : `共 ${data.total} 个候选`}</span>
    </div>

    <p className='field-hint research-landscape-sampling'>{view === '3d'
      ? `三维散点展示 ${axesFor3d.map(axis => landscapeParameterLabel(axis, parameterSpace)).join(' × ')}，当前 ${data.returned_count ?? data.points.length} / ${data.total.toLocaleString()} 个候选。`
      : `二维热图由服务端返回真实参数平面，共 ${data.total.toLocaleString()} 个候选${data.sampled ? `；图中显示 ${data.returned_count ?? slice.length} 个服务端抽样点` : '，当前平面未抽样'}。`}</p>

    {view === '3d' && activeAxes.length > 3 && <div className='research-landscape-controls'>
      <Field label='固定维度'><SelectInput value={fixedAxis} onChange={event => { setFixedAxis(event.target.value); setFixedTick(null) }}>{activeAxes.map(axis => <option key={axis} value={axis}>{landscapeParameterLabel(axis, parameterSpace)}</option>)}</SelectInput></Field>
      <Field label='固定刻度'><div className='landscape-fixed-tick'><input type='range' aria-label='固定刻度滑块' min={0} max={Math.max(0, (data.axis_ticks?.[fixedAxis] ?? []).length - 1)} step={1} value={Math.max(0, (data.axis_ticks?.[fixedAxis] ?? []).indexOf(fixedTick ?? -1))} disabled={!data.axis_ticks?.[fixedAxis]?.length} onChange={event => { const ticks = data.axis_ticks?.[fixedAxis] ?? []; setFixedTick(ticks[Number(event.target.value)] ?? null) }} /><SelectInput value={fixedTick == null ? '' : String(fixedTick)} onChange={event => setFixedTick(Number(event.target.value))}>{(data.axis_ticks?.[fixedAxis] ?? []).map((tick, i) => <option key={tick} value={tick}>{formatLandscapeParameter(fixedAxis, data.axis_values?.[fixedAxis]?.[i] ?? tick)}</option>)}</SelectInput></div></Field>
    </div>}
    {view === '3d' && <Field label='Z 轴参数'><SelectInput value={zName} onChange={event => setZName(event.target.value)}>{activeAxes.filter(axis => axis !== fixedAxis && axis !== xName && axis !== yName).map(axis => <option key={axis} value={axis}>{landscapeParameterLabel(axis, parameterSpace)}</option>)}</SelectInput></Field>}
    {view === '3d' ? activeAxes.length < 3 ? <p className='inline-warning'>当前只有 {activeAxes.length} 个独立参数轴，无法绘制三维散点。</p> : webglFailed ? <p className='inline-warning'>WebGL 不可用，已切回 2D 热图。可在 2D 视图选择参数平面。</p> : <Suspense fallback={<Loading label='加载三维绘图模块' />}><Landscape3D option={scatter3dOption} onFailed={() => { setWebglFailed(true); setView('2d') }} onSelect={raw => { const point = pointFromEvent(raw); if (point) setSelected(point) }} /></Suspense> : validPair ? <div>
      <div>
        <h4>参数切片 · {landscapeParameterLabel(xName, parameterSpace)} × {landscapeParameterLabel(yName, parameterSpace)}</h4>
        {slice.length ? <ChartView option={heatmapOption} height={420} onEvents={{ click: raw => { const point = pointFromEvent(raw); if (point) setSelected(point) } }} /> : <div className='research-chart-empty'><Empty title='当前切片没有候选' detail='尝试切换横纵轴、样本或研究决策。' /></div>}
      </div>
    </div> : activeAxes.length < 2 ? <div className='research-landscape-table'><button type='button' className='secondary-button' onClick={() => exportLandscapeCsv(data.points)}>导出候选 CSV</button><div className='table-scroll'><table><thead><tr><th>候选</th>{['buy_step', 'sell_step', 'buy_shares', 'sell_shares'].map(name => <th key={name}>{landscapeParameterLabel(name, parameterSpace)}</th>)}<th>差值超额</th></tr></thead><tbody>{data.points.slice(0, 100).map(point => { const normalized = canonicalLandscapePoint(point, parameterSpace); return <tr key={point.candidate_id}><td>#{point.candidate_id}</td>{['buy_step', 'sell_step', 'buy_shares', 'sell_shares'].map(name => <td key={name}>{formatLandscapeParameter(name, normalized.params[name])}</td>)}<td>{ratio(point.excess_return)}</td></tr> })}</tbody></table></div></div> : <p className='inline-warning'>可用参数不足两个，无法形成二维切片。</p>}

    <p className='field-hint'>{view === '3d' ? `3D 当前固定条件：${fixedConditions.map(([name, value]) => `${landscapeParameterLabel(name, parameterSpace)}=${formatLandscapeParameter(name, value)}`).join('，') || '无额外固定轴（3 个独立轴均在变化）'}。` : `2D 当前固定条件：${fixedConditions.map(([name, value]) => `${landscapeParameterLabel(name, parameterSpace)}=${formatLandscapeParameter(name, value)}`).join('，') || '服务端按最佳候选确定其余参数'}。`}</p>

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
