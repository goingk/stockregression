import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { get } from './api'
import ChartView from './ChartView'
import { Badge, Card, Empty, Field, Loading, SelectInput } from './components'
import {
  chooseLandscapeAxes,
  formatLandscapeAxis,
  formatLandscapeNumber,
  formatLandscapeParameter,
  landscapeLabels,
  landscapeSliceKey,
  landscapeValue,
  type LandscapePoint,
  type LandscapeResponse,
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
    {Object.entries(point.params).filter(([name, value]) => name !== 'initial_shares' && value != null).map(([name, value]) =>
      <span key={name}><small>{landscapeLabels[name] ?? name}</small><strong>{formatLandscapeParameter(name, value)}</strong></span>)}
  </div>
}

function pointFromEvent(raw: unknown): LandscapePoint | undefined {
  return ((raw as { data?: { item?: LandscapePoint } }).data ?? {}).item
}

export default function ResearchLandscape({ resultId, defaultParams, decisionOptions, regionOptions = [], qualified = false, trainingOnly = false, defaultSample = trainingOnly ? 'training' : 'validation', allowValidation = !trainingOnly, scoreKind = 'legacy', validationLabel = '内部验证' }: {
  resultId: string
  defaultParams?: Dict
  decisionOptions: Array<{ value: string; label: string }>
  regionOptions?: Array<{ value: string; label: string }>
  qualified?: boolean
  trainingOnly?: boolean
  defaultSample?: 'training' | 'validation'
  allowValidation?: boolean
  scoreKind?: 'legacy' | 'hypercube' | 'robust_j'
  validationLabel?: string
}) {
  const [decision, setDecision] = useState(decisionOptions[0]?.value ?? 'current')
  const [sample, setSample] = useState<'training' | 'validation'>(allowValidation ? defaultSample : 'training')
  const [cost, setCost] = useState<'original' | 'stress'>('original')
  const [fold, setFold] = useState('total')
  const [metric, setMetric] = useState<'excess' | 'grid'>('excess')
  const [region, setRegion] = useState('all')
  const [xName, setXName] = useState('')
  const [yName, setYName] = useState('')
  const [data, setData] = useState<LandscapeResponse | null>(null)
  const [selected, setSelected] = useState<LandscapePoint | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const activeSample = allowValidation ? sample : 'training'

  useEffect(() => {
    setSample(allowValidation ? defaultSample : 'training')
  }, [allowValidation, defaultSample, resultId])

  const [decisionKind, decisionIndexText] = decision.split(':')
  const decisionIndex = Number(decisionIndexText || 0)
  useEffect(() => {
    let active = true
    setLoading(true); setError('')
    const query = new URLSearchParams({ decision: decisionKind, decision_index: String(decisionIndex), sample: activeSample, cost, limit: '1000' })
    if (fold !== 'total') query.set('fold', fold)
    if (region !== 'all') query.set('region_id', region)
    void get<LandscapeResponse>(`/results/${encodeURIComponent(resultId)}/landscape?${query}`)
      .then(value => {
        if (!active) return
        setData(value)
        setSelected(null)
      })
      .catch(cause => { if (active) setError(cause instanceof Error ? cause.message : String(cause)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [resultId, decisionKind, decisionIndex, activeSample, cost, fold, region])

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
        return `${point.is_focus ? '<b>相对最佳单点</b><br/>' : ''}${point.is_center ? '<b>推荐区域中心</b><br/>' : ''}${Object.entries(point.params).filter(([key, value]) => key !== 'initial_shares' && value != null).map(([key, value]) => `${landscapeLabels[key] ?? key}: ${formatLandscapeParameter(key, value)}`).join('<br/>')}<br/>相对全仓超额: ${ratio(point.excess_return)}<br/>网格收益: ${ratio(point.grid_return)}<br/>最大回撤: ${ratio(point.drawdown)}<br/>成交笔数: ${point.trade_count == null ? '—' : number(Number(point.trade_count), 0)}`
      } },
      grid: { left: 78, right: 34, top: 42, bottom: 92 },
      xAxis: {
        type: 'category', name: landscapeLabels[xName] ?? xName, nameLocation: 'middle', nameGap: 40,
        data: xs.map(value => formatLandscapeAxis(xName, value)),
        axisLabel: { rotate: xs.length > 10 ? 35 : 0, hideOverlap: true, fontSize: 10, margin: 11 },
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category', name: landscapeLabels[yName] ?? yName, nameLocation: 'middle', nameGap: 48,
        data: ys.map(value => formatLandscapeAxis(yName, value)),
        axisLabel: { fontSize: 10 }, splitArea: { show: true },
      },
      visualMap: {
        type: 'continuous', dimension: 2, seriesIndex: 0, min: minValue, max: maxValue,
        calculable: true, orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 12, itemHeight: 190,
        text: ['较高', '较低'], textGap: 8,
        inRange: { color: ['#b42318', '#f5f2ef', '#067647'] },
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

  if (loading && !data) return <Loading label='读取参数收益分布' />
  if (error) return <Empty title='参数收益分布不可用' detail={error} />
  if (!data || !data.points.length) return <Empty title='暂无参数收益分布' detail='本次研究没有可展示的训练或验证候选。' />

  const ranking = (data.top_candidates.length ? data.top_candidates : data.focus ? [data.focus] : []).slice(0, 3)
  const focusValue = data.focus?.excess_return ?? null
  const neighborhood = data.focus_neighborhood
  return <Card
    title={trainingOnly ? '训练候选收益分布' : '参数收益分布'}
    subtitle={trainingOnly
      ? '训练搜索候选仅用于发现入口；正式参数域与两个代表点以内部验证结果为准。'
      : scoreKind === 'robust_j'
        ? '当前显示训练样本；稳健 J 的资格看 J、中心约束和峰值核验，正式推荐见峰区表。'
        : scoreKind === 'hypercube'
          ? '当前显示训练样本；超正方体按邻域平均超额评分，不要求每个邻点为正。'
          : '先看相对最佳参数，再用二维切片判断它周围是否稳定。横纵轴以外的参数固定为相对最佳单点。'}
    action={<Badge tone='blue'>{loading ? '读取中…' : `${data.total.toLocaleString()} 个候选 · 当前切片 ${slice.length.toLocaleString()}`}</Badge>}
  >
    {data.focus && !trainingOnly && <div className={`research-landscape-focus ${qualified ? 'verified' : 'diagnostic'}`}>
      <div className='research-landscape-focus-copy'>
        <small>{scoreKind === 'legacy' ? (qualified ? '全量候选中的相对最佳单点' : '相对最佳单点 · 仅作研究对照') : '当前样本中的相对最佳单点'}</small>
        <strong>相对全仓超额 {ratio(focusValue)}</strong>
        <span>从全部 {data.total.toLocaleString()} 个候选中按当前筛选选出；候选 #{data.focus.candidate_id ?? '—'}。{qualified
          ? (scoreKind === 'legacy' ? '是否成为正式推荐仍取决于完整邻域验证。' : '正式推荐以峰区表为准；这里用于观察参数分布。')
          : (scoreKind === 'legacy' ? '它没有通过完整正超额邻域验证，不能当作合格推荐。' : '它没有通过当前策略的资格核验，不能当作合格推荐。')}</span>
      </div>
      <ParameterChips point={data.focus} />
    </div>}

    {scoreKind === 'legacy' && neighborhood && neighborhood.available && (
      neighborhood.metric_available === false ?
        <div className='research-neighborhood-check'>
          <div><strong>该样本没有收益数据</strong><span>邻点参数存在，但当前样本没有可用的收益指标；这不代表邻点收益为负。</span></div>
          <div className='research-neighborhood-stats'>
            <span><small>已计算邻点</small><strong>{neighborhood.evaluated_count} / {neighborhood.expected_count}</strong></span>
            <span><small>收益指标</small><strong>缺失</strong></span>
          </div>
        </div>
      : <div className={`research-neighborhood-check ${neighborhood.all_positive ? 'passed' : 'failed'}`}>
        <div><strong>{neighborhood.all_positive ? '所选细搜索邻域全为正' : '所选细搜索邻域已计算，但没有通过'}</strong><span>二维切片只是其中一个截面；下面的数字来自本次勾选参与邻域移动的参数，未勾选参数保持中心值。</span></div>
        <div className='research-neighborhood-stats'>
          <span><small>已计算邻点</small><strong>{neighborhood.evaluated_count} / {neighborhood.expected_count}</strong></span>
          <span><small>正收益邻点</small><strong>{neighborhood.positive_count ?? '—'}（{ratio(neighborhood.positive_ratio)}）</strong></span>
          <span><small>邻点第 25 百分位</small><strong>{ratio(neighborhood.q25_excess)}</strong></span>
        </div>
        <p>{neighborhood.all_positive ? '最小即时邻域内收益均为正；是否成为正式推荐仍以完整邻域表的体积与资格为准。' : '只要即时邻点中存在负收益，就不构成完整正超额邻域；该点不能据此作为推荐。'}</p>
      </div>
    )}

    {scoreKind === 'robust_j' && data.focus && <div className='research-neighborhood-check passed'>
      <div><strong>稳健 J 邻域（训练样本）</strong><span>四维一格联合邻域；邻点不需要全部为正。</span></div>
      <div className='research-neighborhood-stats'>
        <span><small>J</small><strong>{ratio(data.focus.j_score)}</strong></span>
        <span><small>中心超额</small><strong>{ratio(data.focus.excess_return)}</strong></span>
        <span><small>邻域均值</small><strong>{ratio(data.focus.neighborhood_mean)}</strong></span>
        <span><small>正收益占比</small><strong>{ratio(data.focus.pos_win_rate ?? data.focus.neighbor_positive_ratio)}</strong></span>
        <span><small>邻域点数</small><strong>{data.focus.neighborhood_size ?? '—'}</strong></span>
        <span><small>评分完整</small><strong>{data.focus.score_complete === true ? '是' : data.focus.score_complete === false ? '否' : '—'}</strong></span>
      </div>
      <p>推荐资格由 J&gt;0、回撤/成交/持仓约束和局部峰值核验共同决定；邻点负收益会降低 J，不会直接取消资格。</p>
    </div>}

    {scoreKind === 'hypercube' && data.focus && <div className='research-neighborhood-check passed'>
      <div><strong>超正方体平均超额</strong><span>训练样本内按固定边长计算邻域均值。</span></div>
      <div className='research-neighborhood-stats'>
        <span><small>中心超额</small><strong>{ratio(data.focus.excess_return)}</strong></span>
        <span><small>最大回撤</small><strong>{ratio(data.focus.drawdown)}</strong></span>
        <span><small>成交笔数</small><strong>{data.focus.trade_count == null ? '—' : number(Number(data.focus.trade_count), 0)}</strong></span>
      </div>
      <p>当前门槛是平均超额为正且中心满足约束；不要求每个邻点都为正。</p>
    </div>}

    <div className='research-chart-guide'>
      <span><b>每个格子</b>是一组参数</span>
      <span><b>绿色</b>表示当前筛选内相对更高，红色更低；数值仍可能为负</span>
      <span><b>空白</b>表示该组合不合法、未计算或超出切片显示上限，不做插值</span>
    </div>

    <div className='form-grid four research-landscape-controls'>
      <Field label='研究决策'><SelectInput value={decision} onChange={event => setDecision(event.target.value)}>{decisionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>
      <Field label='样本用途'><SelectInput value={activeSample} disabled={!allowValidation} onChange={event => setSample(event.target.value as 'training' | 'validation')}>{allowValidation && <option value='validation'>{validationLabel}</option>}<option value='training'>训练搜索</option></SelectInput></Field>
      <Field label='成本情景'><SelectInput value={cost} onChange={event => setCost(event.target.value as 'original' | 'stress')}><option value='original'>原始成本</option><option value='stress'>双倍压力</option></SelectInput></Field>
      <Field label='验证窗口'><SelectInput value={fold} onChange={event => setFold(event.target.value)} disabled={activeSample === 'training'}><option value='total'>整个验证段</option>{[0, 1, 2, 3].map(index => <option key={index} value={index}>窗口 {index + 1}</option>)}</SelectInput></Field>
      <Field label='着色指标'><SelectInput value={metric} onChange={event => setMetric(event.target.value as 'excess' | 'grid')}><option value='excess'>相对全仓超额</option><option value='grid'>网格收益</option></SelectInput></Field>
      {regionOptions.length > 1 && <Field label='推荐区域'><SelectInput value={region} onChange={event => setRegion(event.target.value)}><option value='all'>全部候选</option>{regionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>}
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

    <p className='field-hint research-landscape-sampling'>{activeSlice ? `参数切片读取固定其余参数后的完整平面，共 ${activeSlice.total.toLocaleString()} 个候选${activeSlice.sampled ? `；当前超过显示上限，图中等距展示 ${activeSlice.points.length.toLocaleString()} 个` : '，邻域判断不依赖全空间抽样'}。` : '当前组合没有可读取的二维平面，已回退到全空间抽样点。'}</p>

    {validPair ? <div>
      <div>
        <h4>参数切片 · {landscapeLabels[xName] ?? xName} × {landscapeLabels[yName] ?? yName}</h4>
        {slice.length ? <ChartView option={heatmapOption} height={420} onEvents={{ click: raw => { const point = pointFromEvent(raw); if (point) setSelected(point) } }} /> : <div className='research-chart-empty'><Empty title='当前切片没有候选' detail='尝试切换横纵轴、样本或研究决策。' /></div>}
      </div>
    </div> : <p className='inline-warning'>可用参数不足两个，无法形成二维切片。相对最佳参数仍显示在上方。</p>}

    <p className='field-hint'>当前切片固定于 {fixedNames.filter(name => center[name] != null).map(name => `${landscapeLabels[name] ?? name}=${formatLandscapeParameter(name, center[name])}`).join('，') || '相对最佳单点的其余参数'}。</p>

    <section className='research-candidate-ranking'>
      <div className='research-candidate-ranking-head'>
        <div><strong>全量候选排名</strong><small>按当前筛选的相对全仓超额排序，只显示前 {ranking.length} 名</small></div>
        <span>相对最佳 #{data.focus?.candidate_id ?? '—'}</span>
      </div>
      {ranking.length ? <div className='table-wrap'><table><thead><tr><th>排名</th><th>收益</th><th>参数</th><th>状态</th></tr></thead><tbody>{ranking.map(candidate => {
        const candidateValue = candidate.excess_return
        return <tr key={candidate.candidate_id} className={candidate.is_focus ? 'focus-row' : ''}>
          <td>{candidate.rank ?? (candidate.is_focus ? 1 : '—')}</td>
          <td><strong>{ratio(candidateValue)}</strong></td>
          <td><div className='research-inline-params'>{Object.entries(candidate.params).filter(([name, value]) => name !== 'initial_shares' && value != null).map(([name, value]) => <span key={name}>{landscapeLabels[name] ?? name} {formatLandscapeParameter(name, value)}</span>)}</div></td>
          <td><button type='button' className='text-link' onClick={() => setSelected(candidate)}>{candidate.is_focus ? '相对最佳 · 查看' : candidate.is_center ? '推荐中心 · 查看' : '查看'}</button></td>
        </tr>
      })}</tbody></table></div> : <Empty title='没有可排名的候选' />}
    </section>

    {selected && <section className='research-selected-candidate'>
      <div className='research-selected-head'>
        <div><strong>候选 #{selected.candidate_id ?? '—'}</strong><span>{selected.is_focus ? '相对最佳单点' : selected.is_center ? '推荐区域中心' : selected.is_baseline ? '对照策略' : '普通候选'} · {selected.status === 'completed' ? '已计算' : selected.status}</span></div>
        <button type='button' className='text-link' onClick={() => setSelected(null)}>收起</button>
      </div>
      <ParameterChips point={selected} />
      <div className='research-selected-metrics'>
        <span><small>相对全仓超额</small><strong>{ratio(selected.excess_return)}</strong></span>
        <span><small>网格收益</small><strong>{ratio(selected.grid_return)}</strong></span>
        <span><small>最大回撤</small><strong>{ratio(selected.drawdown)}</strong></span>
        <span><small>成交笔数</small><strong>{selected.trade_count == null ? '—' : number(Number(selected.trade_count), 0)}</strong></span>
        <span><small>邻点正收益占比</small><strong>{ratio(selected.is_focus && data.focus_neighborhood ? data.focus_neighborhood.positive_ratio : selected.neighbor_positive_ratio)}</strong></span>
        <span><small>邻点第 25 百分位</small><strong>{ratio(selected.is_focus && data.focus_neighborhood ? data.focus_neighborhood.q25_excess : selected.neighbor_q25_excess)}</strong></span>
      </div>
      {selected.folds.length > 0 && <details><summary>查看 {selected.folds.length} 个验证窗口的收益证据</summary><pre className='json-preview'>{JSON.stringify(selected.folds, null, 2)}</pre></details>}
    </section>}
  </Card>
}
