import { useEffect, useMemo, useState } from 'react'
import { get } from './api'
import ChartView from './ChartView'
import { Badge, Button, Card, Empty, ErrorBox, Field, Loading, SelectInput, percent } from './components'
import { formatLandscapeParameter, landscapeLabels } from './landscapeData'
import { pointInSlice, regionChartOption, sameRegionPoint } from './connectedRegionData'
import type { Parameters, RegionPoint, RegionView } from './connectedRegionData'
import { v4ValidationLoadInfo } from './researchV4Validation'

function ParametersView({ params }: { params: Parameters }) {
  return <div className='research-inline-params'>{Object.entries(params).filter(([name]) => name !== 'initial_shares').map(([name,value]) =>
    <span key={name}>{landscapeLabels[name] ?? name} {formatLandscapeParameter(name,value)}</span>)}</div>
}

export default function ConnectedRegionResults({ resultId, decisionOptions, onLoad }: {
  resultId: string; decisionOptions: Array<{ value: string; label: string }>
  onLoad: (params: Parameters, asof?: string, dateRange?: { start_date: string; end_date: string }) => void
}) {
  const [decision,setDecision] = useState(decisionOptions[0]?.value ?? 'current')
  const [regionId,setRegionId] = useState<number | null>(null)
  const [page,setPage] = useState(0)
  const [plotPage,setPlotPage] = useState(0)
  const [view,setView] = useState<'slice' | 'projection'>('projection')
  const [anchor,setAnchor] = useState<'stable' | 'peak'>('stable')
  const [x,setX] = useState('')
  const [y,setY] = useState('')
  const [cost,setCost] = useState('original')
  const [fold,setFold] = useState('total')
  const [metric,setMetric] = useState<'excess_return' | 'grid_return'>('excess_return')
  const [data,setData] = useState<RegionView | null>(null)
  const [plot,setPlot] = useState<RegionView | null>(null)
  const [error,setError] = useState('')
  const [loading,setLoading] = useState(false)
  const [plotLoading,setPlotLoading] = useState(false)
  const [decisionKind,indexText] = decision.split(':')
  const baseQuery = new URLSearchParams({ decision: decisionKind, decision_index: indexText || '0', cost })
  if (regionId != null) baseQuery.set('region_id',String(regionId))
  if (fold !== 'total') baseQuery.set('fold',fold)
  const queryString = baseQuery.toString()
  useEffect(() => { setRegionId(null); setPage(0); setPlotPage(0); setData(null); setPlot(null); setError('') },[resultId,decision])
  useEffect(() => {
    let active = true
    setLoading(true)
    void get<RegionView>(`/results/${encodeURIComponent(resultId)}/landscape?${queryString}&view=members&limit=100&offset=${page*100}`)
      .then(value => {
        if (!active) return
        setData(value); setError('')
        if (regionId == null && value.regions.length) { setRegionId(value.regions[0].region_id); setPage(0) }
        if (!x && value.parameters.length) setX(value.parameters[0])
        if (!y && value.parameters.length > 1) setY(value.parameters[1])
      }).catch(cause => { if (active) setError(String(cause)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  },[resultId,queryString,page])
  useEffect(() => {
    if (regionId == null || !x) return
    let active = true
    setPlotLoading(true); setPlot(null)
    const query = new URLSearchParams(queryString)
    query.set('view',view); query.set('x',x); if (y) query.set('y',y)
    query.set('anchor',anchor); query.set('offset',String(plotPage*10000)); query.set('limit','10000')
    void get<RegionView>(`/results/${encodeURIComponent(resultId)}/landscape?${query}`)
      .then(value => { if (active) setPlot(value) })
      .catch(cause => { if (active) setError(String(cause)) })
      .finally(() => { if (active) setPlotLoading(false) })
    return () => { active = false }
  },[resultId,queryString,regionId,view,x,y,anchor,plotPage])
  const option = useMemo(() => plot ? regionChartOption(plot,metric) : {},[plot,metric])
  const dateInfo = v4ValidationLoadInfo(data?.diagnostics)
  const dateRange = dateInfo.kind === 'fresh' ? dateInfo.dateRange : undefined
  const load = (point: RegionPoint) => onLoad(point.params,data?.as_of,dateRange)
  const selectRegion = (id: number) => { setRegionId(id); setPage(0); setPlotPage(0); setPlot(null) }
  const resetPlot = () => setPlotPage(0)
  const exportBase = `/api/v1/results/${encodeURIComponent(resultId)}/regions/${regionId}/export?decision=${decisionKind}&decision_index=${indexText || '0'}`
  const stable = data?.region?.stable_point
  const peak = data?.region?.peak_point
  return <Card title='完整正收益参数域' subtitle='按实际连通成员数排名，允许弯曲、不对称与空洞。每片域均沿逐维相邻点探查至边界闭合。'>
    <div className='form-grid four'>
      <Field label='研究决策'><SelectInput value={decision} onChange={event => setDecision(event.target.value)}>{decisionOptions.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</SelectInput></Field>
      <Field label='着色成本'><SelectInput value={cost} onChange={event => { setCost(event.target.value); setPage(0); resetPlot() }}><option value='original'>原始成本</option><option value='stress'>双倍压力成本</option></SelectInput></Field>
      <Field label='着色时间段'><SelectInput value={fold} onChange={event => { setFold(event.target.value); setPage(0); resetPlot() }}><option value='total'>整个内部验证段</option>{[0,1,2,3].map(i => <option key={i} value={i}>窗口 {i+1}</option>)}</SelectInput></Field>
      <Field label='着色指标'><SelectInput value={metric} onChange={event => setMetric(event.target.value as typeof metric)}><option value='excess_return'>超额收益</option><option value='grid_return'>网格收益</option></SelectInput></Field>
    </div>
    {error && <ErrorBox message={error} />}
    {loading && !data && <Loading label='读取连通参数域' />}
    {data && <>
      <p className='field-hint'>区域和代表点始终按原成本整个内部验证段确定，切换着色不会重新择参。回撤／成交限制仅为诊断。{data.diagnostics.global_exhaustive ? '已穷尽全参数空间。' : '已发现区域完整，不保证发现全部孤立区域。'}</p>
      {data.regions.length ? <div className='table-wrap connected-region-table'><table><thead><tr><th>参数域</th><th>成员 / 域占比</th><th>平均 / 最低超额</th><th>参数最稳定点</th><th>超额收益最高点</th><th>完整性 / 诊断</th></tr></thead><tbody>{data.regions.map(region => <tr key={region.region_id} className={region.region_id === regionId ? 'focus-row' : ''}>
        <td><Button variant='quiet' onClick={() => selectRegion(region.region_id)}>域 {region.region_id} · 查看整片域</Button></td>
        <td><strong>{region.combination_count.toLocaleString()} 组</strong><small>{percent(region.normalized_volume,4)}（当前固定参数切片）</small></td>
        <td>{percent(region.mean_excess_return,4)}<small>最低 {percent(region.min_excess_return,4)}</small></td>
        <td><ParametersView params={region.stable_point.params} /><small>超额 {percent(region.stable_point.excess_return,4)} · 稳定距离 {region.stable_point.stability_margin.toFixed(6)}</small></td>
        <td><ParametersView params={region.peak_point.params} /><small>超额 {percent(region.peak_point.excess_return,4)}{sameRegionPoint(region.stable_point,region.peak_point) ? ' · 与最稳定点重合' : ''}</small></td>
        <td><Badge tone={region.complete && region.boundary_closed ? 'green' : 'amber'}>{region.boundary_closed ? '边界已闭合' : '未完成'}</Badge><small>{region.touches_search_boundary ? '触及搜索范围边缘' : '位于搜索范围内部'}</small><small>{region.constraint_failure_count} 组有约束诊断</small><small>压力：{region.stress?.status ?? '未计算'}{region.stress?.mean_excess_return != null ? ' · 平均超额 '+percent(region.stress.mean_excess_return,4) : ''}</small></td>
      </tr>)}</tbody></table></div> : <Empty title='没有可正式推荐的完整正收益域' detail={String(data.diagnostics.status === 'completed' ? '本次入口探测没有发现正收益区域。' : '任务未完成，未知或失败点不会作为负收益边界。')} />}
      {data.region && data.region.region_id === regionId && stable && peak && <section className='connected-region-detail'>
        <div className='research-candidate-ranking-head'><strong>域 {data.region.region_id} · 整片参数域</strong><div className='connected-region-actions'>
          <Button variant='secondary' disabled={!dateRange || loading} onClick={() => load(stable)}>载入最稳定点验证回测</Button>
          <Button variant='quiet' disabled={!dateRange || loading} onClick={() => load(peak)}>载入最高收益点验证回测</Button>
          <a className='text-link' href={`${exportBase}&format=csv`}>全部成员 CSV</a><a className='text-link' href={`${exportBase}&format=json`}>全部成员 JSON</a>
        </div></div>
        <p className='field-hint'>稳定性是各参数按有效搜索范围归一化后，距域内边界的最短欧氏距离；边界点为 0。它衡量参数扰动容忍度。{dateRange ? `验证窗口 ${dateRange.start_date} 至 ${dateRange.end_date}。` : '验证日期信息不完整，无法载入复现。'}</p>
        <details><summary>各维包围范围（范围的任意组合不一定属于此域）</summary><div className='research-inline-params'>{Object.entries(data.region.bounds).map(([name,bounds]) => <span key={name}>{landscapeLabels[name]} {formatLandscapeParameter(name,bounds.low)} ～ {formatLandscapeParameter(name,bounds.high)}</span>)}</div></details>
        <div className='form-grid four connected-region-controls'>
          <Field label='观察方式'><SelectInput value={view} onChange={event => { setView(event.target.value as typeof view); resetPlot() }}><option value='projection'>整片域的二维投影聚合</option><option value='slice'>固定其他参数的精确切片</option></SelectInput></Field>
          <Field label='切片定位'><SelectInput value={anchor} disabled={view === 'projection'} onChange={event => { setAnchor(event.target.value as typeof anchor); resetPlot() }}><option value='stable'>最稳定点</option><option value='peak'>最高超额收益点</option></SelectInput></Field>
          <Field label='横轴'><SelectInput value={x} onChange={event => { if (event.target.value === y) setY(x); setX(event.target.value); resetPlot() }}>{data.parameters.map(name => <option key={name} value={name}>{landscapeLabels[name]}</option>)}</SelectInput></Field>
          <Field label='纵轴'><SelectInput value={y} onChange={event => { if (event.target.value === x) setX(y); setY(event.target.value); resetPlot() }}>{data.parameters.map(name => <option key={name} value={name}>{landscapeLabels[name]}</option>)}</SelectInput></Field>
        </div>
        <p className='field-hint'>{view === 'projection' ? '投影聚合全部高维成员，相同横纵坐标的颜色为均值；投影重叠不表示高维参数可以任意组合。' : '精确切片固定其余全部参数，按真实刻度定位；空洞保留为空白，不做插值。'}蓝色菱形为最稳定点，橙色三角形为最高超额收益点。</p>
        {plotLoading ? <Loading label='读取参数域图形' /> : plot && <>
          {view === 'slice' && plot.peak_point && !pointInSlice(plot.peak_point,plot) && <p className='field-hint'>最高收益点不在当前切片。<button type='button' className='text-link' onClick={() => { setAnchor('peak'); resetPlot() }}>切换到最高收益点切片</button></p>}
          {view === 'slice' && <p className='field-hint'>固定参数：{plot.parameters.filter(name => name !== x && name !== y).map(name => `${landscapeLabels[name]} ${formatLandscapeParameter(name,plot.fixed_params[name])}`).join('，')}</p>}
          <ChartView option={option} height={430} />
          <div className='connected-region-actions'><span>图形共 {plot.view_total.toLocaleString()} 个{view === 'projection' ? '投影坐标' : '切片成员'}，当前 {Math.min(plot.view_total,plotPage*10000+1).toLocaleString()}～{Math.min(plot.view_total,(plotPage+1)*10000).toLocaleString()}；未显示页不补点。</span><Button variant='quiet' disabled={!plotPage} onClick={() => setPlotPage(plotPage-1)}>上一图页</Button><Button variant='quiet' disabled={(plotPage+1)*10000 >= plot.view_total} onClick={() => setPlotPage(plotPage+1)}>下一图页</Button></div>
        </>}
        <div className='research-candidate-ranking-head'><strong>全部域成员 · {data.total.toLocaleString()} 组</strong><div className='connected-region-actions'><Button variant='quiet' disabled={!page || loading} onClick={() => setPage(page-1)}>上一页</Button><span>第 {page+1} / {Math.max(1,Math.ceil(data.total/100))} 页</span><Button variant='quiet' disabled={(page+1)*100 >= data.total || loading} onClick={() => setPage(page+1)}>下一页</Button></div></div>
        <div className='table-wrap'><table><thead><tr><th>候选</th><th>完整参数</th><th>当前超额</th><th>稳定距离</th><th>标记 / 诊断</th></tr></thead><tbody>{data.points.map(point => <tr key={point.candidate_id}><td>#{point.candidate_id}</td><td><ParametersView params={point.params} /></td><td>{percent(point.excess_return,4)}</td><td>{point.stability_margin?.toFixed(6)}</td><td>{point.is_stable_point && <Badge tone='blue'>最稳定</Badge>}{point.is_peak_point && <Badge tone='amber'>最高收益</Badge>}<small>{point.constraint_reasons?.join('；') || '无约束诊断'}</small></td></tr>)}</tbody></table></div>
      </section>}
    </>}
  </Card>
}
