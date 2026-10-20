import type { EChartsOption } from 'echarts'
import { formatLandscapeParameter, landscapeLabels } from './landscapeData.ts'

export type Parameters = Record<string, unknown>
export type RegionPoint = {
  candidate_id: number; params: Parameters; excess_return: number | null; grid_return?: number | null
  stability_margin: number; is_stable_point?: boolean; is_peak_point?: boolean
  constraint_eligible?: boolean; constraint_reasons?: string[]
}
export type ConnectedRegion = {
  region_id: number; geometry: 'connected_lattice'; combination_count: number; normalized_volume: number
  mean_excess_return: number; min_excess_return: number; max_excess_return: number
  stable_point: RegionPoint; peak_point: RegionPoint; complete: boolean; boundary_closed: boolean
  global_exhaustive: boolean; touches_search_boundary: boolean; constraint_failure_count: number
  bounds: Record<string, { low: number; high: number }>; stress?: { status?: string; mean_excess_return?: number }
}
export type RegionView = {
  geometry: string; regions: ConnectedRegion[]; region: ConnectedRegion | null; parameters: string[]
  total: number; view_total: number; offset: number; limit: number; view: string; x: string; y: string
  points: RegionPoint[]; projection: Array<{ x: number; y: number; count: number; excess_return: number | null; grid_return: number | null }>
  stable_point: RegionPoint | null; peak_point: RegionPoint | null; fixed_params: Parameters
  diagnostics: Record<string, unknown>; as_of?: string
}

export function sameRegionPoint(a: RegionPoint, b: RegionPoint) {
  return a.candidate_id === b.candidate_id
}

export function pointInSlice(point: RegionPoint, data: Pick<RegionView, 'fixed_params' | 'parameters' | 'x' | 'y'>) {
  return data.parameters.every(name => name === data.x || name === data.y || point.params[name] === data.fixed_params[name])
}

export function regionChartOption(data: RegionView, metric: 'excess_return' | 'grid_return'): EChartsOption {
  const project = data.view === 'projection'
  const items = project
    ? data.projection.map(cell => ({ value: [cell.x, cell.y, cell[metric]], count: cell.count }))
    : data.points.map(point => ({ value: [Number(point.params[data.x]), data.y ? Number(point.params[data.y]) : 0, point[metric]], point }))
  const finite = items.filter(item => typeof item.value[2] === 'number' && Number.isFinite(item.value[2]))
  const values = finite.map(item => Number(item.value[2]))
  const unit = (name: string) => ['buy_step', 'sell_step'].includes(name) ? .0001 : ['min_shares', 'max_shares'].includes(name) ? 500 : 100
  const span = (axis: number, name: string) => {
    const coords = items.map(item => Number(item.value[axis])).filter(Number.isFinite)
    return coords.length ? (Math.max(...coords)-Math.min(...coords))/unit(name)+1 : 1
  }
  const symbolSize = Math.min(18, 650/(span(0,data.x)+1), 320/(span(1,data.y)+1))
  const stable = data.stable_point
  const peak = data.peak_point
  const visible = (point: RegionPoint | null): point is RegionPoint => Boolean(point && (project || pointInSlice(point,data)))
  const combined = Boolean(stable && peak && sameRegionPoint(stable,peak))
  const marker = (point: RegionPoint, label: string, color: string, symbol: string) => ({
    name: label, type: 'scatter', symbol, symbolSize: 17, z: 10,
    itemStyle: { color, borderColor: '#fff', borderWidth: 2 },
    label: { show: true, formatter: label, position: 'top', color, fontSize: 11 },
    data: [{ value: [Number(point.params[data.x]), data.y ? Number(point.params[data.y]) : 0, point[metric]], point }],
  })
  const series = [{ name: project ? '域成员投影' : '精确切片成员', type: 'scatter', symbol: 'rect', symbolSize, data: finite },
    ...(visible(stable) ? [marker(stable,combined ? '最稳定 · 最高收益' : '最稳定','#2563eb','diamond')] : []),
    ...(visible(peak) && !combined ? [marker(peak,'最高收益','#c66d17','triangle')] : [])]
  return {
    animation: false, grid: { left: 70, right: 95, top: 45, bottom: 65 },
    xAxis: { type: 'value', name: landscapeLabels[data.x] ?? data.x, scale: true,
      axisLabel: { formatter: (value: number) => formatLandscapeParameter(data.x,value) } },
    yAxis: { type: 'value', name: landscapeLabels[data.y] ?? data.y, scale: true,
      axisLabel: { formatter: (value: number) => data.y ? formatLandscapeParameter(data.y,value) : '' } },
    dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' }, { type: 'inside', yAxisIndex: 0, filterMode: 'none' }],
    visualMap: { type: 'continuous', dimension: 2, seriesIndex: [0], right: 5, top: 65,
      min: values.length ? Math.min(...values) : 0, max: values.length ? Math.max(...values) : .01,
      inRange: { color: ['#f6c4bb','#f3f5e5','#2b9470'] }, formatter: (value: number) => (value*100).toFixed(2)+'%' },
    tooltip: { formatter: (raw: unknown) => {
      const item = (raw as { data?: { value?: unknown[]; count?: number; point?: RegionPoint } }).data
      if (!item?.value) return ''
      return `${landscapeLabels[data.x] ?? data.x} ${formatLandscapeParameter(data.x,item.value[0])}<br/>`
        + `${landscapeLabels[data.y] ?? data.y} ${formatLandscapeParameter(data.y,item.value[1])}<br/>`
        + `${metric === 'excess_return' ? '超额收益' : '网格收益'} ${typeof item.value[2] === 'number' ? (item.value[2]*100).toFixed(4)+'%' : '未计算'}`
        + (item.count ? `<br/>高维成员数 ${item.count}` : '')
        + (item.point ? `<br/>候选 #${item.point.candidate_id} · 稳定距离 ${item.point.stability_margin?.toFixed(6) ?? '—'}` : '')
    } }, series,
  } as unknown as EChartsOption
}
