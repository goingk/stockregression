import assert from 'node:assert/strict'
import test from 'node:test'
import { pointInSlice, regionChartOption, sameRegionPoint } from '../src/connectedRegionData.ts'

function point(id,x,y,shares=100) { return { candidate_id:id,params:{buy_step:x,sell_step:y,buy_shares:shares},excess_return:.01*id,stability_margin:.1 } }
function view() {
  return { view:'slice',x:'buy_step',y:'sell_step',parameters:['buy_step','sell_step','buy_shares'],
    fixed_params:{buy_shares:100},points:[point(1,.001,.002),point(2,.0013,.002)],projection:[],
    stable_point:point(1,.001,.002),peak_point:point(3,.001,.002,200) }
}
test('exact slice chart keeps real coordinates and does not fill missing lattice ticks',() => {
  const data=view()
  const chart=regionChartOption(data,'excess_return')
  assert.equal(chart.xAxis.type,'value')
  assert.deepEqual(chart.series[0].data.map(item=>item.value[0]),[.001,.0013])
  assert.equal(chart.series[0].data.length,2)
  assert.equal(chart.series.length,2) // stable is visible; peak belongs to another slice
  assert.equal(pointInSlice(data.peak_point,data),false)
})
test('coincident representatives receive one combined marker',() => {
  const data=view(); data.peak_point=data.stable_point
  const chart=regionChartOption(data,'excess_return')
  assert.equal(sameRegionPoint(data.peak_point,data.stable_point),true)
  assert.equal(chart.series.length,2)
  assert.equal(chart.series[1].name,'最稳定 · 最高收益')
})
test('projection uses aggregated coordinates and displays both high dimensional representatives',() => {
  const data=view(); data.view='projection'
  data.projection=[{x:.001,y:.002,count:17,excess_return:.03,grid_return:.04}]
  const chart=regionChartOption(data,'excess_return')
  assert.equal(chart.series[0].data[0].count,17)
  assert.equal(chart.series.length,3)
  assert.deepEqual(chart.series[1].data[0].point.params,data.stable_point.params)
  assert.deepEqual(chart.series[2].data[0].point.params,data.peak_point.params)
})
