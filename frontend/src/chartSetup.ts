import * as echarts from 'echarts/core'
import { BarChart, CandlestickChart, LineChart, ScatterChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([BarChart, CandlestickChart, LineChart, ScatterChart, DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent, CanvasRenderer])

export default echarts
