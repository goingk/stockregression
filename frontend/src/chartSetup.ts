import * as echarts from 'echarts/core'
import { BarChart, CandlestickChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([BarChart, CandlestickChart, HeatmapChart, LineChart, ScatterChart, DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent, VisualMapComponent, CanvasRenderer])

export default echarts
