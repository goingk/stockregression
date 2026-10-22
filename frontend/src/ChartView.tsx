import { useEffect, useMemo, useRef } from 'react'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'
import echarts from './chartSetup'

/** 与全局样式一致的画布字体与面板底色，保证图表文字和导出的 PNG 与页面观感统一。 */
const chartFont = "'DM Sans', 'Noto Sans SC', 'Microsoft YaHei', sans-serif"
const chartSurface = '#ffffff'

export default function ChartView({ option, height, stretch = false, onEvents }: { option: EChartsCoreOption; height: number; stretch?: boolean; onEvents?: Record<string, (params: unknown) => void> }) {
  const element = useRef<HTMLDivElement>(null)
  const instance = useRef<EChartsType | null>(null)
  const mergedOption = useMemo(() => {
    const source = (option ?? {}) as unknown as Record<string, unknown>
    const textStyle = (source.textStyle ?? {}) as Record<string, unknown>
    return {
      backgroundColor: chartSurface,
      ...source,
      textStyle: { fontFamily: chartFont, color: '#475569', ...textStyle },
    }
  }, [option])
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current, undefined, { renderer: 'canvas' })
    instance.current = chart
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => { observer.disconnect(); chart.dispose(); instance.current = null }
  }, [])
  useEffect(() => { instance.current?.setOption(mergedOption, { notMerge: true }) }, [mergedOption])
  useEffect(() => {
    const chart = instance.current
    if (!chart || !onEvents) return
    for (const [name, handler] of Object.entries(onEvents)) chart.on(name, handler)
    return () => { for (const [name, handler] of Object.entries(onEvents)) chart.off(name, handler) }
  }, [onEvents])
  const exportChart = async () => {
    if (!instance.current) return
    const dataUrl = instance.current.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: chartSurface })
    const blob = await (await fetch(dataUrl)).blob()
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'stockregression-chart.png'
    document.body.append(anchor)
    anchor.click()
    anchor.remove()
    setTimeout(() => URL.revokeObjectURL(url), 60_000)
  }
  return <div className={`chart-view ${stretch ? 'chart-stretch' : ''}`} style={{ position: 'relative', minHeight: height }}><div ref={element} style={{ width: '100%', height: stretch ? '100%' : height, minHeight: height }} role='img' aria-label='数据图表' /><button type='button' className='chart-download' aria-label='下载图表 PNG' onClick={() => void exportChart()}>导出 PNG</button></div>
}
