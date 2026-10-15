import { useEffect, useRef } from 'react'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'
import echarts from './chartSetup'

export default function ChartView({ option, height }: { option: EChartsCoreOption; height: number }) {
  const element = useRef<HTMLDivElement>(null)
  const instance = useRef<EChartsType | null>(null)
  useEffect(() => {
    if (!element.current) return
    const chart = echarts.init(element.current, undefined, { renderer: 'canvas' })
    instance.current = chart
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => { observer.disconnect(); chart.dispose(); instance.current = null }
  }, [])
  useEffect(() => { instance.current?.setOption(option, { notMerge: true }) }, [option])
  const exportChart = async () => {
    if (!instance.current) return
    const dataUrl = instance.current.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' })
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
  return <div style={{ position: 'relative' }}><div ref={element} style={{ width: '100%', height }} role='img' aria-label='数据图表' /><button type='button' className='chart-download' aria-label='下载图表 PNG' onClick={() => void exportChart()}>导出 PNG</button></div>
}
