import { useEffect, useRef, useState } from 'react'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'
import echarts from './chartSetup'
import './landscape3d.css'

const registerGL = async () => {
  const [charts, components] = await Promise.all([import('echarts-gl/charts'), import('echarts-gl/components')])
  echarts.use([charts.Scatter3DChart, components.Grid3DComponent] as never[])
}

type Camera = { alpha: number; beta: number; distance: number; orthographicSize: number; center: number[] }
const defaultCamera: Camera = { alpha: 25, beta: 40, distance: 160, orthographicSize: 180, center: [0, 0, 0] }
const cameras: Record<'俯视' | '正视' | '侧视', Camera> = {
  俯视: { ...defaultCamera, alpha: 90, beta: 0 },
  正视: { ...defaultCamera, alpha: 0, beta: 0 },
  侧视: { ...defaultCamera, alpha: 0, beta: 90 },
}

export default function Landscape3D({ option, height = 480, onFailed, onSelect }: { option: EChartsCoreOption; height?: number; onFailed: () => void; onSelect?: (raw: unknown) => void }) {
  const element = useRef<HTMLDivElement>(null)
  const instance = useRef<EChartsType | null>(null)
  const cameraRef = useRef<Camera>(defaultCamera)
  const onFailedRef = useRef(onFailed)
  const onSelectRef = useRef(onSelect)
  const [ready, setReady] = useState(false)
  onFailedRef.current = onFailed
  onSelectRef.current = onSelect
  useEffect(() => {
    let alive = true
    let observer: ResizeObserver | null = null
    void registerGL().then(() => {
      if (!alive || !element.current) return
      const probe = document.createElement('canvas')
      const hasWebGL = Boolean(probe.getContext('webgl2') || probe.getContext('webgl') || probe.getContext('experimental-webgl'))
      if (!hasWebGL) { onFailedRef.current(); return }
      try {
        const chart = echarts.init(element.current, undefined, { renderer: 'canvas' })
        instance.current = chart
        chart.setOption(option)
        chart.on('grid3dcamerachanged', (raw: unknown) => {
          const event = raw as Partial<Camera>
          if (Number.isFinite(event.alpha) && Number.isFinite(event.beta) && Number.isFinite(event.distance)) {
            cameraRef.current = { ...cameraRef.current, alpha: event.alpha!, beta: event.beta!, distance: event.distance!, center: event.center ?? cameraRef.current.center }
            const chartInternals = chart as unknown as {
              getModel(): { getComponent(mainType: string): unknown }
              getViewOfComponentModel(model: unknown): { _control?: { getOrthographicSize?: () => number; getCenter?: () => number[] } }
            }
            const model = chartInternals.getModel().getComponent('grid3D')
            const view = model ? chartInternals.getViewOfComponentModel(model) : undefined
            const control = view?._control
            const orthographicSize = control?.getOrthographicSize?.()
            if (Number.isFinite(orthographicSize)) cameraRef.current.orthographicSize = orthographicSize!
            const center = control?.getCenter?.()
            if (Array.isArray(center)) cameraRef.current.center = center
          }
        })
        if (onSelectRef.current) chart.on('click', raw => onSelectRef.current?.(raw))
        observer = new ResizeObserver(() => chart.resize())
        observer.observe(element.current)
        setReady(true)
      } catch { onFailedRef.current() }
    }).catch(() => onFailedRef.current())
    return () => { alive = false; observer?.disconnect(); instance.current?.dispose(); instance.current = null }
  // Initial option is applied below after GL has registered.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => {
    if (!ready || !instance.current) return
    const next = { ...(option as Record<string, unknown>) }
    const grid = next.grid3D as Record<string, unknown> | undefined
    if (grid) next.grid3D = { ...grid, viewControl: { ...(grid.viewControl as Record<string, unknown> ?? {}), ...cameraRef.current, projection: 'orthographic', rotateSensitivity: 1, zoomSensitivity: 1, panMouseButton: 'middle' } }
    instance.current.setOption(next as EChartsCoreOption, { notMerge: false, lazyUpdate: true })
  }, [option, ready])
  const resetCamera = (preset: Camera) => {
    cameraRef.current = preset
    instance.current?.setOption({ grid3D: { viewControl: { ...preset, projection: 'orthographic' } } })
  }
  const resetDefaultCamera = () => resetCamera(defaultCamera)
  const exportPng = async () => {
    const chart = instance.current
    if (!chart) return
    const dataUrl = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' })
    const blob = await (await fetch(dataUrl)).blob()
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'stockregression-landscape-3d.png'
    document.body.append(anchor)
    anchor.click()
    anchor.remove()
    setTimeout(() => URL.revokeObjectURL(url), 60_000)
  }
  return <div className='landscape3d'>
    <div className='landscape3d-tools'><span>左键旋转 · 滚轮缩放 · 中键平移</span><button type='button' onClick={resetDefaultCamera}>重置</button>{Object.entries(cameras).map(([label, preset]) => <button key={label} type='button' onClick={() => resetCamera(preset)}>{label}</button>)}<button type='button' onClick={exportPng}>导出 PNG</button></div>
    <div ref={element} style={{ width: '100%', height }} role='img' aria-label='三维参数空间散点图' />
  </div>
}
