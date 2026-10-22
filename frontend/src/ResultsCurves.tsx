import { useMemo } from 'react'
import type { EChartsOption } from 'echarts'
import ChartView from './ChartView'
import { money, percent } from './components'
import { buildResultCurveModel, resultCurveRowAt } from './resultPresentation'
import type { BacktestResult } from './types'

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]!)

export default function ResultsCurves({ result }: { result: BacktestResult }) {
  const model = useMemo(() => buildResultCurveModel(result), [result])
  const option: EChartsOption = useMemo(() => {
    const bounds = model.rows.length ? { min: model.rows[0].time, max: model.rows[model.rows.length - 1].time } : {}
    const series = model.strategies.flatMap((strategy) => ([
      {
        id: `${strategy.key}-return`, name: strategy.name, type: 'line' as const, xAxisIndex: 0, yAxisIndex: 0,
        showSymbol: false, connectNulls: false, data: model.rows.map((row) => [row.time, row.values[strategy.key]?.returnRate ?? null]),
        lineStyle: { color: strategy.color, width: strategy.key === 'grid' ? 2.3 : 1.7 }, itemStyle: { color: strategy.color },
      },
      {
        id: `${strategy.key}-drawdown`, name: strategy.name, type: 'line' as const, xAxisIndex: 1, yAxisIndex: 1,
        showSymbol: false, connectNulls: false, data: model.rows.map((row) => [row.time, row.values[strategy.key]?.drawdown ?? null]),
        lineStyle: { color: strategy.color, width: 1.6 }, itemStyle: { color: strategy.color },
        areaStyle: strategy.key === 'grid' ? { opacity: .06 } : undefined,
      },
    ]))
    if (model.strategies.some((strategy) => strategy.key === 'grid')) {
      series.push({
        id: 'grid-position', name: model.strategies.find((strategy) => strategy.key === 'grid')!.name,
        type: 'line', xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, connectNulls: false,
        data: model.rows.map((row) => [row.time, row.position]), lineStyle: { color: '#2563eb', width: 1.8 }, itemStyle: { color: '#2563eb' },
      })
    }
    return {
      animation: false,
      legend: { data: model.strategies.map((strategy) => strategy.name), top: 0, left: 'center', itemWidth: 18, itemHeight: 9, textStyle: { fontSize: 11 } },
      grid: [
        { left: 62, right: 25, top: 58, height: '30%' },
        { left: 62, right: 25, top: '45%', height: '19%' },
        { left: 62, right: 25, top: '73%', bottom: 65 },
      ],
      axisPointer: { link: [{ xAxisIndex: [0, 1, 2] }], label: { backgroundColor: '#65758d' } },
      tooltip: {
        trigger: 'axis', confine: true, axisPointer: { type: 'line', snap: true },
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;',
        formatter: (raw) => {
          const entries = Array.isArray(raw) ? raw : [raw]
          const first = entries[0] as { axisValue?: string | number; value?: unknown[] } | undefined
          const row = resultCurveRowAt(model, first?.value?.[0] ?? first?.axisValue)
          if (!row) return ''
          const date = escapeHtml(row.timestamp.replace('T', ' ').slice(0, 19))
          const rows = model.strategies.map((strategy) => {
            const values = row.values[strategy.key]
            const marker = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${strategy.color};margin-right:5px"></span>`
            return `${marker}${escapeHtml(strategy.name)}　峰值本金收益 ${percent(values?.returnRate)}　回撤 ${percent(values?.drawdown)}<br/><span style="padding-left:13px;color:#94a3b8">资产 ${money(values?.asset)} 元　累计投入 ${money(values?.invested)} 元</span>`
          })
          const position = model.positionMode === 'ratio' ? percent(row.position) : row.position == null ? '—' : `${money(row.position, 0)} ${model.shareUnit}`
          return [`<strong>${date}</strong>`, ...rows, `网格${model.positionMode === 'ratio' ? '仓位' : '持有'}　${position}`].join('<br/>')
        },
      },
      xAxis: [0, 1, 2].map((gridIndex) => ({
        type: 'time' as const, gridIndex, ...bounds, boundaryGap: [0, 0],
        axisLabel: { show: gridIndex === 2, hideOverlap: true, fontSize: 11 },
        axisTick: { show: gridIndex === 2 }, axisLine: { lineStyle: { color: '#cdd5e1' } },
        splitLine: { show: false }, axisPointer: { show: true, snap: true, label: { show: gridIndex === 2 } },
      })),
      yAxis: [
        { name: '峰值本金收益率', gridIndex: 0, axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` } },
        { name: '回撤', gridIndex: 1, axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` } },
        { name: model.positionMode === 'ratio' ? '网格仓位' : `网格持有（${model.shareUnit}）`, gridIndex: 2, axisLabel: { formatter: (value: number) => model.positionMode === 'ratio' ? `${(value * 100).toFixed(0)}%` : value.toLocaleString('zh-CN') } },
      ].map((axis) => ({ ...axis, type: 'value' as const, nameGap: 12, nameLocation: 'end' as const, nameTextStyle: { align: 'left' as const, fontSize: 11 }, axisLabel: { ...axis.axisLabel, fontSize: 10, color: '#64748b' }, splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } } })),
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1, 2], filterMode: 'none' },
        {
          type: 'slider', xAxisIndex: [0, 1, 2], filterMode: 'none', height: 20, bottom: 7, left: 62, right: 25, brushSelect: false,
          borderColor: '#e2e8f0', fillerColor: 'rgba(37, 99, 235, 0.10)', backgroundColor: '#f8fafc',
          handleStyle: { color: '#ffffff', borderColor: '#2563eb' },
        },
      ],
      series,
    }
  }, [model])

  return <div className='results-curves' aria-label='收益、回撤与网格持仓共享日期轴，拖动底部滑块同时缩放三条曲线'><ChartView option={option} height={620} /></div>
}
