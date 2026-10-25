import type { EChartsOption } from 'echarts'
import ChartView from './ChartView'
import echarts from './chartSetup'
import { money } from './components'
import { assetInvestmentReturn } from './equityMath'
import type { BacktestResult, EquityPoint } from './types'

export { assetInvestmentRatio, assetInvestmentReturn, costBasisDrawdowns, costBasisDrawdowns as contributionAdjustedDrawdowns } from './equityMath'

const curves = [
  { key: 'grid', name: '网格交易', color: '#2563eb', width: 2.4 },
  { key: 'same_account_passive', name: '同账户被动持有', color: '#0d9488', width: 1.9 },
  { key: 'buy_hold', name: '全仓持有', color: '#64748b', width: 1.7 },
  { key: 'dca', name: '定投', color: '#d97706', width: 1.9 },
] as const

/** 柔和的网格曲线渐变填充：从主色半透明渐隐到完全透明。 */
const gridAreaGradient = () => new echarts.graphic.LinearGradient(0, 0, 0, 1, [
  { offset: 0, color: 'rgba(37, 99, 235, 0.12)' },
  { offset: 1, color: 'rgba(37, 99, 235, 0.0)' },
])

function denominator(point: EquityPoint, result: BacktestResult) {
  if (typeof point.total_invested === 'number' && Number.isFinite(point.total_invested)) return Math.max(0, point.total_invested)
  if (result.accounting_basis === 'independent_contributions_v1') return null
  return result.funding?.initial_equity ?? result.config.initial_cash ?? null
}

export default function EquityCurve({ result, height, stretch = false }: { result: BacktestResult; height: number; stretch?: boolean }) {
  const option: EChartsOption = {
    animation: false,
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.92)',
      borderColor: 'rgba(255, 255, 255, 0.12)',
      borderWidth: 1,
      textStyle: { color: '#f8fafc', fontSize: 12 },
      extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;',
      formatter: (raw) => {
        const rows = Array.isArray(raw) ? raw : [raw]
        const first = rows[0] as { axisValueLabel?: string; axisValue?: string | number } | undefined
        const date = String(first?.axisValueLabel ?? first?.axisValue ?? '').replace('T', ' ').slice(0, 16)
        const details = rows.map((entry) => {
          const item = entry as { marker?: string; seriesName?: string; data?: { asset?: number; invested?: number | null; returnRate?: number | null } }
          const data = item.data
          const returnRate = data?.returnRate
          return `${item.marker ?? ''}${item.seriesName ?? ''}　资产 ${money(data?.asset)} 元　累计投入 ${money(data?.invested)} 元　收益率 ${returnRate == null ? '—' : `${(returnRate * 100).toFixed(2)}%`}`
        })
        return [date, ...details].join('<br/>')
      },
    },
    legend: { top: 2, right: 4 },
    grid: { left: 57, right: 20, top: 42, bottom: 34 },
    xAxis: { type: 'time', axisLabel: { hideOverlap: true, color: '#64748b', fontSize: 11 } },
    yAxis: { type: 'value', scale: false, name: '当前峰值本金收益率', axisLabel: { color: '#64748b', fontSize: 11, formatter: (value: number) => `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } } },
    dataZoom: [{ type: 'inside' }],
    series: curves.filter((curve) => result.strategies[curve.key]).map((curve) => {
      const strategy = result.strategies[curve.key]
      const points = strategy?.equity ?? []
        return {
        name: curve.name,
        type: 'line' as const,
        showSymbol: false,
        connectNulls: false,
        data: points.map((point) => {
          const returnRate = assetInvestmentReturn(point, strategy, result)
          const invested = denominator(point, result)
          return {
            value: [point.timestamp, returnRate],
            asset: point.equity,
            invested,
            returnRate,
          }
        }),
        itemStyle: { color: curve.color },
        lineStyle: { color: curve.color, width: curve.width },
        areaStyle: curve.key === 'grid' ? { color: gridAreaGradient() } : undefined,
      }
    }),
  }

  return <ChartView option={option} height={height} stretch={stretch} />
}
