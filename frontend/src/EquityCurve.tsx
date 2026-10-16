import type { EChartsOption } from 'echarts'
import ChartView from './ChartView'
import { money } from './components'
import { assetInvestmentReturn } from './equityMath'
import type { BacktestResult, EquityPoint, StrategyResult } from './types'

export { assetInvestmentRatio, assetInvestmentReturn, costBasisDrawdowns, costBasisDrawdowns as contributionAdjustedDrawdowns } from './equityMath'

const curves = [
  { key: 'grid', name: '网格交易', color: '#2864dc', width: 2.3 },
  { key: 'buy_hold', name: '全仓持有', color: '#8a98af', width: 1.7 },
  { key: 'dca', name: '定投', color: '#e29136', width: 1.9 },
] as const

function denominator(point: EquityPoint, strategy: StrategyResult, result: BacktestResult) {
  if (typeof point.total_invested === 'number' && Number.isFinite(point.total_invested)) return Math.max(0, point.total_invested)
  if (result.accounting_basis === 'independent_contributions_v1') {
    if (typeof point.normalized_equity === 'number' && Number.isFinite(point.normalized_equity) && point.normalized_equity > 0) return point.equity / point.normalized_equity
    return null
  }
  if (strategy.return_basis === 'cost' && typeof point.normalized_equity === 'number' && point.normalized_equity > 0) return point.equity / point.normalized_equity
  if (strategy.return_basis === 'cost') return null
  return result.funding?.initial_equity ?? result.config.initial_cash ?? null
}

export default function EquityCurve({ result, height }: { result: BacktestResult; height: number }) {
  const option: EChartsOption = {
    animation: false,
    tooltip: {
      trigger: 'axis',
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
    xAxis: { type: 'time', axisLabel: { hideOverlap: true } },
    yAxis: { type: 'value', scale: false, name: '投入收益率', axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: '#eef1f5' } } },
    dataZoom: [{ type: 'inside' }],
    series: curves.map((curve) => {
      const strategy = result.strategies[curve.key]
      const points = strategy?.equity ?? []
        return {
        name: curve.name,
        type: 'line' as const,
        showSymbol: false,
        connectNulls: false,
        data: points.map((point) => {
          const returnRate = assetInvestmentReturn(point, strategy, result)
          const invested = denominator(point, strategy, result)
          return {
            value: [point.timestamp, returnRate],
            asset: point.equity,
            invested,
            returnRate,
          }
        }),
        itemStyle: { color: curve.color },
        lineStyle: { color: curve.color, width: curve.width },
      }
    }),
  }

  return <ChartView option={option} height={height} />
}
