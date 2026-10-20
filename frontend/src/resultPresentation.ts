import { assetInvestmentReturn, costBasisDrawdowns } from './equityMath.ts'
import type { BacktestResult, EquityPoint, StrategyResult } from './types.ts'

export type ResultParameter = { key: string; label: string; value: string; note?: string }
type SavedRecord = Record<string, unknown>

const record = (value: unknown): SavedRecord => value != null && typeof value === 'object' && !Array.isArray(value) ? value as SavedRecord : {}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const numberText = (value: unknown, digits = 2) => finite(value) ? value.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—'
const amountText = (value: unknown, unit: string, digits = 2) => finite(value) ? `${numberText(value, digits)} ${unit}` : '—'
const percentText = (value: unknown) => finite(value) ? `${(value * 100).toFixed(2)}%` : '—'

export function resultShareUnit(symbol: unknown) {
  return /^[15]/.test(String(symbol ?? '')) ? '份' : '股'
}

function quantityText(value: unknown, unit: string) {
  const quantity = record(value)
  if (!finite(quantity.value)) return '—'
  if (quantity.mode === 'shares') return amountText(quantity.value, unit, 0)
  if (quantity.mode === 'cash') return amountText(quantity.value, '元')
  if (quantity.mode === 'equity_ratio') return `${percentText(quantity.value)} 净资产`
  return quantity.mode ? `${numberText(quantity.value)}（${String(quantity.mode)}）` : '—'
}

/** Read only the saved result. Missing historical fields must never acquire editor defaults. */
export function savedGridParameters(value: unknown): ResultParameter[] {
  const result = record(value)
  const config = record(result.config)
  const funding = record(result.funding)
  const unit = resultShareUnit(result.symbol ?? config.symbol)
  const version = config.schema_version ?? result.schema_version
  const shareBased = typeof version === 'number' && version >= 2 || (version !== 1 && ['initial_shares', 'min_shares', 'max_shares', 'buy_shares', 'sell_shares'].some((key) => finite(config[key])))
  const modeNames: Record<string, string> = { trailing: '移动网格', arithmetic: '等差网格', geometric: '等比网格' }
  const mode = typeof config.grid_mode === 'string' ? config.grid_mode : ''
  const rows: ResultParameter[] = [{ key: 'grid_mode', label: '网格模式', value: modeNames[mode] ?? (mode || '—') }]
  if (finite(config.buy_step) || finite(config.sell_step)) {
    rows.push({ key: 'buy_step', label: '买入间距', value: percentText(config.buy_step) }, { key: 'sell_step', label: '卖出间距', value: percentText(config.sell_step) })
  } else if ((mode === 'arithmetic' || mode === 'geometric') && finite(config.grid_step)) {
    rows.push({ key: 'grid_step', label: '旧版网格间距', value: percentText(config.grid_step) })
  }
  const reference = finite(funding.reference_price) ? funding.reference_price : config.reference_price
  const referenceMode = config.reference_price_mode === 'auto' ? '自动参考' : config.reference_price_mode === 'manual' ? '手动参考' : ''
  const referenceDate = typeof funding.reference_date === 'string' ? funding.reference_date : ''
  rows.push(
    { key: 'reference_price', label: '参考价格', value: amountText(reference, '元', 3), note: [referenceMode, referenceDate].filter(Boolean).join(' · ') || undefined },
    { key: 'lower_price', label: '价格下限', value: config.lower_price === null ? '不限' : amountText(config.lower_price, '元', 3) },
    { key: 'upper_price', label: '价格上限', value: config.upper_price === null ? '不限' : amountText(config.upper_price, '元', 3) },
  )
  if (shareBased) {
    rows.push(
      { key: 'buy_shares', label: '每次买入', value: amountText(config.buy_shares, unit, 0) },
      { key: 'sell_shares', label: '每次卖出', value: amountText(config.sell_shares, unit, 0) },
      { key: 'initial_shares', label: '初始持有', value: amountText(config.initial_shares, unit, 0) },
      { key: 'min_shares', label: '最低持有', value: amountText(config.min_shares, unit, 0) },
      { key: 'max_shares', label: '最高持有', value: config.max_shares === null ? '不限' : amountText(config.max_shares, unit, 0) },
      ...(version !== 3 ? [{ key: 'reserve_cash', label: '网格预留现金', value: amountText(finite(funding.reserve_cash) ? funding.reserve_cash : config.reserve_cash, '元') }] : []),
      { key: 'initial_equity', label: '网格起始资产', value: amountText(funding.initial_equity, '元') },
      ...(version === 3 ? [
        { key: 'peak_invested', label: '峰值净投入', value: amountText(funding.peak_invested, '元') },
        { key: 'topups', label: '累计外部补款', value: amountText(funding.topups, '元') },
      ] : []),
    )
  } else {
    rows.push(
      { key: 'buy_quantity', label: '每次买入', value: quantityText(config.buy_quantity, unit) },
      { key: 'sell_quantity', label: '每次卖出', value: quantityText(config.sell_quantity, unit) },
      { key: 'initial_position', label: '初始仓位', value: percentText(config.initial_position) },
      { key: 'min_position', label: '最低仓位', value: percentText(config.min_position) },
      { key: 'max_position', label: '最高仓位', value: percentText(config.max_position) },
      { key: 'initial_cash', label: '初始资金', value: amountText(config.initial_cash, '元') },
    )
  }
  return rows
}

export const resultCurveStyles = [
  { key: 'grid', name: '网格策略', color: '#2864dc' },
  { key: 'same_account_passive', name: '同账户被动持有', color: '#13957b' },
  { key: 'dca', name: '定投', color: '#e29136' },
  { key: 'buy_hold', name: '买入并持有', color: '#8795aa' },
] as const

export type ResultCurveValues = { returnRate: number | null; drawdown: number | null; asset: number | null; invested: number | null }
export type ResultCurveRow = { timestamp: string; time: number; values: Record<string, ResultCurveValues>; position: number | null }
export type ResultCurveModel = {
  strategies: Array<{ key: string; name: string; color: string }>
  rows: ResultCurveRow[]
  positionMode: 'ratio' | 'shares'
  shareUnit: string
}

function pointTime(timestamp: unknown) {
  return typeof timestamp === 'string' && timestamp.trim() ? Date.parse(timestamp.trim().replace(' ', 'T')) : Number.NaN
}

function investmentAmount(point: EquityPoint, strategy: StrategyResult, result: BacktestResult): number | null {
  if (finite(point.total_invested)) return Math.max(0, point.total_invested)
  if (result.accounting_basis === 'independent_contributions_v1' || strategy.return_basis === 'cost') {
    return finite(point.normalized_equity) && point.normalized_equity > 0 && finite(point.equity) ? point.equity / point.normalized_equity : null
  }
  const base = result.funding?.initial_equity ?? result.config?.initial_cash
  return finite(base) ? base : null
}

/** A shared timestamp union prevents one strategy's nearest observation from being shown as another date. */
export function buildResultCurveModel(result: BacktestResult): ResultCurveModel {
  const available = Object.entries(result.strategies ?? {}).filter(([, strategy]) => strategy != null)
  const known = resultCurveStyles.filter((style) => available.some(([key]) => style.key === key))
  const extra = available.filter(([key]) => !resultCurveStyles.some((style) => style.key === key))
  const strategies = [
    ...known.map((style) => ({ ...style })),
    ...extra.map(([key]) => ({ key, name: key, color: '#69758b' })),
  ].map((style) => ({ ...style, name: result.strategies[style.key]?.return_basis === 'cost' ? '定投（成本收益）' : style.name }))
  const rows = new Map<number, ResultCurveRow>()
  const gridPoints = result.strategies?.grid?.equity ?? []
  // A partial ratio history stays a ratio history with gaps; older share-only results keep their units.
  const positionMode = gridPoints.some((point) => finite(point.position_ratio)) ? 'ratio' : 'shares'
  for (const { key } of strategies) {
    const strategy = result.strategies[key]
    const points = strategy.equity ?? []
    const drawdowns = strategy.return_basis === 'cost' || key === 'dca' ? costBasisDrawdowns(points, strategy, result) : null
    points.forEach((point, index) => {
      const time = pointTime(point.timestamp)
      if (!Number.isFinite(time)) return
      const row = rows.get(time) ?? { timestamp: point.timestamp, time, values: {}, position: null }
      const drawdown = drawdowns ? drawdowns[index][1] : point.drawdown
      row.values[key] = {
        returnRate: assetInvestmentReturn(point, strategy, result),
        drawdown: finite(drawdown) ? drawdown : null,
        asset: finite(point.equity) ? point.equity : null,
        invested: investmentAmount(point, strategy, result),
      }
      if (key === 'grid') {
        const position = positionMode === 'ratio' ? point.position_ratio : point.position
        row.position = finite(position) ? position : null
      }
      rows.set(time, row)
    })
  }
  return { strategies, rows: [...rows.values()].sort((a, b) => a.time - b.time), positionMode, shareUnit: resultShareUnit(result.symbol) }
}

export function resultCurveRowAt(model: ResultCurveModel, timestamp: unknown): ResultCurveRow | undefined {
  const time = typeof timestamp === 'number' ? timestamp : pointTime(timestamp)
  return Number.isFinite(time) ? model.rows.find((row) => row.time === time) : undefined
}
