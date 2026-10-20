export type LandscapeParams = Record<string, unknown>

export type LandscapePoint = {
  candidate_id: number
  rank?: number
  params: LandscapeParams
  is_center: boolean
  is_baseline: boolean
  is_focus: boolean
  status: string
  excess_return: number | null
  grid_return: number | null
  baseline_return: number | null
  drawdown: number | null
  trade_count: number | null
  peak_invested: number | null
  neighbor_candidate_ids: number[]
  neighbor_of_candidate_ids: number[]
  robustness_status: string | null
  neighbor_positive_ratio: number | null
  neighbor_q25_excess: number | null
  island_suspect: boolean
  folds: LandscapeParams[]
}

export type LandscapeResponse = {
  parameters: string[]
  points: LandscapePoint[]
  total: number
  bounds: { min: number; max: number } | null
  focus: LandscapePoint | null
  focus_neighborhood: LandscapeNeighborhood | null
  top_candidates: LandscapePoint[]
  slices: Record<string, LandscapeSlice>
  sampled: boolean
  sample_strategy: string
  center_count: number
}

export type LandscapeNeighborhood = {
  available: boolean
  dimensions?: string[]
  expected_count: number
  evaluated_count: number
  missing_count?: number
  positive_count?: number
  positive_ratio?: number | null
  q25_excess?: number | null
  min_excess?: number | null
  max_excess?: number | null
  all_positive?: boolean
}

export type LandscapeSlice = {
  x: string
  y: string
  points: LandscapePoint[]
  total: number
  sampled: boolean
}

export const landscapeLabels: Record<string, string> = {
  buy_step: '买入间距', sell_step: '卖出间距', grid_step: '旧版网格间距',
  min_shares: '最低持股', max_shares: '最高持股', buy_shares: '每次买入', sell_shares: '每次卖出',
  initial_shares: '初始持股',
}

export function distinctParameterValues(points: LandscapePoint[], name: string) {
  return new Set(points.map(point => point.params[name]).filter(value => value != null).map(String))
}

export function chooseLandscapeAxes(
  parameters: string[],
  points: LandscapePoint[],
  currentX: string,
  currentY: string,
): [string, string] {
  const available = parameters.filter(Boolean)
  if (!available.length) return ['', '']
  if (currentX && currentY && currentX !== currentY && available.includes(currentX) && available.includes(currentY)) {
    return [currentX, currentY]
  }
  if (available.includes('buy_step') && available.includes('sell_step')) return ['buy_step', 'sell_step']
  const varying = available.filter(name => distinctParameterValues(points, name).size > 1)
  const preferred = [
    ['grid_step', 'sell_shares'],
    ['grid_step', 'buy_shares'],
    ['min_shares', 'buy_shares'],
    ['sell_shares', 'buy_shares'],
  ]
  for (const pair of preferred) {
    if (pair.every(name => varying.includes(name))) return [pair[0], pair[1]]
  }
  if (varying.length >= 2) return [varying[0], varying[1]]
  if (available.length >= 2) return [available[0], available[1]]
  return [available[0], '']
}

export function landscapeValue(point: LandscapePoint, metric: 'excess' | 'grid') {
  return metric === 'grid' ? point.grid_return : point.excess_return
}

export function landscapeSliceKey(x: string, y: string) {
  return [x, y].filter(Boolean).sort().join('|')
}

export function formatLandscapeNumber(value: number, digits = 2) {
  return value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
}

export function formatLandscapeParameter(name: string, value: unknown) {
  if (value == null) return '不限'
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  if (name === 'grid_step' || name === 'buy_step' || name === 'sell_step') return `${formatLandscapeNumber(numeric * 100, 2)}%`
  return formatLandscapeNumber(numeric, 0)
}

export function formatLandscapeAxis(name: string, value: unknown) {
  return formatLandscapeParameter(name, value)
}
