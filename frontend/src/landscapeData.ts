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
  /** Peak-principal drawdown of the candidate. */
  drawdown: number | null
  trade_count: number | null
  peak_invested: number | null
  neighbor_candidate_ids: number[]
  neighbor_of_candidate_ids: number[]
  robustness_status: string | null
  neighbor_positive_ratio: number | null
  neighbor_q25_excess: number | null
  j_score?: number | null
  neighborhood_mean?: number | null
  downside_risk_tau?: number | null
  pos_penalty?: number | null
  neighborhood_size?: number | null
  neighborhood_support_size?: number | null
  pos_win_rate?: number | null
  safe_win_rate?: number | null
  score_complete?: boolean | null
  score_missing_count?: number | null
  constraint_eligible?: boolean | null
  constraint_reasons?: string[]
  actual_min_pos_ratio?: number | null
  actual_max_pos_ratio?: number | null
  neighborhood_max_eta?: number | null
  floor_episode_count?: number | null
  floor_episode_grids?: number[]
  floor_penalty?: number | null
  floor_penalty_basis?: string
  stockout_count?: number | null
  tplus1_block_count?: number | null
  local_peak_verified?: boolean
  platform_size?: number | null
  stress?: { status?: string; mean_excess_return?: number | null }
  island_suspect: boolean
}

/** The decision branch a completion revision covers. */
export type LandscapeRevisionScope = {
  revision?: number
  decision?: string | null
  decision_index?: number
}

export type LandscapeCompletion = {
  status?: 'available' | 'complete' | string
  endpoint?: string
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
  j_focus?: LandscapePoint | null
  j_ranking?: LandscapePoint[]
  j_candidate_count?: number
  recommendation_count?: number
  decision_as_of?: string | null
  training_start?: string | null
  training_end?: string | null
  coverage_status?: string | null
  /** The completion revision this landscape was read from. */
  revision?: number | null
  /** The branch the applied completion revision covers. */
  revision_scope?: LandscapeRevisionScope | null
  completion?: LandscapeCompletion | null
  parameter_space?: ParameterSpace | null
  axis_values?: Record<string, unknown[]>
  axis_ticks?: Record<string, number[]>
  axis_bounds?: Record<string, { min: number; max: number }>
  fixed_parameters?: Record<string, unknown>
  fixed_axis?: string | null
  fixed_tick?: number | null
  returned_count?: number
  metric_bounds?: Record<'excess' | 'grid', { min: number; max: number }>
}

export type ParameterSpace = {
  version: 1
  parameter_links: { spacing: boolean; shares: boolean }
  independent_axes: Array<{ id: string; members: string[] }>
  active_axes: string[]
}

export const canonicalParameter = (name: string, space?: ParameterSpace | null) =>
  space?.independent_axes.find(axis => axis.members.includes(name))?.id ?? name

export function parameterSpaceAxes(parameters: string[], space?: ParameterSpace | null) {
  if (space && Array.isArray(space.active_axes)) return space.active_axes.filter(axis => parameters.includes(axis))
  return [...new Set(parameters.map(name => canonicalParameter(name, space)))]
}

export function canonicalLandscapePoint(point: LandscapePoint, space?: ParameterSpace | null): LandscapePoint {
  const params = { ...point.params }
  for (const axis of space?.independent_axes ?? []) {
    const found = axis.members.find(member => params[member] != null)
    if (found) for (const member of axis.members) params[member] = params[found]
  }
  return { ...point, params }
}

export type LandscapeNeighborhood = {
  available: boolean
  dimensions?: string[]
  expected_count: number
  evaluated_count: number
  missing_count?: number
  metric_count?: number
  metric_missing_count?: number
  metric_available?: boolean
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
  buy_step: '买入间距', sell_step: '卖出间距',
  buy_shares: '每次买入', sell_shares: '每次卖出',
}

export function landscapeParameterLabel(name: string, space?: ParameterSpace | null) {
  if (space?.parameter_links.spacing && (name === 'buy_step' || name === 'sell_step')) return '统一间距'
  if (space?.parameter_links.shares && (name === 'buy_shares' || name === 'sell_shares')) return '统一份额'
  return landscapeLabels[name] ?? name
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
    ['sell_step', 'buy_shares'],
    ['buy_step', 'buy_shares'],
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

/**
 * A completion revision only covers one decision branch, so a landscape read
 * may only carry the revision when the selected decision is that branch.
 */
export function revisionMatchesScope(scope: LandscapeRevisionScope | null | undefined, decision: string, decisionIndex: number) {
  if (!scope) return false
  return scope.decision === decision && Number(scope.decision_index) === Number(decisionIndex)
}

export function formatLandscapeNumber(value: number, digits = 2) {
  return value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
}

export function formatLandscapeParameter(name: string, value: unknown) {
  if (value == null) return '不限'
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  if (name === 'buy_step' || name === 'sell_step') return `${formatLandscapeNumber(numeric * 100, 2)}%`
  return formatLandscapeNumber(numeric, 0)
}

export function formatLandscapeAxis(name: string, value: unknown) {
  return formatLandscapeParameter(name, value)
}
