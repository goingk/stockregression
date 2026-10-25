export interface EquityMathPoint {
  timestamp: string
  equity: number
  normalized_equity?: number | null
  total_invested?: number | null
  /** v5 peak-principal accounting: 1 + cumulative profit / current peak principal. */
  principal_curve?: number | null
  /** v5 peak-principal accounting: drawdown of the principal profit curve. */
  principal_drawdown?: number | null
}

export interface EquityMathStrategy {
  return_basis?: string
}

export interface EquityMathResult {
  accounting_basis?: string
  funding?: { initial_equity?: number | null }
  config?: { initial_cash?: number | null }
}

const positiveFinite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

export function assetInvestmentRatio(
  point: EquityMathPoint,
  strategy: EquityMathStrategy,
  result: EquityMathResult,
  requireInvestmentData = false,
): number | null {
  const invested = typeof point.total_invested === 'number' && Number.isFinite(point.total_invested)
    ? point.total_invested
    : null

  // A peak-principal series is measured against the capital contributed by
  // that point.  Derive it here as well so saved v5 results that used the old
  // whole-period maximum display the corrected curve without a rerun.
  if (strategy.return_basis === 'peak_principal' && invested != null) {
    if (invested <= 0) return null
    if (!Number.isFinite(point.equity)) return null
    // External principal is cumulative and never withdrawn, so the value at
    // this point is also the high-water mark reached by this point.
    const ratio = 1 + (point.equity - invested) / invested
    return Number.isFinite(ratio) ? ratio : null
  }

  // Other accounting bases state the normalized ratio directly.
  if (typeof point.principal_curve === 'number' && Number.isFinite(point.principal_curve)) {
    return point.principal_curve
  }

  if (invested != null && invested <= 0) return null

  if (typeof point.normalized_equity === 'number' && Number.isFinite(point.normalized_equity)) {
    return point.normalized_equity
  }
  if (invested != null) {
    if (!Number.isFinite(point.equity)) return null
    const ratio = point.equity / invested
    return Number.isFinite(ratio) ? ratio : null
  }
  if (requireInvestmentData || result.accounting_basis === 'independent_contributions_v1' || strategy.return_basis === 'cost') {
    return null
  }

  // Older fixed-capital results have no contribution fields; use their starting principal when available.
  const base = positiveFinite(result.funding?.initial_equity)
    ? result.funding!.initial_equity!
    : positiveFinite(result.config?.initial_cash) ? result.config!.initial_cash! : null
  if (base == null || !Number.isFinite(point.equity)) return null
  const ratio = point.equity / base
  return Number.isFinite(ratio) ? ratio : null
}

export function assetInvestmentReturn(
  point: EquityMathPoint,
  strategy: EquityMathStrategy,
  result: EquityMathResult,
): number | null {
  const ratio = assetInvestmentRatio(point, strategy, result)
  return ratio == null ? null : ratio - 1
}

export function costBasisDrawdowns(
  points: EquityMathPoint[],
  strategy: EquityMathStrategy,
  result: EquityMathResult,
): Array<[string, number | null]> {
  let peakRatio = 1
  return points.map((point): [string, number | null] => {
    // v5 reports the principal profit drawdown itself; older results keep the peak-based fallback.
    const reported = typeof point.principal_drawdown === 'number' && Number.isFinite(point.principal_drawdown)
      ? point.principal_drawdown
      : null
    const ratio = assetInvestmentRatio(point, strategy, result, true)
    if (ratio == null || !Number.isFinite(ratio) || ratio < 0) return [point.timestamp, reported]
    peakRatio = Math.max(peakRatio, ratio)
    if (reported != null) return [point.timestamp, reported]
    return [point.timestamp, (peakRatio - ratio) / peakRatio]
  })
}
