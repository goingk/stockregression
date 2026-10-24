export type AdjustmentPageIdentity = {
  version?: string | number
  adjustment?: 'none' | 'qfq'
  adjustment_anchor?: string | null
  adjustment_version?: string | null
}

export function adjustmentPageChanged(current: AdjustmentPageIdentity, incoming: AdjustmentPageIdentity): boolean {
  return current.version !== incoming.version
    || current.adjustment !== incoming.adjustment
    || current.adjustment_anchor !== incoming.adjustment_anchor
    || current.adjustment_version !== incoming.adjustment_version
}

export function adjustmentPageParams(adjustment: 'none' | 'qfq', anchor?: string | null): string {
  const params = new URLSearchParams({ adjustment })
  if (adjustment === 'qfq' && anchor) params.set('adjustment_anchor', anchor)
  return params.toString()
}

export function appendMarketPage<T extends AdjustmentPageIdentity & { bars: unknown[] }>(
  current: T,
  earlier: T,
): T | null {
  if (adjustmentPageChanged(current, earlier)) return null
  return { ...earlier, bars: [...earlier.bars, ...current.bars] }
}
