import type { BacktestConfig } from './types'
import type { SearchRange } from './workbenchDefaults'

/** Apply link state without changing the canonical four research fields. */
export function setParameterLink(config: BacktestConfig, axis: keyof BacktestConfig['parameter_links'], linked: boolean): BacktestConfig {
  const parameter_links = { ...(config.parameter_links ?? { spacing: false, shares: false }), [axis]: linked }
  if (!linked) return { ...config, parameter_links }
  return axis === 'spacing'
    ? { ...config, parameter_links, sell_step: config.buy_step }
    : { ...config, parameter_links, sell_shares: config.buy_shares }
}

/** Mirror the buy range into the sell range when research spacing is linked. */
export function linkResearchRanges<T extends Record<string, SearchRange>>(ranges: T, links: BacktestConfig['parameter_links']): T {
  return {
    ...ranges,
    ...(links.spacing ? { sell_step: { ...ranges.buy_step } } : {}),
    ...(links.shares ? { sell_shares: { ...ranges.buy_shares } } : {}),
  }
}

export function syncLinkedResearchRange<T extends Record<string, SearchRange>>(
  ranges: T,
  field: 'buy_step' | 'sell_step' | 'buy_shares' | 'sell_shares',
  next: SearchRange,
  links: BacktestConfig['parameter_links'],
): T {
  const result: Record<string, SearchRange> = { ...ranges, [field]: next }
  if (links.spacing && field === 'buy_step') result.sell_step = { ...next }
  if (links.spacing && field === 'sell_step') result.buy_step = { ...next }
  if (links.shares && field === 'buy_shares') result.sell_shares = { ...next }
  if (links.shares && field === 'sell_shares') result.buy_shares = { ...next }
  return result as T
}
