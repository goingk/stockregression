import type { Frequency, TradingCalendar } from './types'

/** Local market data coverage is independent from whether slider dates exist. */
export function hasLocalCalendarCoverage(calendar: Pick<TradingCalendar, 'first_date' | 'last_date'>): boolean {
  return Boolean(calendar.first_date && calendar.last_date)
}

export function isCalendarStateCurrent(state: { key: string }, symbol: string, frequency: Frequency): boolean {
  return state.key === `${symbol}|${frequency}`
}
