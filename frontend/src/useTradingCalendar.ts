import { useEffect, useState } from 'react'
import { get } from './api'
import type { Frequency, TradingCalendar } from './types'
import { hasLocalCalendarCoverage, isCalendarStateCurrent } from './calendarAvailability'

export interface TradingCalendarState {
  key: string
  calendar: TradingCalendar | null
  loading: boolean
  ready: boolean
  /** Local market data has a known covered start/end; independent of slider calendar support. */
  available: boolean
  error: string
}

/** Read the symbol/frequency calendar with stale-response protection. */
export function useTradingCalendar(symbol: string, frequency: Frequency): TradingCalendarState {
  const key = `${symbol}|${frequency}`
  const [state, setState] = useState<TradingCalendarState>({ key: '', calendar: null, loading: false, ready: false, available: false, error: '' })
  useEffect(() => {
    let active = true
    setState({ key, calendar: null, loading: Boolean(symbol), ready: false, available: false, error: '' })
    if (!symbol) return () => { active = false }
    get<TradingCalendar>(`/data/calendar/${encodeURIComponent(symbol)}?frequency=${encodeURIComponent(frequency)}`).then(calendar => {
      if (active) setState({ key, calendar, loading: false, ready: true, available: hasLocalCalendarCoverage(calendar), error: '' })
    }).catch(error => {
      if (active) setState({ key, calendar: null, loading: false, ready: true, available: false, error: error instanceof Error ? error.message : String(error) })
    })
    return () => { active = false }
  }, [key, symbol, frequency])
  // Effects run after paint. Mask the prior request synchronously in the render
  // where symbol/frequency changed so it can never initialize the new config.
  return isCalendarStateCurrent(state, symbol, frequency) ? state : { key, calendar: null, loading: Boolean(symbol), ready: false, available: false, error: '' }
}
