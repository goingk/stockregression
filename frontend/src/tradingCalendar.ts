/** Number of trading dates inclusively contained in [start, end], found in O(log n). */
export function tradingDayCount(dates: readonly string[], start: string, end: string): number {
  const upperBound = (target: string) => {
    let low = 0, high = dates.length
    while (low < high) {
      const mid = (low + high) >>> 1
      if (dates[mid] <= target) low = mid + 1
      else high = mid
    }
    return low
  }
  return Math.max(0, upperBound(end) - upperBound(previousCalendarDay(start)))
}

/** Index of the last trading date on or before date, or -1 when none exists. */
export function tradingDateIndexAtOrBefore(dates: readonly string[], date: string): number {
  let low = 0, high = dates.length
  while (low < high) {
    const mid = (low + high) >>> 1
    if (dates[mid] <= date) low = mid + 1
    else high = mid
  }
  return low - 1
}

export function trailingTradingRange(dates: readonly string[], end: string, count: number): { start: string; endIndex: number } | null {
  const endIndex = tradingDateIndexAtOrBefore(dates, end)
  if (endIndex < 0 || !Number.isFinite(count)) return null
  return { start: dates[Math.max(0, endIndex - Math.max(1, Math.trunc(count)) + 1)], endIndex }
}

/** Default window: subtract calendar years from the covered end, then snap forward to a trading date. */
export function defaultTradingRange(dates: readonly string[], lastDate: string, years = 1): { start: string; end: string; count: number } | null {
  const endIndex = tradingDateIndexAtOrBefore(dates, lastDate)
  if (endIndex < 0) return null
  const target = calendarYearsBefore(dates[endIndex], years)
  if (!target) return null
  let low = 0, high = endIndex + 1
  while (low < high) {
    const mid = (low + high) >>> 1
    if (dates[mid] < target) low = mid + 1
    else high = mid
  }
  const startIndex = Math.min(low, endIndex)
  return { start: dates[startIndex], end: dates[endIndex], count: endIndex - startIndex + 1 }
}

export function calendarYearsBefore(value: string, years = 1): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match || !Number.isSafeInteger(years) || years < 0) return ''
  const year = Number(match[1]) - years
  const month = Number(match[2])
  const day = Number(match[3])
  if (year < 1 || month < 1 || month > 12 || day < 1) return ''
  const clamped = Math.min(day, new Date(Date.UTC(year, month, 0)).getUTCDate())
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(clamped).padStart(2, '0')}`
}

/** Fill only a missing end from local coverage; preserve the causal start exactly. */
export function resolveMissingEndDate(
  start: string,
  end: string | null | undefined,
  datesExplicit: boolean,
  lastDate: string | null | undefined,
): { start: string; end: string } | null {
  if (end || datesExplicit || !lastDate || lastDate < start) return null
  return { start, end: lastDate }
}

export function shouldInitializeDefaultWindow(
  firstResolution: boolean,
  datesExplicit: boolean,
  autoEnd: boolean,
  end: string | null | undefined,
  lastDate: string | null | undefined,
): boolean {
  return firstResolution && !datesExplicit && autoEnd && Boolean(end) && Boolean(lastDate)
}

function previousCalendarDay(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return ''
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) - 1))
  return Number.isFinite(date.getTime()) ? date.toISOString().slice(0, 10) : ''
}
