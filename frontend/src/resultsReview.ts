/**
 * Pure helpers behind the results page.
 *
 * Two independent concerns live here so they can be unit tested without React:
 *  - the year / month contribution rows rebuilt from the peak-principal equity curve,
 *  - the K-line bar windows and the trade-to-bar mapping used by the candle chart.
 */

export type PeriodUnit = 'year' | 'month'

export interface PeriodEquityPoint {
  timestamp: string
  /** v5 peak-principal ratio: 1 + cumulative profit / peak principal. */
  principal_curve?: number | null
  /** Legacy unit net asset value, only used when the peak-principal curve is absent. */
  normalized_equity?: number | null
}

export interface PeriodReturnStrategy {
  key: string
  equity: PeriodEquityPoint[]
}

export interface PeriodReturnRow {
  period: string
  /** Strategy key -> additive period contribution in the peak-principal caliber. */
  values: Record<string, number | null>
  /** Strategy keys whose opening baseline is unknown, so their row cannot be computed. */
  uncomputable: string[]
}

export interface PeriodReturnOptions {
  /** Opening net asset value of the first period; null when the result has no known baseline. */
  baseline: number | null
}

export interface ResultBasisInfo {
  accounting_basis?: string | null
  result_schema_version?: number | null
  schema_version?: number | null
  config?: { schema_version?: number | null } | null
}

/** The peak-principal caliber is used by every complete v3 (share account) run. */
export function hasPeakPrincipalBasis(result: ResultBasisInfo | null | undefined): boolean {
  if (!result) return false
  if (result.accounting_basis) return result.accounting_basis === 'peak_principal_v2'
  return result.result_schema_version === 2 || result.schema_version === 3 || result.config?.schema_version === 3
}

/**
 * A complete v3 run starts from a 1.0 peak-principal curve, so the first period may use it.
 * Historical results without that anchor report no baseline and stay uncomputed.
 */
export function periodReturnBaseline(result: ResultBasisInfo | null | undefined): number | null {
  return hasPeakPrincipalBasis(result) ? 1 : null
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

/** The peak-principal contribution caliber wins; legacy unit net asset values are the fallback. */
export function periodNav(point: PeriodEquityPoint): number | null {
  if (finite(point.principal_curve)) return point.principal_curve
  if (finite(point.normalized_equity)) return point.normalized_equity
  return null
}

/** Naive (timezone free) stamp: the calendar day plus the minutes since midnight. */
export interface NaiveStamp {
  day: string
  minute: number
  value: number
}

const STAMP_PATTERN = /^(\d{4}-\d{2}-\d{2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?/

export function naiveStamp(timestamp: unknown): NaiveStamp | null {
  const text = String(timestamp ?? '').trim()
  const match = STAMP_PATTERN.exec(text)
  if (!match) return null
  const day = match[1]
  const hour = Number(match[2] ?? '0')
  const minute = Number(match[3] ?? '0')
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null
  const clock = hour * 60 + minute
  const anchor = Date.parse(day + 'T00:00:00Z')
  if (!Number.isFinite(anchor)) return null
  return { day, minute: clock, value: anchor + clock * 60000 }
}

function periodKey(timestamp: unknown, unit: PeriodUnit): string {
  const day = String(timestamp ?? '').slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) return ''
  return unit === 'year' ? day.slice(0, 4) : day.slice(0, 7)
}

function compareStamps(a: string, b: string): number {
  const left = naiveStamp(a)
  const right = naiveStamp(b)
  if (left && right) return left.value - right.value
  if (left) return -1
  if (right) return 1
  return String(a).localeCompare(String(b))
}

/**
 * Rebuild the year / month contribution rows from the equity curves.
 *
 * Points are sorted by time, every period opens at the previous period's closing net asset
 * value, and the first period opens at the run baseline (1.0 for a complete v3 run). The
 * contributions are additive in the peak-principal caliber, so the periods of one strategy
 * add up to the whole segment.
 */
export function periodReturns(
  strategies: PeriodReturnStrategy[],
  unit: PeriodUnit,
  options: PeriodReturnOptions,
): PeriodReturnRow[] {
  const rows = new Map<string, PeriodReturnRow>()
  const order: string[] = []
  const rowFor = (period: string) => {
    let row = rows.get(period)
    if (!row) {
      row = { period, values: {}, uncomputable: [] }
      rows.set(period, row)
      order.push(period)
    }
    return row
  }

  // The opening of a period is the previous period's close of the same strategy.
  const closes = new Map<string, Map<string, number>>()
  for (const strategy of strategies) {
    const grouped = new Map<string, number>()
    const points = (strategy.equity ?? [])
      .map((point) => ({ point, nav: periodNav(point), period: periodKey(point?.timestamp, unit) }))
      .filter((item): item is { point: PeriodEquityPoint; nav: number; period: string } => item.nav != null && item.period !== '')
      .sort((a, b) => compareStamps(a.point.timestamp, b.point.timestamp))
    for (const item of points) grouped.set(item.period, item.nav)
    closes.set(strategy.key, grouped)
  }

  const periods = new Set<string>()
  for (const grouped of closes.values()) for (const period of grouped.keys()) periods.add(period)
  const firstPeriod = [...periods].sort()[0] ?? ''

  for (const strategy of strategies) {
    const grouped = closes.get(strategy.key) ?? new Map<string, number>()
    const strategyPeriods = [...grouped.keys()].sort()
    let previousClose: number | null = null
    for (const period of strategyPeriods) {
      const close = grouped.get(period)!
      const row = rowFor(period)
      let opening: number | null
      if (previousClose != null) opening = previousClose
      else if (period === firstPeriod) opening = options.baseline
      else opening = null
      if (opening == null) {
        row.values[strategy.key] = null
        if (!row.uncomputable.includes(strategy.key)) row.uncomputable.push(strategy.key)
      } else {
        row.values[strategy.key] = close - opening
      }
      previousClose = close
    }
  }

  return order.sort().map((period) => rows.get(period)!)
}

/* -------------------------------------------------------------------------- */
/* K-line bars and trade mapping                                              */
/* -------------------------------------------------------------------------- */

export const MINUTE_FREQUENCY_MINUTES: Record<string, number> = { '1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60 }

/** Backend session boundaries: the morning opens at 09:30 and closes at 11:30, the afternoon opens at 13:00, the closing auction runs 14:57 to 15:00. */
export const SESSION_MORNING_OPEN = 9 * 60 + 30
export const SESSION_MORNING_CLOSE = 11 * 60 + 30
export const SESSION_AFTERNOON_OPEN = 13 * 60
export const SESSION_AUCTION_START = 14 * 60 + 57
export const SESSION_CLOSE = 15 * 60

const MINUTE_MS = 60000
const DAY_MS = 24 * 60 * MINUTE_MS
const DAILY_SESSION_MS = (SESSION_CLOSE - SESSION_MORNING_OPEN) * MINUTE_MS

function minuteSize(frequency: string): number | null {
  const size = MINUTE_FREQUENCY_MINUTES[String(frequency ?? '').toLowerCase()]
  return size ?? null
}

function sessionCloseMinute(openMinute: number): number {
  return openMinute < SESSION_AFTERNOON_OPEN ? SESSION_MORNING_CLOSE : SESSION_CLOSE
}

/**
 * Opening time of a bar, mirroring the backend: the bar label minus its own length, clamped
 * to the session open (09:30 / 13:00). Daily bars open at 09:30 of their day.
 */
export function barOpenStamp(timestamp: unknown, frequency: string): number | null {
  const stamp = naiveStamp(timestamp)
  if (!stamp) return null
  const anchor = Date.parse(stamp.day + 'T00:00:00Z')
  if (!Number.isFinite(anchor)) return null
  const size = minuteSize(frequency)
  if (size == null) return anchor + SESSION_MORNING_OPEN * MINUTE_MS
  const candidate = stamp.minute - size
  const sessionOpen = stamp.minute <= SESSION_MORNING_CLOSE ? SESSION_MORNING_OPEN : SESSION_AFTERNOON_OPEN
  const open = candidate < sessionOpen ? sessionOpen : candidate
  return anchor + open * MINUTE_MS
}

export interface BarWindow {
  /** Index of the bar in the source array. */
  index: number
  /** Opening time of the bar. */
  open: number
  /** Exclusive closing time of the bar. */
  end: number
  /** Raw bar timestamp. */
  label: string
}

/**
 * Windows of the displayed bars. A bar owns the span from its own open to the next bar's open,
 * but never crosses its own length or the session close, so trades inside a data gap stay
 * unmapped instead of being clamped onto a neighbouring bar.
 */
export function barWindows(bars: Array<Record<string, unknown> | null | undefined>, frequency: string): BarWindow[] {
  const size = minuteSize(frequency)
  const span = size == null ? DAILY_SESSION_MS : size * MINUTE_MS
  const entries = bars
    .map((bar, index) => ({ index, label: String(bar?.timestamp ?? ''), open: barOpenStamp(bar?.timestamp, frequency) }))
    .filter((entry): entry is { index: number; label: string; open: number } => entry.open != null)
    .sort((a, b) => a.open - b.open)
  return entries.map((entry, position) => {
    const next = entries[position + 1]
    const dayStart = Math.floor(entry.open / DAY_MS) * DAY_MS
    const openMinute = Math.round((entry.open - dayStart) / MINUTE_MS)
    // Daily bars cover the whole trading session of their day, minute bars stop at the close.
    const closeMinute = size == null ? SESSION_CLOSE : sessionCloseMinute(openMinute)
    const sessionClose = dayStart + closeMinute * MINUTE_MS
    const end = Math.min(entry.open + span, sessionClose, next ? next.open : Number.POSITIVE_INFINITY)
    return { index: entry.index, open: entry.open, end, label: entry.label }
  })
}

export interface TradeMarker {
  /** Index of the bar the trade belongs to. */
  barIndex: number
  /** Exact number of trades behind this marker. */
  count: number
  buyCount: number
  sellCount: number
  /** Price used to anchor the marker on the price axis. */
  price: number
  side: 'buy' | 'sell'
  firstTime: number
  lastTime: number
}

export interface MappedTrade {
  barIndex: number
  time: number
  price: number
  side: 'buy' | 'sell'
  timestamp: string
}

export function tradeSide(side: unknown): 'buy' | 'sell' {
  const text = String(side ?? '').toLowerCase()
  return text.includes('buy') || text.includes('买') ? 'buy' : 'sell'
}

/**
 * Map trades onto the displayed bars. A trade belongs to the bar whose window contains it;
 * trades outside the loaded window or inside a data gap are dropped instead of being clamped
 * onto the first or last bar.
 */
export function mapTradesToBars(
  trades: Array<{ timestamp?: unknown; side?: unknown; price?: unknown }>,
  bars: Array<Record<string, unknown> | null | undefined>,
  frequency: string,
): MappedTrade[] {
  const windows = barWindows(bars, frequency)
  if (!windows.length) return []
  const mapped: MappedTrade[] = []
  for (const trade of trades ?? []) {
    const stamp = naiveStamp(trade?.timestamp)
    if (!stamp) continue
    const price = finite(trade?.price) ? trade.price : Number.NaN
    const window = findWindow(windows, stamp.value)
    if (!window) continue
    mapped.push({ barIndex: window.index, time: stamp.value, price, side: tradeSide(trade?.side), timestamp: String(trade?.timestamp ?? '') })
  }
  return mapped
}

/** Last bar window that contains the given stamp, searched by binary search. */
function findWindow(windows: BarWindow[], value: number): BarWindow | null {
  let low = 0
  let high = windows.length - 1
  let candidate: BarWindow | null = null
  while (low <= high) {
    const middle = (low + high) >>> 1
    const window = windows[middle]
    if (window.open <= value) {
      candidate = window
      low = middle + 1
    } else high = middle - 1
  }
  if (candidate && value < candidate.end) return candidate
  return null
}

/** Merge the trades of one bar into a single marker with an exact count. */
export function groupTradeMarkers(trades: MappedTrade[]): TradeMarker[] {
  const groups = new Map<number, TradeMarker>()
  for (const trade of trades) {
    const current = groups.get(trade.barIndex)
    if (!current) {
      groups.set(trade.barIndex, {
        barIndex: trade.barIndex,
        count: 1,
        buyCount: trade.side === 'buy' ? 1 : 0,
        sellCount: trade.side === 'sell' ? 1 : 0,
        price: trade.price,
        side: trade.side,
        firstTime: trade.time,
        lastTime: trade.time,
      })
      continue
    }
    current.count += 1
    if (trade.side === 'buy') current.buyCount += 1
    else current.sellCount += 1
    if (Number.isFinite(trade.price)) current.price = trade.price
    current.side = current.buyCount > current.sellCount ? 'buy' : current.sellCount > current.buyCount ? 'sell' : trade.side
    current.firstTime = Math.min(current.firstTime, trade.time)
    current.lastTime = Math.max(current.lastTime, trade.time)
  }
  return [...groups.values()].sort((a, b) => a.barIndex - b.barIndex)
}

/**
 * Limit the number of markers by merging neighbouring ones into buckets. The trade counts of
 * every bucket stay exact, only the number of drawn symbols shrinks.
 */
export function bucketTradeMarkers(markers: TradeMarker[], maxMarkers: number): TradeMarker[] {
  const limit = Math.max(1, Math.floor(maxMarkers))
  if (markers.length <= limit) return markers
  const size = Math.ceil(markers.length / limit)
  const buckets: TradeMarker[] = []
  for (let index = 0; index < markers.length; index += size) {
    const slice = markers.slice(index, index + size)
    const last = slice[slice.length - 1]
    const count = slice.reduce((total, marker) => total + marker.count, 0)
    const buyCount = slice.reduce((total, marker) => total + marker.buyCount, 0)
    const sellCount = slice.reduce((total, marker) => total + marker.sellCount, 0)
    buckets.push({
      barIndex: last.barIndex,
      count,
      buyCount,
      sellCount,
      price: last.price,
      side: buyCount > sellCount ? 'buy' : sellCount > buyCount ? 'sell' : last.side,
      firstTime: slice[0].firstTime,
      lastTime: last.lastTime,
    })
  }
  return buckets
}

/** Marker label used by the candle chart. */
export function markerLabel(marker: TradeMarker): string {
  if (marker.count <= 1) return marker.side === 'buy' ? '买' : '卖'
  return marker.count + '笔'
}

/* -------------------------------------------------------------------------- */
/* Result history pagination                                                  */
/* -------------------------------------------------------------------------- */

export const RESULT_PAGE_SIZE = 50

/** The results endpoint keeps returning a plain array, so paging is offset based. */
export function resultsPageQuery(offset: number, limit = RESULT_PAGE_SIZE, kind = 'backtest'): string {
  const safeOffset = Math.max(0, Math.floor(offset) || 0)
  const safeLimit = Math.max(1, Math.floor(limit) || RESULT_PAGE_SIZE)
  const safeKind = /^[a-z_]{1,64}$/.test(kind) ? kind : 'backtest'
  return '/results?kind=' + safeKind + '&offset=' + safeOffset + '&limit=' + safeLimit
}

/** Merge freshly loaded bars with the ones already displayed, keeping one row per timestamp. */
export function mergeBars(current: Array<Record<string, unknown>>, incoming: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  const merged = new Map<string, Record<string, unknown>>()
  for (const bar of [...current, ...incoming]) {
    const stamp = naiveStamp(bar?.timestamp)
    const key = stamp ? stamp.day + ' ' + String(stamp.minute).padStart(4, '0') : String(bar?.timestamp ?? '')
    merged.set(key, bar)
  }
  return [...merged.values()].sort((a, b) => compareStamps(String(a?.timestamp ?? ''), String(b?.timestamp ?? '')))
}

/** Earliest bar timestamp of a page, used as the exclusive upper bound of the next earlier page. */
export function earliestBarTimestamp(bars: Array<Record<string, unknown>>): string {
  let earliest: { value: number; label: string } | null = null
  for (const bar of bars ?? []) {
    const stamp = naiveStamp(bar?.timestamp)
    if (!stamp) continue
    if (!earliest || stamp.value < earliest.value) earliest = { value: stamp.value, label: String(bar?.timestamp ?? '') }
  }
  return earliest?.label ?? ''
}

/** Displayed range of the loaded bars. */
export function barTimestampRange(bars: Array<Record<string, unknown>>): { first: string; last: string; count: number } {
  let first: { value: number; label: string } | null = null
  let last: { value: number; label: string } | null = null
  for (const bar of bars ?? []) {
    const stamp = naiveStamp(bar?.timestamp)
    if (!stamp) continue
    if (!first || stamp.value < first.value) first = { value: stamp.value, label: String(bar?.timestamp ?? '') }
    if (!last || stamp.value > last.value) last = { value: stamp.value, label: String(bar?.timestamp ?? '') }
  }
  return { first: first?.label ?? '', last: last?.label ?? '', count: bars?.length ?? 0 }
}
