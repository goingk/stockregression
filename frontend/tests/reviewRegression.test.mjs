import assert from 'node:assert/strict'
import test from 'node:test'
import {
  barOpenStamp,
  barWindows,
  bucketTradeMarkers,
  earliestBarTimestamp,
  groupTradeMarkers,
  mapTradesToBars,
  mergeBars,
  naiveStamp,
  periodReturnBaseline,
  periodReturns,
  resultsPageQuery,
} from '../src/resultsReview.ts'

const point = (timestamp, principal_curve, values = {}) => ({ timestamp, principal_curve, ...values })
const close = (value, expected, message) => assert.ok(Math.abs(value - expected) < 1e-12, message + ': ' + value + ' != ' + expected)

test('monthly contributions chain on the previous close and sum to the whole segment', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      point('2024-01-31T15:00:00', 1.10),
      point('2024-02-01T15:00:00', 1.20),
      point('2024-02-29T15:00:00', 1.05),
    ] }],
    'month',
    { baseline: 1 },
  )

  assert.deepEqual(rows.map((row) => row.period), ['2024-01', '2024-02'])
  close(rows[0].values.grid, 0.10, 'january opens at the 1.0 baseline')
  close(rows[1].values.grid, -0.05, 'february opens at the january close')
  close(rows.reduce((total, row) => total + row.values.grid, 0), 0.05, 'the periods add up to the whole segment')
})

test('yearly contributions cross the new year boundary', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      point('2023-12-29T15:00:00', 1.20),
      point('2024-01-05T15:00:00', 1.10),
    ] }],
    'year',
    { baseline: 1 },
  )

  assert.deepEqual(rows.map((row) => row.period), ['2023', '2024'])
  close(rows[0].values.grid, 0.20, 'the first year opens at the baseline')
  close(rows[1].values.grid, -0.10, 'the second year opens at the first close')
  close(rows.reduce((total, row) => total + row.values.grid, 0), 0.10, 'the years add up to the whole segment')
})

test('a period with a single point is still calculated', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      point('2024-01-31T15:00:00', 1.10),
      point('2024-02-01T15:00:00', 1.25),
    ] }],
    'month',
    { baseline: 1 },
  )

  assert.equal(rows.length, 2)
  close(rows[1].values.grid, 0.15, 'the single february point closes against the january close')
})

test('a missing baseline leaves the first period uncomputed and the rest chained', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      point('2024-01-31T15:00:00', 1.10),
      point('2024-02-01T15:00:00', 1.20),
    ] }],
    'month',
    { baseline: null },
  )

  assert.equal(rows[0].values.grid, null)
  assert.deepEqual(rows[0].uncomputable, ['grid'])
  close(rows[1].values.grid, 0.10, 'february still opens at the january close')
})

test('the equity curve is sorted by time before the periods are built', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      point('2024-02-29T15:00:00', 1.05),
      point('2024-01-31T15:00:00', 1.10),
    ] }],
    'month',
    { baseline: 1 },
  )

  assert.deepEqual(rows.map((row) => row.period), ['2024-01', '2024-02'])
  close(rows[0].values.grid, 0.10, 'the earlier point opens the first period')
  close(rows[1].values.grid, -0.05, 'the later point closes the second period')
})

test('the peak principal caliber wins over the legacy unit net asset value', () => {
  const rows = periodReturns(
    [{ key: 'grid', equity: [
      { timestamp: '2024-01-31T15:00:00', principal_curve: 1.10, normalized_equity: 3 },
      { timestamp: '2024-02-01T15:00:00', principal_curve: 1.20, normalized_equity: 9 },
    ] }],
    'month',
    { baseline: 1 },
  )

  close(rows[0].values.grid, 0.10, 'the peak principal curve is used')
  close(rows[1].values.grid, 0.10, 'the peak principal curve is used')
})

test('a strategy that starts later has no opening baseline for its first period', () => {
  const rows = periodReturns(
    [
      { key: 'grid', equity: [point('2024-01-31T15:00:00', 1.10), point('2024-02-01T15:00:00', 1.20)] },
      { key: 'dca', equity: [point('2024-02-01T15:00:00', 1.05)] },
    ],
    'month',
    { baseline: null },
  )

  assert.deepEqual(rows.map((row) => row.period), ['2024-01', '2024-02'])
  assert.deepEqual(rows[0].uncomputable, ['grid'])
  assert.deepEqual(rows[1].uncomputable, ['dca'])
  close(rows[1].values.grid, 0.10, 'grid still chains from january')
})

test('only a complete peak principal run may use the 1.0 baseline', () => {
  assert.equal(periodReturnBaseline({ accounting_basis: 'peak_principal_v2', schema_version: 3 }), 1)
  assert.equal(periodReturnBaseline({ schema_version: 3, config: { schema_version: 3 } }), 1)
  assert.equal(periodReturnBaseline({ accounting_basis: 'independent_contributions_v1', schema_version: 2 }), null)
  assert.equal(periodReturnBaseline({}), null)
})

test('minute bar opens follow the backend session boundaries', () => {
  const open = (label, frequency) => barOpenStamp(label, frequency)
  const at = (text) => naiveStamp(text).value

  assert.equal(open('2024-01-02T09:31:00', '1m'), at('2024-01-02T09:30:00'))
  assert.equal(open('2024-01-02T11:30:00', '1m'), at('2024-01-02T11:29:00'))
  assert.equal(open('2024-01-02T13:01:00', '1m'), at('2024-01-02T13:00:00'))
  assert.equal(open('2024-01-02T15:00:00', '1m'), at('2024-01-02T14:59:00'))
  assert.equal(open('2024-01-02T09:35:00', '5m'), at('2024-01-02T09:30:00'))
  assert.equal(open('2024-01-02T11:30:00', '5m'), at('2024-01-02T11:25:00'))
  assert.equal(open('2024-01-02T10:30:00', '60m'), at('2024-01-02T09:30:00'))
  assert.equal(open('2024-01-02T11:30:00', '60m'), at('2024-01-02T10:30:00'))
  assert.equal(open('2024-01-02T15:00:00', '60m'), at('2024-01-02T14:00:00'))
  // A first bar of a session is clamped to the session open.
  assert.equal(open('2024-01-02T13:01:00', '5m'), at('2024-01-02T13:00:00'))
})

test('a trade is mapped onto the bar that opened at its own time', () => {
  const bars = [
    { timestamp: '2024-01-02T09:31:00' },
    { timestamp: '2024-01-02T09:32:00' },
    { timestamp: '2024-01-02T09:33:00' },
  ]
  const mapped = mapTradesToBars(
    [
      { timestamp: '2024-01-02T09:30:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T09:31:00', side: 'sell', price: 11 },
    ],
    bars,
    '1m',
  )

  assert.deepEqual(mapped.map((trade) => trade.barIndex), [0, 1])
  assert.deepEqual(mapped.map((trade) => trade.side), ['buy', 'sell'])
})

test('trades outside the window or inside a gap are dropped instead of clamped', () => {
  const bars = [
    { timestamp: '2024-01-02T09:31:00' },
    { timestamp: '2024-01-02T09:32:00' },
    { timestamp: '2024-01-02T09:35:00' },
  ]
  const mapped = mapTradesToBars(
    [
      { timestamp: '2024-01-02T09:00:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T09:33:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T09:40:00', side: 'buy', price: 10 },
    ],
    bars,
    '1m',
  )

  assert.deepEqual(mapped, [])
})

test('the lunch break between the sessions is not drawn on a bar', () => {
  const bars = [
    { timestamp: '2024-01-02T11:29:00' },
    { timestamp: '2024-01-02T11:30:00' },
    { timestamp: '2024-01-02T13:01:00' },
  ]
  const mapped = mapTradesToBars(
    [
      { timestamp: '2024-01-02T11:29:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T13:00:00', side: 'sell', price: 12 },
      { timestamp: '2024-01-02T12:00:00', side: 'buy', price: 11 },
    ],
    bars,
    '1m',
  )

  assert.deepEqual(mapped.map((trade) => trade.barIndex), [1, 2])
})

test('daily bars only accept trades of the same day', () => {
  const bars = [
    { timestamp: '2024-01-02' },
    { timestamp: '2024-01-03' },
  ]
  const mapped = mapTradesToBars(
    [
      { timestamp: '2024-01-02T09:30:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T14:00:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T20:00:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-03T09:30:00', side: 'sell', price: 10 },
    ],
    bars,
    '1d',
  )

  assert.deepEqual(mapped.map((trade) => trade.barIndex), [0, 0, 1])
})

test('bar windows never reach over a missing bar', () => {
  const windows = barWindows(
    [{ timestamp: '2024-01-02T09:31:00' }, { timestamp: '2024-01-02T09:32:00' }, { timestamp: '2024-01-02T09:35:00' }],
    '1m',
  )
  const at = (text) => naiveStamp(text).value

  assert.equal(windows[1].end, at('2024-01-02T09:32:00'))
  assert.equal(windows[2].open, at('2024-01-02T09:34:00'))
  assert.equal(windows[2].end, at('2024-01-02T09:35:00'))
})

test('markers of one bar are merged and bucketed with exact counts', () => {
  const bars = [
    { timestamp: '2024-01-02T09:31:00' },
    { timestamp: '2024-01-02T09:32:00' },
    { timestamp: '2024-01-02T09:33:00' },
  ]
  const mapped = mapTradesToBars(
    [
      { timestamp: '2024-01-02T09:30:00', side: 'buy', price: 10 },
      { timestamp: '2024-01-02T09:30:00', side: 'buy', price: 10.5 },
      { timestamp: '2024-01-02T09:31:00', side: 'sell', price: 11 },
    ],
    bars,
    '1m',
  )
  const markers = groupTradeMarkers(mapped)

  assert.equal(markers.length, 2)
  assert.equal(markers[0].count, 2)
  assert.equal(markers[0].buyCount, 2)
  assert.equal(markers[1].count, 1)

  const bucketed = bucketTradeMarkers(markers, 1)
  assert.equal(bucketed.length, 1)
  assert.equal(bucketed[0].count, 3)
  assert.equal(bucketed[0].buyCount, 2)
  assert.equal(bucketed[0].sellCount, 1)
  assert.equal(bucketed[0].barIndex, 1)
})

test('the results history is paged with offset and a page size of fifty', () => {
  assert.equal(resultsPageQuery(0), '/results?kind=backtest&offset=0&limit=50')
  assert.equal(resultsPageQuery(50), '/results?kind=backtest&offset=50&limit=50')
  assert.equal(resultsPageQuery(-10), '/results?kind=backtest&offset=0&limit=50')
  assert.equal(resultsPageQuery(0, 25, 'optimization'), '/results?kind=optimization&offset=0&limit=25')
})

test('loaded bars are merged without duplicates and stay sorted', () => {
  const merged = mergeBars(
    [{ timestamp: '2024-01-02T09:32:00' }, { timestamp: '2024-01-02T09:31:00' }],
    [{ timestamp: '2024-01-02T09:32:00' }, { timestamp: '2024-01-02T09:30:00' }],
  )

  assert.deepEqual(merged.map((bar) => bar.timestamp), [
    '2024-01-02T09:30:00',
    '2024-01-02T09:31:00',
    '2024-01-02T09:32:00',
  ])
  assert.equal(earliestBarTimestamp(merged), '2024-01-02T09:30:00')
})
