import assert from 'node:assert/strict'
import test from 'node:test'
import { calendarYearsBefore, defaultTradingRange, resolveMissingEndDate, shouldInitializeDefaultWindow, tradingDayCount, tradingDateIndexAtOrBefore, trailingTradingRange } from '../src/tradingCalendar.ts'
import { hasLocalCalendarCoverage, isCalendarStateCurrent } from '../src/calendarAvailability.ts'

const dates = ['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07']

test('manual non-trading bounds count trading sessions inclusively', () => {
  assert.equal(tradingDayCount(dates, '2026-01-03', '2026-01-06'), 2)
  assert.equal(tradingDayCount(dates, '2026-01-02', '2026-01-04'), 1)
  assert.equal(tradingDateIndexAtOrBefore(dates, '2026-01-04'), 0)
  assert.equal(tradingDateIndexAtOrBefore(dates, '2025-12-31'), -1)
})

test('slider window ends at the last trading date on or before a manual end date', () => {
  assert.deepEqual(trailingTradingRange(dates, '2026-01-06', 2), { start: '2026-01-05', endIndex: 2 })
  assert.deepEqual(trailingTradingRange(dates, '2026-01-06', 99), { start: '2026-01-02', endIndex: 2 })
  assert.equal(trailingTradingRange(dates, '2025-12-31', 2), null)
})

test('first default window is based on covered last_date, not the wall-clock defaults', () => {
  const recentHistory = ['2025-09-26', '2025-09-29', '2025-09-30', '2026-09-25', '2026-09-28']
  assert.deepEqual(defaultTradingRange(recentHistory, '2026-09-28'), {
    start: '2025-09-29', end: '2026-09-28', count: 4,
  })
  const delayed2022History = [
    '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-10',
    '2022-06-01', '2022-12-29', '2022-12-30',
  ]
  const delayed = defaultTradingRange(delayed2022History, '2022-12-30')
  assert.deepEqual(delayed, { start: '2022-01-04', end: '2022-12-30', count: 8 })
  assert.ok(delayed.count > 1)
  assert.equal(calendarYearsBefore('2024-02-29'), '2023-02-28')
})

test('missing end resolution preserves causal start and respects explicit intent', () => {
  assert.deepEqual(resolveMissingEndDate('2026-03-01', null, false, '2026-09-28'), {
    start: '2026-03-01', end: '2026-09-28',
  })
  assert.equal(resolveMissingEndDate('2026-03-01', null, false, '2026-02-27'), null)
  assert.equal(resolveMissingEndDate('2026-03-01', null, true, '2026-09-28'), null)
  assert.equal(resolveMissingEndDate('2026-03-01', '2026-05-01', false, '2026-09-28'), null)
  assert.equal(shouldInitializeDefaultWindow(true, false, true, null, '2026-02-27'), false)
  assert.equal(shouldInitializeDefaultWindow(true, false, true, '2026-03-10', '2026-02-27'), true)
  assert.equal(shouldInitializeDefaultWindow(true, true, true, '2026-03-10', '2026-02-27'), false)
})

test('local data coverage readiness is independent from slider calendar availability', () => {
  const response = {
    symbol: '510300.SH', frequency: '1d', data_version: 'v1',
    first_date: '2025-01-02', last_date: '2026-01-30', trading_dates: [], calendar_available: false,
  }
  assert.equal(hasLocalCalendarCoverage(response), true)
  assert.equal(hasLocalCalendarCoverage({ first_date: null, last_date: null }), false)
  assert.equal(isCalendarStateCurrent({ key: '510300.SH|1d' }, '510300.SH', '1d'), true)
  assert.equal(isCalendarStateCurrent({ key: '510300.SH|1d' }, '600000.SH', '1d'), false)
  assert.equal(isCalendarStateCurrent({ key: '510300.SH|1d' }, '510300.SH', '5m'), false)
})
