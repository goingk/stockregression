import assert from 'node:assert/strict'
import test from 'node:test'
import { adjustmentPageChanged, adjustmentPageParams, appendMarketPage } from '../src/marketPagination.ts'

test('an earlier page request carries the original adjustment anchor unchanged', () => {
  const originalAnchor = '2024-02-01'
  const params = new URLSearchParams(adjustmentPageParams('qfq', originalAnchor))
  assert.equal(params.get('adjustment'), 'qfq')
  assert.equal(params.get('adjustment_anchor'), originalAnchor)
})

test('an adjustment version change asks the chart to reload instead of appending', () => {
  const current = { adjustment: 'qfq', adjustment_anchor: '2024-02-01', adjustment_version: 'v1', bars: [{ timestamp: '2024-01-02' }] }
  const older = { ...current, adjustment_version: 'v2', bars: [{ timestamp: '2023-12-29' }] }
  assert.equal(adjustmentPageChanged(current, older), true)
  assert.equal(appendMarketPage(current, older), null)
})

test('a raw dataset revision also prevents pages from being mixed', () => {
  const current = { version: 1, adjustment: 'none', bars: [1] }
  assert.equal(appendMarketPage(current, { ...current, version: 2, bars: [0] }), null)
})

test('pages with the same adjustment identity append in chronological order', () => {
  const current = { adjustment: 'qfq', adjustment_anchor: '2024-02-01', adjustment_version: 'v1', bars: [{ timestamp: '2024-01-02' }] }
  const older = { ...current, bars: [{ timestamp: '2023-12-29' }] }
  assert.deepEqual(appendMarketPage(current, older).bars.map(bar => bar.timestamp), ['2023-12-29', '2024-01-02'])
})
