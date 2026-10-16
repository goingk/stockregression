import assert from 'node:assert/strict'
import test from 'node:test'
import { assetInvestmentRatio, assetInvestmentReturn, costBasisDrawdowns } from '../src/equityMath.ts'

const costStrategy = { return_basis: 'cost' }
const legacyStrategy = {}
const result = { accounting_basis: 'independent_contributions_v1', config: {} }
const point = (timestamp, normalized_equity, total_invested, equity = 100) => ({
  timestamp,
  equity,
  normalized_equity,
  total_invested,
})

test('1.10 invested ratio is a 10% return from break-even', () => {
  const ratio = assetInvestmentRatio(point('2024-01-01', 1.1, 100, 110), costStrategy, result)
  const investmentReturn = assetInvestmentReturn(point('2024-01-01', 1.1, 100, 110), costStrategy, result)

  assert.equal(ratio, 1.1)
  assert.ok(Math.abs(investmentReturn - 0.1) < 1e-12)
})

test('the first observed loss is measured from the 1.0 baseline', () => {
  const values = costBasisDrawdowns([point('2024-01-01', 0.95, 100, 95)], costStrategy, result)

  assert.equal(values[0][0], '2024-01-01')
  assert.ok(Math.abs(values[0][1] - 0.05) < 1e-12)
})

test('uninvested points stay null, then the first loss still uses a 1.0 baseline', () => {
  const values = costBasisDrawdowns([
    point('2024-01-01', null, 0, 0),
    point('2024-01-02', 0.95, 100, 95),
  ], costStrategy, result)

  assert.deepEqual(values.map(([timestamp]) => timestamp), ['2024-01-01', '2024-01-02'])
  assert.equal(values[0][1], null)
  assert.ok(Math.abs(values[1][1] - 0.05) < 1e-12)
})

test('cost-basis drawdown follows normalized equity across new contributions', () => {
  const values = costBasisDrawdowns([
    point('2024-01-01', 1.1, 100, 110),
    point('2024-01-02', 1.2, 200, 240),
    point('2024-01-03', 1.05, 300, 315),
  ], costStrategy, result)

  assert.equal(values[0][1], 0)
  assert.equal(values[1][1], 0)
  assert.ok(Math.abs(values[2][1] - 0.125) < 1e-12)
})

test('older points can use normalized equity when total_invested is absent', () => {
  const olderPoint = { timestamp: '2024-01-01', equity: 110, normalized_equity: 1.1 }

  assert.equal(assetInvestmentRatio(olderPoint, costStrategy, { config: {} }), 1.1)
  const values = costBasisDrawdowns([olderPoint, { ...olderPoint, timestamp: '2024-01-02', normalized_equity: 1 }], costStrategy, { config: {} })
  assert.equal(values[0][1], 0)
  assert.ok(Math.abs(values[1][1] - (1 - 1 / 1.1)) < 1e-12)
})

test('older cost-basis points without normalized data or invested amount remain gaps', () => {
  const olderPoint = { timestamp: '2024-01-01', equity: 95 }

  assert.equal(assetInvestmentRatio(olderPoint, costStrategy, { config: { initial_cash: 100 } }), null)
  assert.deepEqual(costBasisDrawdowns([olderPoint], costStrategy, { config: { initial_cash: 100 } }), [['2024-01-01', null]])
})

test('missing or nonpositive legacy base does not turn currency into a ratio', () => {
  const value = { timestamp: '2024-01-01', equity: 100 }

  assert.equal(assetInvestmentRatio(value, legacyStrategy, { config: {} }), null)
  assert.equal(assetInvestmentRatio(value, legacyStrategy, { config: { initial_cash: 0 } }), null)
})

test('nonfinite normalized values and derived asset ratios remain null', () => {
  assert.equal(assetInvestmentRatio(point('2024-01-01', Infinity, 100, 110), costStrategy, result), 1.1)
  assert.equal(assetInvestmentRatio(point('2024-01-01', NaN, 100, Infinity), costStrategy, result), null)
  assert.equal(assetInvestmentRatio({ timestamp: '2024-01-01', equity: NaN }, legacyStrategy, { config: { initial_cash: 100 } }), null)
  assert.deepEqual(
    costBasisDrawdowns([point('2024-01-01', NaN, 100, Infinity)], costStrategy, result),
    [['2024-01-01', null]],
  )
})
