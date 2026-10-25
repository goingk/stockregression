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
const peakPoint = (timestamp, principal_curve, values = {}) => ({
  timestamp,
  equity: 100,
  total_invested: 100,
  principal_curve,
  ...values,
})

test('the peak-principal curve is the ratio and the return follows it', () => {
  const value = peakPoint('2024-01-01', 1.25)

  assert.equal(assetInvestmentRatio(value, costStrategy, result), 1.25)
  assert.ok(Math.abs(assetInvestmentReturn(value, costStrategy, result) - 0.25) < 1e-12)
})

test('the peak-principal curve wins over the legacy invested-capital fallback', () => {
  const value = { timestamp: '2024-01-01', equity: 110, total_invested: 100, normalized_equity: 1.1, principal_curve: 1.4 }

  assert.equal(assetInvestmentRatio(value, costStrategy, result), 1.4)
  // A point without contributions still reports its peak-principal curve.
  assert.equal(assetInvestmentRatio({ timestamp: '2024-01-02', equity: 0, total_invested: 0, principal_curve: 0.9 }, costStrategy, result), 0.9)
})

test('peak-principal returns use the capital reached by that point', () => {
  const peakStrategy = { return_basis: 'peak_principal' }
  const early = peakPoint('2024-01-01', 1.001, { equity: 110, total_invested: 100 })
  const later = peakPoint('2024-01-02', 1.001, { equity: 1110, total_invested: 1100 })

  // A later contribution must not dilute the return already earned.
  assert.ok(Math.abs(assetInvestmentReturn(early, peakStrategy, result) - 0.1) < 1e-12)
  assert.ok(Math.abs(assetInvestmentReturn(later, peakStrategy, result) - (1110 / 1100 - 1)) < 1e-12)
})

test('a missing peak-principal curve falls back to the legacy fields', () => {
  assert.equal(assetInvestmentRatio(point('2024-01-01', 1.1, 100, 110), costStrategy, result), 1.1)
  assert.equal(assetInvestmentRatio({ timestamp: '2024-01-01', equity: 110, total_invested: 100 }, costStrategy, result), 1.1)
})

test('the reported principal drawdown is used as it is', () => {
  const values = costBasisDrawdowns([
    peakPoint('2024-01-01', 1.1, { principal_drawdown: 0 }),
    peakPoint('2024-01-02', 1.2, { principal_drawdown: 0.03 }),
    peakPoint('2024-01-03', 1.05, { principal_drawdown: 0.12 }),
  ], costStrategy, result)

  assert.deepEqual(values.map(([, value]) => value), [0, 0.03, 0.12])
})

test('a reported principal drawdown survives a point whose ratio is missing', () => {
  const values = costBasisDrawdowns([{ timestamp: '2024-01-01', equity: 0, total_invested: 0, principal_drawdown: 0.04 }], costStrategy, result)

  assert.deepEqual(values, [['2024-01-01', 0.04]])
})

test('legacy points without a reported drawdown keep the peak baseline', () => {
  const values = costBasisDrawdowns([
    point('2024-01-01', 1.1, 100, 110),
    point('2024-01-02', 1.2, 200, 240),
    point('2024-01-03', 1.05, 300, 315),
  ], costStrategy, result)

  assert.equal(values[0][1], 0)
  assert.equal(values[1][1], 0)
  assert.ok(Math.abs(values[2][1] - 0.125) < 1e-12)
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
