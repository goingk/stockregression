import assert from 'node:assert/strict'
import test from 'node:test'
import { buildResultCurveModel, resultCurveRowAt, resultStrategyEntries, savedGridParameters } from '../src/resultPresentation.ts'

const parameterMap = (result) => Object.fromEntries(savedGridParameters(result).map((row) => [row.key, row]))
const strategy = (equity, values = {}) => ({ metrics: {}, equity, trades: [], rejections: [], ...values })

test('saved trailing grid parameters show only active steps and the actual saved funding', () => {
  const rows = parameterMap({
    symbol: '600000.SH', schema_version: 2,
    config: { grid_mode: 'trailing', buy_step: .02, sell_step: .04, grid_step: .99, initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 200, reference_price: null, reference_price_mode: 'auto', lower_price: null, upper_price: 12, reserve_cash: 8000 },
    funding: { reference_price: 10.25, reference_date: '2024-01-02', reserve_cash: 9000, initial_equity: 19250 },
  })
  assert.equal(rows.grid_mode.value, '移动网格')
  assert.equal(rows.buy_step.value, '2.00%')
  assert.equal(rows.sell_step.value, '4.00%')
  assert.equal(rows.grid_step, undefined)
  assert.equal(rows.reference_price.value, '10.250 元')
  assert.equal(rows.reference_price.note, '自动参考 · 2024-01-02')
  assert.equal(rows.lower_price.value, '不限')
  assert.equal(rows.upper_price.value, '12.000 元')
  assert.equal(rows.min_shares.value, '0 股')
  assert.equal(rows.reserve_cash.value, '9,000.00 元')
  assert.equal(rows.initial_equity.value, '19,250.00 元')
})

test('arithmetic and geometric grids show separate buy and sell spacing', () => {
  for (const grid_mode of ['arithmetic', 'geometric']) {
    const rows = parameterMap({ schema_version: 2, config: { grid_mode, buy_step: .03, sell_step: .04 } })
    assert.equal(rows.grid_step, undefined)
    assert.equal(rows.buy_step.value, '3.00%')
    assert.equal(rows.sell_step.value, '4.00%')
    for (const key of ['reference_price', 'lower_price', 'upper_price', 'initial_shares', 'min_shares', 'max_shares', 'buy_shares', 'sell_shares', 'reserve_cash', 'initial_equity']) assert.equal(rows[key].value, '—', key)
  }
  assert.equal(parameterMap({ config: {} }).grid_mode.value, '—')
  assert.equal(parameterMap({ config: {} }).grid_step, undefined)
  const legacy = parameterMap({ config: { grid_mode: 'geometric', grid_step: .05 } })
  assert.equal(legacy.grid_step.value, '5.00%')
  assert.equal(legacy.buy_step, undefined)
  assert.equal(legacy.sell_step, undefined)
})

test('legacy parameters keep cash and ratio units without new share defaults', () => {
  const rows = parameterMap({
    symbol: '600000.SH',
    config: { schema_version: 1, grid_mode: 'geometric', grid_step: .05, buy_quantity: { mode: 'cash', value: 5000 }, sell_quantity: { mode: 'equity_ratio', value: .1 }, initial_cash: 100000, initial_position: .5, min_position: 0, max_position: .8 },
  })
  assert.equal(rows.buy_quantity.value, '5,000.00 元')
  assert.equal(rows.sell_quantity.value, '10.00% 净资产')
  assert.equal(rows.initial_position.value, '50.00%')
  assert.equal(rows.min_position.value, '0.00%')
  assert.equal(rows.initial_cash.value, '100,000.00 元')
  assert.equal(rows.initial_shares, undefined)
  assert.equal(rows.reserve_cash, undefined)
  assert.equal(parameterMap({ config: { schema_version: 1 } }).buy_quantity.value, '—')
})

test('ETF holdings use fund units and a saved zero cash reserve is preserved', () => {
  const rows = parameterMap({ symbol: '159611.SZ', config: { initial_shares: 1000, buy_shares: 100, reserve_cash: 0 } })
  assert.equal(rows.initial_shares.value, '1,000 份')
  assert.equal(rows.buy_shares.value, '100 份')
  assert.equal(rows.reserve_cash.value, '0.00 元')
})

test('v3 unlimited holdings hide the removed cash budget and show peak contributions', () => {
  const rows = parameterMap({
    symbol: '159611.SZ', schema_version: 3,
    config: { schema_version: 3, grid_mode: 'trailing', buy_step: .03, sell_step: .03, initial_shares: 1000, min_shares: 0, max_shares: null, buy_shares: 100, sell_shares: 100 },
    funding: { reference_price: 1, reference_date: '2024-01-02', initial_equity: 1000, peak_invested: 1300, topups: 300 },
    strategies: { grid: { metrics: { external_topups: 300 } } },
  })
  assert.equal(rows.max_shares.value, '不限')
  assert.equal(rows.reserve_cash, undefined)
  assert.equal(rows.peak_invested.value, '1,300.00 元')
  assert.equal(rows.topups.value, '300.00 元')
  assert.equal(rows.external_topups.value, '300.00 元')
})

test('older results keep their cash reserve and show missing peak fields as dashes', () => {
  const rows = parameterMap({
    symbol: '600000.SH', schema_version: 2,
    config: { schema_version: 2, grid_mode: 'trailing', initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 200 },
    funding: { initial_equity: 19250, reserve_cash: 9000 },
  })
  assert.equal(rows.reserve_cash.value, '9,000.00 元')
  assert.equal(rows.initial_equity.value, '19,250.00 元')
  assert.equal(rows.peak_invested.value, '—')
  assert.equal(rows.topups.value, '—')
  assert.equal(rows.external_topups.value, '—')
})

test('peak-principal curves drive the curve model returns and drawdowns', () => {
  const model = buildResultCurveModel({
    symbol: '600000.SH', accounting_basis: 'peak_principal_v2', config: {},
    strategies: {
      grid: strategy([{ timestamp: '2024-01-02', equity: 110, total_invested: 100, principal_curve: 1.1, principal_drawdown: .02, position: 10 }]),
      same_account_passive: strategy([{ timestamp: '2024-01-02', equity: 105, total_invested: 100, principal_curve: 1.05, principal_drawdown: .01 }]),
    },
  })
  assert.ok(Math.abs(model.rows[0].values.grid.returnRate - .1) < 1e-12)
  assert.equal(model.rows[0].values.grid.drawdown, .02)
  assert.ok(Math.abs(model.rows[0].values.same_account_passive.returnRate - .05) < 1e-12)
  assert.equal(model.rows[0].values.same_account_passive.drawdown, .01)
  assert.equal(model.strategies.find((style) => style.key === 'same_account_passive').name, '同账户被动持有')
})

test('results hide the matched-funding baseline without changing saved research data or other strategies', () => {
  const result = {
    symbol: '600000.SH', accounting_basis: 'peak_principal_v2', config: {},
    funding: { matched_funding_hold_invested: 100 },
    strategies: {
      matched_funding_hold: strategy([
        { timestamp: '2023-01-01', equity: 100, total_invested: 100, principal_curve: 1 },
        { timestamp: '2024-01-03', equity: 105, total_invested: 100, principal_curve: 1.05 },
      ], { metrics: { peak_principal_return: .05, total_invested: 100 } }),
      grid: strategy([{ timestamp: '2024-01-02', equity: 110, total_invested: 100, principal_curve: 1.1 }], { metrics: { matched_funding_hold_return: .05, matched_funding_hold_invested: 100 } }),
      same_account_passive: strategy([{ timestamp: '2024-01-02', equity: 105, total_invested: 100, principal_curve: 1.05 }]),
      historical_strategy: strategy([{ timestamp: '2024-01-04', equity: 108, total_invested: 100, principal_curve: 1.08 }]),
      dca: null,
    },
  }
  const saved = structuredClone(result)
  const entries = resultStrategyEntries(result)
  assert.deepEqual(entries.map(([key]) => key), ['grid', 'same_account_passive', 'historical_strategy'])
  assert.equal(entries[0][1], result.strategies.grid)
  const model = buildResultCurveModel(result)
  assert.deepEqual(model.strategies.map(({ key }) => key), ['grid', 'same_account_passive', 'historical_strategy'])
  assert.deepEqual(model.rows.map(({ timestamp }) => timestamp), ['2024-01-02', '2024-01-04'])
  assert.ok(model.rows.every((row) => !Object.hasOwn(row.values, 'matched_funding_hold')))
  assert.equal(resultCurveRowAt(model, '2024-01-03'), undefined)
  assert.deepEqual(result, saved)
})

test('missing strategies and historical results containing only the hidden baseline stay empty', () => {
  assert.deepEqual(resultStrategyEntries(null), [])
  assert.deepEqual(resultStrategyEntries(undefined), [])
  assert.deepEqual(resultStrategyEntries({}), [])
  const model = buildResultCurveModel({ strategies: { matched_funding_hold: strategy([{ timestamp: '2024-01-01', equity: 100 }]) } })
  assert.deepEqual(model.strategies, [])
  assert.deepEqual(model.rows, [])
})

test('legacy results keep their own drawdown when no principal drawdown is reported', () => {
  const model = buildResultCurveModel({
    symbol: '600000.SH', config: {},
    strategies: { grid: strategy([{ timestamp: '2024-01-02', equity: 110, drawdown: .03 }]) },
  })
  assert.equal(model.rows[0].values.grid.drawdown, .03)
  assert.equal(model.rows[0].values.grid.returnRate, null)
})

test('aligned curves use equityMath returns and cost-basis drawdowns across contributions', () => {
  const model = buildResultCurveModel({
    symbol: '600000.SH', accounting_basis: 'independent_contributions_v1', config: {},
    strategies: {
      grid: strategy([{ timestamp: '2024-01-02', equity: 110, total_invested: 100, drawdown: .02, position: 10 }]),
      dca: strategy([
        { timestamp: '2024-01-01', equity: 0, total_invested: 0, normalized_equity: null },
        { timestamp: '2024-01-02', equity: 240, total_invested: 200, normalized_equity: 1.2 },
        { timestamp: '2024-01-03', equity: 315, total_invested: 300, normalized_equity: 1.05 },
      ], { return_basis: 'cost' }),
    },
  })
  assert.deepEqual(model.rows.map((row) => row.timestamp), ['2024-01-01', '2024-01-02', '2024-01-03'])
  assert.ok(Math.abs(model.rows[1].values.grid.returnRate - .1) < 1e-12)
  assert.equal(model.rows[1].values.grid.drawdown, .02)
  assert.equal(model.rows[0].values.dca.returnRate, null)
  assert.equal(model.rows[0].values.dca.drawdown, null)
  assert.ok(Math.abs(model.rows[2].values.dca.drawdown - .125) < 1e-12)
  assert.equal(model.rows[2].values.dca.invested, 300)
})

test('curve gaps and a missing comparison date are never replaced by zero or the nearest value', () => {
  const model = buildResultCurveModel({
    symbol: '600000.SH', config: { initial_cash: 100 }, strategies: {
      grid: strategy([{ timestamp: '2024-01-01', equity: 100, position: 100 }, { timestamp: '2024-01-03', equity: 90 }]),
      buy_hold: strategy([{ timestamp: '2024-01-02', equity: 110, drawdown: .01 }]),
    },
  })
  assert.equal(model.rows[0].values.grid.returnRate, 0)
  assert.equal(model.rows[0].values.grid.drawdown, null)
  assert.equal(model.rows[0].values.buy_hold, undefined)
  assert.equal(model.rows[1].values.grid, undefined)
  assert.equal(model.rows[1].position, null)
  assert.equal(model.rows[2].position, null)
  assert.equal(resultCurveRowAt(model, '2024-01-02'), model.rows[1])
  assert.equal(resultCurveRowAt(model, '2024-01-02T12:00:00'), undefined)
})

test('share-only position histories retain their units while partial ratios retain their gaps', () => {
  const base = { symbol: '159611.SZ', config: {}, strategies: {} }
  const shares = buildResultCurveModel({ ...base, strategies: { grid: strategy([{ timestamp: '2024-01-01', equity: 100, position: 1000 }]) } })
  assert.equal(shares.positionMode, 'shares')
  assert.equal(shares.shareUnit, '份')
  assert.equal(shares.rows[0].position, 1000)
  const ratio = buildResultCurveModel({ ...base, strategies: { grid: strategy([{ timestamp: '2024-01-01', equity: 100, position: 1000, position_ratio: .6 }, { timestamp: '2024-01-02', equity: 100, position: 1000 }]) } })
  assert.equal(ratio.positionMode, 'ratio')
  assert.equal(ratio.rows[0].position, .6)
  assert.equal(ratio.rows[1].position, null)
})

test('partial historical results can be opened without invented contribution data', () => {
  const model = buildResultCurveModel({ symbol: '600000.SH', strategies: { dca: strategy([{ timestamp: '2024-01-01', equity: 100 }], { return_basis: 'cost' }) } })
  assert.equal(model.rows[0].values.dca.returnRate, null)
  assert.equal(model.rows[0].values.dca.drawdown, null)
  assert.equal(model.rows[0].values.dca.invested, null)
  assert.deepEqual(buildResultCurveModel({ symbol: '600000.SH' }).rows, [])
})
