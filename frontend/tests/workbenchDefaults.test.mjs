import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultWorkbenchDefaults, normalizeWorkbenchDefaults, researchRangesForSymbol, strategyDefaults, validWorkbenchGroup, workbenchConfig } from '../src/workbenchDefaults.ts'

test('all four groups keep valid zero values and independent default copies', () => {
  const value = defaultWorkbenchDefaults()
  value.position.initial_shares = 0
  value.position.max_shares = 0
  value.dca.initial_shares = 0
  value.research.seed = 0
  value.research.min_trades = 0
  value.research.min_minute_ratio = 0
  for (const group of ['grid', 'position', 'dca', 'research']) assert.equal(validWorkbenchGroup(group, value[group]), true)
  assert.deepEqual(normalizeWorkbenchDefaults(value), value)
  value.research.ranges.buy_shares.high = 3000
  assert.equal(defaultWorkbenchDefaults().research.ranges.buy_shares.high, 1000)
})

test('invalid remembered groups fall back independently', () => {
  const value = defaultWorkbenchDefaults()
  value.grid.buy_step = NaN
  value.position.initial_shares = 3000
  value.dca.frequency = 'weekly'
  value.research.n_trials = 333
  const loaded = normalizeWorkbenchDefaults(value)
  assert.equal(loaded.grid.buy_step, .03)
  assert.equal(loaded.position.initial_shares, 1000)
  assert.equal(loaded.dca.frequency, 'weekly')
  assert.equal(loaded.research.n_trials, 333)
  assert.deepEqual(normalizeWorkbenchDefaults(null), defaultWorkbenchDefaults())
})

test('remembered fields and inactive grid search ranges all survive blank workbench construction', () => {
  const memory = defaultWorkbenchDefaults()
  memory.grid.grid_mode = 'geometric'
  memory.grid.buy_step = .025
  memory.grid.grid_step = .045
  memory.grid.lower_price = 8
  memory.grid.upper_price = 30
  memory.position.initial_shares = 1200
  memory.dca = { frequency: 'weekly', shares: 300, initial_shares: 600 }
  memory.research.ranges.buy_step = { low: .02, high: .12, step: .01 }
  const base = { symbol: '600000.SH', start_date: '2025-10-16', end_date: '2026-10-16', frequency: '1m', reference_price: 15, reference_price_mode: 'manual', fees: { commission_rate: .0007, min_commission: .2, slippage_bps: 5 } }
  const config = workbenchConfig(base, memory)
  assert.equal(config.symbol, '')
  assert.equal(config.reference_price, null)
  assert.equal(config.reference_price_mode, 'auto')
  assert.equal(config.start_date, base.start_date)
  assert.deepEqual(config.fees, base.fees)
  assert.deepEqual(strategyDefaults(config), { grid: memory.grid, position: memory.position, dca: memory.dca })
  assert.equal(memory.research.ranges.buy_step.low, .02)
  assert.equal(base.symbol, '600000.SH')
})

test('invalid numeric drafts and ranges cannot replace valid remembered groups', () => {
  const value = defaultWorkbenchDefaults()
  for (const invalid of [0, NaN, Infinity, null, '']) assert.equal(validWorkbenchGroup('grid', { ...value.grid, buy_step: invalid }), false)
  assert.equal(validWorkbenchGroup('grid', { ...value.grid, lower_price: 20, upper_price: 10 }), false)
  assert.equal(validWorkbenchGroup('position', { ...value.position, buy_shares: 150 }), false)
  assert.equal(validWorkbenchGroup('research', { ...value.research, n_trials: 0 }), false)
  assert.equal(validWorkbenchGroup('research', { ...value.research, n_trials: 5001 }), false)
  value.research.ranges.buy_step.low = .2
  assert.equal(validWorkbenchGroup('research', value.research), false)
  value.research.ranges.buy_step.low = .005
  value.research.ranges.buy_shares.step = 0
  assert.equal(validWorkbenchGroup('research', value.research), false)
  value.research.ranges.buy_shares.step = 50
  assert.equal(validWorkbenchGroup('research', value.research), false)
})

test('STAR minimums affect the displayed and submitted range without overwriting shared preferences', () => {
  const memory = defaultWorkbenchDefaults()
  const star = researchRangesForSymbol(memory.research.ranges, '688001.SH')
  assert.equal(star.buy_shares.low, 200)
  assert.equal(memory.research.ranges.buy_shares.low, 100)
  const edited = { ...memory.research, mode: 'historical' }
  assert.equal(edited.ranges.buy_shares.low, 100)
  assert.equal(researchRangesForSymbol(edited.ranges, '600000.SH').buy_shares.low, 100)
  memory.research.ranges.buy_shares.high = 100
  assert.deepEqual(researchRangesForSymbol(memory.research.ranges, '689001.SH').buy_shares, { low: 200, high: 200, step: 100 })
  assert.equal(memory.research.ranges.buy_shares.high, 100)
})
