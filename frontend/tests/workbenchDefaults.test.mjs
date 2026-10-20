import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultWorkbenchDefaults, normalizeRememberedResearchDefaults, normalizeWorkbenchDefaults, researchConfigForSubmission, researchRangesForSymbol, strategyDefaults, validResearchDefaults, validWorkbenchGroup, workbenchConfig } from '../src/workbenchDefaults.ts'

test('v4 uses the four natural-position axes and aligns quantities to 100 shares', () => {
  const value = defaultWorkbenchDefaults().research
  value.ranges.buy_shares.low = 200
  assert.equal(validResearchDefaults(value), true)
  value.ranges.buy_shares.high = 1050
  assert.equal(validResearchDefaults(value), false)
  const old = defaultWorkbenchDefaults().research
  old.fixed_half_widths.min_shares = 100
  old.fixed_half_widths.max_shares = 600
  const migrated = normalizeRememberedResearchDefaults(old)
  assert.equal(migrated.fixed_half_widths.min_shares, 500)
  assert.equal(migrated.fixed_half_widths.max_shares, 1000)
  assert.equal(validResearchDefaults(migrated), true)
})

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

test('an empty maximum holding is a valid remembered draft', () => {
  const value = defaultWorkbenchDefaults()
  value.position.max_shares = null
  assert.equal(validWorkbenchGroup('position', value.position), true)
  assert.equal(normalizeWorkbenchDefaults(value).position.max_shares, null)
})

test('new workspaces use the v4 geometric neighborhood defaults', () => {
  const research = defaultWorkbenchDefaults().research
  assert.equal(research.protocol_version, 4)
  assert.equal(research.ranking_mode, 'fixed_mean')
  assert.equal(research.strategy_version, 4)
  assert.equal(research.hypercube_edge, 3)
  assert.deepEqual(research.granularity, {
    buy_step: .0001, sell_step: .0001, sell_shares: 100, buy_shares: 100,
  })
  assert.equal(research.max_shares_mode, 'unlimited')
  assert.equal(research.training_budget, 20000)
  assert.equal(research.validation_budget, 100000)
  assert.equal(research.selection_target, 'both')
  assert.equal(research.rebalance_days, 63)
  assert.equal(research.bootstrap_reps, 10000)
  assert.equal(research.n_trials, 1000)
  assert.equal(research.local_search_budget, 3000)
  assert.deepEqual(research.neighborhood_fields, ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'])
})

test('legacy fixed-radius ranking normalizes to strategy version one', () => {
  const saved = defaultWorkbenchDefaults().research
  saved.ranking_mode = 'fixed_mean'
  saved.strategy_version = 1
  const normalized = normalizeRememberedResearchDefaults(saved)
  assert.equal(normalized.ranking_mode, 'fixed_mean')
  assert.equal(normalized.strategy_version, 1)
})

test('older saved research defaults gain all fine-search fields while empty selections are rejected', () => {
  const saved = defaultWorkbenchDefaults().research
  delete saved.neighborhood_fields
  assert.deepEqual(normalizeWorkbenchDefaults({ research: saved }).research.neighborhood_fields, ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'])
  const empty = defaultWorkbenchDefaults().research
  empty.neighborhood_fields = []
  assert.equal(validResearchDefaults(empty), false)
  assert.deepEqual(normalizeRememberedResearchDefaults(empty).neighborhood_fields, ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'])
})

test('v4 submission ignores a stale hidden maximum holding when research mode changes', () => {
  const base = { initial_shares: 100000, max_shares: 2000, grid_mode: 'trailing' }
  const unlimited = researchConfigForSubmission(base, 4, 'unlimited')
  assert.equal(unlimited.max_shares, null)
  assert.equal(unlimited.grid_mode, 'geometric')
  const searched = researchConfigForSubmission(base, 4, 'search')
  assert.equal(searched.max_shares, 100000)
})

test('legacy v3 settings fill extended fields while preserving its seven ranges', () => {
  const saved = defaultWorkbenchDefaults().research
  saved.strategy_version = 3
  saved.ranges.initial_shares = { low: 300, high: 2300, step: 200 }
  saved.ranges.min_shares = { low: 0, high: 1000, step: 100 }
  saved.ranges.max_shares = { low: 1000, high: 5000, step: 100 }
  saved.ranges.buy_step = { low: .01, high: .15, step: .01 }
  for (const key of ['protocol_version', 'selection_target', 'rebalance_days', 'bootstrap_reps', 'local_search_budget']) delete saved[key]
  const migrated = normalizeWorkbenchDefaults({ research: saved }).research
  assert.equal(migrated.protocol_version, 4)
  assert.equal(migrated.ranking_mode, 'fixed_mean')
  assert.equal(migrated.strategy_version, 3)
  assert.equal(migrated.selection_target, 'both')
  assert.equal(migrated.rebalance_days, 63)
  assert.equal(migrated.bootstrap_reps, 10000)
  assert.equal(migrated.local_search_budget, 3000)
  assert.equal(Object.keys(migrated.ranges).length, 7)
  assert.deepEqual(migrated.ranges.buy_step, saved.ranges.buy_step)
  assert.deepEqual(migrated.ranges.initial_shares, saved.ranges.initial_shares)
  assert.deepEqual(normalizeRememberedResearchDefaults(saved).ranges, migrated.ranges)
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
  memory.grid.sell_step = .045
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
  value.research.granularity.buy_shares = 0
  assert.equal(validWorkbenchGroup('research', value.research), false)
  value.research.granularity.buy_shares = 50
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
  assert.deepEqual(researchRangesForSymbol(memory.research.ranges, '689001.SH').buy_shares, { low: 200, high: 200 })
  assert.equal(memory.research.ranges.buy_shares.high, 100)
})

test('v4 research ranges use 0.01 percentage point and 100 share lattice endpoints', () => {
  const value = defaultWorkbenchDefaults()
  value.research.fixed_half_widths.buy_step = .0001
  value.research.fixed_half_widths.sell_step = .0001
  value.research.ranges.buy_step = { low: .0001, high: .0211, step: .0001 }
  assert.equal(validWorkbenchGroup('research', value.research), true)
  value.research.ranges.buy_step.high = .02155
  assert.equal(validWorkbenchGroup('research', value.research), false)
  value.research.ranges.buy_step.high = .021
  value.research.ranges.buy_shares.high = 1050
  assert.equal(validWorkbenchGroup('research', value.research), false)
  const remembered = normalizeWorkbenchDefaults({ research: { ...value.research, ranges: { ...value.research.ranges, grid_step: { low: .01055, high: .021, step: .0005 }, buy_step: undefined, sell_step: undefined } } }).research
  assert.deepEqual(remembered.ranges.buy_step, { low: .01055, high: .021 })
  assert.deepEqual(remembered.ranges.sell_step, { low: .01055, high: .021 })
  assert.equal(validResearchDefaults(remembered), false)
})

test('legacy single grid settings migrate to separate buy and sell spacing', () => {
  const saved = defaultWorkbenchDefaults().research
  saved.strategy_version = 1
  delete saved.ranges.buy_step
  delete saved.ranges.sell_step
  saved.ranges.grid_step = { low: .008, high: .024, step: .001 }
  saved.neighborhood_fields = ['grid_step', 'buy_shares']
  delete saved.fixed_half_widths.buy_step
  delete saved.fixed_half_widths.sell_step
  saved.fixed_half_widths.grid_step = .002
  const migrated = normalizeRememberedResearchDefaults(saved)
  assert.deepEqual(migrated.ranges.buy_step, saved.ranges.grid_step)
  assert.deepEqual(migrated.ranges.sell_step, saved.ranges.grid_step)
  assert.deepEqual(migrated.neighborhood_fields, ['buy_step', 'sell_step', 'buy_shares'])
  assert.equal(migrated.fixed_half_widths.buy_step, .002)
  assert.equal(migrated.fixed_half_widths.sell_step, .002)
})
