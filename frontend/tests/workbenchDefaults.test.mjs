import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultWorkbenchDefaults, mergeEditedResearchDefaults, normalizeRememberedResearchDefaults, normalizeWorkbenchDefaults, researchConfigForSubmission, researchRangesForSymbol, strategyDefaults, validResearchDefaults, validWorkbenchGroup, workbenchConfig } from '../src/workbenchDefaults.ts'

const ROBUST_AXES = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares']
const RETIRED_FIELDS = [
  'bootstrap_reps', 'bootstrap_block_lengths', 'significance_alpha', 'outer_validation_days',
  'mode', 'n_trials', 'local_search_budget', 'max_drawdown',
]

/** A research configuration saved by the retired v4 protocol. */
function v4SavedResearch() {
  return {
    ...defaultWorkbenchDefaults().research,
    protocol_version: 4,
    strategy_version: 4,
    ranking_mode: 'fixed_mean',
    selection_target: 'fixed',
    top_candidates: 100,
    min_history_days: 504,
    training_budget: 30000,
    recommendation_count: 5,
    slippage_mode: 'zero',
    bootstrap_reps: 10000,
    bootstrap_block_lengths: [20, 10, 40],
    significance_alpha: .05,
    outer_validation_days: 126,
    mode: 'both',
    n_trials: 1000,
    local_search_budget: 3000,
    max_drawdown: .3,
    granularity: { buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100, min_shares: 500, max_shares: 500 },
    fixed_half_widths: { buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100, min_shares: 500, max_shares: 500 },
    ranges: {
      buy_step: { low: .005, high: .05, step: null },
      sell_step: { low: .005, high: .05, step: null },
      buy_shares: { low: 2500, high: 7500, step: null },
      sell_shares: { low: 2500, high: 7500, step: null },
      initial_shares: { low: 300, high: 2300, step: 200 },
      min_shares: { low: 0, high: 1000, step: 100 },
      max_shares: { low: 1000, high: 5000, step: 100 },
    },
  }
}

test('new workspaces use the v5 research protocol defaults', () => {
  const research = defaultWorkbenchDefaults().research
  assert.equal(research.protocol_version, 5)
  assert.equal(research.strategy_version, 5)
  assert.equal(research.ranking_mode, 'robust_j')
  assert.equal(research.strategy_id, 'geometric_robust_j')
  assert.equal(research.max_shares_mode, 'unlimited')
  assert.equal(research.selection_target, 'both')
  assert.equal(research.rebalance_days, 63)
  assert.equal(research.seed, 42)
  assert.equal(research.top_candidates, 20)
  assert.equal(research.max_principal_drawdown, .2)
  assert.equal(research.min_trades, 10)
  assert.equal(research.min_history_days, 250)
  assert.equal(research.min_minute_ratio, .95)
  assert.equal(research.hypercube_edge, 3)
  assert.deepEqual(research.granularity, {
    buy_step: .0001, sell_step: .0001, sell_shares: 100, buy_shares: 100,
  })
  assert.equal(research.search_method, 'auto')
  assert.equal(research.training_budget, 20000)
  assert.equal(research.validation_budget, 100000)
  assert.equal(research.reporting_budget, 1000)
  assert.equal(research.total_evaluation_budget, null)
  assert.equal(research.max_runtime_seconds, 3600)
  assert.equal(research.recommendation_count, 10)
  assert.equal(research.slippage_mode, 'configured')
  assert.deepEqual(research.neighborhood_fields, ROBUST_AXES)
  assert.deepEqual(research.fixed_half_widths, {
    buy_step: .001, sell_step: .001, sell_shares: 100, buy_shares: 100,
  })
  assert.equal(research.sigma, 1)
  assert.equal(research.alpha, .3)
  assert.equal(research.beta, 1.2)
  assert.equal(research.tau, .01)
  assert.equal(research.eta_limit, 2)
  assert.equal(research.eta_hard_cap, 3.5)
  assert.equal(research.gamma_max, .4)
  assert.equal(research.eta_floor, 0)
  assert.equal(research.gamma_min, 0)
  assert.equal(research.floor_k_ref, 10)
  assert.equal(research.training_min_days, 250)
  assert.equal(research.training_max_days, 500)
  assert.equal(research.training_window_mode, 'same_price')
  assert.equal(research.peak_tolerance, 1e-12)
  assert.equal(research.start_min, 20)
  assert.equal(research.start_max, 64)
  assert.equal(research.starts_per_candidate, 4)
  assert.deepEqual(research.initial_strides, [1, 2])
  assert.equal(research.stride_multiplier, 2)
  assert.equal(research.max_stride_fraction, .25)
  assert.equal(research.validation_reserve_fraction, .25)
  assert.equal(research.exhaustive_memory_limit_mb, 512)
  assert.equal(research.estimated_bytes_per_point, 512)
  assert.equal(research.stress_cost_multiplier, 2)
  assert.equal(research.stress_region_count, 3)
  assert.deepEqual(Object.keys(research.ranges).sort(), [...ROBUST_AXES].sort())
  assert.equal(validResearchDefaults(research), true)
})

test('a saved v4 configuration migrates to the v5 protocol without legacy fields', () => {
  const migrated = normalizeRememberedResearchDefaults(v4SavedResearch())
  assert.equal(migrated.protocol_version, 5)
  assert.equal(migrated.strategy_version, 5)
  assert.equal(migrated.ranking_mode, 'robust_j')
  assert.equal(migrated.strategy_id, 'geometric_robust_j')
  assert.equal(migrated.max_shares_mode, 'unlimited')
  assert.deepEqual(migrated.neighborhood_fields, ROBUST_AXES)
  assert.equal(migrated.selection_target, 'fixed')
  assert.equal(migrated.top_candidates, 100)
  assert.equal(migrated.slippage_mode, 'zero')
  for (const field of RETIRED_FIELDS) assert.equal(field in migrated, false, field + ' must be removed')
  assert.deepEqual(Object.keys(migrated.granularity).sort(), [...ROBUST_AXES].sort())
  assert.deepEqual(Object.keys(migrated.fixed_half_widths).sort(), [...ROBUST_AXES].sort())
  assert.deepEqual(Object.keys(migrated.ranges).sort(), [...ROBUST_AXES].sort())
  assert.deepEqual(migrated.ranges.buy_step, { low: .005, high: .05, step: null })
  assert.deepEqual(migrated.ranges.buy_shares, { low: 2500, high: 7500, step: null })
  assert.equal(validResearchDefaults(migrated), true)
})

test('the retired unit-NAV drawdown is never promoted to the principal drawdown', () => {
  const saved = v4SavedResearch()
  delete saved.max_principal_drawdown
  saved.max_drawdown = .3
  const migrated = normalizeRememberedResearchDefaults(saved)
  assert.equal(migrated.max_principal_drawdown, .2)
  assert.equal('max_drawdown' in migrated, false)
  const kept = { ...defaultWorkbenchDefaults().research, max_principal_drawdown: .35 }
  assert.equal(normalizeRememberedResearchDefaults(kept).max_principal_drawdown, .35)
})

test('v5 budgets and the training window default to the protocol values and keep saved ones', () => {
  const legacy = v4SavedResearch()
  delete legacy.reporting_budget
  delete legacy.total_evaluation_budget
  delete legacy.max_runtime_seconds
  delete legacy.training_window_mode
  const migrated = normalizeRememberedResearchDefaults(legacy)
  assert.equal(migrated.reporting_budget, 1000)
  assert.equal(migrated.total_evaluation_budget, null)
  assert.equal(migrated.max_runtime_seconds, 3600)
  assert.equal(migrated.training_window_mode, 'same_price')
  const saved = {
    ...defaultWorkbenchDefaults().research,
    reporting_budget: 250,
    total_evaluation_budget: 50000,
    max_runtime_seconds: 120,
    training_window_mode: 'fixed',
  }
  const normalized = normalizeRememberedResearchDefaults(saved)
  assert.equal(normalized.reporting_budget, 250)
  assert.equal(normalized.total_evaluation_budget, 50000)
  assert.equal(normalized.max_runtime_seconds, 120)
  assert.equal(normalized.training_window_mode, 'fixed')
})

test('research submission always uses an unlimited geometric grid', () => {
  const base = { initial_shares: 100000, max_shares: 2000, grid_mode: 'trailing' }
  const submitted = researchConfigForSubmission(base)
  assert.equal(submitted.max_shares, null)
  assert.equal(submitted.grid_mode, 'geometric')
  assert.equal(submitted.initial_shares, 100000)
  assert.equal(base.max_shares, 2000)
})

test('only the v5 protocol passes research validation', () => {
  const value = defaultWorkbenchDefaults().research
  assert.equal(validResearchDefaults(value), true)
  assert.equal(validResearchDefaults({ ...value, protocol_version: 4 }), false)
  assert.equal(validResearchDefaults({ ...value, strategy_version: 4 }), false)
  assert.equal(validResearchDefaults({ ...value, ranking_mode: 'fixed_mean' }), false)
  assert.equal(validResearchDefaults({ ...value, strategy_id: 'geometric_excess_neighborhood' }), false)
  assert.equal(validResearchDefaults({ ...value, max_shares_mode: 'search' }), false)
  assert.equal(validResearchDefaults({ ...value, training_window_mode: 'other' }), false)
  assert.equal(validResearchDefaults({ ...value, training_window_mode: 'fixed' }), true)
})

test('v5 neighbourhoods, half widths, granularities and ranges cover exactly the four axes', () => {
  const value = defaultWorkbenchDefaults().research
  assert.equal(validResearchDefaults({ ...value, neighborhood_fields: ['buy_step', 'sell_step', 'buy_shares'] }), false)
  assert.equal(validResearchDefaults({ ...value, neighborhood_fields: [...ROBUST_AXES, 'min_shares'] }), false)
  assert.equal(validResearchDefaults({ ...value, fixed_half_widths: { ...value.fixed_half_widths, min_shares: 500 } }), false)
  assert.equal(validResearchDefaults({ ...value, granularity: { ...value.granularity, max_shares: 500 } }), false)
  assert.equal(validResearchDefaults({ ...value, ranges: { ...value.ranges, grid_step: { low: .01, high: .02, step: null } } }), false)
  assert.equal(validResearchDefaults({ ...value, ranges: { ...value.ranges, buy_step: { low: .01, high: .02, step: .001 } } }), false)
  assert.equal(validResearchDefaults({ ...value, ranges: { ...value.ranges, buy_shares: { low: 100, high: 1050, step: null } } }), false)
  assert.equal(validResearchDefaults({ ...value, ranges: { ...value.ranges, buy_step: { low: 0, high: .1, step: null } } }), false)
  assert.equal(validResearchDefaults({ ...value, ranges: { ...value.ranges, buy_step: { low: .005, high: 1.5, step: null } } }), false)
  assert.equal(validResearchDefaults({ ...value, fixed_half_widths: { ...value.fixed_half_widths, buy_step: .0005 } }), false)
  assert.equal(validResearchDefaults({ ...value, fixed_half_widths: { ...value.fixed_half_widths, buy_step: .002 } }), true)
  assert.equal(validResearchDefaults({ ...value, fixed_half_widths: { ...value.fixed_half_widths, sell_shares: 150 } }), false)
  assert.equal(validResearchDefaults({ ...value, fixed_half_widths: { ...value.fixed_half_widths, sell_shares: 200 } }), true)
  assert.equal(validResearchDefaults({ ...value, granularity: { ...value.granularity, buy_shares: 50 } }), false)
  assert.equal(validResearchDefaults({ ...value, granularity: { ...value.granularity, buy_shares: 200 } }), true)
})

test('v5 budget bounds are enforced', () => {
  const value = defaultWorkbenchDefaults().research
  assert.equal(validResearchDefaults({ ...value, total_evaluation_budget: null }), true)
  assert.equal(validResearchDefaults({ ...value, total_evaluation_budget: 1 }), true)
  assert.equal(validResearchDefaults({ ...value, total_evaluation_budget: 10000000 }), true)
  assert.equal(validResearchDefaults({ ...value, total_evaluation_budget: 0 }), false)
  assert.equal(validResearchDefaults({ ...value, total_evaluation_budget: 10000001 }), false)
  assert.equal(validResearchDefaults({ ...value, max_runtime_seconds: 1 }), true)
  assert.equal(validResearchDefaults({ ...value, max_runtime_seconds: 86400 }), true)
  assert.equal(validResearchDefaults({ ...value, max_runtime_seconds: 0 }), false)
  assert.equal(validResearchDefaults({ ...value, max_runtime_seconds: 86401 }), false)
  assert.equal(validResearchDefaults({ ...value, reporting_budget: 0 }), true)
  assert.equal(validResearchDefaults({ ...value, reporting_budget: 2000000 }), true)
  assert.equal(validResearchDefaults({ ...value, reporting_budget: 2000001 }), false)
  assert.equal(validResearchDefaults({ ...value, reporting_budget: -1 }), false)
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

test('missing fine-search fields are restored and empty selections are rejected', () => {
  const saved = defaultWorkbenchDefaults().research
  delete saved.neighborhood_fields
  assert.deepEqual(normalizeWorkbenchDefaults({ research: saved }).research.neighborhood_fields, ROBUST_AXES)
  const empty = defaultWorkbenchDefaults().research
  empty.neighborhood_fields = []
  assert.equal(validResearchDefaults(empty), false)
  assert.deepEqual(normalizeRememberedResearchDefaults(empty).neighborhood_fields, ROBUST_AXES)
})

test('invalid remembered groups fall back independently', () => {
  const value = defaultWorkbenchDefaults()
  value.grid.buy_step = NaN
  value.position.initial_shares = 3000
  value.dca.frequency = 'weekly'
  value.research.seed = 7
  const loaded = normalizeWorkbenchDefaults(value)
  assert.equal(loaded.grid.buy_step, .03)
  assert.equal(loaded.position.initial_shares, 1000)
  assert.equal(loaded.dca.frequency, 'weekly')
  assert.equal(loaded.research.seed, 7)
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
  memory.research.ranges.buy_step = { low: .02, high: .12, step: null }
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
  assert.equal(validWorkbenchGroup('research', { ...value.research, protocol_version: 4 }), false)
  assert.equal(validWorkbenchGroup('research', { ...value.research, strategy_version: 3 }), false)
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
  const edited = { ...memory.research, selection_target: 'fixed' }
  assert.equal(edited.ranges.buy_shares.low, 100)
  assert.equal(researchRangesForSymbol(edited.ranges, '600000.SH').buy_shares.low, 100)
  memory.research.ranges.buy_shares.high = 100
  assert.deepEqual(researchRangesForSymbol(memory.research.ranges, '689001.SH').buy_shares, { low: 200, high: 200, step: null })
  assert.equal(memory.research.ranges.buy_shares.high, 100)
})

test('legacy range names do not survive the v5 migration', () => {
  const saved = v4SavedResearch()
  saved.ranges.grid_step = { low: .008, high: .024, step: .001 }
  const migrated = normalizeRememberedResearchDefaults(saved)
  assert.deepEqual(Object.keys(migrated.ranges).sort(), [...ROBUST_AXES].sort())
  assert.equal('grid_step' in migrated.ranges, false)
  assert.equal('initial_shares' in migrated.ranges, false)
  assert.equal(migrated.ranges.buy_step.step, null)
})

test('source-imported constraints are only remembered after their own edit', () => {
  const remembered = defaultWorkbenchDefaults().research
  const sourceDraft = { ...remembered, min_trades: 22, max_principal_drawdown: .31 }
  const rangeEdit = {
    ...sourceDraft,
    ranges: { ...sourceDraft.ranges, buy_step: { ...sourceDraft.ranges.buy_step, high: .12 } },
  }
  const merged = mergeEditedResearchDefaults(sourceDraft, rangeEdit, remembered)
  assert.equal(merged.min_trades, remembered.min_trades)
  assert.equal(merged.max_principal_drawdown, remembered.max_principal_drawdown)
  assert.equal(merged.ranges.buy_step.high, .12)
  const constraintEdit = { ...rangeEdit, min_trades: 35 }
  const next = mergeEditedResearchDefaults(rangeEdit, constraintEdit, merged)
  assert.equal(next.min_trades, 35)
  assert.equal(next.max_principal_drawdown, remembered.max_principal_drawdown)
})
