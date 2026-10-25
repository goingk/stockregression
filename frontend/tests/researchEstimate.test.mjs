import assert from 'node:assert/strict'
import test from 'node:test'
import { researchEstimateKey } from '../src/researchEstimateKey.ts'
import {
  normalizeResearchEstimateResponse,
  researchEstimateCleanupPath,
  researchEstimateFailure,
  researchEstimateSubmission,
  tagResearchEstimateKey,
} from '../src/researchEstimateLifecycle.ts'

const request = () => ({
  base: { symbol: '600000.SH', start_date: '2024-01-01', end_date: '2025-01-01', frequency: '1d', selection_asof: '2024-01-02', parameter_links: { spacing: false, shares: true },
    reference_price: 10, fees: { commission_rate: .001 }, dca: { shares: 100 } },
  ranges: { buy_step: { low: .01, high: .03 } }, selection_target: 'excess_return', rebalance_days: 5,
  seed: 7, top_candidates: 20, source_etf: '510300.SH', prior_policy: 'conservative', requested_data_version: 'v1',
  strategy: { granularity: .01, training_window_mode: 'rolling', training_budget: 10, validation_budget: 5, reporting_budget: 2,
    total_evaluation_budget: 17, search_method: 'adaptive', hypercube_edge: 3, exhaustive_memory_limit_mb: 128,
    estimated_bytes_per_point: 2048, stress_region_count: 4, alpha: .1, beta: .2, sigma: .3 },
})

test('estimate identity ignores output-only knobs and execution costs', () => {
  const original = request()
  const changed = structuredClone(original)
  changed.seed = 99
  changed.top_candidates = 3
  changed.base.fees.commission_rate = .002
  changed.base.dca.shares = 200
  changed.strategy.alpha = 8
  changed.strategy.beta = 9
  changed.strategy.sigma = 10
  assert.equal(researchEstimateKey(original), researchEstimateKey(changed))
})

test('estimate identity changes with data, linked axes, source and resource inputs', () => {
  const original = request()
  for (const change of [
    value => { value.base.start_date = '2023-01-01' },
    value => { value.base.parameter_links.spacing = true },
    value => { value.ranges.buy_step.high = .04 },
    value => { value.strategy.training_budget += 1 },
    value => { value.strategy.estimated_bytes_per_point += 1 },
    value => { value.strategy.stress_region_count += 1 },
    value => { value.source_etf = '510500.SH' },
    value => { value.requested_data_version = 'v2' },
  ]) {
    const changed = structuredClone(original)
    change(changed)
    assert.notEqual(researchEstimateKey(original), researchEstimateKey(changed))
  }
  assert.notEqual(researchEstimateKey(original, 10), researchEstimateKey(original, 20))
})

test('research estimate dispatch uses the latest request snapshot with its active revision', () => {
  const latestRequest = { base: { symbol: '000001.SZ' }, strategy: { training_budget: 22 }, fees: { commission_rate: .004 } }
  assert.deepEqual(researchEstimateSubmission('session/a', 8, latestRequest, 'stable-key'), {
    session_id: 'session/a', revision: 8, config: latestRequest, estimate_key: 'stable-key',
  })
  assert.equal(researchEstimateCleanupPath('session/a', 8), '/optimizations/preview-sessions/session%2Fa?through_revision=8')
})

test('estimate lifecycle normalizes manager completion result and retains diagnostic detail', () => {
  const ready = normalizeResearchEstimateResponse({ status: 'completed', result: { status: 'ready', preview: { estimate: 12 }, message: 'ok' } })
  assert.deepEqual(ready, { status: 'ready', preview: { estimate: 12 }, message: 'ok' })

  const blocked = normalizeResearchEstimateResponse({ status: 'completed', result: {
    status: 'blocked', reasons: ['样本数据缺失'], detail: { code: 'DATA_NOT_READY', gaps: ['x'] },
  } })
  assert.equal(blocked.status, 'blocked')
  assert.equal(blocked.message, '样本数据缺失')
  assert.deepEqual(blocked.error, { code: 'DATA_NOT_READY', gaps: ['x'] })

  for (const innerStatus of ['data_issue', 'insufficient_history', 'insufficient_range', 'budget_exceeded', 'no_qualified']) {
    const terminal = normalizeResearchEstimateResponse({ status: 'completed', result: { status: innerStatus, reasons: [`${innerStatus} reason`] } })
    assert.equal(terminal.status, 'blocked', innerStatus)
    assert.equal(terminal.message, `${innerStatus} reason`, innerStatus)
  }

  const direct = normalizeResearchEstimateResponse({ status: 'ready', preview: { estimate: 15 }, message: 'direct' })
  assert.deepEqual(direct, { status: 'ready', preview: { estimate: 15 }, message: 'direct' })
  assert.equal(normalizeResearchEstimateResponse({ status: 'running' }), null)
  assert.deepEqual(normalizeResearchEstimateResponse({ status: 'cancelled', message: 'cancelled by manager' }), {
    status: 'blocked', preview: { status: 'cancelled', message: 'cancelled by manager' }, message: 'cancelled by manager',
  })

  const tagged = tagResearchEstimateKey(direct, 'key-for-completed-input')
  assert.equal(tagged.key, 'key-for-completed-input')
  assert.equal(tagged.status, 'ready')
  assert.equal(tagged.key === 'new-current-key', false)
  assert.deepEqual(researchEstimateFailure('failed-key', 'request failed', 503, { code: 'UPSTREAM', message: 'diagnostic' }), {
    status: 'blocked', preview: null, message: 'request failed', key: 'failed-key',
    error: { status: 503, detail: { code: 'UPSTREAM', message: 'diagnostic' } },
  })
})
