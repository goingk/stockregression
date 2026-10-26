import assert from 'node:assert/strict'
import test from 'node:test'
import {
  archivedResearchSource, clearResearchDraftSource, dateNextDay, dateYearsAgo, defaultResearchDraftContext, etfSourceCandidateIds,
  isMissingResearchSourceError, requestedDataVersionFromPreview, researchDraftFromEtfSource, resolveAfterCutoffEnd,
  researchSourceRequestFields, setResearchPriorPolicy, shanghaiDateToday, sourceResearchDraftContext,
} from '../src/researchDraft.ts'

const settings = {
  strategy_id: 'geometric_robust_j', protocol_version: 5, strategy_version: 5, ranking_mode: 'robust_j',
  selection_target: 'both', rebalance_days: 63, seed: 42, top_candidates: 20,
  max_principal_drawdown: .2, min_trades: 10, min_history_days: 250, min_minute_ratio: .95,
  hypercube_edge: 3, granularity: { buy_step: .0001, sell_step: .0001, buy_shares: 100, sell_shares: 100 },
  search_method: 'auto', training_budget: 21000, validation_budget: 101000, reporting_budget: 1200,
  total_evaluation_budget: null, max_runtime_seconds: 7200, recommendation_count: 10,
  max_shares_mode: 'unlimited', slippage_mode: 'configured', neighborhood_fields: ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'],
  fixed_half_widths: { buy_step: .001, sell_step: .001, buy_shares: 100, sell_shares: 100 },
  sigma: 1, alpha: .3, beta: 1.2, tau: .01, eta_limit: 2, eta_hard_cap: 3.5, gamma_max: .4,
  eta_floor: 0, gamma_min: 0, floor_k_ref: 10, training_min_days: 250, training_max_days: 500,
  training_window_mode: 'same_price', peak_tolerance: 1e-12, start_min: 20, start_max: 64,
  starts_per_candidate: 4, initial_strides: [1, 2], stride_multiplier: 2, max_stride_fraction: .25,
  validation_reserve_fraction: .25, exhaustive_memory_limit_mb: 512, estimated_bytes_per_point: 512,
  stress_cost_multiplier: 2, stress_region_count: 3,
  ranges: {
    buy_step: { low: .006, high: .08 }, sell_step: { low: .007, high: .09 },
    buy_shares: { low: 200, high: 1200 }, sell_shares: { low: 100, high: 900 },
  },
}

const current = {
  schema_version: 3, symbol: '600000.SH', start_date: '2023-01-01', end_date: '2024-01-01',
  frequency: '1m', initial_shares: 1000, min_shares: 0, max_shares: 3000,
  buy_shares: 200, sell_shares: 300, dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
  reference_price_mode: 'manual', reference_price: 9.5, lower_price: 8, upper_price: 11,
  selection_asof: '2024-01-01', grid_mode: 'geometric', buy_step: .02, sell_step: .03,
  fees: { commission_rate: .0003, min_commission: .1, slippage_bps: 2 },
}

const candidate = (id, qualified) => ({
  id, base_step: .02, buy_step: .02, sell_step: .03, reference_price: 1, lower_price: .8, upper_price: 1.2,
  grids_down_count: 10, grids_up_count: 7, asymmetric: false, metrics: {}, qualified, reasons: [], replay_config: current,
})

const guidance = {
  method_version: 'etf_parameter_guidance_v1', as_of: '2026-02-28', data_version: 'mkt-source-version', status: 'ready',
  ranges: { buy_step: { low: .01, high: .05 }, sell_step: { low: .01, high: .05 } },
  granularity: { buy_step: .002, sell_step: .002, buy_shares: 100, sell_shares: 100 },
  tau: { value: .016, sensitivity: [.008, .016, .024], reference_days: 250 },
}

test('applying historical hints carries ranges, granularity and tau into a future draft only', () => {
  const { result, row } = sourceFixture()
  row.parameter_guidance = guidance
  const input = { result, row, current, settings, apply_guidance: true }
  assert.match(researchDraftFromEtfSource(input).error, /截止日之后/)
  const { draft } = researchDraftFromEtfSource({ ...input, range_mode: 'after_cutoff' })
  assert.equal(draft.config.start_date, '2026-03-01')
  assert.equal(draft.config.end_date, null)
  assert.equal(draft.settings.tau, .016)
  assert.deepEqual(draft.settings.ranges.buy_step, guidance.ranges.buy_step)
  assert.deepEqual(draft.settings.granularity, guidance.granularity)
  assert.deepEqual(draft.settings.ranges.buy_shares, settings.ranges.buy_shares)
  assert.equal(draft.settings.training_budget, settings.training_budget)
  assert.equal(draft.context.source_summary.applied_guidance.as_of, guidance.as_of)
  assert.equal(researchSourceRequestFields(draft.context).source_etf.apply_parameter_guidance, true)
  assert.equal(settings.tau, .01)
  assert.deepEqual(settings.ranges.buy_step, { low: .006, high: .08 })
})

test('stale, missing and malformed historical guidance cannot be applied', () => {
  for (const patch of [null, { status: 'unavailable' }, { data_version: 'other' }, { as_of: '2026-03-01' }, { tau: { value: NaN } }, { granularity: { ...guidance.granularity, buy_step: .00015 } }]) {
    const { result, row } = sourceFixture()
    row.parameter_guidance = patch === null ? null : { ...guidance, ...patch }
    const converted = researchDraftFromEtfSource({ result, row, current, settings, apply_guidance: true, range_mode: 'after_cutoff' })
    assert.ok(converted.error)
    assert.equal(converted.draft, undefined)
  }
})

function sourceFixture() {
  return {
    result: {
      id: 'etf-result-1', kind: 'etf_research', method_version: 'etf_min_grid_v3', result_schema_version: 1,
      status: 'completed', as_of: '2026-02-28', ranking_complete: true, rows_count: 1, warnings: [],
      counts: { total: 1, scored: 1, eligible: 0, validated: 1, recommended: 0, ready: 1, pending_validation: 0, insufficient_history: 0, data_not_ready: 0 },
      universe: { package_id: 'package', generation: 'generation', versions: {} },
      request: {
        scope: 'selected', symbols: ['510210.SH'], as_of: '2026-02-28', window_days: 120, validation_top_k: 1,
        execution: {
          initial_shares: 10000, min_shares: 0, max_shares: 30000, buy_shares: 1200, sell_shares: 1000,
          fees: { commission_rate: .0008, min_commission: .2, slippage_bps: 7 },
        },
        min_trades: 22, max_principal_drawdown: .31, max_runtime_seconds: 3600,
      },
    },
    row: {
      symbol: '510210.SH', symbol_name: '上证 ETF', status: 'completed', data_status: 'limited',
      version: 'mkt-source-version', window_start: '2025-12-01', window_end: '2026-02-28',
      metrics: {}, metric_details: {}, gaps: [], eligible: false, filter_reasons: ['variance_ratio_240m_gt_1.10'],
      warnings: [], diagnostics: {}, grid_basis: {}, gss_score: 71.25, rank: null, score_cohort_size: 5,
      score_contributions: {},
      validation: {
        status: 'no_qualified', candidates: [candidate(7, false), candidate('7', false), candidate(9, true)],
        recommendation: null, forward_config: null,
      },
    },
  }
}

test('source drafts use the cutoff date and preserve research workspace settings', () => {
  const { result, row } = sourceFixture()
  const converted = researchDraftFromEtfSource({ result, row, current, settings })
  assert.equal(converted.error, undefined)
  const draft = converted.draft
  assert.ok(draft)
  assert.equal(draft.config.symbol, '510210.SH')
  assert.equal(draft.config.start_date, '2025-02-28')
  assert.equal(draft.config.end_date, '2026-02-28')
  assert.equal(draft.config.initial_shares, 10000)
  assert.equal(draft.config.fees.commission_rate, .0008)
  assert.equal(draft.config.fees.min_commission, .2)
  assert.equal(draft.config.fees.slippage_bps, 7)
  assert.equal(draft.config.reference_price_mode, 'auto')
  assert.equal(draft.config.reference_price, null)
  assert.equal(draft.config.lower_price, null)
  assert.equal(draft.config.upper_price, null)
  assert.equal(draft.config.selection_asof, null)
  assert.equal(draft.settings.min_trades, 22)
  assert.equal(draft.settings.max_principal_drawdown, .31)
  assert.equal(draft.settings.training_budget, 21000)
  assert.deepEqual(draft.settings.ranges, settings.ranges)
  assert.notEqual(draft.settings.ranges, settings.ranges)
  assert.deepEqual(draft.context.source_etf, { result_id: 'etf-result-1', candidate_ids: [9] })
  assert.equal(draft.context.prior_policy, 'daily_state_v1')
  assert.equal(draft.context.source_summary?.candidate_count, 3)
  assert.equal(draft.context.source_summary?.qualified_count, 1)
  assert.equal(draft.context.source_summary?.data_version, 'mkt-source-version')
  assert.equal(draft.context.source_summary?.gss_score, 71.25)
  assert.equal(draft.context.source_summary?.score_cohort_size, 5)
  assert.equal(draft.context.source_summary?.data_status, 'limited')
  assert.equal(draft.context.source_summary?.eligible, false)
  assert.equal(draft.context.source_summary?.validation_status, 'no_qualified')
  assert.equal('as_of' in draft.context.source_etf, false)
})

test('limited, filtered and no-qualified rows still create a draft', () => {
  const { result, row } = sourceFixture()
  row.data_status = 'data_not_ready'
  row.eligible = false
  row.validation = { status: 'unavailable', candidates: [], recommendation: null, forward_config: null }
  const converted = researchDraftFromEtfSource({ result, row, current, settings })
  assert.ok(converted.draft)
  assert.deepEqual(converted.draft.context.source_etf, { result_id: 'etf-result-1' })
  assert.equal(converted.draft.context.source_summary?.candidate_count, 0)
})

test('ordinary drafts default to off and clearing a source leaves no residual identity', () => {
  assert.deepEqual(defaultResearchDraftContext(), { source_etf: null, source_summary: null, prior_policy: 'off' })
  const sourced = sourceResearchDraftContext({ result_id: 'r1', candidate_ids: ['1', '1', 2] })
  assert.deepEqual(sourced.source_etf, { result_id: 'r1', candidate_ids: [1, 2] })
  assert.equal(sourced.prior_policy, 'daily_state_v1')
  assert.equal(setResearchPriorPolicy(sourced, 'off').prior_policy, 'off')
  assert.deepEqual(clearResearchDraftSource(), { source_etf: null, source_summary: null, prior_policy: 'off' })
  assert.equal(setResearchPriorPolicy(defaultResearchDraftContext(), 'daily_state_v1').prior_policy, 'daily_state_v1')
  const enabledIndependent = setResearchPriorPolicy(defaultResearchDraftContext(), 'daily_state_v1')
  assert.deepEqual(researchSourceRequestFields(enabledIndependent), { source_etf: null, prior_policy: 'daily_state_v1' })
  assert.deepEqual(clearResearchDraftSource(), { source_etf: null, source_summary: null, prior_policy: 'off' })
  assert.equal(enabledIndependent.source_etf, null)
})

test('calendar-year subtraction clamps leap day', () => {
  assert.equal(dateYearsAgo('2024-02-29'), '2023-02-28')
  assert.equal(dateYearsAgo('2028-02-29'), '2027-02-28')
  assert.equal(dateYearsAgo('2025-03-01'), '2024-03-01')
  assert.equal(dateYearsAgo('not-a-date'), '')
})

test('after-cutoff drafts wait for the target trading calendar to resolve their end date', () => {
  const { result, row } = sourceFixture()
  const converted = researchDraftFromEtfSource({
    result, row, current, settings, range_mode: 'after_cutoff', shanghai_today: '2026-03-10',
  })
  assert.ok(converted.draft)
  assert.equal(converted.draft.config.start_date, '2026-03-01')
  assert.equal(converted.draft.config.end_date, null)
  assert.equal(converted.draft.context.prior_policy, 'daily_state_v1')
  const noLaterData = researchDraftFromEtfSource({
    result, row, current, settings, range_mode: 'after_cutoff', shanghai_today: '2026-02-15',
  })
  assert.equal(noLaterData.draft.config.start_date, '2026-03-01')
  assert.equal(noLaterData.draft.config.end_date, null)
  assert.equal(resolveAfterCutoffEnd('2026-03-01', { calendar_available: true, last_date: '2026-03-06' }), '2026-03-06')
  assert.equal(resolveAfterCutoffEnd('2026-03-01', { calendar_available: false, last_date: '2026-03-06' }), null)
  assert.equal(resolveAfterCutoffEnd('2026-03-01', { calendar_available: true, last_date: '2026-02-27' }), null)
})

test('next-day and Shanghai-date helpers cross month and timezone boundaries', () => {
  assert.equal(dateNextDay('2024-02-28'), '2024-02-29')
  assert.equal(dateNextDay('2024-02-29'), '2024-03-01')
  assert.equal(dateNextDay('2026-12-31'), '2027-01-01')
  assert.equal(dateNextDay('bad'), '')
  assert.equal(shanghaiDateToday(new Date('2026-10-24T17:00:00Z')), '2026-10-25')
  assert.equal(shanghaiDateToday(new Date('2026-10-25T15:59:59Z')), '2026-10-25')
})

test('submission data version comes only from the latest preview', () => {
  assert.equal(requestedDataVersionFromPreview({ data_version: 'preview-target', source_etf: { data_version: 'source-version' } }), 'preview-target')
  assert.equal(requestedDataVersionFromPreview({ source_etf: { data_version: 'source-version' } }), undefined)
  assert.equal(requestedDataVersionFromPreview({ data_version: '  ' }), undefined)
})

test('archived results restore their frozen ETF source from research_origin', () => {
  const result = {
    research_origin: {
      etf_result_id: 'etf-archived', symbol: '510050.SH', source_as_of: '2025-01-03',
      source_data_version: 'source-retired', method_version: 'etf_min_grid_v4', gss_score: 62.5,
      seeds: [{ candidate_id: 8, params: {} }, { candidate_id: 3, params: {} }, { candidate_id: 8, params: {} }],
    },
    config: { prior_policy: 'off', research_origin: { etf_result_id: 'stale-nested', symbol: '000001.SZ' } },
  }
  const restored = archivedResearchSource(result)
  assert.deepEqual(restored?.source_etf, { result_id: 'etf-archived', candidate_ids: [8, 3] })
  assert.equal(restored?.prior_policy, 'off')
  assert.equal(restored?.source_summary?.symbol, '510050.SH')
  assert.equal(restored?.source_summary?.as_of, '2025-01-03')
  assert.equal(restored?.source_summary?.data_version, 'source-retired')
  assert.equal(restored?.source_summary?.gss_score, 62.5)
  assert.equal(restored?.source_summary?.qualified_count, 2)
  assert.equal(restored?.source_summary?.archived, true)
})

test('archived source falls back to config.research_origin and the source prior default', () => {
  const restored = archivedResearchSource({
    config: {
      research_origin: {
        etf_result_id: 'legacy-archived', symbol: '159611.SZ', source_as_of: '2024-12-31', seeds: [],
      },
    },
  })
  assert.deepEqual(restored?.source_etf, { result_id: 'legacy-archived' })
  assert.equal(restored?.prior_policy, 'daily_state_v1')
  assert.equal(restored?.source_summary?.validation_status, 'no_qualified')
  assert.match(restored?.source_summary?.validation_reason ?? '', /没有入选候选/)
  assert.equal(archivedResearchSource({ research_origin: null, config: {} }), null)
})

test('missing source errors remain distinguishable and only qualified candidates seed a source', () => {
  assert.equal(isMissingResearchSourceError({ status: 404, detail: { code: 'SOURCE_ETF_NOT_FOUND', message: 'missing' } }), true)
  assert.equal(isMissingResearchSourceError({ status: 422, detail: { message: '来源 ETF 研究结果不存在' } }), true)
  assert.equal(isMissingResearchSourceError({ status: 404, message: '来源 ETF 研究结果不可读取，请解除来源后继续独立研究' }), true)
  assert.equal(isMissingResearchSourceError({ status: 404, detail: { code: 'RESEARCH_SOURCE_INVALID', message: '来源 ETF 研究结果不存在' } }), true)
  assert.equal(isMissingResearchSourceError({ status: 404, detail: { message: 'result not found' } }), false)
  assert.deepEqual(etfSourceCandidateIds(sourceFixture().row), [9])
})

test('ETF source candidates are recommendation-first qualified and capped at three', () => {
  const { row } = sourceFixture()
  row.validation.candidates = [candidate(2, true), candidate(4, true), candidate(5, true), candidate(9, true), candidate(12, false)]
  row.validation.recommendation = row.validation.candidates[2]
  assert.deepEqual(etfSourceCandidateIds(row), [5, 2, 4])
})

test('invalid ETF symbols are rejected without rebinding a source', () => {
  const { result, row } = sourceFixture()
  row.symbol = 'not-a-symbol'
  const converted = researchDraftFromEtfSource({ result, row, current, settings })
  assert.match(converted.error, /有效/)
  assert.equal(converted.draft, undefined)
})
