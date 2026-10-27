import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildValidationRequest, canSubmitValidationPlan, isResearchValidationResult, parseValidationSymbols,
  requestedVersionsFromPlan, researchResultHistoryLabel, researchV5FromDraft, validationEvidenceRecorded,
  validationOverviewItems, validationPlanSymbols, validationPrimaryArm, validationPrimaryMetrics,
  validationSchemeEntries, validationBlocks, validationAsyncTicketCurrent, validationDraftKey, validationRequestKey,
  validationAccountRows, validationSegmentGroups,
} from '../src/researchValidationModel.ts'

const draft = {
  base: {
    schema_version: 3, symbol: '600000.SH', start_date: '2022-01-04', end_date: null, frequency: '1m',
    initial_shares: 1000, min_shares: 0, max_shares: null, buy_shares: 100, sell_shares: 100,
    grid_mode: 'geometric', buy_step: .03, sell_step: .03,
    fees: { commission_rate: .0003, min_commission: .1, slippage_bps: 2 },
    reference_price_mode: 'auto', reference_price: null, lower_price: null, upper_price: null,
    dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
    parameter_links: { spacing: false, shares: false }, selection_asof: null,
  },
  protocol_version: 5,
  selection_target: 'both',
  rebalance_days: 63,
  seed: 42,
  ranges: { buy_step: { low: .01, high: .08 }, sell_step: { low: .01, high: .08 } },
  strategy: { strategy_id: 'geometric_robust_j', strategy_version: 5 },
  prior_policy: 'off',
  source_etf: { result_id: 'etf-1', candidate_ids: [7] },
}

function validInput(overrides = {}) {
  return {
    research: researchV5FromDraft(draft),
    symbols: ['600000', '510300.SH'],
    historyPolicy: 'auto',
    accounts: {
      '600000.SH': { total_budget: 200000, max_shares: 3000, contributions: [] },
      '510300.SH': {
        total_budget: 150000,
        max_shares: 2000,
        contributions: [{ available_at: '2024-06-03T09:30', amount: 20000 }],
      },
    },
    initialShares: 1000,
    planHash: 'frozen-plan-hash',
    ...overrides,
  }
}

test('current optimization-shaped draft becomes a v5 research contract without backtest-only fields', () => {
  const research = researchV5FromDraft(draft)
  assert.equal(research.research_context.symbol, '600000.SH')
  assert.equal(research.research_context.initial_shares, 1000)
  assert.equal('selection_asof' in research.research_context, false)
  assert.equal('base' in research, false)
  assert.equal(research.protocol_version, undefined)
  assert.deepEqual(research.source_etf, { result_id: 'etf-1', candidate_ids: [7] })
  assert.equal(research.prior_policy, 'off')
})

test('validation submission canonicalises the pool and keeps every account independent', () => {
  const built = buildValidationRequest(validInput())
  assert.equal(built.error, undefined)
  assert.deepEqual(built.request.symbols, ['600000.SH', '510300.SH'])
  assert.equal(built.request.suite_version, 'research_validation_v1')
  assert.equal(built.request.primary_branch, 'adaptive')
  assert.equal(built.request.plan_hash, 'frozen-plan-hash')
  assert.equal(built.request.max_runtime_seconds, 86400)
  assert.equal(built.request.accounts['600000.SH'].total_budget, 200000)
  assert.equal(built.request.accounts['510300.SH'].total_budget, 150000)
  assert.deepEqual(built.request.accounts['510300.SH'].contributions, [{ available_at: '2024-06-03T09:30', amount: 20000 }])
  assert.equal(built.request.accounts['600000.SH'].participation_rate, .01)
})

test('missing money and insufficient holding limits are rejected before any task is submitted', () => {
  const missing = buildValidationRequest(validInput({
    accounts: { '600000.SH': { total_budget: null, max_shares: 1000, contributions: [] }, '510300.SH': { total_budget: 1, max_shares: 1000, contributions: [] } },
  }))
  assert.match(missing.error, /总预算/)
  const below = buildValidationRequest(validInput({
    accounts: { '600000.SH': { total_budget: 100000, max_shares: 900, contributions: [] }, '510300.SH': { total_budget: 100000, max_shares: 1000, contributions: [] } },
  }))
  assert.match(below.error, /最高持股/)
  const blankLimit = buildValidationRequest(validInput({
    accounts: { '600000.SH': { total_budget: 100000, max_shares: null, contributions: [] }, '510300.SH': { total_budget: 100000, max_shares: 1000, contributions: [] } },
  }))
  assert.match(blankLimit.error, /最高持股/)
  const noHash = buildValidationRequest(validInput({ planHash: '', requirePlanHash: true }))
  assert.match(noHash.error, /方案检查/)
  const explicitWithoutEnd = buildValidationRequest(validInput({ historyPolicy: 'explicit' }))
  assert.match(explicitWithoutEnd.error, /结束日期/)
})

test('canonical draft fingerprints preserve an unchanged plan across parent rerenders', () => {
  const research = researchV5FromDraft(draft)
  const accounts = {
    '600000.SH': {
      total_budget: 200000,
      max_shares: 3000,
      contributions: [
        { available_at: '2024-07-01T09:30', amount: 30000 },
        { available_at: '2024-06-03T09:30', amount: 20000 },
      ],
    },
    '510300.SH': { total_budget: 150000, max_shares: 2000, contributions: [] },
  }
  const first = validationDraftKey({ research, symbols: ['600000.SH', '510300.SH'], historyPolicy: 'auto', accounts, familyId: ' study ', maxRuntimeSeconds: 86400 })
  const rebuilt = validationDraftKey({
    research: { ...structuredClone(research), configuration_revision: 'server-revision-9' },
    symbols: ['510300', '600000'], historyPolicy: 'auto',
    accounts: {
      '510300.SH': { contributions: [], max_shares: 2000, total_budget: 150000 },
      '600000.SH': {
        max_shares: 3000,
        contributions: [
          { amount: 20000, available_at: '2024-06-03T09:30' },
          { amount: 30000, available_at: '2024-07-01T09:30' },
        ],
        total_budget: 200000,
      },
    },
    familyId: 'study', maxRuntimeSeconds: 86400,
  })
  assert.equal(rebuilt, first)
  const dateChanged = validationDraftKey({
    research: { ...research, research_context: { ...research.research_context, end_date: '2024-12-31' } },
    symbols: ['600000.SH', '510300.SH'], historyPolicy: 'explicit', accounts, familyId: 'study', maxRuntimeSeconds: 86400,
  })
  assert.notEqual(dateChanged, first)
  const accountChanged = validationDraftKey({ research, symbols: ['600000.SH', '510300.SH'], historyPolicy: 'auto', accounts: { ...accounts, '510300.SH': { total_budget: 150001, max_shares: 2000, contributions: [] } }, familyId: 'study', maxRuntimeSeconds: 86400 })
  assert.notEqual(accountChanged, first)
  const versionChanged = validationDraftKey({ research: { ...research, requested_data_version: 'mkt-2026-10-27' }, symbols: ['600000.SH', '510300.SH'], historyPolicy: 'auto', accounts, familyId: 'study', maxRuntimeSeconds: 86400 })
  assert.notEqual(versionChanged, first)
})

test('request key and async revision reject stale preflight work', () => {
  const built = buildValidationRequest(validInput())
  const firstRequest = validationRequestKey(built.request, false)
  const otherHash = validationRequestKey({ ...built.request, plan_hash: 'another-plan' }, false)
  assert.equal(firstRequest, otherHash)
  const ticket = { revision: 7, phase: 'preview', draftKey: 'draft-a', requestKey: firstRequest }
  assert.equal(validationAsyncTicketCurrent(ticket, 7, 'draft-a'), true)
  assert.equal(validationAsyncTicketCurrent(ticket, 8, 'draft-a'), false)
  assert.equal(validationAsyncTicketCurrent(ticket, 7, 'draft-b'), false)
})

test('pool symbols, preflight versions and submit gate accept the current public plan shapes', () => {
  const preview = {
    plan: {
      plan_hash: 'abc123', status: 'ready', primary_arm: 'adaptive',
      symbols: [
        { symbol: '600000.SH', requested_data_version: 'mkt-600000', training_days: 410, gaps: [] },
        { symbol: '510300.SH', requested_data_version: 'mkt-510300', training_days: 380, gaps: ['2021-02-10 停牌'] },
      ],
    },
  }
  assert.deepEqual(validationPlanSymbols(preview).map(item => item.symbol), ['600000.SH', '510300.SH'])
  assert.deepEqual(requestedVersionsFromPlan(preview), { '600000.SH': 'mkt-600000', '510300.SH': 'mkt-510300' })
  assert.equal(canSubmitValidationPlan(preview), true)
  assert.equal(canSubmitValidationPlan({ plan: { ...preview.plan, status: 'blocked' } }), false)
  assert.equal(canSubmitValidationPlan({ plan: { ...preview.plan, status: 'data_issue' } }), false)
  assert.equal(canSubmitValidationPlan({ plan: { ...preview.plan, status: 'budget_exceeded' } }), false)
  assert.equal(canSubmitValidationPlan({ plan: { ...preview.plan, plan_hash: '' } }), false)
})

test('completion, candidate, budget qualification and evidence remain four independent conclusions', () => {
  const states = validationOverviewItems({
    completion_status: 'complete',
    candidate_status: 'no_qualified',
    qualification_status: 'execution_limited',
    evidence: { status: 'limited', reason: '部分标的缺少共同后续区间' },
  })
  assert.equal(states.completion.label, '验证完整')
  assert.equal(states.completion.tone, 'green')
  assert.equal(states.candidate.label, '没有合格候选')
  assert.equal(states.candidate.tone, 'amber')
  assert.equal(states.qualification.label, '执行受限')
  assert.equal(states.qualification.tone, 'amber')
  assert.equal(states.evidence.label, '证据有限')
  assert.match(states.evidence.note, /共同后续区间/)
})

test('root not_qualified and no_candidate states stay distinct from pending', () => {
  const notQualified = validationOverviewItems({
    completion_status: 'complete', candidate_status: 'not_qualified',
    qualification_status: 'no_candidate', evidence: { status: 'recorded' },
  })
  assert.equal(notQualified.candidate.label, '没有合格候选')
  assert.equal(notQualified.candidate.tone, 'amber')
  assert.equal(notQualified.qualification.label, '没有可评价候选')
  assert.equal(notQualified.qualification.tone, 'amber')
  const noCandidate = validationOverviewItems({ completion_status: 'complete', candidate_status: 'no_candidate', qualification_status: 'risk_breached' })
  assert.equal(noCandidate.candidate.label, '没有合格候选')
  assert.equal(noCandidate.qualification.label, '风险超限')
  assert.equal(noCandidate.qualification.tone, 'red')
})

test('the result distinguishes complete validation from an older v5 parameter study', () => {
  assert.equal(isResearchValidationResult({ kind: 'research_validation', suite_version: 'research_validation_v1' }), true)
  assert.equal(isResearchValidationResult({ kind: 'optimization', protocol_version: 5, result_schema_version: 2 }), false)
  assert.equal(researchResultHistoryLabel({ kind: 'optimization', title: '等比网格 v5 · 稳健 J' }), '等比网格 v5 · 稳健 J · 未完整验证')
  assert.equal(researchResultHistoryLabel({ kind: 'research_validation', title: '历史套件' }), '完整研究验证')
})

test('eight frozen schemes stay visible and evidence view recording is deduplicated by the returned audit', () => {
  const cells = Object.fromEntries(Array.from({ length: 8 }, (_, index) => [String(index), {
    window_mode: index < 4 ? 'same_price' : 'fixed', prior_policy: index % 2 ? 'daily_state_v1' : 'off',
    selection_target: index % 4 < 2 ? 'adaptive' : 'fixed', status: 'incomplete', qualification_status: 'unevaluable',
  }]))
  const schemes = validationSchemeEntries({ cells })
  assert.equal(schemes.length, 8)
  assert.match(schemes[1].label, /同价训练窗.*日线先验开启.*自适应参数/)
  assert.match(schemes[7].label, /固定训练窗.*日线先验开启.*固定参数/)
  assert.equal(validationEvidenceRecorded({ audit: { view_events: [{ section: 'overview', first_viewed_at: '2026-01-02T01:00:00' }] } }), true)
  assert.equal(validationEvidenceRecorded({ audit: { view_events: [{ section: 'blocks' }] } }), false)
  assert.equal(validationEvidenceRecorded({ audit: { evidence_views: [{ section: 'blocks' }] } }, 'blocks'), true)
  assert.equal(validationEvidenceRecorded({ evidence_views: [{ kind: 'cells' }] }, 'cells'), true)
  assert.deepEqual(parseValidationSymbols('600000, 510300.SH 600000'), { symbols: ['600000.SH', '510300.SH'], invalid: [] })
})

test('the runner result shape still exposes primary metrics, eight schemes, blocks and honest fallback states', () => {
  const arms = ['same_price:off', 'same_price:daily_state_v1', 'fixed:off', 'fixed:daily_state_v1'].map((group, index) => {
    const [window_mode, prior_policy] = group.split(':')
    return {
      arm_group: group, window_mode, prior_policy,
      schemes: {
        fixed: { status: 'completed', selected: index === 0 ? { buy_step: .01 } : null, qualification: { status: 'completed', strategy_metrics: { peak_principal_return: index ? .02 : .05, max_principal_drawdown: .1 } } },
        adaptive: { status: 'completed', schedule: [{ paused: false, effective_date: '2024-02-01' }], qualification: { status: 'completed', strategy_metrics: { peak_principal_return: .06, return_on_committed_budget: .04 } } },
      },
    }
  })
  const result = {
    kind: 'research_validation', suite_version: 'research_validation_v1', status: 'completed', completion_status: 'complete',
    primary_arm: 'same_price:off:adaptive',
    symbols: [{ symbol: '600000.SH', blocks: [{ start: '2023-01-01', end: '2023-06-30' }], arms }],
    cells: [{ kind: 'qualification', status: 'completed', result: { status: 'completed' } }],
    gss: { status: 'incomplete', historical_universe_verified: false },
    warnings: ['部分ETF缺少共同后续区间'],
    audit: { plan: { plan_hash: 'p1', primary_arm: 'same_price:off:adaptive', symbols: [{ symbol: '600000.SH', gaps: [] }] }, human_untouched_verified: false, view_events: [] },
  }
  assert.equal(validationPrimaryArm(result), '同价训练窗 · 先验关闭 · 自适应参数主方案')
  assert.equal(validationSchemeEntries(result).length, 8)
  assert.equal(validationBlocks(result)[0].symbol, '600000.SH')
  assert.equal(validationPrimaryMetrics(result).peak_principal_return, .06)
  const states = validationOverviewItems(result)
  assert.equal(states.candidate.label, '存在合格候选')
  assert.equal(states.qualification.label, '资格状态待确认')
  assert.equal(states.evidence.label, '证据有限')
  assert.deepEqual(validationPlanSymbols({ plan: result.audit.plan }).map(row => row.symbol), ['600000.SH'])
})

test('real qualification ledgers drive primary segment returns and per-symbol budget diagnostics', () => {
  const qualification = {
    status: 'completed',
    segment_returns: { segments: [{ block: 1, start: '2025-01-02', end: '2025-03-31', days: 58, principal_return: .12, max_principal_drawdown: .04, capital_added: 10000, starting_equity: 200000, ending_equity: 234000, status: 'completed' }] },
    strategy_metrics: {
      total_budget: 200000, initial_equity: 200000, initial_holdings_value: 30000, initial_cash: 170000,
      scheduled_contributions: 0, external_topups: 10000, total_invested: 210000, peak_invested: 210000,
      final_cash: 12345.67, final_equity: 234000, final_position: 2600, total_fees: 345.67,
      trade_count: 18,
      funding_diagnostics: { unspent_cash: 12345.67, cash_shortfall_action: 'cancel_remainder' },
      execution_diagnostics: {
        trade_fill_count: 18, rejection_count: 4, rejections_by_reason: { insufficient_cash: 2, limit_up_locked: 2 },
        day_end_cancel: true, participation_rate: .01, ttl_bars: 3,
      },
    },
    bounded_ledger: { trade_count: 18, rejection_count: 4, ending_cash: 12345.67, ending_equity: 234000, ending_position: 2600, ending_principal: 210000 },
  }
  const result = {
    kind: 'research_validation', primary_arm: 'same_price:off:adaptive',
    plan: { symbols: [{ symbol: '600000.SH', initial_cash: 1, blocks: [{ start: '2025-01-02', end: '2025-06-30' }] }] },
    symbols: [{ symbol: '600000.SH', blocks: [{ start: '2025-01-02', end: '2025-06-30' }], arms: [{
      arm_group: 'same_price:off', window_mode: 'same_price', prior_policy: 'off', primary: true,
      schemes: {
        adaptive: { qualification },
        fixed: { qualification: { segment_returns: { segments: [{ block: 1, start: '2025-01-02', end: '2025-03-31', principal_return: .08, status: 'completed' }] } } },
      },
    }] }],
  }
  const segments = validationSegmentGroups(result)
  assert.equal(segments.primary.length, 1)
  assert.equal(segments.primary[0].segments[0].principal_return, .12)
  assert.equal(segments.alternatives.length, 1)
  assert.match(segments.alternatives[0].label, /固定参数/)
  const accounts = validationAccountRows(result)
  assert.equal(accounts.length, 1)
  assert.equal(accounts[0].total_budget, 200000)
  assert.equal(accounts[0].total_invested, 210000)
  assert.equal(accounts[0].final_cash, 12345.67)
  assert.equal(accounts[0].fill_ratio, 18 / 22)
  assert.equal(accounts[0].insufficient_cash_rejections, 2)
  assert.equal(accounts[0].cancelled_orders, '日内剩余委托撤销')
  assert.equal(accounts[0].cash_shortfall_action, 'cancel_remainder')
  assert.equal(validationBlocks(result)[0].symbol, '600000.SH', 'plan dates remain available only as fallback')
})
