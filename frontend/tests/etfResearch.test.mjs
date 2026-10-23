import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildEtfResearchRequest, dataStatusLabel, etfDataStatusCounts, etfMetricRows, etfMetricText,
  etfSymbolValidationPath, executionError, executionFromBacktest, filterReasonLabel, filterReasonsText,
  gapText, isEtfResearchResult, isEtfValidationResult, itemsPageCount, parseSelectedEtfSymbols,
  resultStatusLabel, safeItemsOffset, scoreDisplay, settingsError, stopReasonLabel,
} from '../src/etfResearch.ts'

const config = {
  schema_version: 3,
  symbol: '510210.SH',
  start_date: '2024-01-02',
  end_date: '2024-06-28',
  frequency: '1m',
  initial_shares: 2000,
  min_shares: 0,
  max_shares: 8000,
  buy_shares: 300,
  sell_shares: 400,
  fees: {
    commission_rate: .0003,
    min_commission: .2,
    slippage_bps: 8,
    share_face_value: 1,
    legacy_sh_transfer_rate: .001,
    legacy_sh_min_transfer: 1,
    legacy_sz_transfer_rate: .0000255,
  },
}

const settings = {
  scope: 'selected',
  symbolsText: '510210, 159611.SZ 510210.SH',
  as_of: '',
  window_days: 120,
  validation_top_k: 10,
  min_trades: 10,
  max_principal_drawdown: .2,
  max_runtime_seconds: 3600,
}

test('ETF execution copies the current share and fee draft without legacy cash fields', () => {
  const execution = executionFromBacktest(config)
  assert.equal(execution.initial_shares, 2000)
  assert.equal(execution.buy_shares, 300)
  assert.equal(execution.fees.commission_rate, .0003)
  assert.equal(execution.fees.share_face_value, 1)
  assert.equal(execution.fees.legacy_sh_transfer_rate, .001)
  assert.equal(execution.fees.legacy_sh_min_transfer, 1)
  assert.equal(execution.fees.legacy_sz_transfer_rate, .0000255)
  assert.equal('initial_cash' in execution, false)
  assert.equal('symbol' in execution, false)
  assert.equal(executionError(execution), '')
  execution.fees.commission_rate = 0
  assert.equal(config.fees.commission_rate, .0003)
})

test('ETF execution rejects invalid lots and holding bounds', () => {
  const execution = executionFromBacktest(config)
  assert.match(executionError({ ...execution, buy_shares: 350 }), /100 的整数倍/)
  assert.match(executionError({ ...execution, min_shares: 2500 }), /最低持股不能高于初始持股/)
  assert.match(executionError({ ...execution, max_shares: 1000 }), /初始持股不能高于最高持股/)
  assert.match(executionError({ ...execution, fees: { ...execution.fees, slippage_bps: -1 } }), /滑点/)
  assert.match(executionError({ ...execution, fees: { ...execution.fees, share_face_value: 0 } }), /股票面值/)
})

test('selected ETF symbols are normalized and deduplicated', () => {
  assert.deepEqual(parseSelectedEtfSymbols(settings.symbolsText), {
    symbols: ['510210.SH', '159611.SZ'], invalid: [],
  })
  assert.deepEqual(parseSelectedEtfSymbols('600000.SH 510210.SZ bad'), {
    symbols: [], invalid: ['600000.SH', '510210.SZ', 'bad'],
  })
})

test('ETF requests carry the fixed execution envelope and no research-v5 seed', () => {
  const execution = executionFromBacktest(config)
  const request = buildEtfResearchRequest(settings, execution, 'config-revision-1')
  assert.equal(request.as_of, null)
  assert.deepEqual(request.symbols, ['510210.SH', '159611.SZ'])
  assert.equal(request.configuration_revision, 'config-revision-1')
  assert.deepEqual(Object.keys(request.execution).sort(), ['buy_shares', 'fees', 'initial_shares', 'max_shares', 'min_shares', 'sell_shares'])
  assert.equal('ranges' in request, false)
  assert.equal('protocol_version' in request, false)
})

test('single-symbol validation targets the saved result without rebuilding a latest request', () => {
  assert.equal(
    etfSymbolValidationPath('result/one', '510210.SH'),
    '/etf-research/result%2Fone/validate/510210.SH',
  )
})

test('ETF settings require a selected symbol while watchlist and local scopes do not', () => {
  const execution = executionFromBacktest(config)
  assert.match(settingsError({ ...settings, symbolsText: '' }, execution), /至少需要一只 ETF/)
  assert.equal(settingsError({ ...settings, scope: 'watchlist', symbolsText: '' }, execution), '')
  assert.equal(settingsError({ ...settings, validation_top_k: 0 }, execution), '')
  assert.match(settingsError({ ...settings, validation_top_k: 101 }, execution), /0–100/)
  assert.match(settingsError({ ...settings, min_trades: 1_000_001 }, execution), /0–1000000/)
  assert.match(settingsError({ ...settings, max_runtime_seconds: 86401 }, execution), /1–86400/)
})

test('metric rows preserve declared sampling labels, values and missing reasons', () => {
  const rows = etfMetricRows(
    { rv_annualized: .285, path_redundancy: .95, variance_ratio_240m: .82, ou_half_life_days: 14.5 },
    { rv_annualized: { status: 'ok', sample_count: 120 }, ou_half_life_days: { status: 'insufficient', reason: '样本不足' } },
  )
  assert.equal(rows.find(row => row.key === 'rv_annualized').value, '28.50%')
  assert.equal(rows.find(row => row.key === 'rv_annualized').detail.sample_count, 120)
  assert.equal(rows.find(row => row.key === 'ou_half_life_days').value, '14.5 日')
  assert.equal(rows.find(row => row.key === 'ou_half_life_days').detail.reason, '样本不足')
  assert.equal(etfMetricText('missing', null), '—')
})

test('single-symbol cohorts display no invented score and pagination stays aligned', () => {
  assert.equal(scoreDisplay({ gss_score: null, score_cohort_size: 0 }), '评分样本不足')
  assert.equal(scoreDisplay({ gss_score: null, score_cohort_size: 1 }), '单标的无横截面分数')
  assert.equal(scoreDisplay({ gss_score: 84.2, score_cohort_size: 12 }), '84.20')
  assert.equal(itemsPageCount(0, 50), 1)
  assert.equal(itemsPageCount(101, 50), 3)
  assert.equal(safeItemsOffset(149, 101, 50), 100)
  assert.equal(dataStatusLabel('pending_validation'), '待校验')
  assert.equal(resultStatusLabel('failed'), '失败')
})

test('saved rows translate filter reasons, data statuses, gaps and stop reasons', () => {
  assert.equal(filterReasonLabel('ou_invalid'), 'OU 半衰期无效或不可估计')
  assert.equal(filterReasonLabel('variance_ratio_240m_gt_1.10'), '240 分钟方差比高于 1.10')
  assert.equal(filterReasonLabel('status_unavailable'), '统计计算不可用')
  assert.equal(filterReasonsText(['ou_invalid', 'mean_raw_price_lt_0.5']), 'OU 半衰期无效或不可估计；平均不复权价格低于 0.5 元')
  assert.equal(dataStatusLabel('unavailable'), '统计不可用')
  assert.equal(dataStatusLabel('pending'), '待计算')
  assert.equal(gapText({ timeframe: '1min', kind: 'quality', message: '缺少分钟行情' }), '1min · quality：缺少分钟行情')
  assert.equal(stopReasonLabel('runtime_limit'), '达到最长运行时间')
  assert.equal(stopReasonLabel('workspace_limit'), '工作内存达到预算')
})

test('result counts separate ranking outcomes from dynamic data-status counts', () => {
  const counts = etfDataStatusCounts({
    total: 10, scored: 7, eligible: 4, validated: 3, recommended: 2,
    ready: 5, pending_validation: 2, insufficient_history: 1, data_not_ready: 1, unavailable: 1,
  })
  assert.deepEqual(counts.map(item => [item.status, item.label, item.count]), [
    ['ready', '数据就绪', 5],
    ['pending_validation', '待校验', 2],
    ['insufficient_history', '历史不足', 1],
    ['data_not_ready', '数据未就绪', 1],
    ['unavailable', '统计不可用', 1],
  ])
})

test('ETF result kinds keep validation reports out of ranking rendering', () => {
  assert.equal(isEtfValidationResult({ kind: 'etf_validation' }), true)
  assert.equal(isEtfValidationResult({ kind: 'etf_research' }), false)
  assert.equal(isEtfResearchResult({ kind: 'etf_research' }), true)
  assert.equal(isEtfResearchResult({ kind: 'etf_validation' }), false)
  assert.equal(isEtfValidationResult(null), false)
})
