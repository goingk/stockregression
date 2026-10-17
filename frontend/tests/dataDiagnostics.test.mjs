import assert from 'node:assert/strict'
import test from 'node:test'
import { actualRepairMinutes, coverageDayHasIssue, diagnosticsStatusLabel, effectiveMinuteCompletenessLabel, sourceSpanLabel, uniqueCurrentWarnings, uniqueSourceEvents } from '../src/dataDiagnostics.ts'

test('source spans identify manual imports only from an explicit import source kind', () => {
  assert.equal(sourceSpanLabel({ source: 'local-file', source_kind: 'import' }), '本地文件导入')
  assert.equal(sourceSpanLabel({ source: 'import', source_kind: 'unknown' }), '来源类型未识别 · import')
  assert.equal(sourceSpanLabel({ source: 'hf', source_kind: 'api' }), '在线接口 · Hugging Face')
  assert.equal(sourceSpanLabel({ source: 'sina', source_kind: 'provider' }), '在线接口 · 新浪')
  assert.equal(sourceSpanLabel({ source: 'tdx', source_kind: 'provider' }), '在线接口 · 通达信')
})

test('current warnings are deduplicated independently from historical source events', () => {
  const warnings = uniqueCurrentWarnings(['仍缺少 2 个分钟交易日'], ['仍缺少 2 个分钟交易日', '待校准'])
  const events = uniqueSourceEvents(
    [{ kind: 'request_failure', source: 'hf', status: 'resolved', message: '请求失败后改用本地数据' }],
    [{ kind: 'request_failure', source: 'hf', status: 'resolved', message: '请求失败后改用本地数据' }],
  )
  assert.deepEqual(warnings, ['仍缺少 2 个分钟交易日', '待校准'])
  assert.equal(events.length, 1)
  assert.equal(events[0].status, 'resolved')
})

test('diagnostic status labels reflect the local validation state', () => {
  assert.equal(diagnosticsStatusLabel('complete'), '已完整核验')
  assert.equal(diagnosticsStatusLabel('partial'), '部分核验')
  assert.equal(diagnosticsStatusLabel('unavailable'), '暂无可核验数据')
})

test('auction placeholders are excluded from actual repaired minute counts', () => {
  assert.equal(actualRepairMinutes(14, 14), 0)
  assert.equal(actualRepairMinutes(21, 14), 7)
  assert.equal(actualRepairMinutes(2, 14), 0)
  assert.equal(actualRepairMinutes(14, undefined), null)
})

test('effective minute completeness uses specific diagnoses instead of labeling every incomplete state as a gap', () => {
  assert.equal(effectiveMinuteCompletenessLabel({ raw_complete: true, missing_raw_minutes: 0 }), '已完整')
  assert.equal(effectiveMinuteCompletenessLabel({ raw_complete: false, pending_calibration_days: 2 }), '待校准')
  assert.equal(effectiveMinuteCompletenessLabel({ raw_complete: false, missing_raw_minutes: 3 }), '存在原始缺口')
  assert.equal(effectiveMinuteCompletenessLabel({ raw_complete: false, daily_days: 4 }), '有日线降级')
  assert.equal(effectiveMinuteCompletenessLabel({ raw_complete: false }), '待完善')
  assert.equal(effectiveMinuteCompletenessLabel({}), '状态未提供')
})

test('coverage issue filter excludes pure auction placeholders but retains unknown reasons and real repairs', () => {
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_minutes: 2, missing_data_minutes: 0, auction_placeholders: 2, reason: '缺失2分钟' }), false)
  assert.equal(coverageDayHasIssue({ mode: 'complete', missing_data_minutes: 0, auction_placeholders: 2, reason: '14:58、14:59收盘集合竞价零量占位' }), false)
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_data_minutes: 0, auction_placeholders: 2, reason: '分钟源校验失败' }), true)
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_data_minutes: 1, auction_placeholders: 2 }), true)
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_minutes: 2 }), true)
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_data_minutes: 0, reason: '因果修复4根分钟条' }), true)
  assert.equal(coverageDayHasIssue({ mode: 'daily', missing_data_minutes: 0 }), true)
  assert.equal(coverageDayHasIssue({ mode: 'suspended', missing_data_minutes: 0 }), true)
  assert.equal(coverageDayHasIssue({ mode: 'minute', missing_data_minutes: 0, reason: '完整' }), false)
})
