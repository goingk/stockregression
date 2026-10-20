import assert from 'node:assert/strict'
import test from 'node:test'
import { reportFailures, reportReason, syncReportRuns } from '../src/dataReportModel.ts'

test('historical sync reports retain per-period counts and identify the failure period once', () => {
  const failure = { symbol: '159516.SZ', reason: '[WinError 5] 拒绝访问' }
  const report = { runs: [
    { timeframe: '1day', status: 'completed', symbols: [{ symbol: '159516.SZ' }], skipped: [{ symbol: 'other', message: '已完成，跳过' }], failures: [] },
    { timeframe: '1min', status: 'partial', symbols: [], skipped: [], failures: [failure] },
  ], failures: [failure] }
  const runs = syncReportRuns(report)
  assert.equal(runs[0].successes.length, 1)
  assert.equal(runs[0].skipped.length, 1)
  assert.equal(reportReason(runs[0].skipped[0]), '已完成，跳过')
  assert.deepEqual(reportFailures(report), [{ ...failure, timeframe: '1min' }])
  assert.equal(reportReason(reportFailures(report)[0]), failure.reason)
})

test('non-sync reports and missing optional fields remain readable', () => {
  assert.deepEqual(syncReportRuns({}), [])
  assert.deepEqual(reportFailures({ failures: [{ error: 'bad file' }] }), [{ error: 'bad file' }])
  assert.equal(reportReason({ error: 'bad file' }), 'bad file')
  assert.deepEqual(syncReportRuns({ runs: [null, 'invalid', { timeframe: '1day' }] })[0].skipped, [])
})
