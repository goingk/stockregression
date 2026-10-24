import assert from 'node:assert/strict'
import test from 'node:test'
import { canResumeMarketSync, syncProgressLines } from '../src/types.ts'

test('sync progress shows backend phase, verified days, all actual outcome counts and resources', () => {
  assert.deepEqual(syncProgressLines({
    stage: '建立历史账本', symbol: '510300.SH', checked_days: 800, total_days: 1249,
    counts: { total: 20, processed: 8, succeeded: 3, failed: 1, reused: 4, skipped: 0, unprocessed: 12 },
    resources: { download_workers: 2, baostock_sessions: 2, write_workers: 1 },
    cutoff_date: '2026-10-23',
  }), [
    '建立历史账本 · 510300.SH 800/1249 日',
    '目标 20 · 已处理 8 · 成功 3 · 失败 1 · 复用 4 · 跳过 0 · 未处理 12',
    '下载并发 2 · BaoStock 会话 2 · 写入并发 1',
    '本次截止日 2026-10-23',
  ])
})

test('sync progress gracefully handles old jobs without progress details', () => {
  assert.deepEqual(syncProgressLines(undefined), [])
  assert.deepEqual(syncProgressLines({ stage: '下载行情' }), ['下载行情'])
  assert.deepEqual(syncProgressLines({ stage: 'history_validation', checked_days: 800, total_days: 1249 }), ['建立与核对历史账本 800/1249 日'])
})

test('only terminal market sync failures, cancellations, and partial jobs can resume', () => {
  for (const status of ['failed', 'cancelled', 'partial']) assert.equal(canResumeMarketSync({ kind: 'market_sync', status }), true)
  for (const status of ['queued', 'running', 'completed']) assert.equal(canResumeMarketSync({ kind: 'market_sync', status }), false)
  assert.equal(canResumeMarketSync({ kind: 'market_import', status: 'failed' }), false)
})
