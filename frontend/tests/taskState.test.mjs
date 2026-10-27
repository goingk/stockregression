import assert from 'node:assert/strict'
import test from 'node:test'
import {
  canResumeResearchValidation, etfResearchResultId, isEtfResearchJob, isResearchJob, mergeJobSnapshots,
  newlyNotifiableJobs, submissionJobIds, taskLabel, taskNoticeMessage, validationResumeExtension,
  validationResumeLimits, validationResumeNeedsIncrease, validationResumeReason,
} from '../src/taskState.ts'

const job = (id, status, values = {}) => ({ id, kind: 'backtest', status, progress: 0, message: '', ...values })
const ids = (jobs) => jobs.map((entry) => entry.id)

test('opening and paging history after live initialization never replays old failures', () => {
  const active = job('tracked', 'running')
  const oldFailures = [job('old-1', 'failed'), job('old-2', 'failed'), job('old-3', 'failed')]
  const next = mergeJobSnapshots([active], oldFailures)
  assert.deepEqual(newlyNotifiableJobs([active], next, true, 'history'), [])
  const finished = mergeJobSnapshots(next, [{ ...active, status: 'completed' }])
  assert.deepEqual(ids(newlyNotifiableJobs(next, finished, true, 'history')), ['tracked'])
  assert.deepEqual(newlyNotifiableJobs(finished, finished, true, 'history'), [])
})

test('failed notices show the actual import or database error', () => {
  assert.equal(taskNoticeMessage(job('import', 'failed', { message: '任务失败，查看原因后可重试', error: '行情缺少字段：timestamp' })), '行情缺少字段：timestamp')
  assert.equal(taskNoticeMessage(job('sync', 'running', { message: '1124/2762' })), '1124/2762')
})

test('research completion jobs are treated as research tasks and labelled as such', () => {
  assert.equal(isResearchJob({ kind: 'research_completion' }), true)
  assert.ok(taskLabel({ kind: 'research_completion', payload: {} }).includes('补算'))
  assert.equal(isResearchJob({ kind: 'optimization' }), true)
  assert.equal(isResearchJob({ kind: 'optimize' }), true)
  assert.equal(isResearchJob({ kind: 'backtest' }), false)
})

test('complete validation is a research task with its frozen primary symbol and a resumable terminal state', () => {
  const validation = {
    id: '', kind: 'research_validation', status: 'partial', progress: .8, message: '', created_at: '',
    payload: {
      research: { research_context: { symbol: '600000.SH' } },
      plan: { symbols: [{ symbol: '600000.SH' }, { symbol: '510300.SH' }] },
    },
  }
  assert.equal(isResearchJob(validation), true)
  assert.equal(taskLabel(validation), '完整研究验证 · 600000.SH · 2 个标的')
  assert.equal(canResumeResearchValidation(validation), true)
  assert.equal(canResumeResearchValidation({ ...validation, status: 'running' }), false)
  assert.equal(canResumeResearchValidation({ ...validation, kind: 'optimization' }), false)
})

test('validation resume shows frozen limits and accepts only increased exhausted budgets', () => {
  const partial = {
    ...job('validation-limited', 'partial', {
      payload: {
        config: { max_runtime_seconds: 3600, total_evaluation_budget: 1000 },
        plan: { estimated_work: { max_runtime_seconds: 7200, total_evaluation_budget: 1200 } },
      },
    }),
  }
  const limits = validationResumeLimits(partial)
  assert.deepEqual(limits, { max_runtime_seconds: 3600, total_evaluation_budget: 1000 })
  assert.equal(validationResumeReason(partial, { stop_reason: 'runtime_limit' }), 'runtime_limit')
  assert.equal(validationResumeNeedsIncrease('runtime_limit'), true)
  assert.match(validationResumeExtension(limits, limits, true).error, /至少提高一项/)
  assert.deepEqual(validationResumeExtension(limits, { max_runtime_seconds: 4800, total_evaluation_budget: 1000 }, true), {
    body: { max_runtime_seconds: 4800 },
  })
  assert.deepEqual(validationResumeExtension(limits, { max_runtime_seconds: 3600, total_evaluation_budget: 1500 }, true), {
    body: { total_evaluation_budget: 1500 },
  })
  assert.match(validationResumeExtension(limits, { max_runtime_seconds: 3000, total_evaluation_budget: 1500 }, true).error, /运行时间只能增加/)
  assert.match(validationResumeExtension(limits, { max_runtime_seconds: 3600, total_evaluation_budget: 999 }, true).error, /评价预算只能增加/)
})

test('cancelled validation can resume on the original limits while exhaustion is detected from plan or message', () => {
  const cancelled = job('validation-cancelled', 'cancelled', {
    payload: { config: { max_runtime_seconds: 7200, total_evaluation_budget: null }, plan: { estimated_work: { total_evaluation_budget: 4321 } } },
  })
  const limits = validationResumeLimits(cancelled)
  assert.deepEqual(limits, { max_runtime_seconds: 7200, total_evaluation_budget: 4321 })
  assert.equal(validationResumeReason(cancelled), 'cancelled')
  assert.deepEqual(validationResumeExtension(limits, limits, false), { body: undefined })
  assert.equal(validationResumeReason({ ...cancelled, status: 'partial', message: '评价预算耗尽，等待恢复' }), 'evaluation_budget')
  assert.equal(validationResumeReason({ ...cancelled, status: 'failed', message: '运行时间耗尽' }), 'runtime_limit')
  assert.equal(validationResumeReason({ ...cancelled, status: 'failed', message: '进程异常退出' }, { stop_reason: 'workflow_crash' }), 'failed')
})

test('ETF research and data-validation jobs open the ETF workspace', () => {
  assert.equal(isEtfResearchJob({ kind: 'etf_research' }), true)
  assert.equal(isEtfResearchJob({ kind: 'etf_validation' }), true)
  assert.equal(isEtfResearchJob({ kind: 'optimization' }), false)
  assert.equal(isResearchJob({ kind: 'etf_research' }), false)
  assert.equal(taskLabel({ kind: 'etf_research', payload: {} }), 'ETF研究')
  assert.equal(taskLabel({ kind: 'etf_validation', payload: {} }), 'ETF数据校验')
  assert.equal(etfResearchResultId({ kind: 'etf_research', result_id: 'etf-result' }), 'etf-result')
  assert.equal(etfResearchResultId({ kind: 'etf_validation', result_id: 'validation-result' }), 'validation-result')
  assert.equal(etfResearchResultId({ kind: 'etf_validation' }), null)
  assert.equal(etfResearchResultId({ kind: 'etf_research', result_id: undefined }), null)
})

test('first history load restores queued and running notices without replaying old terminal results', () => {
  const initial = [
    job('completed-before-load', 'completed', { progress: 1, result_id: 'old-result' }),
    job('failed-before-load', 'failed', { error: 'old error' }),
    job('cancelled-before-load', 'cancelled'),
    job('queued-at-load', 'queued'),
    job('running-at-load', 'running', { progress: .4 }),
  ]
  const history = mergeJobSnapshots([], initial)
  assert.equal(history.length, initial.length, 'terminal jobs remain available in history')
  assert.deepEqual(ids(newlyNotifiableJobs([], history, false)), ['queued-at-load', 'running-at-load'])
  assert.deepEqual(newlyNotifiableJobs(history, history, true), [], 'unchanged first-load tasks do not replay on the next poll')
})

test('progress and message updates refresh a job while only status changes create another notice', () => {
  const queued = job('tracked', 'queued', { updated_at: '2026-10-17T01:00:00Z' })
  const running = { ...queued, status: 'running', progress: .1, updated_at: '2026-10-17T01:00:01Z' }
  const started = mergeJobSnapshots([queued], [running])
  assert.deepEqual(newlyNotifiableJobs([queued], started, true), [running])

  const progressed = { ...running, progress: .8, message: '80%', updated_at: '2026-10-17T01:00:02Z' }
  const updated = mergeJobSnapshots(started, [progressed])
  assert.equal(updated[0].progress, .8)
  assert.equal(updated[0].message, '80%')
  assert.deepEqual(newlyNotifiableJobs(started, updated, true), [])

  const completed = { ...progressed, status: 'completed', progress: 1, result_id: 'new-result', updated_at: '2026-10-17T01:00:03Z' }
  const finished = mergeJobSnapshots(updated, [completed])
  assert.deepEqual(newlyNotifiableJobs(updated, finished, true), [completed])
  assert.deepEqual(newlyNotifiableJobs(finished, mergeJobSnapshots(finished, [completed]), true), [])
})

test('a new terminal job discovered after initialization is not mistaken for first-load history', () => {
  const previous = [job('already-known', 'completed')]
  const finishedBetweenPolls = job('finished-between-polls', 'completed', { progress: 1, result_id: 'available' })
  const next = mergeJobSnapshots(previous, [finishedBetweenPolls])
  assert.deepEqual(newlyNotifiableJobs(previous, next, true), [finishedBetweenPolls])
})

test('dismissing an active notice does not suppress its later completion notification', () => {
  const running = job('dismissed-notice', 'running', { progress: .2 })
  const history = mergeJobSnapshots([], [running])
  // Toast dismissal removes the notice, while task history continues to track this snapshot.
  const progressed = { ...running, progress: .9 }
  const afterDismissal = mergeJobSnapshots(history, [progressed])
  assert.deepEqual(newlyNotifiableJobs(history, afterDismissal, true), [])

  const completed = { ...progressed, status: 'completed', progress: 1, result_id: 'open-me' }
  const finished = mergeJobSnapshots(afterDismissal, [completed])
  assert.deepEqual(newlyNotifiableJobs(afterDismissal, finished, true), [completed])
})

test('submission ids combine single and batch responses once and ignore invalid entries', () => {
  assert.deepEqual(submissionJobIds({ job_id: 'first', job_ids: ['first', 'second', 'second', '', null, 2, 'third'] }), ['first', 'second', 'third'])
  assert.deepEqual(submissionJobIds({ job_ids: ['batch-a', 'batch-b', 'batch-a'] }), ['batch-a', 'batch-b'])
  assert.deepEqual(submissionJobIds({ job_id: 'single', job_ids: 'not-an-array' }), ['single'])
  for (const invalid of [null, undefined, [], 'job-id', 7, {}, { job_id: false, job_ids: [null, 0, {}] }]) assert.deepEqual(submissionJobIds(invalid), [])
})

test('snapshot merging keeps the newest update despite delayed polls and duplicate batch entries', () => {
  const newest = job('shared', 'running', { progress: .8, message: 'newest', updated_at: '2026-10-17T01:02:00Z' })
  const stale = { ...newest, progress: .3, message: 'stale', updated_at: '2026-10-17T01:01:00Z' }
  assert.deepEqual(mergeJobSnapshots([newest], [stale]), [newest])
  assert.deepEqual(mergeJobSnapshots([], [newest, stale, newest]), [newest])
  assert.deepEqual(mergeJobSnapshots([], [stale, newest]), [newest])
  assert.deepEqual(newlyNotifiableJobs([newest], mergeJobSnapshots([newest], [stale]), true), [])
  assert.equal(stale.progress, .3, 'merging does not mutate an incoming snapshot')
})

test('a terminal job never regresses to queued or running, even without timestamps or with a later active timestamp', () => {
  for (const status of ['completed', 'failed', 'cancelled']) {
    const terminal = job('finished', status, { progress: 1, updated_at: '2026-10-17T01:01:00Z' })
    for (const activeStatus of ['queued', 'running']) {
      const delayedActive = job(terminal.id, activeStatus, { updated_at: '2026-10-17T01:02:00Z' })
      const next = mergeJobSnapshots([terminal], [delayedActive])
      assert.deepEqual(next, [terminal], status + ' -> ' + activeStatus)
      assert.deepEqual(newlyNotifiableJobs([terminal], next, true), [])
      const withoutTimes = job('finished-without-time', status, { progress: 1 })
      assert.deepEqual(mergeJobSnapshots([withoutTimes], [job(withoutTimes.id, activeStatus)]), [withoutTimes])
    }
  }
})

test('merging partial polls retains other tasks and orders history by newest creation time', () => {
  const older = job('older', 'completed', { created_at: '2026-10-16T12:00:00Z' })
  const newer = job('newer', 'running', { created_at: '2026-10-17T12:00:00Z' })
  const next = mergeJobSnapshots([older], [newer])
  assert.deepEqual(ids(next), ['newer', 'older'])
  assert.deepEqual(mergeJobSnapshots(next, []), next)
})
