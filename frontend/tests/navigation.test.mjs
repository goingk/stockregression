import test from 'node:test'
import assert from 'node:assert/strict'
import { navigationFromHistory, navigationFromUrl, navigationHistoryState, navigationUrl } from '../src/navigation.ts'

test('copied URLs restore the page, selections and relevant filters', () => {
  const url = navigationUrl('', {
    page: 'results', result: 'abc123', compare: 'def456', research: null, report: null,
    resultView: 'month', researchDecision: null, taskFilter: 'failed', taskKind: 'optimization',
  })
  assert.equal(url, '?page=results&result=abc123&compare=def456&result_view=month&task_filter=failed&task_kind=optimization')
  assert.deepEqual(navigationFromUrl(url), {
    page: 'results', result: 'abc123', compare: 'def456', research: null, report: null,
    resultView: 'month', researchDecision: null, taskFilter: 'failed', taskKind: 'optimization',
  })
})

test('a selection is omitted from pages that cannot consume it', () => {
  const url = navigationUrl('', {
    page: 'backtest', result: 'abc', compare: 'def', research: 'ghi', report: 'jkl',
    resultView: 'year', researchDecision: 'adaptive:2', taskFilter: 'all', taskKind: '',
  })
  assert.equal(url, '?page=backtest')
  assert.equal(navigationFromUrl(url).result, null)
})

test('unknown pages and malformed ids fail closed', () => {
  const state = navigationFromUrl('?page=unknown&result=../../secret&compare=%20&task_kind=Not Kind')
  assert.equal(state.page, 'data')
  assert.equal(state.result, null)
  assert.equal(state.compare, null)
  assert.equal(state.taskKind, '')
})

test('History API state is validated before it is applied', () => {
  const state = navigationFromUrl('?page=research&research=result-1&research_decision=adaptive%3A1')
  assert.deepEqual(navigationFromHistory(navigationHistoryState(state)), state)
  assert.equal(navigationFromHistory({ stockregression: { page: 'results', result: '../bad' } }).result, null)
})
