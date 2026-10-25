import assert from 'node:assert/strict'
import test from 'node:test'
import { canSubmitResearchEstimate } from '../src/researchEstimateFreshness.ts'

test('submission waits for both the newest completed estimate and page validation', () => {
  const old = { data_version: 'old' }
  const next = { data_version: 'next' }
  const validated = { status: 'ready', preview: old }
  assert.equal(canSubmitResearchEstimate({ key: 'old', status: 'ready', preview: old }, 'next', validated, true), false)
  assert.equal(canSubmitResearchEstimate({ key: 'next', status: 'loading', preview: old }, 'next', validated, true), false)
  assert.equal(canSubmitResearchEstimate({ key: 'next', status: 'ready', preview: next }, 'next', validated, true), false)
  const current = { status: 'ready', preview: next }
  const estimate = { key: 'next', status: 'ready', preview: next }
  assert.equal(canSubmitResearchEstimate(estimate, 'next', current, true), true)
  assert.equal(canSubmitResearchEstimate(estimate, 'next', current, false), false)
  assert.equal(canSubmitResearchEstimate(estimate, 'next', current, true, 'invalid account'), false)
})

test('a completed estimate without a frozen target version cannot authorize submission', () => {
  const preview = { data_version: ' ' }
  assert.equal(canSubmitResearchEstimate({ key: 'k', status: 'ready', preview }, 'k', { status: 'ready', preview }, true), false)
})
