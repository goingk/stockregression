import assert from 'node:assert/strict'
import test from 'node:test'
import { formatPValue, reviewBootstrapEvidence } from '../src/researchEvidence.ts'

const primary = { requested_reps: 10000, completed_reps: 10000, alpha: .05, p_value: .014, selected_simultaneous_lower_log_excess: .0002 }

test('insufficient-history bootstrap payload is not presented as completed evidence', () => {
  const review = reviewBootstrapEvidence({
    status: 'insufficient_history',
    bootstrap: { primary },
  })
  assert.equal(review.state, 'insufficient')
  assert.equal(review.completed, false)
})

test('family-level result needs a completed primary test, significant p-value, and positive adjusted lower bound', () => {
  assert.equal(reviewBootstrapEvidence({ status: 'completed', bootstrap: { primary } }).state, 'positive')
  assert.equal(reviewBootstrapEvidence({ status: 'completed', bootstrap: { primary: { ...primary, p_value: .08 } } }).state, 'not_positive')
  assert.equal(reviewBootstrapEvidence({ status: 'completed', bootstrap: { primary: { ...primary, selected_simultaneous_lower_log_excess: 0 } } }).state, 'not_positive')
  assert.equal(reviewBootstrapEvidence({ status: 'completed', bootstrap: { primary: { ...primary, completed_reps: 9999 } } }).state, 'insufficient')
})

test('p-values retain probability notation rather than percentage formatting', () => {
  assert.equal(formatPValue(.014), '0.0140')
  assert.equal(formatPValue(.000001), '<0.0001')
  assert.equal(formatPValue(null), '—')
})
