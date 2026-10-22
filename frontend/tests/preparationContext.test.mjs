import test from 'node:test'
import assert from 'node:assert/strict'
import { newPreparationContext, preparationFingerprint } from '../src/preparationContext.ts'

test('preparation fingerprints ignore object key order and retain request identity', () => {
  const first = { symbol: '600000.SH', config: { symbol: '600000.SH', start_date: '2024-01-01' } }
  const second = { config: { start_date: '2024-01-01', symbol: '600000.SH' }, symbol: '600000.SH' }
  assert.equal(preparationFingerprint(first), preparationFingerprint(second))
  const context = newPreparationContext(first, 'research', () => 'request-1')
  assert.equal(context.requestId, 'request-1')
  assert.equal(context.origin, 'research')
  assert.equal(context.fingerprint, preparationFingerprint(first))
  first.config.start_date = '2025-01-01'
  assert.equal(context.request.config.start_date, '2024-01-01')
})
