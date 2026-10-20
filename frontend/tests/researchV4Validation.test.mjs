import assert from 'node:assert/strict'
import test from 'node:test'
import { v4ValidationLoadInfo } from '../src/researchV4Validation.ts'

test('new v4 diagnostics expose their exact validation window for region loading', () => {
  assert.deepEqual(v4ValidationLoadInfo({
    validation_account_mode: 'fresh_initial_config',
    validation_start: '2025-04-01',
    validation_end: '2025-12-31',
  }), { kind: 'fresh', dateRange: { start_date: '2025-04-01', end_date: '2025-12-31' } })
})

test('legacy v4 diagnostics without the account-mode marker keep legacy loading behavior', () => {
  assert.deepEqual(v4ValidationLoadInfo({ validation_start: '2025-04-01', validation_end: '2025-12-31' }), { kind: 'legacy' })
  assert.deepEqual(v4ValidationLoadInfo(undefined), { kind: 'legacy' })
})

test('marked v4 diagnostics with missing or invalid dates do not invent a validation range', () => {
  assert.deepEqual(v4ValidationLoadInfo({ validation_account_mode: 'fresh_initial_config', validation_start: '2025-04-01' }), { kind: 'incomplete' })
  assert.deepEqual(v4ValidationLoadInfo({ validation_account_mode: 'fresh_initial_config', validation_start: '2025-12-31', validation_end: '2025-04-01' }), { kind: 'incomplete' })
})
