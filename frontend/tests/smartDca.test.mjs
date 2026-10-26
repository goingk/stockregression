import assert from 'node:assert/strict'
import test from 'node:test'
import { defaultDca, dcaError, dcaSummary, normalizeDca } from '../src/smartDca.ts'
import { defaultWorkbenchDefaults, normalizeWorkbenchDefaults, validWorkbenchGroup } from '../src/workbenchDefaults.ts'

test('legacy ordinary DCA retains its fixed-share defaults', () => {
  const legacy = { frequency: 'weekly', shares: 300, initial_shares: 100 }
  const normalized = normalizeDca(legacy)
  assert.equal(normalized.mode, 'fixed_shares')
  assert.equal(normalized.shares, 300)
  assert.equal(normalized.initial_shares, 100)
  assert.equal(dcaError(legacy), '')
  assert.equal(dcaSummary(legacy), '每周 300 股')
})

test('both smart modes and external choices survive default persistence', () => {
  for (const mode of ['ma', 'price_change']) for (const reference of ['target', 'security', 'index']) {
    const dca = { ...defaultDca(), mode, reference, base_amount: 3000, ma_window: 120, change_window: 5,
      reference_symbol: reference === 'security' ? '510300.SH' : reference === 'index' ? '000001.SH' : null,
      reference_version: reference === 'target' ? null : 'locked-v1', min_multiplier: 0, max_multiplier: 3 }
    assert.equal(dcaError(dca), '')
    assert.equal(validWorkbenchGroup('dca', dca), true)
    assert.deepEqual(normalizeWorkbenchDefaults({ ...defaultWorkbenchDefaults(), dca }).dca, dca)
    assert.match(dcaSummary(dca), /智能.*3000 元/)
  }
})

test('invalid smart parameters prevent submission and invalid defaults are rejected', () => {
  for (const patch of [
    { mode: 'other' }, { base_amount: 0 }, { sensitivity: Infinity }, { min_multiplier: -1 },
    { max_multiplier: .9 }, { ma_window: 1 }, { change_window: 1.5 },
    { reference: 'security', reference_symbol: '' },
    { reference: 'security', reference_symbol: '000001.SH' },
    { reference: 'index', reference_symbol: '000001.SZ' },
  ]) {
    const dca = { ...defaultDca(), mode: 'ma', ...patch }
    assert.notEqual(dcaError(dca), '')
    assert.equal(validWorkbenchGroup('dca', dca), false)
  }
})
