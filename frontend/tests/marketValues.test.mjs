import assert from 'node:assert/strict'
import test from 'node:test'
import { chartValue, priceText, validPrice, validVolume } from '../src/marketValues.ts'

test('invalid prices never become zero or a formatted number', () => {
  for (const value of [0, -1, null, undefined, Number.NaN, Number.POSITIVE_INFINITY, '']) {
    assert.equal(validPrice(value), null)
    assert.equal(chartValue(value), '-')
    assert.equal(priceText(value, 3), '—')
  }
  assert.equal(validPrice('1.039'), 1.039)
  assert.equal(priceText('1.039', 3), '1.039')
})

test('zero volume is valid while negative or missing volume is a gap', () => {
  assert.equal(validVolume(0), 0)
  assert.equal(validVolume(undefined), null)
  assert.equal(validVolume(-1), null)
})
