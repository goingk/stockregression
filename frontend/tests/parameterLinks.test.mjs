import assert from 'node:assert/strict'
import test from 'node:test'
import { setParameterLink, linkResearchRanges, syncLinkedResearchRange } from '../src/parameterLinks.ts'
import { defaultWorkbenchDefaults, normalizeWorkbenchDefaults, strategyDefaults, workbenchConfig } from '../src/workbenchDefaults.ts'

const config = {
  buy_step: .02, sell_step: .04, buy_shares: 300, sell_shares: 700,
  parameter_links: { spacing: false, shares: false },
}

test('enabling either parameter link copies buy values to sell and preserves four axes', () => {
  const spacing = setParameterLink(config, 'spacing', true)
  assert.equal(spacing.sell_step, .02)
  assert.deepEqual(spacing.parameter_links, { spacing: true, shares: false })
  const shares = setParameterLink(config, 'shares', true)
  assert.equal(shares.sell_shares, 300)
  assert.deepEqual(Object.keys({ buy_step: 1, sell_step: 1, buy_shares: 1, sell_shares: 1 }), ['buy_step', 'sell_step', 'buy_shares', 'sell_shares'])
})

test('linked research ranges mirror both values and edits', () => {
  const ranges = {
    buy_step: { low: .01, high: .05 }, sell_step: { low: .02, high: .08 },
    buy_shares: { low: 100, high: 1000 }, sell_shares: { low: 200, high: 2000 },
  }
  const linked = linkResearchRanges(ranges, { spacing: true, shares: true })
  assert.deepEqual(linked.sell_step, ranges.buy_step)
  assert.deepEqual(linked.sell_shares, ranges.buy_shares)
  const edited = syncLinkedResearchRange(linked, 'buy_step', { low: .03, high: .07 }, { spacing: true, shares: false })
  assert.deepEqual(edited.sell_step, { low: .03, high: .07 })
})

test('personal link defaults persist under grid and position, and old documents default to false', () => {
  const defaults = defaultWorkbenchDefaults()
  assert.equal(defaults.grid.link_spacing, false)
  assert.equal(defaults.position.link_shares, false)
  const legacy = structuredClone(defaults)
  delete legacy.grid.link_spacing
  delete legacy.position.link_shares
  const normalized = normalizeWorkbenchDefaults(legacy)
  assert.equal(normalized.grid.link_spacing, false)
  assert.equal(normalized.position.link_shares, false)
  const linkedConfig = { ...workbenchConfig({ ...config, symbol: '600000.SH' }, { ...normalized, grid: { ...normalized.grid, link_spacing: true }, position: { ...normalized.position, link_shares: true } }) }
  assert.deepEqual(linkedConfig.parameter_links, { spacing: true, shares: true })
  assert.deepEqual(strategyDefaults(linkedConfig).grid.link_spacing, true)
  assert.deepEqual(strategyDefaults(linkedConfig).position.link_shares, true)
})
