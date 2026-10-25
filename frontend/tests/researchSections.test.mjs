import assert from 'node:assert/strict'
import test from 'node:test'
import {
  defaultResearchSectionOrder,
  dropResearchSection,
  isResearchSectionOrder,
  moveResearchSection,
  normalizeResearchSectionOrder,
  researchSectionIds,
} from '../src/researchSections.ts'

test('research sections have one complete default order', () => {
  assert.deepEqual(defaultResearchSectionOrder, [...researchSectionIds])
  assert.equal(new Set(researchSectionIds).size, researchSectionIds.length)
  assert.equal(isResearchSectionOrder(defaultResearchSectionOrder), true)
  assert.equal(isResearchSectionOrder(defaultResearchSectionOrder.slice(1)), false)
  assert.equal(isResearchSectionOrder([...defaultResearchSectionOrder, 'unknown']), false)
  assert.equal(isResearchSectionOrder([...defaultResearchSectionOrder.slice(1), researchSectionIds[0]]), true)
  assert.equal(isResearchSectionOrder([...defaultResearchSectionOrder, researchSectionIds[0]]), false)
})

test('stored section order keeps known ids once and appends newly introduced sections', () => {
  const normalized = normalizeResearchSectionOrder([
    'research-fees', 'unknown', 'research-fees', 'research-basics', 7,
  ])
  assert.deepEqual(normalized.slice(0, 2), ['research-fees', 'research-basics'])
  assert.equal(normalized.length, researchSectionIds.length)
  assert.deepEqual(new Set(normalized), new Set(researchSectionIds))
  assert.deepEqual(normalizeResearchSectionOrder(null), [...researchSectionIds])
  assert.deepEqual(normalizeResearchSectionOrder({ invalid: true }), [...researchSectionIds])
})

test('moving a section stops at the boundaries and respects the requested step', () => {
  const moved = moveResearchSection(defaultResearchSectionOrder, 'research-fees', -1)
  assert.deepEqual(moved.slice(0, 3), ['research-basics', 'research-fees', 'research-price-bounds'])
  assert.deepEqual(moveResearchSection(defaultResearchSectionOrder, 'research-basics', -1), [...defaultResearchSectionOrder])
  assert.deepEqual(moveResearchSection(defaultResearchSectionOrder, 'research-constraints', 1), [...defaultResearchSectionOrder])
  assert.deepEqual(moveResearchSection(defaultResearchSectionOrder, 'research-fees', 0), [...defaultResearchSectionOrder])
})

test('drag drops insert before or after the target', () => {
  const movedBefore = dropResearchSection(defaultResearchSectionOrder, 'research-fees', 'research-basics', false)
  assert.deepEqual(movedBefore.slice(0, 3), ['research-fees', 'research-basics', 'research-price-bounds'])
  const movedAfter = dropResearchSection(defaultResearchSectionOrder, 'research-basics', 'research-fees', true)
  assert.deepEqual(movedAfter.slice(0, 3), ['research-price-bounds', 'research-fees', 'research-basics'])
  assert.deepEqual(dropResearchSection(defaultResearchSectionOrder, 'research-fees', 'research-fees', true), [...defaultResearchSectionOrder])
  assert.deepEqual(dropResearchSection(defaultResearchSectionOrder, 'unknown', 'research-fees', true), [...defaultResearchSectionOrder])
})

test('restoring the default order only depends on the layout defaults', () => {
  const customized = moveResearchSection(defaultResearchSectionOrder, 'research-constraints', -4)
  assert.notDeepEqual(customized, defaultResearchSectionOrder)
  assert.deepEqual(
    normalizeResearchSectionOrder(defaultResearchSectionOrder, customized),
    [...researchSectionIds],
  )
})
