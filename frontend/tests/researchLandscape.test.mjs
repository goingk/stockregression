import assert from 'node:assert/strict'
import test from 'node:test'
import { canonicalLandscapePoint, chooseLandscapeAxes, landscapeSliceKey, parameterSpaceAxes, revisionMatchesScope } from '../src/landscapeData.ts'

function point(params) {
  return { params, candidate_id: 1, is_center: false, is_baseline: false, is_focus: false, status: 'completed', excess_return: 0, grid_return: 0, baseline_return: null, drawdown: 0, trade_count: 1, peak_invested: null, neighbor_candidate_ids: [], neighbor_of_candidate_ids: [], robustness_status: null, neighbor_positive_ratio: null, neighbor_q25_excess: null, island_suspect: false }
}

const fourAxes = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares']

test('v5 uses the two spacing axes and never repeats one parameter', () => {
  const points = [
    point({ buy_step: .01, sell_step: .02, buy_shares: 500, sell_shares: 500 }),
    point({ buy_step: .02, sell_step: .03, buy_shares: 600, sell_shares: 700 }),
  ]
  assert.deepEqual(chooseLandscapeAxes(fourAxes, points, '', ''), ['buy_step', 'sell_step'])
  assert.deepEqual(chooseLandscapeAxes(fourAxes, points, 'sell_step', 'sell_step'), ['buy_step', 'sell_step'])
})

test('a valid custom pair is preserved', () => {
  const points = [point({ buy_step: .01, sell_step: .02, buy_shares: 500, sell_shares: 500 })]
  assert.deepEqual(chooseLandscapeAxes(fourAxes, points, 'buy_shares', 'sell_shares'), ['buy_shares', 'sell_shares'])
  assert.deepEqual(chooseLandscapeAxes(fourAxes, points, 'buy_shares', 'buy_shares'), ['buy_step', 'sell_step'])
})

test('share axes are used when the spacing axes are unavailable', () => {
  const points = [
    point({ buy_shares: 500, sell_shares: 500 }),
    point({ buy_shares: 600, sell_shares: 700 }),
  ]
  assert.deepEqual(chooseLandscapeAxes(['buy_shares', 'sell_shares'], points, '', ''), ['sell_shares', 'buy_shares'])
})

test('slice keys are order independent', () => {
  assert.equal(landscapeSliceKey('buy_step', 'sell_step'), landscapeSliceKey('sell_step', 'buy_step'))
})

test('a revision is only sent for the decision branch it covers', () => {
  const scope = { revision: 3, decision: 'current', decision_index: 0 }
  assert.equal(revisionMatchesScope(scope, 'current', 0), true)
  assert.equal(revisionMatchesScope(scope, 'fixed', 0), false)
  assert.equal(revisionMatchesScope(scope, 'adaptive', 0), false)
  assert.equal(revisionMatchesScope(scope, 'current', 1), false)
})

test('without a scope no revision may be applied', () => {
  assert.equal(revisionMatchesScope(null, 'current', 0), false)
  assert.equal(revisionMatchesScope(undefined, 'current', 0), false)
})

test('an adaptive revision only covers its own decision index', () => {
  const scope = { revision: 2, decision: 'adaptive', decision_index: 1 }
  assert.equal(revisionMatchesScope(scope, 'adaptive', 1), true)
  assert.equal(revisionMatchesScope(scope, 'adaptive', 0), false)
})

test('linked candidate fields collapse to metadata active canonical axes', () => {
  const space = { version: 1, parameter_links: { spacing: true, shares: false }, independent_axes: [
    { id: 'buy_step', members: ['buy_step', 'sell_step'] },
    { id: 'buy_shares', members: ['buy_shares'] },
    { id: 'sell_shares', members: ['sell_shares'] },
  ], active_axes: ['buy_step', 'buy_shares', 'sell_shares'] }
  assert.deepEqual(parameterSpaceAxes(fourAxes, space), space.active_axes)
  const normalized = canonicalLandscapePoint(point({ buy_step: .025, sell_step: .025, buy_shares: 500, sell_shares: 500 }), space)
  assert.equal(normalized.params.sell_step, .025)
})

test('legacy landscape metadata derives distinct parameters without assuming links', () => {
  assert.deepEqual(parameterSpaceAxes(fourAxes), fourAxes)
  assert.deepEqual(parameterSpaceAxes(fourAxes, { version: 1, parameter_links: { spacing: false, shares: false }, independent_axes: [], active_axes: [] }), [])
  assert.deepEqual(parameterSpaceAxes(['buy_step', 'sell_step'], { version: 1, parameter_links: { spacing: true, shares: true }, independent_axes: [{ id: 'buy_step', members: ['buy_step', 'sell_step'] }], active_axes: ['buy_step'] }), ['buy_step'])
})
