import assert from 'node:assert/strict'
import test from 'node:test'
import { chooseLandscapeAxes, landscapeSliceKey } from '../src/landscapeData.ts'

function point(params) {
  return { params, candidate_id: 1, is_center: false, is_baseline: false, is_focus: false, status: 'completed', excess_return: 0, grid_return: 0, baseline_return: null, drawdown: 0, trade_count: 1, peak_invested: null, neighbor_candidate_ids: [], neighbor_of_candidate_ids: [], robustness_status: null, neighbor_positive_ratio: null, neighbor_q25_excess: null, island_suspect: false, folds: [] }
}

test('v4 axes use distinct varying dimensions instead of selecting the same parameter twice', () => {
  const points = [
    point({ buy_step: .01, sell_step: .02, sell_shares: 500, buy_shares: 500, min_shares: 0 }),
    point({ buy_step: .02, sell_step: .03, sell_shares: 700, buy_shares: 600, min_shares: 500 }),
  ]
  const parameters = ['buy_step', 'sell_step', 'min_shares', 'buy_shares', 'sell_shares']
  assert.deepEqual(chooseLandscapeAxes(parameters, points, '', ''), ['buy_step', 'sell_step'])
  assert.deepEqual(chooseLandscapeAxes(parameters, points, 'sell_step', 'sell_step'), ['buy_step', 'sell_step'])
})

test('axes avoid a fixed slice dimension and preserve a valid custom pair', () => {
  const points = [
    point({ grid_step: .01, sell_shares: 500, buy_shares: 500, min_shares: 0 }),
    point({ grid_step: .01, sell_shares: 700, buy_shares: 600, min_shares: 500 }),
  ]
  assert.deepEqual(chooseLandscapeAxes(['grid_step', 'min_shares', 'buy_shares', 'sell_shares'], points, 'grid_step', 'sell_shares'), ['grid_step', 'sell_shares'])
  assert.deepEqual(chooseLandscapeAxes(['grid_step', 'min_shares', 'buy_shares', 'sell_shares'], points, 'grid_step', 'grid_step'), ['min_shares', 'buy_shares'])
  assert.deepEqual(chooseLandscapeAxes(['grid_step', 'sell_shares'], points, 'sell_shares', 'grid_step'), ['sell_shares', 'grid_step'])
})

test('slice keys are order independent', () => {
  assert.equal(landscapeSliceKey('buy_step', 'sell_step'), landscapeSliceKey('sell_step', 'buy_step'))
})
