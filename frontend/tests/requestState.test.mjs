import assert from 'node:assert/strict'
import test from 'node:test'

import { LatestRequest, PromiseCache } from '../src/requestState.ts'

test('only the latest request for the active key may update the view', () => {
  const latest = new LatestRequest()
  const first = latest.begin('A')
  const second = latest.begin('B')

  assert.equal(latest.isCurrent(first), false)
  assert.equal(latest.isCurrent(second), true)
  latest.invalidate('A')
  assert.equal(latest.isCurrent(second), true)
  latest.invalidate('B')
  assert.equal(latest.isCurrent(second), false)
})

test('a failed read is removed from the cache and can be retried', async () => {
  const cache = new PromiseCache()
  let calls = 0
  const load = () => cache.getOrCreate('result', async () => {
    calls += 1
    if (calls === 1) throw new Error('temporary')
    return 'ok'
  })

  await assert.rejects(load(), /temporary/)
  assert.equal(cache.get('result'), undefined)
  assert.equal(await load(), 'ok')
  assert.equal(await load(), 'ok')
  assert.equal(calls, 2)
})

test('a stale rejection never removes a newer cached request', async () => {
  const cache = new PromiseCache()
  let rejectFirst
  const first = cache.getOrCreate('result', () => new Promise((_, reject) => { rejectFirst = reject }))
  cache.forget('result', first)
  const second = cache.getOrCreate('result', async () => 'new')

  rejectFirst(new Error('old failure'))
  await assert.rejects(first, /old failure/)
  assert.equal(cache.get('result'), second)
  assert.equal(await second, 'new')
})
