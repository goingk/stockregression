import assert from 'node:assert/strict'
import test from 'node:test'
import { amplitudeBucketCount, buildAmplitudeHistogram } from '../src/amplitudeMath.ts'

const sample = (amplitude, upside = amplitude, downside = amplitude) => ({
  date: '2026-01-01',
  previous_close: 10,
  high: 10 + amplitude * 10,
  low: 10 - amplitude * 10,
  amplitude,
  upside,
  downside,
})

test('bucket count follows ceil(sqrt(n)) and is clamped to 5–12', () => {
  assert.equal(amplitudeBucketCount(0), 5)
  assert.equal(amplitudeBucketCount(1), 5)
  assert.equal(amplitudeBucketCount(20), 5)
  assert.equal(amplitudeBucketCount(60), 8)
  assert.equal(amplitudeBucketCount(250), 12)
})

test('an ordinary distribution shares edges and counts every metric', () => {
  const samples = [sample(0), sample(0.009), sample(0.01), sample(0.025), sample(0.05)]
  const histogram = buildAmplitudeHistogram(samples, 5)

  assert.equal(histogram.upperBound, 0.05)
  assert.equal(histogram.edges.length, 6)
  ;[0, 0.01, 0.02, 0.03, 0.04, 0.05].forEach((edge, index) => {
    assert.ok(Math.abs(histogram.edges[index] - edge) < 1e-12)
  })
  assert.deepEqual(histogram.bins.amplitude.map(bin => bin.count), [2, 1, 1, 0, 1])
  assert.deepEqual(histogram.bins.upside.map(bin => bin.count), [2, 1, 1, 0, 1])
  assert.equal(histogram.totals.amplitude, samples.length)
})

test('the shared upper edge is included in the final bucket', () => {
  const histogram = buildAmplitudeHistogram([sample(0.02), sample(0.1)], 5)
  const last = histogram.bins.amplitude.at(-1)

  assert.equal(last.end, 0.1)
  assert.equal(last.includeUpper, true)
  assert.equal(last.count, 1)
  assert.equal(histogram.bins.amplitude.reduce((total, bin) => total + bin.count, 0), 2)
})

test('all-zero samples stay representable in the final bucket', () => {
  const histogram = buildAmplitudeHistogram([sample(0), sample(0), sample(0)], 5)

  assert.equal(histogram.allZero, true)
  assert.equal(histogram.upperBound, 0)
  assert.deepEqual(histogram.bins.amplitude.map(bin => bin.count), [0, 0, 0, 0, 3])
  assert.deepEqual(histogram.bins.amplitude.at(-1).start, 0)
  assert.deepEqual(histogram.bins.amplitude.at(-1).end, 0)
  assert.equal(histogram.bins.amplitude.at(-1).share, 1)
})

test('a single sample uses five buckets and lands in the final one', () => {
  const histogram = buildAmplitudeHistogram([sample(0.03)], amplitudeBucketCount(1))

  assert.equal(histogram.bucketCount, 5)
  assert.deepEqual(histogram.bins.amplitude.map(bin => bin.count), [0, 0, 0, 0, 1])
  assert.equal(histogram.bins.amplitude.at(-1).share, 1)
})
