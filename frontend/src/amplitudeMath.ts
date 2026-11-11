import type { AmplitudeMetricKey, AmplitudeSample } from './types'

export const AMPLITUDE_METRICS: readonly AmplitudeMetricKey[] = ['amplitude', 'upside', 'downside']
export const AMPLITUDE_BUCKET_MIN = 5
export const AMPLITUDE_BUCKET_MAX = 12

export interface AmplitudeBucket {
  index: number
  start: number
  end: number
  includeUpper: boolean
  count: number
  share: number
}

export interface AmplitudeHistogram {
  bucketCount: number
  upperBound: number
  edges: number[]
  bins: Record<AmplitudeMetricKey, AmplitudeBucket[]>
  totals: Record<AmplitudeMetricKey, number>
  allZero: boolean
}

/** Choose the shared histogram bucket count from the response's sample count. */
export function amplitudeBucketCount(sampleCount: number): number {
  const count = Number.isFinite(sampleCount) ? Math.max(0, Math.floor(sampleCount)) : 0
  return Math.max(AMPLITUDE_BUCKET_MIN, Math.min(AMPLITUDE_BUCKET_MAX, Math.ceil(Math.sqrt(count))))
}

function clampBucketCount(bucketCount: number): number {
  const count = Number.isFinite(bucketCount) ? Math.floor(bucketCount) : AMPLITUDE_BUCKET_MIN
  return Math.max(AMPLITUDE_BUCKET_MIN, Math.min(AMPLITUDE_BUCKET_MAX, count))
}

function bucketPosition(value: number, upperBound: number, bucketCount: number): number {
  if (upperBound <= 0) return bucketCount - 1
  if (value >= upperBound) return bucketCount - 1
  const scaled = (value * bucketCount) / upperBound
  const nearestEdge = Math.round(scaled)
  const stableScaled = Math.abs(scaled - nearestEdge) < 1e-12 ? nearestEdge : scaled
  return Math.max(0, Math.min(bucketCount - 1, Math.floor(stableScaled)))
}

/**
 * Build one shared edge set for all three amplitude metrics. The upper edge is
 * inclusive in the final bucket; zero-only samples occupy that final bucket so
 * the display can still report their frequency.
 */
export function buildAmplitudeHistogram(
  samples: readonly AmplitudeSample[],
  requestedBucketCount = amplitudeBucketCount(samples.length),
): AmplitudeHistogram {
  const bucketCount = clampBucketCount(requestedBucketCount)
  let upperBound = 0

  for (const sample of samples) {
    for (const metric of AMPLITUDE_METRICS) {
      const value = sample[metric]
      if (Number.isFinite(value) && value >= 0 && value > upperBound) upperBound = value
    }
  }

  const edges = Array.from({ length: bucketCount + 1 }, (_, index) =>
    index === bucketCount ? upperBound : (upperBound * index) / bucketCount,
  )
  const bins = {} as Record<AmplitudeMetricKey, AmplitudeBucket[]>
  const totals = {} as Record<AmplitudeMetricKey, number>

  for (const metric of AMPLITUDE_METRICS) {
    const counts = Array.from({ length: bucketCount }, () => 0)
    let total = 0

    for (const sample of samples) {
      const value = sample[metric]
      if (!Number.isFinite(value) || value < 0) continue
      counts[bucketPosition(value, upperBound, bucketCount)] += 1
      total += 1
    }

    totals[metric] = total
    bins[metric] = counts.map((count, index) => ({
      index,
      start: edges[index],
      end: edges[index + 1],
      includeUpper: index === bucketCount - 1,
      count,
      share: total ? count / total : 0,
    }))
  }

  return { bucketCount, upperBound, edges, bins, totals, allZero: upperBound === 0 }
}
