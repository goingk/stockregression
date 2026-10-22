import type { DataPreparationRequest } from './types'

export type PreparationOrigin = 'backtest' | 'research'
export interface PreparationContext {
  request: DataPreparationRequest
  origin: PreparationOrigin
  requestId: string
  fingerprint: string
  preparationJobId?: string
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']'
  if (value && typeof value === 'object') {
    return '{' + Object.keys(value as Record<string, unknown>).sort().map(key =>
      JSON.stringify(key) + ':' + stable((value as Record<string, unknown>)[key]),
    ).join(',') + '}'
  }
  return JSON.stringify(value) ?? 'null'
}

/** Fingerprint the original request so later screen edits cannot redefine it. */
export function preparationFingerprint(request: DataPreparationRequest): string {
  let hash = 0x811c9dc5
  const text = stable(request)
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash.toString(16).padStart(8, '0')
}

export function newPreparationContext(
  request: DataPreparationRequest,
  origin: PreparationOrigin,
  createId: () => string = () => crypto.randomUUID(),
): PreparationContext {
  return {
    request: structuredClone(request), origin, requestId: createId(),
    fingerprint: preparationFingerprint(request),
  }
}
