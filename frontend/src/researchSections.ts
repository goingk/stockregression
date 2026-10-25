export const researchSectionIds = [
  'research-basics',
  'research-price-bounds',
  'research-fees',
  'research-dca',
  'research-plan',
  'research-search-ranges',
  'research-robust-j',
  'research-robust-search',
  'research-robust-window',
  'research-constraints',
] as const

export type ResearchSectionId = typeof researchSectionIds[number]
export const defaultResearchSectionOrder: ResearchSectionId[] = [...researchSectionIds]
export const RESEARCH_SECTION_ORDER_KEY = 'stockregression-ui:research-section-order-v1'

/** True when every available section appears exactly once and no unknown id leaks in. */
export function isResearchSectionOrder(
  value: unknown,
  available: readonly string[] = researchSectionIds,
): value is string[] {
  if (!Array.isArray(value) || value.length !== available.length) return false
  const seen = new Set<string>()
  for (const raw of value) {
    if (typeof raw !== 'string' || seen.has(raw) || !available.includes(raw)) return false
    seen.add(raw)
  }
  return seen.size === available.length
}

/** Keep known ids once, then append any newly introduced sections. */
export function normalizeResearchSectionOrder(
  value: unknown,
  available: readonly string[] = researchSectionIds,
): string[] {
  const allowed = new Set(available)
  const seen = new Set<string>()
  const result: string[] = []
  if (Array.isArray(value)) {
    for (const raw of value) {
      const id = String(raw)
      if (allowed.has(id) && !seen.has(id)) {
        result.push(id)
        seen.add(id)
      }
    }
  }
  for (const id of available) {
    if (!seen.has(id)) result.push(id)
  }
  return result
}

export function moveResearchSection(
  order: readonly string[],
  id: string,
  delta: number,
  available: readonly string[] = researchSectionIds,
): string[] {
  const current = normalizeResearchSectionOrder(order, available)
  const index = current.indexOf(id)
  if (index < 0 || !Number.isFinite(delta) || delta === 0) return current
  const next = Math.max(0, Math.min(current.length - 1, index + Math.trunc(delta)))
  if (next === index) return current
  const [item] = current.splice(index, 1)
  current.splice(next, 0, item)
  return current
}

/** Move one section before or after another section. */
export function dropResearchSection(
  order: readonly string[],
  source: string,
  target: string,
  after: boolean,
  available: readonly string[] = researchSectionIds,
): string[] {
  const current = normalizeResearchSectionOrder(order, available)
  if (source === target || !current.includes(source) || !current.includes(target)) return current
  const next = current.filter(id => id !== source)
  const targetIndex = next.indexOf(target)
  next.splice(targetIndex + (after ? 1 : 0), 0, source)
  return normalizeResearchSectionOrder(next, available)
}
