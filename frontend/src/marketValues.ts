export function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function validPrice(value: unknown): number | null {
  const parsed = finiteNumber(value)
  return parsed != null && parsed > 0 ? parsed : null
}

export function validVolume(value: unknown): number | null {
  const parsed = finiteNumber(value)
  return parsed != null && parsed >= 0 ? parsed : null
}

export function chartValue(value: unknown): number | '-' {
  return validPrice(value) ?? '-'
}

export function priceText(value: unknown, digits: number): string {
  const parsed = validPrice(value)
  return parsed == null ? '—' : parsed.toFixed(digits)
}
