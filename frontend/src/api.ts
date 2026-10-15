const API_ROOT = '/api/v1'

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, detail: unknown) {
    super(formatDetail(detail) || `请求失败（HTTP ${status}）`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (typeof item === 'string') return item
      if (item && typeof item === 'object') {
        const entry = item as Record<string, unknown>
        const loc = Array.isArray(entry.loc) ? `${entry.loc.join(' › ')}：` : ''
        return `${loc}${String(entry.msg ?? JSON.stringify(item))}`
      }
      return String(item)
    }).join('；')
  }
  if (detail && typeof detail === 'object') {
    const item = detail as Record<string, unknown>
    if ('message' in item || 'error' in item) return String(item.message ?? item.error)
    return JSON.stringify(detail)
  }
  return ''
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
  })
  const contentType = response.headers.get('content-type') ?? ''
  const payload: unknown = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload
      ? (payload as { detail: unknown }).detail
      : payload
    throw new ApiError(response.status, detail)
  }
  return payload as T
}

export const get = <T,>(path: string) => request<T>(path)
export const post = <T,>(path: string, body?: unknown) => request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
export const put = <T,>(path: string, body: unknown) => request<T>(path, { method: 'PUT', body: JSON.stringify(body) })
export const del = <T,>(path: string) => request<T>(path, { method: 'DELETE' })

export async function download(path: string, filename: string) {
  const response = await fetch(`${API_ROOT}${path}`)
  if (!response.ok) {
    const type = response.headers.get('content-type') ?? ''
    const body = type.includes('application/json') ? await response.json() : await response.text()
    const detail = body && typeof body === 'object' && 'detail' in body ? (body as { detail: unknown }).detail : body
    throw new ApiError(response.status, detail)
  }
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}

export function useNotice() {
  // Kept as a tiny helper for consistent API error display in pages.
  return (error: unknown) => error instanceof Error ? error.message : String(error)
}
