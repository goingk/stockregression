import { useEffect, useSyncExternalStore } from 'react'
import { get, request } from './api'
import { defaultWorkbenchDefaults, normalizeWorkbenchDefaults, validWorkbenchGroup } from './workbenchDefaults'
import type { WorkbenchDefaults, WorkbenchGroup } from './workbenchDefaults'

const CACHE_KEY = 'stockregression-workbench-defaults-v1'
const groups: WorkbenchGroup[] = ['grid', 'position', 'dca', 'research']
type Pending = Partial<WorkbenchDefaults>

function readCache(): { value: WorkbenchDefaults; pending: Pending } {
  try {
    const cached = JSON.parse(localStorage.getItem(CACHE_KEY) ?? 'null')
    if (cached?.version === 1) {
      const pending: Pending = {}
      for (const group of groups) if (validWorkbenchGroup(group, cached.pending?.[group])) Object.assign(pending, { [group]: cached.pending[group] })
      return { value: normalizeWorkbenchDefaults(cached.value), pending }
    }
  } catch { /* A missing or damaged cache uses the built-in defaults. */ }
  return { value: defaultWorkbenchDefaults(), pending: {} }
}

const cached = readCache()
let snapshot = { value: cached.value, loading: true, error: '' }
let pending = cached.pending
let initialization: Promise<void> | undefined
let saveChain: Promise<void> = Promise.resolve()
let saveTimer: ReturnType<typeof setTimeout> | undefined
const listeners = new Set<() => void>()

function publish(next: typeof snapshot) {
  snapshot = next
  for (const listener of listeners) listener()
}

function writeCache() {
  try { localStorage.setItem(CACHE_KEY, JSON.stringify({ version: 1, value: snapshot.value, pending })) }
  catch { /* The settings API remains available when browser storage is disabled. */ }
}

function flush(keepalive = false) {
  clearTimeout(saveTimer)
  saveTimer = undefined
  saveChain = saveChain.catch(() => undefined).then(async () => {
    if (!Object.keys(pending).length) return
    const patch = structuredClone(pending)
    try {
      await request('/settings', { method: 'PUT', body: JSON.stringify({ workbench_defaults: patch }), keepalive })
      for (const group of groups) {
        if (group in patch && JSON.stringify(pending[group]) === JSON.stringify(patch[group])) delete pending[group]
      }
      writeCache()
      if (snapshot.error) publish({ ...snapshot, error: '' })
    } catch (error) {
      publish({ ...snapshot, error: '上次参数暂未保存到本机设置，将在下次进入时重试：' + (error instanceof Error ? error.message : String(error)) })
    }
  })
  return saveChain
}

function scheduleSave() {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => { void flush() }, 500)
}

async function initialize() {
  if (!initialization) initialization = (async () => {
    try {
      const settings = await get<{ workbench_defaults?: unknown }>('/settings')
      const server = normalizeWorkbenchDefaults(settings.workbench_defaults)
      publish({ value: { ...server, ...pending }, loading: false, error: '' })
    } catch {
      publish({ ...snapshot, loading: false })
    }
    writeCache()
  })()
  await initialization
  if (Object.keys(pending).length) scheduleSave()
}

export function rememberWorkbenchGroup<K extends WorkbenchGroup>(group: K, value: WorkbenchDefaults[K]) {
  if (!validWorkbenchGroup(group, value)) return
  if (JSON.stringify(snapshot.value[group]) === JSON.stringify(value)) return
  const copy = structuredClone(value)
  pending = { ...pending, [group]: copy }
  publish({ ...snapshot, value: { ...snapshot.value, [group]: copy } })
  writeCache()
  scheduleSave()
}

if (typeof window !== 'undefined') window.addEventListener('pagehide', () => { void flush(true) })

export function useWorkbenchDefaults() {
  const state = useSyncExternalStore(listener => { listeners.add(listener); return () => { listeners.delete(listener) } }, () => snapshot)
  useEffect(() => {
    void initialize()
    return () => { void flush() }
  }, [])
  return state
}
