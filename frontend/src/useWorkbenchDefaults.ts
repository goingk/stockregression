import { useEffect, useSyncExternalStore } from 'react'
import { ApiError, get, request } from './api'
import { defaultWorkbenchDefaults, normalizeWorkbenchDefaults, validWorkbenchGroup } from './workbenchDefaults'
import type { WorkbenchDefaults, WorkbenchGroup } from './workbenchDefaults'
import type { components } from './generated/api-schema'

// Bumped for the v5 research protocol: a v4 draft can carry retired research
// fields that the configuration document no longer accepts.
const CACHE_KEY = 'stockregression-configuration-drafts-v5'
const RECOVERY_KEY = 'stockregression-configuration-recovery-v1'
type Document = Record<string, unknown>
type Patch = Record<string, unknown>
type ConfigurationResponse = components['schemas']['ConfigurationEnvelope']
type ConfigurationPatchRequest = components['schemas']['ConfigurationPatchRequest']
type Snapshot = {
  value: WorkbenchDefaults
  document: Document
  serverDocument: Document
  loading: boolean
  error: string
  conflict: boolean
  revision: string
  pendingRestart: string[]
}

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function isObject(value: unknown): value is Record<string, unknown> {
  return record(value) !== null
}

function deepMerge(target: unknown, patch: unknown): unknown {
  if (!isObject(patch)) return structuredClone(patch)
  const result: Document = isObject(target) ? structuredClone(target) : {}
  for (const [key, value] of Object.entries(patch)) {
    result[key] = isObject(value) && isObject(result[key])
      ? deepMerge(result[key], value)
      : structuredClone(value)
  }
  return result
}

function diffLeaves(base: unknown, desired: unknown): Patch {
  if (!isObject(base) || !isObject(desired)) return structuredClone(desired) as Patch
  const result: Patch = {}
  for (const [key, value] of Object.entries(desired)) {
    if (JSON.stringify(base[key]) === JSON.stringify(value)) continue
    if (isObject(base[key]) && isObject(value)) {
      const nested = diffLeaves(base[key], value)
      if (Object.keys(nested).length) result[key] = nested
    } else {
      result[key] = structuredClone(value)
    }
  }
  return result
}

function overlaps(left: unknown, right: unknown): boolean {
  if (!isObject(left) || !isObject(right)) return true
  return Object.keys(left).some(key => key in right && overlaps(left[key], right[key]))
}

function withoutPath(value: Patch, path: string): Patch {
  if (!(path in value)) return structuredClone(value)
  const copy = structuredClone(value)
  delete copy[path]
  return copy
}

function workbenchFromDocument(document: Document): WorkbenchDefaults {
  const backtest = record(document.backtest) ?? {}
  const research = record(document.research) ?? {}
  return normalizeWorkbenchDefaults({
    grid: backtest.grid,
    position: backtest.position,
    dca: backtest.dca,
    research: record(research.defaults) ?? {},
  })
}

function researchDefaultsForFile(value: WorkbenchDefaults['research']): WorkbenchDefaults['research'] {
  return structuredClone(value)
}

function mergePatch(base: Document, patch: Patch): Document {
  return deepMerge(base, patch) as Document
}

function readDraft(): { pending: Patch; revision: string } {
  try {
    const parsed = JSON.parse(localStorage.getItem(CACHE_KEY) ?? 'null')
    if (Number(parsed?.version) >= 3 && typeof parsed.revision === 'string' && isObject(parsed.pending)) {
      return { pending: parsed.pending, revision: parsed.revision }
    }
  } catch { /* A damaged draft never replaces the file-backed configuration. */ }
  return { pending: {}, revision: '' }
}

const recoveredDraft = readDraft()
if (typeof window !== 'undefined' && Object.keys(recoveredDraft.pending).length) {
  try {
    localStorage.setItem(RECOVERY_KEY, JSON.stringify({
      version: 1, revision: recoveredDraft.revision, pending: recoveredDraft.pending,
    }))
    localStorage.removeItem(CACHE_KEY)
  } catch { /* Recovery remains optional; it is never replayed over the file. */ }
}
let snapshot: Snapshot = {
  value: defaultWorkbenchDefaults(), document: {}, serverDocument: {},
  loading: true, error: '', conflict: false, revision: '', pendingRestart: [],
}
let pending: Patch = {}
let draftVersion = 0
let initialization: Promise<void> | undefined
let saveChain: Promise<void> = Promise.resolve()
let saveTimer: ReturnType<typeof setTimeout> | undefined
let pollTimer: ReturnType<typeof setTimeout> | undefined
let autosaveMs = 500
let pollingMs = 2_000
const listeners = new Set<() => void>()

function publish(next: Snapshot) {
  snapshot = next
  for (const listener of listeners) listener()
}

function writeDraft() {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ version: 3, revision: snapshot.revision, pending }))
  } catch { /* The configuration file remains the durable source. */ }
}

function applyIntervals(document: Document) {
  const ui = record(document.ui) ?? {}
  autosaveMs = Math.max(100, Number(ui.autosave_debounce_ms) || autosaveMs)
  pollingMs = Math.max(500, Number(ui.polling_interval_ms) || pollingMs)
}

function responseSnapshot(response: ConfigurationResponse, document: Document = response.config): Snapshot {
  applyIntervals(response.config)
  return {
    ...snapshot,
    value: workbenchFromDocument(document),
    document,
    serverDocument: response.config,
    loading: false,
    error: '',
    conflict: false,
    revision: response.revision,
    pendingRestart: Array.isArray(response.pending_restart) ? response.pending_restart.map(String) : [],
  }
}

function applyResponse(response: ConfigurationResponse) {
  publish(responseSnapshot(response))
  writeDraft()
}

function publishMerged(response: ConfigurationResponse) {
  const desired = mergePatch(response.config, pending)
  publish({ ...responseSnapshot(response, desired), value: workbenchFromDocument(desired) })
  writeDraft()
  return desired
}

async function resolveExternalChange(response: ConfigurationResponse, message: string) {
  const serverChanges = diffLeaves(snapshot.serverDocument, response.config)
  if (Object.keys(pending).length && overlaps(pending, serverChanges)) {
    const desired = mergePatch(response.config, pending)
    publish({
      ...responseSnapshot(response, desired),
      value: workbenchFromDocument(desired),
      error: message + '；同一字段已被其他页面或手工编辑修改。本地值已保留为草稿，请检查后继续编辑以重新保存。',
      conflict: true,
    })
    writeDraft()
    return
  }
  publishMerged(response)
  if (Object.keys(pending).length) scheduleSave()
}

function flush(keepalive = false) {
  clearTimeout(saveTimer)
  saveTimer = undefined
  saveChain = saveChain.catch(() => undefined).then(async () => {
    const patch = structuredClone(pending)
    if (!Object.keys(patch).length) return
    const version = draftVersion
    try {
      const body: ConfigurationPatchRequest = { base_revision: snapshot.revision, patch }
      const response = await request<ConfigurationResponse>('/configuration', {
        method: 'PATCH',
        body: JSON.stringify(body),
        keepalive,
      })
      if (draftVersion === version) {
        pending = {}
        applyResponse(response)
      } else {
        const desired = snapshot.document
        pending = diffLeaves(response.config, desired)
        publish({ ...responseSnapshot(response, desired), value: workbenchFromDocument(desired) })
        writeDraft()
        if (Object.keys(pending).length) scheduleSave()
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        try {
          const latest = await get<ConfigurationResponse>('/configuration')
          await resolveExternalChange(latest, '配置文件已被其他页面或手工编辑修改')
          return
        } catch { /* Report the original conflict below. */ }
      }
      publish({ ...snapshot, error: '配置暂未保存；请检查文件冲突或字段校验：' + (error instanceof Error ? error.message : String(error)) })
    }
  })
  return saveChain
}

function scheduleSave() {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => { void flush() }, autosaveMs)
}

async function refreshFromFile() {
  try {
    const response = await get<ConfigurationResponse>('/configuration')
    if (!Object.keys(pending).length && response.revision === snapshot.revision) return
    if (!Object.keys(pending).length) applyResponse(response)
    else await resolveExternalChange(response, '配置文件存在未合并的外部修改')
  } catch (error) {
    publish({ ...snapshot, loading: false, error: '无法读取项目配置文件：' + (error instanceof Error ? error.message : String(error)) })
  }
}

function armPolling() {
  clearTimeout(pollTimer)
  pollTimer = setTimeout(async () => {
    await refreshFromFile()
    armPolling()
  }, pollingMs)
}

async function initialize() {
  if (!initialization) initialization = (async () => {
    try {
      const response = await get<ConfigurationResponse>('/configuration')
      if (response.revision !== snapshot.revision) { pending = {}; draftVersion = 0 }
      const desired = publishMerged(response)
      publish({ ...snapshot, value: workbenchFromDocument(desired), loading: false })
    } catch (error) {
      publish({ ...snapshot, loading: false, error: '读取项目配置失败：' + (error instanceof Error ? error.message : String(error)) })
    }
    if (typeof window !== 'undefined' && !pollTimer) {
      const refresh = () => { void refreshFromFile() }
      window.addEventListener('focus', refresh)
      document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') refresh() })
      armPolling()
    }
  })()
  await initialization
  if (Object.keys(pending).length) scheduleSave()
}

export function rememberConfigurationPatch(patch: Patch) {
  if (!isObject(patch) || !Object.keys(patch).length) return
  pending = deepMerge(pending, patch) as Patch
  draftVersion += 1
  const desired = mergePatch(snapshot.document, patch)
  publish({ ...snapshot, value: workbenchFromDocument(desired), document: desired, conflict: false })
  writeDraft()
  scheduleSave()
}

export function rememberWorkbenchGroup<K extends WorkbenchGroup>(group: K, value: WorkbenchDefaults[K]) {
  if (!validWorkbenchGroup(group, value)) return
  if (JSON.stringify(snapshot.value[group]) === JSON.stringify(value)) return
  if (group === 'research') {
    const researchValue = value as WorkbenchDefaults['research']
    const baseResearch = record(record(snapshot.serverDocument.research)?.defaults) ?? {}
    const researchPatch = diffLeaves(baseResearch, researchDefaultsForFile(researchValue))
    // The configuration document no longer owns a statistics group, so any
    // leftover statistics path is dropped instead of being written back.
    let next = withoutPath(withoutPath(pending, 'research'), 'statistics')
    if (Object.keys(researchPatch).length) next = deepMerge(next, { research: { defaults: researchPatch } }) as Patch
    pending = next
    const desired = mergePatch(snapshot.serverDocument, pending)
    publish({ ...snapshot, value: { ...snapshot.value, research: structuredClone(researchValue) }, document: desired, conflict: false })
  } else {
    const baseGroup = record(record(snapshot.serverDocument.backtest)?.[group]) ?? {}
    const leafPatch = diffLeaves(baseGroup, value)
    const otherGroups = withoutPath(record(pending.backtest) ?? {}, group)
    pending = withoutPath(pending, 'backtest')
    if (Object.keys(leafPatch).length) pending = deepMerge(pending, { backtest: { ...otherGroups, [group]: leafPatch } }) as Patch
    else if (Object.keys(otherGroups).length) pending = deepMerge(pending, { backtest: otherGroups }) as Patch
    const desired = mergePatch(snapshot.serverDocument, pending)
    publish({ ...snapshot, value: { ...snapshot.value, [group]: structuredClone(value) } as WorkbenchDefaults, document: desired, conflict: false })
  }
  draftVersion += 1
  writeDraft()
  scheduleSave()
}

if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => { void flush(true) })
}

export function useWorkbenchDefaults() {
  const state = useSyncExternalStore(
    listener => { listeners.add(listener); return () => { listeners.delete(listener) } },
    () => snapshot,
  )
  useEffect(() => {
    void initialize()
    return () => { void flush() }
  }, [])
  return state
}

export function useConfigurationDocument() {
  return useWorkbenchDefaults()
}

export async function flushConfiguration() {
  await flush()
  if (snapshot.error) throw new Error(snapshot.error)
  if (snapshot.conflict) throw new Error('配置存在同字段冲突，请先处理后再提交研究')
  return snapshot.revision
}

export async function saveConfigurationPatch(patch: Patch) {
  rememberConfigurationPatch(patch)
  return flushConfiguration()
}

export function configurationRecoveryDraft(): { pending: Patch; revision: string } | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(RECOVERY_KEY) ?? 'null')
    if (parsed?.version === 1 && typeof parsed.revision === 'string' && isObject(parsed.pending)) {
      return { pending: structuredClone(parsed.pending), revision: parsed.revision }
    }
  } catch { /* A damaged recovery draft is ignored. */ }
  return null
}
