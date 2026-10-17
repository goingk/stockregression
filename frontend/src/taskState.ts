import type { Job } from './types'

export const isActiveJob = (job: Pick<Job, 'status'>) => ['queued', 'running'].includes(job.status)
export const isResearchJob = (job: Pick<Job, 'kind'>) => ['optimization', 'optimize'].includes(job.kind)
export const taskStatusLabel = (status: string) => ({ queued: '排队中', running: '运行中', completed: '已完成', cancelled: '已取消', failed: '失败' }[status] ?? status)
export function taskLabel(job: Job) {
  const kind = ({ backtest: '策略回测', batch_backtest: '批量回测', optimization: '参数研究', optimize: '参数研究', sync: '行情同步', import: '文件导入', instruments: '证券清单更新', demo: '演示数据' } as Record<string, string>)[job.kind] ?? job.kind
  const config = job.payload?.config as Record<string, unknown> | undefined
  const base = config?.base as Record<string, unknown> | undefined
  const symbol = base?.symbol ?? config?.symbol
  return `${kind}${typeof symbol === 'string' ? ' · ' + symbol : ''}`
}

/** Do not let a delayed snapshot turn a completed task back into a running task. */
export function mergeJobSnapshots(previous: Job[], incoming: Job[]): Job[] {
  const merged = new Map(previous.map(job => [job.id, job]))
  for (const job of incoming) {
    const before = merged.get(job.id)
    if (before?.updated_at && !job.updated_at) continue
    if (before?.updated_at && job.updated_at && before.updated_at > job.updated_at) continue
    if (before && !isActiveJob(before) && isActiveJob(job)) continue
    merged.set(job.id, job)
  }
  return [...merged.values()].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
}

export function submissionJobIds(response: unknown): string[] {
  if (!response || typeof response !== 'object') return []
  const value = response as { job_id?: unknown; job_ids?: unknown }
  return [...new Set([value.job_id, ...(Array.isArray(value.job_ids) ? value.job_ids : [])].filter((id): id is string => typeof id === 'string' && Boolean(id)))]
}

export function newlyNotifiableJobs(previous: Job[], current: Job[], initialized: boolean): Job[] {
  const known = new Map(previous.map(job => [job.id, job]))
  return current.filter(job => {
    const before = known.get(job.id)
    if (!initialized) return isActiveJob(job)
    return !before || before.status !== job.status
  })
}
