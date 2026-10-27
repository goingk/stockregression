import type { Job } from './types'

export const isActiveJob = (job: Pick<Job, 'status'>) => ['queued', 'running'].includes(job.status)
export const isResearchJob = (job: Pick<Job, 'kind'>) => ['optimization', 'optimize', 'research_completion', 'research_validation'].includes(job.kind)
export const isResearchValidationJob = (job: Pick<Job, 'kind'>) => job.kind === 'research_validation'
export const isEtfResearchJob = (job: Pick<Job, 'kind'>) => ['etf_research', 'etf_validation'].includes(job.kind)
export const etfResearchResultId = (job: Pick<Job, 'kind' | 'result_id'>) => isEtfResearchJob(job) ? job.result_id ?? null : null
export const canResumeResearchValidation = (job: Pick<Job, 'kind' | 'status'>) => isResearchValidationJob(job) && ['failed', 'cancelled', 'partial'].includes(job.status)
export type ValidationResumeReason = 'runtime_limit' | 'evaluation_budget' | 'cancelled' | 'failed' | 'interrupted'
export type ValidationResumeLimits = { max_runtime_seconds: number; total_evaluation_budget: number | null }
export type ValidationResumeExtension = Partial<Pick<ValidationResumeLimits, 'max_runtime_seconds' | 'total_evaluation_budget'>>

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : undefined
}

function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function firstLimit(source: Record<string, unknown> | undefined, key: keyof ValidationResumeLimits): number | null {
  const value = source?.[key]
  return positiveInteger(value) ? value : null
}

/** Original execution limits are shown before a resumed validation is submitted. */
export function validationResumeLimits(job: Pick<Job, 'payload'>): ValidationResumeLimits {
  const payload = recordValue(job.payload) ?? {}
  const config = recordValue(payload.config)
  const plan = recordValue(payload.plan)
  const estimated = recordValue(plan?.estimated_work)
  return {
    max_runtime_seconds: firstLimit(config, 'max_runtime_seconds')
      ?? firstLimit(plan, 'max_runtime_seconds')
      ?? firstLimit(estimated, 'max_runtime_seconds')
      ?? firstLimit(payload, 'max_runtime_seconds')
      ?? 86400,
    total_evaluation_budget: firstLimit(config, 'total_evaluation_budget')
      ?? firstLimit(estimated, 'total_evaluation_budget')
      ?? firstLimit(plan, 'total_evaluation_budget')
      ?? firstLimit(payload, 'total_evaluation_budget'),
  }
}

export function validationResumeReason(job: Pick<Job, 'status' | 'message' | 'error'>, result?: unknown): ValidationResumeReason {
  if (job.status === 'cancelled') return 'cancelled'
  const root = recordValue(result)
  const computation = recordValue(root?.computation)
  const values = [root?.stop_reason, computation?.stop_reason, computation?.budget_reason,
    root?.message, computation?.reason, job.message,
    typeof job.error === 'string' ? job.error : job.error ? JSON.stringify(job.error) : undefined]
    .filter((value): value is string => typeof value === 'string')
    .map(value => value.toLowerCase())
  if (values.some(value => /runtime_limit|timeout|runtime_stopped|timed_out|运行时间耗尽|时间上限/.test(value))) return 'runtime_limit'
  if (values.some(value => /evaluation_budget|budget_exhausted|budget_stopped|评价预算耗尽|预算耗尽/.test(value))) return 'evaluation_budget'
  if (job.status === 'failed') return 'failed'
  return 'interrupted'
}

export const validationResumeNeedsIncrease = (reason: ValidationResumeReason) => reason === 'runtime_limit' || reason === 'evaluation_budget'

export function validationResumeExtension(
  original: ValidationResumeLimits,
  requested: Partial<ValidationResumeLimits>,
  requireIncrease = false,
): { body?: ValidationResumeExtension; error?: string } {
  const runtime = requested.max_runtime_seconds
  if (!positiveInteger(runtime)) return { error: '最长运行时间必须是正整数秒。' }
  if (runtime < original.max_runtime_seconds) return { error: '最长运行时间只能增加，不能低于原上限。' }
  const originalBudget = original.total_evaluation_budget
  const budget = requested.total_evaluation_budget
  if (originalBudget != null && !positiveInteger(budget)) return { error: '评价预算必须是正整数。' }
  if (originalBudget != null && budget! < originalBudget) return { error: '评价预算只能增加，不能低于原上限。' }
  if (budget != null && !positiveInteger(budget)) return { error: '评价预算必须是正整数。' }
  const body: ValidationResumeExtension = {}
  if (runtime > original.max_runtime_seconds) body.max_runtime_seconds = runtime
  if (budget != null && (originalBudget == null || budget > originalBudget)) body.total_evaluation_budget = budget
  if (requireIncrease && !Object.keys(body).length) {
    return { error: '上次因时间或评价预算耗尽而停止，请至少提高一项上限后再继续。' }
  }
  return { body: Object.keys(body).length ? body : undefined }
}
export const taskStatusLabel = (status: string) => ({ queued: '排队中', running: '运行中', completed: '已完成', partial: '部分完成', cancelled: '已取消', failed: '失败' }[status] ?? status)
export function taskLabel(job: Job) {
  const kind = ({ backtest: '策略回测', batch_backtest: '批量回测', optimization: '参数研究', optimize: '参数研究', research_completion: '研究补算', research_validation: '完整研究验证', etf_research: 'ETF研究', etf_validation: 'ETF数据校验', sync: '行情同步', import: '文件导入', market_import: '周期文件导入', market_import_batch: '批量周期导入', market_sync: '周期同步', market_validate: '数据校验', market_aggregate: '本地聚合', instruments: '证券清单更新' } as Record<string, string>)[job.kind] ?? job.kind
  const config = job.payload?.config as Record<string, unknown> | undefined
  const base = config?.base as Record<string, unknown> | undefined
  const research = job.payload?.research as Record<string, unknown> | undefined
  const context = research?.research_context as Record<string, unknown> | undefined
  const plan = job.payload?.plan as Record<string, unknown> | undefined
  const planSymbols = Array.isArray(plan?.symbols) ? plan.symbols : []
  const firstPlanSymbol = planSymbols.find(item => typeof item === 'string')
    ?? planSymbols.map(item => item && typeof item === 'object' ? (item as Record<string, unknown>).symbol : undefined).find(item => typeof item === 'string')
  const symbol = base?.symbol ?? config?.symbol ?? context?.symbol ?? firstPlanSymbol
  const symbolText = typeof symbol === 'string' ? symbol : ''
  const count = isResearchValidationJob(job) && planSymbols.length > 1 ? ` · ${planSymbols.length} 个标的` : ''
  return `${kind}${symbolText ? ' · ' + symbolText : ''}${count}`
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

export function newlyNotifiableJobs(previous: Job[], current: Job[], initialized: boolean, source: 'live' | 'history' = 'live'): Job[] {
  const known = new Map(previous.map(job => [job.id, job]))
  return current.filter(job => {
    const before = known.get(job.id)
    if (source === 'history') return before ? isActiveJob(before) && !isActiveJob(job) : isActiveJob(job)
    if (!initialized) return isActiveJob(job)
    return !before || before.status !== job.status
  })
}

export function taskNoticeMessage(job: Pick<Job, 'status' | 'message' | 'error'>): string {
  if (job.status === 'failed' && job.error != null) {
    return typeof job.error === 'string' ? job.error : JSON.stringify(job.error)
  }
  return job.message || '操作完成'
}
