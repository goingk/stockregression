import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { AlertCircle, ArrowLeft, Bell, CheckCircle2, ChevronRight, ExternalLink, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { get, post } from './api'
import { Badge, Button, Empty, JobProgress, money, percent } from './components'
import { DataTaskReport } from './DataTaskReport'
import { isActiveJob, isEtfResearchJob, mergeJobSnapshots, newlyNotifiableJobs, submissionJobIds, taskLabel, taskStatusLabel, taskNoticeMessage } from './taskState'
import type { Job } from './types'
import { useWorkbenchDefaults } from './useWorkbenchDefaults'
import { LatestRequest } from './requestState'

type Notice = { id: string; label: string; job?: Job; error?: string; submitting?: boolean; expiresAt?: number }
type Tasks = {
  jobs: Job[]; loading: boolean; error: string; refresh: () => Promise<void>; cancel: (id: string) => Promise<void>
  run: <T>(label: string, action: () => Promise<T>) => Promise<T>; openResult: (job: Job) => void; showHistory: () => void
}
type JobHistory = {
  items: Job[]
  total: number
  offset: number
  limit: number
  kinds: Array<{ kind: string; count: number }>
}

type TaskFact = { label: string; value: string; title?: string }
type TaskResult = Record<string, unknown>

const DATA_REPORT_KINDS = new Set(['sync', 'import', 'market_sync', 'market_import', 'market_import_batch', 'market_validate', 'market_aggregate', 'market_prepare', 'market_index_package', 'instruments'])
const COUNT_LABELS: Record<string, string> = {
  total: '范围总数', completed: '完成', partial: '部分完成', failed: '失败', cancelled: '已取消',
  scored: '可评分', eligible: '符合硬过滤', validated: '完成核验', recommended: '有推荐参数', ready: '就绪',
  pending_validation: '待校验', insufficient_history: '历史不足', data_not_ready: '数据未就绪',
}

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value)

function taskTime(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function taskDuration(job: Job) {
  if (!job.created_at || !job.updated_at) return '—'
  const elapsed = new Date(job.updated_at).getTime() - new Date(job.created_at).getTime()
  if (!Number.isFinite(elapsed) || elapsed < 0) return '—'
  if (elapsed < 1000) return '不足 1 秒'
  if (elapsed < 60000) return `${Math.round(elapsed / 1000)} 秒`
  if (elapsed < 3600000) return `${Math.floor(elapsed / 60000)} 分 ${Math.round((elapsed % 60000) / 1000)} 秒`
  return `${Math.floor(elapsed / 3600000)} 小时 ${Math.floor((elapsed % 3600000) / 60000)} 分`
}

function compactList(value: unknown, limit = 8) {
  if (!Array.isArray(value)) return ''
  const items = value.map(item => String(item)).filter(Boolean)
  if (!items.length) return ''
  return items.length > limit ? `${items.slice(0, limit).join('、')} 等 ${items.length} 项` : items.join('、')
}

function textValue(value: unknown) {
  if (value == null || value === '') return ''
  if (Array.isArray(value)) return compactList(value)
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function taskRequestFacts(job: Job): TaskFact[] {
  const payload = isRecord(job.payload) ? job.payload : {}
  const config = isRecord(payload.config) ? payload.config : {}
  const base = isRecord(config.base) ? config.base : {}
  const plan = isRecord(payload.plan) ? payload.plan : {}
  const first = (...keys: string[]) => {
    for (const source of [config, base, plan, payload]) {
      for (const key of keys) {
        const value = source[key]
        if (value != null && value !== '') return value
      }
    }
    return undefined
  }
  const symbol = textValue(config.symbol) || textValue(base.symbol) || textValue(payload.symbol)
  const symbols = compactList(plan.symbols) || compactList(config.symbols) || compactList(payload.symbols)
  const timeframes = compactList(payload.timeframes) || textValue(payload.timeframe) || textValue(first('frequency'))
  const asOf = textValue(plan.as_of) || textValue(first('as_of'))
  const facts: TaskFact[] = []
  const add = (label: string, value: string, title?: string) => { if (value) facts.push({ label, value, title }) }
  add('范围', textValue(first('scope')))
  add('模式', textValue(first('mode')))
  add('标的', symbol || symbols, symbol || symbols)
  add('周期', timeframes)
  add('开始日期', textValue(first('start_date')))
  add('结束日期', textValue(first('end_date')))
  const windowDays = first('window_days')
  add('统计窗口', windowDays == null ? '' : `${textValue(windowDays)} 个交易日`)
  add('截止日', asOf)
  const validationTopK = first('validation_top_k')
  add('自动核验', validationTopK == null ? '' : `前 ${textValue(validationTopK)} 名`)
  add('并发数', textValue(first('workers', 'download_workers', 'validation_workers')))
  return facts
}

function taskErrorText(error: unknown) {
  if (error == null) return ''
  return typeof error === 'string' ? error : JSON.stringify(error, null, 2)
}

function prettyJson(value: unknown) {
  const text = JSON.stringify(value ?? {}, null, 2)
  return text.length > 60000 ? `${text.slice(0, 60000)}\n… 内容过长，已截断；请打开完整结果查看全部数据。` : text
}

function JsonDetails({ label, value }: { label: string; value: unknown }) {
  const [open, setOpen] = useState(false)
  return <details className="task-detail-json" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{label}</summary>
    {open && <pre>{prettyJson(value)}</pre>}
  </details>
}

function taskResultFacts(job: Job, result: TaskResult): TaskFact[] {
  const facts: TaskFact[] = []
  const add = (label: string, value: unknown) => {
    const text = textValue(value)
    if (text) facts.push({ label, value: text, title: text })
  }
  add('结果状态', typeof result.status === 'string' ? taskStatusLabel(result.status) : result.status)
  add('结果说明', result.message === job.message ? undefined : result.message)
  add('数据截止', result.as_of)
  add('记录数', result.rows_count)
  const completion = isRecord(result.completion) ? result.completion : {}
  add('本次修订', completion.selected_revision ?? completion.revision_number)
  add('修订范围', completion.revision_scope)
  if (job.kind === 'backtest' || job.kind === 'batch_backtest') {
    const config = isRecord(result.config) ? result.config : {}
    add('回测标的', config.symbol)
    add('回测区间', config.start_date && config.end_date ? `${textValue(config.start_date)} 至 ${textValue(config.end_date)}` : config.start_date)
    const strategies = isRecord(result.strategies) ? result.strategies : {}
    for (const [name, label] of [['grid', '网格'], ['dca', '定投'], ['buy_hold', '持有']] as const) {
      const strategy = isRecord(strategies[name]) ? strategies[name] : {}
      const metrics = isRecord(strategy.metrics) ? strategy.metrics : {}
      if (!Object.keys(metrics).length) continue
      const parts = [
        metrics.peak_principal_return != null ? `收益 ${percent(Number(metrics.peak_principal_return))}` : '',
        metrics.max_principal_drawdown != null ? `回撤 ${percent(Number(metrics.max_principal_drawdown))}` : '',
        metrics.trade_count != null ? `成交 ${textValue(metrics.trade_count)} 笔` : '',
        metrics.total_fees != null ? `费用 ${money(Number(metrics.total_fees))}` : '',
      ].filter(Boolean)
      if (parts.length) add(`${label}摘要`, parts.join(' · '))
    }
  }
  return facts
}

function ResultCounts({ counts }: { counts: Record<string, unknown> }) {
  const entries = Object.entries(counts).filter(([, value]) => typeof value === 'number')
  if (!entries.length) return null
  return <div className="task-result-counts">{entries.map(([key, value]) => <span key={key}><small>{COUNT_LABELS[key] ?? key}</small><strong>{String(value)}</strong></span>)}</div>
}

function TaskDetail({ job, cancelling, onBack, onCancel, onOpenResult }: {
  job: Job
  cancelling: boolean
  onBack: () => void
  onCancel: (id: string) => void
  onOpenResult: (job: Job) => void
}) {
  const facts = useMemo(() => taskRequestFacts(job), [job.payload])
  const [result, setResult] = useState<TaskResult | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [resultError, setResultError] = useState('')
  const [resultRetry, setResultRetry] = useState(0)
  const errorText = taskErrorText(job.error)
  const resultLabel = isEtfResearchJob(job) ? '打开 ETF 研究' : ['sync', 'market_sync'].includes(job.kind) ? '打开完整报告' : '打开完整结果'
  const dataReport = DATA_REPORT_KINDS.has(job.kind)
  const resultFacts = useMemo(() => result ? taskResultFacts(job, result) : [], [job, result])
  const resultCounts = result && isRecord(result.counts) ? result.counts : null
  const resultWarnings = result && Array.isArray(result.warnings) ? result.warnings.map(item => textValue(item)).filter(Boolean) : []

  useEffect(() => {
    if (!job.result_id || isActiveJob(job)) {
      setResult(null)
      setResultLoading(false)
      setResultError('')
      return
    }
    let active = true
    setResultLoading(true)
    setResultError('')
    get<TaskResult>('/jobs/' + encodeURIComponent(job.id) + '/result')
      .then(value => { if (active) setResult(value) })
      .catch(cause => { if (active) { setResult(null); setResultError(cause instanceof Error ? cause.message : String(cause)) } })
      .finally(() => { if (active) setResultLoading(false) })
    return () => { active = false }
  }, [job.id, job.result_id, job.status, resultRetry])

  return <div className="task-detail-panel">
    <div className="task-detail-top"><Button variant="quiet" onClick={onBack}><ArrowLeft size={14} />返回任务列表</Button><Badge tone={job.status === 'failed' ? 'red' : job.status === 'partial' ? 'amber' : job.status === 'completed' ? 'green' : isActiveJob(job) ? 'blue' : 'neutral'}>{taskStatusLabel(job.status)}</Badge></div>
    <div className="task-detail-heading"><small>任务详情</small><h2>{taskLabel(job)}</h2><p>{job.message || '等待处理'}</p></div>
    <JobProgress progress={job.progress} status={job.status} />
    <section className="task-detail-section" aria-labelledby="task-detail-basics">
      <h3 id="task-detail-basics">任务信息</h3>
      <dl className="task-detail-facts">
        <div><dt>任务编号</dt><dd title={job.id}>{job.id}</dd></div>
        <div><dt>创建时间</dt><dd>{taskTime(job.created_at)}</dd></div>
        <div><dt>更新时间</dt><dd>{taskTime(job.updated_at)}</dd></div>
        <div><dt>耗时</dt><dd>{taskDuration(job)}</dd></div>
        <div><dt>结果编号</dt><dd title={job.result_id || ''}>{job.result_id || '—'}</dd></div>
      </dl>
    </section>
    {errorText && <section className="task-detail-section task-detail-error" aria-labelledby="task-detail-error"><h3 id="task-detail-error">错误详情</h3><pre>{errorText}</pre></section>}
    <section className="task-detail-section" aria-labelledby="task-detail-request">
      <h3 id="task-detail-request">请求参数</h3>
      {facts.length ? <dl className="task-detail-facts">{facts.map(fact => <div key={fact.label}><dt>{fact.label}</dt><dd title={fact.title || fact.value}>{fact.value}</dd></div>)}</dl> : <p className="task-detail-empty">该任务未记录可摘要的请求参数。</p>}
      {job.payload && Object.keys(job.payload).length > 0 && <JsonDetails label="原始请求 JSON" value={job.payload} />}
    </section>
    <section className="task-detail-section" aria-labelledby="task-detail-result">
      <h3 id="task-detail-result">结果</h3>
      {job.result_id ? <>
        {isActiveJob(job) ? <p className="task-detail-empty">任务仍在执行，结束后会在这里读取已保存的结果。</p> : resultLoading ? <p className="task-result-loading" role="status"><LoaderCircle className="spin" size={15} />正在读取保存的结果…</p> : resultError ? <div className="task-result-loading failed" role="alert"><AlertCircle size={15} /><span>结果读取失败：{resultError}</span><button className="text-link" onClick={() => setResultRetry(value => value + 1)}>重试</button></div> : result ? <div className="task-result-body">
          {resultFacts.length > 0 && <dl className="task-detail-facts">{resultFacts.map(fact => <div key={fact.label}><dt>{fact.label}</dt><dd title={fact.title || fact.value}>{fact.value}</dd></div>)}</dl>}
          {resultCounts && <ResultCounts counts={resultCounts} />}
          {resultWarnings.length > 0 && <div className="task-result-warnings"><strong>警告 {resultWarnings.length} 项</strong>{resultWarnings.slice(0, 4).map((warning, index) => <p key={index}>{warning}</p>)}{resultWarnings.length > 4 && <small>另有 {resultWarnings.length - 4} 项，请在完整结果中查看。</small>}</div>}
          {dataReport && <div className="task-result-report"><DataTaskReport report={result} /></div>}
          <JsonDetails label="原始结果 JSON" value={result} />
        </div> : <p className="task-detail-empty">结果已保存，但当前无法读取摘要。</p>}
        <Button variant="secondary" onClick={() => onOpenResult(job)}>{resultLabel}<ExternalLink size={14} /></Button>
      </> : <p className="task-detail-empty">该任务没有保存结果{job.status === 'failed' ? '，请根据错误详情处理后重试。' : '。'}</p>}
    </section>
    {isActiveJob(job) && <div className="task-detail-actions"><Button variant="quiet" busy={cancelling} disabled={job.cancel_requested} onClick={() => onCancel(job.id)}>{job.cancel_requested ? '取消中…' : '取消任务'}</Button><span>任务状态会自动刷新。</span></div>}
  </div>
}

const TaskContext = createContext<Tasks | null>(null)
export function useTasks() {
  const value = useContext(TaskContext)
  if (!value) throw new Error('任务中心尚未初始化')
  return value
}

export function TaskProvider({ children, onOpenResult, filter, kind, onFilterChange }: {
  children: ReactNode
  onOpenResult: (job: Job) => void
  filter: string
  kind: string
  onFilterChange: (filter: string, kind: string) => void
}) {
  const configuration = useWorkbenchDefaults()
  const pollingMs = Number((configuration.document.ui as Record<string, unknown> | undefined)?.task_polling_interval_ms ?? 2500)
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notices, setNotices] = useState<Notice[]>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [history, setHistory] = useState<JobHistory>({ items: [], total: 0, offset: 0, limit: 50, kinds: [] })
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [historyOffset, setHistoryOffset] = useState(0)
  const [selectedJobId, setSelectedJobId] = useState('')
  const [cancelling, setCancelling] = useState<Set<string>>(new Set())
  const jobsRef = useRef<Job[]>([])
  const initialized = useRef(false)
  const pendingRefresh = useRef<Promise<void> | null>(null)
  const mounted = useRef(true)
  const drawer = useRef<HTMLElement>(null)
  const historyRequest = useRef(new LatestRequest<string>())
  const historyFilter = useRef('')
  const activeIds = useRef<Set<string>>(new Set())

  const accept = useCallback((incoming: Job[], source: 'live' | 'history' = 'live') => {
    if (!mounted.current) return
    const next = mergeJobSnapshots(jobsRef.current, incoming)
    const changed = newlyNotifiableJobs(jobsRef.current, next, initialized.current, source)
    jobsRef.current = next
    setJobs(next)
    setNotices(current => {
      const merged = new Map(current.map(notice => [notice.id, notice]))
      for (const job of next) {
        const before = merged.get(job.id)
        if (before) merged.set(job.id, { ...before, job })
      }
      for (const job of changed) merged.set(job.id, {
        id: job.id, label: taskLabel(job), job,
        expiresAt: isActiveJob(job) || job.status === 'failed' || job.status === 'partial' ? undefined : Date.now() + 6000,
      })
      return [...merged.values()]
    })
  }, [])

  const refresh = useCallback((): Promise<void> => {
    if (pendingRefresh.current) return pendingRefresh.current
    const request = (async () => {
      try {
        const list = await get<Job[]>('/jobs/active')
        const ids = new Set(list.map(job => job.id))
        const missing = jobsRef.current.filter(job => isActiveJob(job) && !ids.has(job.id))
        const extra = await Promise.allSettled(missing.map(job => get<Job>('/jobs/' + encodeURIComponent(job.id))))
        accept([...list, ...extra.flatMap(item => item.status === 'fulfilled' ? [item.value] : [])])
        if (mounted.current) {
          setError('')
          initialized.current = true
        }
      } catch (cause) {
        if (mounted.current) setError('连接暂时中断，正在重试：' + (cause instanceof Error ? cause.message : String(cause)))
      } finally {
        if (mounted.current) setLoading(false)
        pendingRefresh.current = null
      }
    })()
    pendingRefresh.current = request
    return request
  }, [accept])

  const loadHistory = useCallback(async (offset = 0, append = false): Promise<void> => {
    const requestKey = `${filter}|${kind}|${offset}`
    const ticket = historyRequest.current.begin(requestKey)
    setHistoryLoading(true)
    setHistoryError('')
    try {
      let page: JobHistory
      if (filter === 'active') {
        const active = await get<Job[]>('/jobs/active')
        const items = kind ? active.filter(job => job.kind === kind) : active
        page = { items, total: items.length, offset: 0, limit: 50, kinds: [] }
      } else {
        const params = new URLSearchParams({ offset: String(offset), limit: '50' })
        if (filter !== 'all') params.set('status', filter)
        if (kind) params.set('kind', kind)
        page = await get<JobHistory>('/jobs/history?' + params.toString())
      }
      if (!historyRequest.current.isCurrent(ticket) || !mounted.current) return
      setHistory(current => append ? {
        ...page,
        items: [...current.items, ...page.items.filter(item => !current.items.some(existing => existing.id === item.id))],
        kinds: page.kinds.length ? page.kinds : current.kinds,
      } : page)
      setHistoryOffset(offset)
      accept(page.items, 'history')
      initialized.current = true
      setLoading(false)
    } catch (cause) {
      if (historyRequest.current.isCurrent(ticket) && mounted.current) {
        // Keep the previous page visible; a history refresh failure only disables the page controls.
        setHistoryError('任务历史读取失败：' + (cause instanceof Error ? cause.message : String(cause)))
        setLoading(false)
      }
    } finally {
      if (historyRequest.current.isCurrent(ticket) && mounted.current) setHistoryLoading(false)
    }
  }, [accept, filter, kind])

  useEffect(() => {
    mounted.current = true
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, Math.max(500, pollingMs))
    return () => { mounted.current = false; window.clearInterval(timer) }
  }, [refresh, pollingMs])
  useEffect(() => {
    if (!historyOpen) {
      historyFilter.current = ''
      setSelectedJobId('')
      activeIds.current = new Set(jobs.filter(isActiveJob).map(job => job.id))
      return
    }
    const key = `${filter}|${kind}`
    if (historyFilter.current === key) return
    historyFilter.current = key
    setHistoryOffset(0)
    void loadHistory(0, false)
  }, [historyOpen, filter, kind, loadHistory, jobs])
  useEffect(() => {
    if (!historyOpen) return
    const next = new Set(jobs.filter(isActiveJob).map(job => job.id))
    const finished = [...activeIds.current].some(id => !next.has(id))
    activeIds.current = next
    if (finished) void loadHistory(historyOffset, false)
  }, [jobs, historyOpen, historyOffset, loadHistory])
  useEffect(() => {
    if (!notices.some(notice => notice.expiresAt)) return
    const timer = window.setInterval(() => setNotices(current => current.filter(notice => !notice.expiresAt || notice.expiresAt > Date.now())), 500)
    return () => window.clearInterval(timer)
  }, [notices])
  useEffect(() => {
    if (!historyOpen) return
    const previous = document.activeElement as HTMLElement | null
    drawer.current?.querySelector<HTMLElement>('button')?.focus()
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (selectedJobId) setSelectedJobId('')
        else setHistoryOpen(false)
      }
      if (event.key !== 'Tab') return
      const controls = drawer.current?.querySelectorAll<HTMLElement>('button:not(:disabled), select, input, summary, [tabindex="0"]')
      if (!controls?.length) return
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', keydown)
    return () => { document.removeEventListener('keydown', keydown); previous?.focus() }
  }, [historyOpen, selectedJobId])

  const run = useCallback(async <T,>(label: string, action: () => Promise<T>): Promise<T> => {
    const id = 'submission-' + crypto.randomUUID()
    setNotices(current => [...current, { id, label, submitting: true }])
    try {
      const response = await action()
      const ids = submissionJobIds(response)
      if (ids.length) {
        accept(ids.map(jobId => ({ id: jobId, kind: label, status: 'queued', progress: 0, message: '任务已提交，等待执行', created_at: new Date().toISOString() })))
        // The submission itself is user initiated, even if the initial history request has not finished.
        setNotices(current => {
          const withoutPending = current.filter(notice => notice.id !== id)
          return [...withoutPending, ...ids.filter(jobId => !withoutPending.some(notice => notice.id === jobId)).map(jobId => ({ id: jobId, label, job: jobsRef.current.find(job => job.id === jobId) }))]
        })
        await refresh()
        if (historyOpen) await loadHistory(0, false)
      } else setNotices(current => current.map(notice => notice.id === id ? { id, label, expiresAt: Date.now() + 6000 } : notice))
      return response
    } catch (cause) {
      setNotices(current => current.map(notice => notice.id === id ? { id, label, error: cause instanceof Error ? cause.message : String(cause) } : notice))
      throw cause
    }
  }, [accept, refresh, loadHistory, historyOpen])
  const cancel = useCallback(async (id: string) => {
    setCancelling(current => new Set(current).add(id))
    try {
      const job = await post<Job>('/jobs/' + encodeURIComponent(id) + '/cancel')
      accept([job]); await refresh()
      if (historyOpen) await loadHistory(historyOffset, false)
    } catch (cause) {
      setNotices(current => [...current.filter(notice => notice.id !== 'cancel-' + id), { id: 'cancel-' + id, label: '取消任务失败', error: cause instanceof Error ? cause.message : String(cause) }])
    } finally { setCancelling(current => { const next = new Set(current); next.delete(id); return next }) }
  }, [accept, refresh, loadHistory, historyOffset, historyOpen])
  const openResult = useCallback((job: Job) => { setSelectedJobId(''); setHistoryOpen(false); onOpenResult(job) }, [onOpenResult])
  const showHistory = useCallback(() => { setSelectedJobId(''); setHistoryOpen(true) }, [])
  const value = useMemo(() => ({ jobs, loading, error, refresh, cancel, run, openResult, showHistory }), [jobs, loading, error, refresh, cancel, run, openResult, showHistory])
  const selectedJob = selectedJobId ? jobs.find(job => job.id === selectedJobId) ?? history.items.find(job => job.id === selectedJobId) : undefined
  const removeNotice = (id: string) => setNotices(current => current.filter(notice => notice.id !== id))
  const jobActions = (job: Job) => <div className="task-actions">
    <button className="text-link" onClick={() => { setSelectedJobId(job.id); setHistoryOpen(true) }}>查看详情<ChevronRight size={13} /></button>
    {isActiveJob(job) && <button className="text-link danger-text" disabled={cancelling.has(job.id) || job.cancel_requested} onClick={() => void cancel(job.id)}>{cancelling.has(job.id) || job.cancel_requested ? '取消中…' : '取消任务'}</button>}
  </div>
  return <TaskContext.Provider value={value}>{children}
    <div className="task-toasts" aria-label="任务通知">
      {notices.slice(-3).reverse().map(notice => <section className={`task-toast ${notice.error || notice.job?.status === 'failed' ? 'failed' : notice.job?.status === 'partial' ? 'partial' : ''}`} key={notice.id}>
        <div className="task-toast-head"><span className="task-toast-icon">{notice.submitting || (notice.job && isActiveJob(notice.job)) ? <LoaderCircle className="spin" size={17} /> : notice.error || notice.job?.status === 'failed' || notice.job?.status === 'partial' ? <AlertCircle size={17} /> : <CheckCircle2 size={17} />}</span><strong>{notice.label}</strong>{notice.job?.status === 'partial' && <Badge tone="amber">部分完成</Badge>}<button className="icon-button" onClick={() => removeNotice(notice.id)} aria-label="收起任务提示"><X size={15} /></button></div>
        <div role="status" className="task-toast-message">{notice.error ?? (notice.submitting ? '正在提交…' : notice.job ? taskNoticeMessage(notice.job) : '操作完成')}</div>
        {notice.job && <><JobProgress progress={notice.job.progress} status={notice.job.status} />{jobActions(notice.job)}</>}
      </section>)}
      {notices.length > 3 && <button className="task-overflow" onClick={showHistory}>另有 {notices.length - 3} 条任务通知 · 查看全部</button>}
    </div>
    {historyOpen && <div className="task-drawer-overlay" onMouseDown={event => { if (event.target === event.currentTarget) setHistoryOpen(false) }}>
      <section ref={drawer} className={`task-drawer ${selectedJobId ? 'task-drawer-detail' : ''}`} role="dialog" aria-modal="true" aria-labelledby="task-center-title">
        <div className="task-drawer-head"><div><h2 id="task-center-title">任务中心</h2><p>{selectedJobId ? '请求、错误与结果入口' : '进度、历史记录与结果入口'}</p></div><button className="icon-button" onClick={() => setHistoryOpen(false)} aria-label="关闭任务中心"><X size={20} /></button></div>
        {selectedJobId ? (selectedJob ? <TaskDetail job={selectedJob} cancelling={cancelling.has(selectedJob.id)} onBack={() => setSelectedJobId('')} onCancel={id => void cancel(id)} onOpenResult={openResult} /> : <div className="task-detail-panel"><Button variant="quiet" onClick={() => setSelectedJobId('')}><ArrowLeft size={14} />返回任务列表</Button><Empty title="任务不存在" detail="该任务可能已从当前筛选或分页中移除。" /></div>) : <>
          <div className="task-drawer-tools"><select className="input" aria-label="筛选任务状态" value={filter} onChange={event => onFilterChange(event.target.value, kind)}><option value="all">全部任务</option><option value="active">正在运行</option><option value="completed">已完成</option><option value="partial">部分完成</option><option value="failed">失败</option><option value="cancelled">已取消</option></select><select className="input" aria-label="筛选任务类型" value={kind} onChange={event => onFilterChange(filter, event.target.value)}><option value="">全部类型</option>{history.kinds.map(item => <option key={item.kind} value={item.kind}>{taskLabel({ id: '', kind: item.kind, status: '', progress: 0, message: '', created_at: '' })}（{item.count}）</option>)}</select><Button variant="quiet" busy={historyLoading} onClick={() => void Promise.all([refresh(), loadHistory(historyOffset, false)])}><RefreshCw size={14} />刷新</Button></div>
          {error && <p className="task-connection-note" role="status">{error}</p>}
          {historyError && <p className="task-connection-note" role="alert">{historyError} <button className="text-link" onClick={() => void loadHistory(historyOffset, false)}>重试</button></p>}
          <div className="task-drawer-list">{!history.items.length ? <Empty title={loading || historyLoading ? '正在读取任务…' : '暂无任务'} /> : history.items.map(job => <article className="task-history-item" key={job.id}>
            <div className="task-history-heading"><strong>{taskLabel(job)}</strong><Badge tone={job.status === 'failed' ? 'red' : job.status === 'partial' ? 'amber' : job.status === 'completed' ? 'green' : isActiveJob(job) ? 'blue' : 'neutral'}>{taskStatusLabel(job.status)}</Badge></div>
            <small>{job.created_at ? new Date(job.created_at).toLocaleString('zh-CN', { hour12: false }) : ''} · {job.id.slice(0, 12)}</small>
            <JobProgress progress={job.progress} status={job.status} /><p>{job.message || '等待处理'}</p>
            <div className="task-actions"><button className="text-link" onClick={() => setSelectedJobId(job.id)}>查看详情<ChevronRight size={13} /></button>{isActiveJob(job) && <button className="text-link danger-text" disabled={cancelling.has(job.id) || job.cancel_requested} onClick={() => void cancel(job.id)}>{cancelling.has(job.id) || job.cancel_requested ? '取消中…' : '取消任务'}</button>}</div>
          </article>)}</div>
          {filter !== 'active' && history.total > 0 && <div className="pagination-row"><span>第 {Math.floor(historyOffset / 50) + 1} / {Math.max(1, Math.ceil(history.total / 50))} 页 · 共 {history.total} 条</span><div><Button variant="quiet" disabled={historyLoading || historyOffset <= 0} onClick={() => void loadHistory(Math.max(0, historyOffset - 50), false)}>上一页</Button><Button variant="quiet" disabled={historyLoading || historyOffset + 50 >= history.total} onClick={() => void loadHistory(historyOffset + 50, false)}>下一页</Button></div></div>}
        </>}
      </section>
    </div>}
  </TaskContext.Provider>
}

export function TaskButton() {
  const { jobs, error, showHistory } = useTasks()
  const count = jobs.filter(isActiveJob).length
  return <button className={`task-center-button ${count ? 'has-active' : ''}`} onClick={showHistory} title={error || '查看任务进度和历史记录'}><Bell size={16} /><span>任务</span>{count > 0 && <b>{count}</b>}{error && <i aria-label="正在重新连接" />}</button>
}
