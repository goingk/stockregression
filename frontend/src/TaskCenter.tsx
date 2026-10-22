import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { AlertCircle, Bell, CheckCircle2, ChevronRight, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { get, post } from './api'
import { Badge, Button, Empty, JobProgress } from './components'
import { isActiveJob, mergeJobSnapshots, newlyNotifiableJobs, submissionJobIds, taskLabel, taskStatusLabel } from './taskState'
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
  const [cancelling, setCancelling] = useState<Set<string>>(new Set())
  const jobsRef = useRef<Job[]>([])
  const initialized = useRef(false)
  const pendingRefresh = useRef<Promise<void> | null>(null)
  const mounted = useRef(true)
  const drawer = useRef<HTMLElement>(null)
  const historyRequest = useRef(new LatestRequest<string>())
  const historyFilter = useRef('')
  const activeIds = useRef<Set<string>>(new Set())

  const accept = useCallback((incoming: Job[]) => {
    if (!mounted.current) return
    const next = mergeJobSnapshots(jobsRef.current, incoming)
    const changed = newlyNotifiableJobs(jobsRef.current, next, initialized.current)
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
      accept(page.items)
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
      if (event.key === 'Escape') setHistoryOpen(false)
      if (event.key !== 'Tab') return
      const controls = drawer.current?.querySelectorAll<HTMLElement>('button:not(:disabled), select, input, [tabindex="0"]')
      if (!controls?.length) return
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', keydown)
    return () => { document.removeEventListener('keydown', keydown); previous?.focus() }
  }, [historyOpen])

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
  const openResult = useCallback((job: Job) => { setHistoryOpen(false); onOpenResult(job) }, [onOpenResult])
  const showHistory = useCallback(() => setHistoryOpen(true), [])
  const value = useMemo(() => ({ jobs, loading, error, refresh, cancel, run, openResult, showHistory }), [jobs, loading, error, refresh, cancel, run, openResult, showHistory])
  const removeNotice = (id: string) => setNotices(current => current.filter(notice => notice.id !== id))
  const jobActions = (job: Job) => <div className="task-actions">
    {job.result_id && <button className="text-link" onClick={() => openResult(job)}>查看{job.kind === 'sync' ? '报告' : '结果'}<ChevronRight size={13} /></button>}
    {isActiveJob(job) && <button className="text-link danger-text" disabled={cancelling.has(job.id) || job.cancel_requested} onClick={() => void cancel(job.id)}>{cancelling.has(job.id) || job.cancel_requested ? '取消中…' : '取消任务'}</button>}
  </div>
  return <TaskContext.Provider value={value}>{children}
    <div className="task-toasts" aria-label="任务通知">
      {notices.slice(-3).reverse().map(notice => <section className={`task-toast ${notice.error || notice.job?.status === 'failed' ? 'failed' : notice.job?.status === 'partial' ? 'partial' : ''}`} key={notice.id}>
        <div className="task-toast-head"><span className="task-toast-icon">{notice.submitting || (notice.job && isActiveJob(notice.job)) ? <LoaderCircle className="spin" size={17} /> : notice.error || notice.job?.status === 'failed' || notice.job?.status === 'partial' ? <AlertCircle size={17} /> : <CheckCircle2 size={17} />}</span><strong>{notice.label}</strong>{notice.job?.status === 'partial' && <Badge tone="amber">部分完成</Badge>}<button className="icon-button" onClick={() => removeNotice(notice.id)} aria-label="收起任务提示"><X size={15} /></button></div>
        <div role="status" className="task-toast-message">{notice.error ?? (notice.submitting ? '正在提交…' : notice.job?.message || '操作完成')}</div>
        {notice.job && <><JobProgress progress={notice.job.progress} status={notice.job.status} />{jobActions(notice.job)}</>}
      </section>)}
      {notices.length > 3 && <button className="task-overflow" onClick={showHistory}>另有 {notices.length - 3} 条任务通知 · 查看全部</button>}
    </div>
    {historyOpen && <div className="task-drawer-overlay" onMouseDown={event => { if (event.target === event.currentTarget) setHistoryOpen(false) }}>
      <section ref={drawer} className="task-drawer" role="dialog" aria-modal="true" aria-labelledby="task-center-title">
        <div className="task-drawer-head"><div><h2 id="task-center-title">任务中心</h2><p>进度、历史记录与结果入口</p></div><button className="icon-button" onClick={() => setHistoryOpen(false)} aria-label="关闭任务中心"><X size={20} /></button></div>
        <div className="task-drawer-tools"><select className="input" aria-label="筛选任务状态" value={filter} onChange={event => onFilterChange(event.target.value, kind)}><option value="all">全部任务</option><option value="active">正在运行</option><option value="completed">已完成</option><option value="partial">部分完成</option><option value="failed">失败</option><option value="cancelled">已取消</option></select><select className="input" aria-label="筛选任务类型" value={kind} onChange={event => onFilterChange(filter, event.target.value)}><option value="">全部类型</option>{history.kinds.map(item => <option key={item.kind} value={item.kind}>{taskLabel({ id: '', kind: item.kind, status: '', progress: 0, message: '', created_at: '' })}（{item.count}）</option>)}</select><Button variant="quiet" busy={historyLoading} onClick={() => void Promise.all([refresh(), loadHistory(historyOffset, false)])}><RefreshCw size={14} />刷新</Button></div>
        {error && <p className="task-connection-note" role="status">{error}</p>}
        {historyError && <p className="task-connection-note" role="alert">{historyError} <button className="text-link" onClick={() => void loadHistory(historyOffset, false)}>重试</button></p>}
        <div className="task-drawer-list">{!history.items.length ? <Empty title={loading || historyLoading ? '正在读取任务…' : '暂无任务'} /> : history.items.map(job => <article className="task-history-item" key={job.id}>
          <div className="task-history-heading"><strong>{taskLabel(job)}</strong><Badge tone={job.status === 'failed' ? 'red' : job.status === 'partial' ? 'amber' : job.status === 'completed' ? 'green' : isActiveJob(job) ? 'blue' : 'neutral'}>{taskStatusLabel(job.status)}</Badge></div>
          <small>{job.created_at ? new Date(job.created_at).toLocaleString('zh-CN', { hour12: false }) : ''} · {job.id.slice(0, 12)}</small>
          <JobProgress progress={job.progress} status={job.status} /><p>{job.message || '等待处理'}</p>
          {job.error != null && <details className="task-error"><summary>错误详情</summary><pre>{typeof job.error === 'string' ? job.error : JSON.stringify(job.error, null, 2)}</pre></details>}{jobActions(job)}
        </article>)}</div>
        {filter !== 'active' && history.total > 0 && <div className="pagination-row"><span>第 {Math.floor(historyOffset / 50) + 1} / {Math.max(1, Math.ceil(history.total / 50))} 页 · 共 {history.total} 条</span><div><Button variant="quiet" disabled={historyLoading || historyOffset <= 0} onClick={() => void loadHistory(Math.max(0, historyOffset - 50), false)}>上一页</Button><Button variant="quiet" disabled={historyLoading || historyOffset + 50 >= history.total} onClick={() => void loadHistory(historyOffset + 50, false)}>下一页</Button></div></div>}
      </section>
    </div>}
  </TaskContext.Provider>
}

export function TaskButton() {
  const { jobs, error, showHistory } = useTasks()
  const count = jobs.filter(isActiveJob).length
  return <button className={`task-center-button ${count ? 'has-active' : ''}`} onClick={showHistory} title={error || '查看任务进度和历史记录'}><Bell size={16} /><span>任务</span>{count > 0 && <b>{count}</b>}{error && <i aria-label="正在重新连接" />}</button>
}
