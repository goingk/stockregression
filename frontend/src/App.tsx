import { Component, lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Activity, BarChart3, Database, FlaskConical, HardDrive, Layers3, Menu, RotateCcw, ScanSearch, TrendingUp } from 'lucide-react'
import { get, post } from './api'
import { Button, defaultConfig, Loading } from './components'
import { TaskButton, TaskProvider } from './TaskCenter'
import { WorkbenchProvider, useWorkspaces } from './WorkbenchState'
import { etfResearchResultId, isEtfResearchJob, isResearchJob } from './taskState'
import {
  navigationFromHistory, navigationFromUrl, navigationHistoryState, navigationUrl,
} from './navigation'
import type { AppPage, NavigationState } from './navigation'
import { newPreparationContext } from './preparationContext'
import type { PreparationContext } from './preparationContext'
import type { BacktestConfig, DataPreparationRequest, Job } from './types'

const DataCenterPage = lazy(() => import('./pages/DataCenterPage'))
const DataManagementPage = lazy(() => import('./pages/DataManagementPage'))
const BacktestPage = lazy(() => import('./pages/BacktestPage'))
const ResultsPage = lazy(() => import('./pages/ResultsPage'))
const ResearchPage = lazy(() => import('./pages/ResearchPage'))
const EtfResearchPage = lazy(() => import('./pages/EtfResearchPage'))

export type PageKey = AppPage
const nav: Array<{ id: PageKey; label: string; hint: string; icon: typeof Database }> = [
  { id: 'data', label: '行情', hint: '股票 · ETF · 自选', icon: Database },
  { id: 'data-admin', label: '数据管理', hint: '周期 · 导入 · 校验', icon: HardDrive },
  { id: 'etf', label: 'ETF 研究', hint: '范围 · 评分 · 核验', icon: ScanSearch },
  { id: 'backtest', label: '策略回测', hint: '网格 · 实时预览', icon: Layers3 },
  { id: 'research', label: '参数研究', hint: '择参 · 样本外验证', icon: FlaskConical },
  { id: 'results', label: '结果分析', hint: '绩效与交易', icon: BarChart3 },
]

class PageErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('工作区页面载入失败', error, info)
  }

  render() {
    if (this.state.failed) {
      return <div className="page-error-boundary" role="alert" aria-live="polite">
        <div className="page-error-mark">!</div>
        <h2>页面已更新，请刷新</h2>
        <p>当前页面没有完整载入。刷新后即可继续使用。</p>
        <button className="button primary" onClick={() => window.location.reload()}>刷新页面</button>
      </div>
    }
    return this.props.children
  }
}

export default function App() {
  return <WorkbenchProvider><AppWindow /></WorkbenchProvider>
}

function AppWindow() {
  const workspace = useWorkspaces()
  const [navigation, setNavigation] = useState<NavigationState>(() => navigationFromUrl(
    typeof window === 'undefined' ? '' : window.location.search,
  ))
  const navigationRef = useRef(navigation)
  navigationRef.current = navigation
  const page = navigation.page
  const [mobileOpen, setMobileOpen] = useState(false)
  const [transferNotice, setTransferNotice] = useState<{ page: PageKey; message: string } | null>(null)
  const [dataPreparation, setDataPreparation] = useState<PreparationContext | null>(null)
  const [restartState, setRestartState] = useState<'idle' | 'confirm' | 'running' | 'failed'>('idle')
  const [restartMessage, setRestartMessage] = useState('')
  const previousInstance = useRef<string | null>(null)
  const commitNavigation = useCallback((
    update: NavigationState | ((previous: NavigationState) => NavigationState),
    options: { replace?: boolean; scroll?: boolean } = {},
  ) => {
    const previous = navigationRef.current
    const next = typeof update === 'function' ? update(previous) : update
    if (typeof window !== 'undefined') {
      const url = navigationUrl(window.location.search, next)
      const method = options.replace ? 'replaceState' : 'pushState'
      window.history[method](navigationHistoryState(next), '', url)
    }
    navigationRef.current = next
    setNavigation(next)
    setMobileOpen(false)
    if (options.scroll !== false) window.scrollTo({ top: 0, behavior: 'instant' })
  }, [])
  useEffect(() => {
    const initial = navigationRef.current
    window.history.replaceState(navigationHistoryState(initial), '', navigationUrl(window.location.search, initial))
    const restore = (event: PopStateEvent) => {
      const next = navigationFromHistory(event.state) ?? navigationFromUrl(window.location.search)
      navigationRef.current = next
      setNavigation(next)
      setMobileOpen(false)
      window.scrollTo({ top: 0, behavior: 'instant' })
    }
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])
  const go = useCallback((next: PageKey) => {
    commitNavigation(previous => ({ ...previous, page: next }))
  }, [commitNavigation])
  const openJobResult = useCallback((job: Job) => {
    if (isEtfResearchJob(job)) commitNavigation(previous => ({ ...previous, page: 'etf', etfResult: etfResearchResultId(job) }))
    else if (isResearchJob(job)) commitNavigation(previous => ({ ...previous, page: 'research', research: job.id }))
    else if (['backtest', 'batch_backtest'].includes(job.kind)) commitNavigation(previous => ({ ...previous, page: 'results', result: job.result_id ?? null }))
    else commitNavigation(previous => ({ ...previous, page: 'data-admin', report: job.result_id ?? null }))
  }, [commitNavigation])
  const clearRequestedReport = useCallback(() => commitNavigation(previous => ({ ...previous, report: null }), { replace: true, scroll: false }), [commitNavigation])
  const prepareData = useCallback((request: DataPreparationRequest, origin: 'backtest' | 'research') => {
    setDataPreparation(newPreparationContext(request, origin))
    commitNavigation(previous => ({ ...previous, page: 'data-admin' }))
  }, [commitNavigation])
  const finishDataPreparation = useCallback(() => {
    const origin = dataPreparation?.origin ?? 'backtest'
    setDataPreparation(null)
    commitNavigation(previous => ({ ...previous, page: origin, report: null }))
  }, [commitNavigation, dataPreparation?.origin])
  const researchFromConfig = (config: BacktestConfig) => {
    workspace.transfer('research', config)
    setTransferNotice({ page: 'research', message: '已复制回测配置；可调整本页的评估区间与搜索设置后启动研究。' })
    go('research')
  }
  const backtestFromResearch = (config: BacktestConfig, message?: string) => {
    workspace.transfer('backtest', config)
    setTransferNotice({ page: 'backtest', message: message ?? `研究参数已载入回测。${config.selection_asof ? '选参截止日：' + config.selection_asof + '。' : ''}预览将自动更新。` })
    go('backtest')
  }
  const unsupportedRestartMessage = '当前后台仍是旧版本，无法通过网页重启。请关闭当前服务后重新运行 start.cmd；新版服务启动后，这个按钮即可直接重启项目。'
  const openRestartDialog = async () => {
    setRestartMessage('')
    try {
      const health = await get<{ restart_supported?: boolean }>('/health')
      if (health.restart_supported !== true) {
        setRestartState('failed')
        setRestartMessage(unsupportedRestartMessage)
        return
      }
      setRestartState('confirm')
    } catch (cause) {
      setRestartState('failed')
      setRestartMessage(cause instanceof Error ? cause.message : String(cause))
    }
  }
  const waitForRestart = async (instanceId: string) => {
    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      await new Promise(resolve => window.setTimeout(resolve, 1_000))
      try {
        const response = await fetch('/api/v1/health', { cache: 'no-store' })
        if (!response.ok) continue
        const health = await response.json() as { instance_id?: string }
        if (health.instance_id && health.instance_id !== instanceId) {
          window.location.reload()
          return
        }
      } catch {
        // The old process is expected to disappear during the restart.
      }
    }
    setRestartState('failed')
    setRestartMessage('服务未在 120 秒内恢复。请查看 .local/restart.log，或关闭窗口后重新运行 start.cmd。')
  }
  const restartProject = async () => {
    setRestartState('running')
    setRestartMessage('正在关闭当前服务并重新启动，页面会自动恢复。')
    try {
      const result = await post<{ previous_instance_id: string }>('/system/restart', { force: true })
      previousInstance.current = result.previous_instance_id
      await waitForRestart(result.previous_instance_id)
    } catch (cause) {
      setRestartState('failed')
      const message = cause instanceof Error ? cause.message : String(cause)
      setRestartMessage(message === 'Method Not Allowed' ? unsupportedRestartMessage : message)
    }
  }
  const retryRestartCheck = async () => {
    if (!previousInstance.current) return
    setRestartState('running')
    setRestartMessage('正在等待新服务响应…')
    await waitForRestart(previousInstance.current)
  }
  return <TaskProvider
    onOpenResult={openJobResult}
    filter={navigation.taskFilter}
    kind={navigation.taskKind}
    onFilterChange={(filter, kind) => commitNavigation(previous => ({ ...previous, taskFilter: filter, taskKind: kind }), { replace: true, scroll: false })}
  ><div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark"><TrendingUp size={19} strokeWidth={2.4} /></div><div><strong>StockRegression</strong><span>量化研究工作台</span></div></div>
      <div className="nav-caption">研究空间</div>
      <nav className="side-nav" aria-label="主导航">
        {nav.map(({ id, label, hint, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? 'active' : ''}`} onClick={() => go(id)}><span className="nav-icon"><Icon size={18} /></span><span className="nav-copy"><b>{label}</b><small>{hint}</small></span>{page === id && <i />}</button>)}
      </nav>
      <div className="sidebar-spacer" />
      <div className="local-card"><div className="local-icon"><Activity size={16} /></div><div><strong>本机研究环境</strong><span>数据保存在本地目录</span></div><span className="online-dot" /></div>
      <div className="sidebar-foot"><span className="tiny-dot" /> 本机研究工作台</div>
    </aside>
    {mobileOpen && <button className="mobile-scrim" aria-label="关闭导航" onClick={() => setMobileOpen(false)} />}
    <main className="main-shell">
      <header className="topbar"><button className="mobile-menu" aria-label="打开导航" onClick={() => setMobileOpen(!mobileOpen)}><Menu size={20} /></button><div className="breadcrumb"><span>研究工作台</span><i>/</i><strong>{nav.find((n) => n.id === page)?.label}</strong></div><div className="topbar-right"><TaskButton /><button className="project-restart" onClick={() => void openRestartDialog()} title="重启项目服务"><RotateCcw size={14} /><span>重启项目</span></button><span className="local-pill"><span />本地模式</span><div className="avatar">SR</div></div></header>
      <div className={`content-area page-${page}`}>
        {transferNotice?.page === page && <div className="notice success"><span>{transferNotice.message}</span><button className="text-link" onClick={() => setTransferNotice(null)}>收起</button></div>}
        <PageErrorBoundary>
          <Suspense fallback={<div className="route-loading">载入工作区…</div>}>
            {page === 'data' && <DataCenterPage requestedReport={navigation.report} onReportHandled={clearRequestedReport} onOpenDataManagement={() => go('data-admin')} />}
            {page === 'data-admin' && <DataManagementPage requestedReport={navigation.report} onReportHandled={clearRequestedReport} preparation={dataPreparation} onPreparationJob={(jobId) => setDataPreparation(current => current ? { ...current, preparationJobId: jobId } : current)} onReturn={dataPreparation ? finishDataPreparation : undefined} />}
            {page === 'etf' && <EtfResearchPage requestedResult={navigation.etfResult} onSelectedResultChange={(result) => commitNavigation(previous => ({ ...previous, etfResult: result }), { replace: true, scroll: false })} onLoadBacktest={backtestFromResearch} />}
            {page === 'backtest' && <BacktestPage onResearch={researchFromConfig} onPrepareData={(request) => prepareData(request, 'backtest')} />}
            {page === 'research' && (workspace.research && workspace.researchSettings ? <ResearchPage base={workspace.research} settings={workspace.researchSettings} onBaseChange={workspace.updateResearch} onSettingsChange={workspace.updateResearchSettings} onLoadConfig={backtestFromResearch} requestedResource={navigation.research} onSelectedResourceChange={(resource) => commitNavigation(previous => ({ ...previous, research: resource }), { replace: true, scroll: false })} completionDecision={navigation.researchDecision} onCompletionDecisionChange={(decision) => commitNavigation(previous => ({ ...previous, researchDecision: decision }), { replace: true, scroll: false })} onFeeDefaultsEdit={workspace.rememberFees} onDefaultsEdit={workspace.rememberStrategy} onRestore={() => workspace.restore('research')} onPrepareData={(request) => prepareData(request, 'research')} /> : <Loading label="读取本机研究设置…" />)}
            {page === 'results' && <ResultsPage selectedResultId={navigation.result} selectedCompareId={navigation.compare} view={navigation.resultView} onSelectionChange={(result, compare) => commitNavigation(previous => ({ ...previous, result, compare }), { replace: true, scroll: false })} onViewChange={(resultView) => commitNavigation(previous => ({ ...previous, resultView }), { replace: true, scroll: false })} />}
          </Suspense>
        </PageErrorBoundary>
      </div>
    </main>
    {restartState === 'confirm' && <div className="modal-scrim" onMouseDown={event => { if (event.target === event.currentTarget) setRestartState('idle') }}><div className="confirm-modal restart-confirm" role="dialog" aria-modal="true" aria-labelledby="restart-title"><div className="restart-confirm-icon"><RotateCcw size={20} /></div><h3 id="restart-title">重启项目</h3><p>将关闭并重新启动本地服务。正在运行的任务会被中断，已排队任务会在服务恢复后继续。</p><div className="button-row"><Button variant="quiet" onClick={() => setRestartState('idle')}>取消</Button><Button variant="danger" onClick={() => void restartProject()}>确认重启</Button></div></div></div>}
    {(restartState === 'running' || restartState === 'failed') && <div className="restart-overlay" role="alert" aria-live="assertive"><div className="restart-panel"><div className={`restart-spinner ${restartState === 'running' ? 'running' : ''}`}><RotateCcw size={24} /></div><h2>{restartState === 'running' ? '正在重启项目' : '自动重启未完成'}</h2><p>{restartMessage}</p>{restartState === 'running' ? <div className="restart-track"><i /></div> : <div className="button-row"><Button variant="quiet" onClick={() => setRestartState('idle')}>关闭</Button>{previousInstance.current && <Button onClick={() => void retryRestartCheck()}>重新检查</Button>}</div>}</div></div>}
  </div></TaskProvider>
}

export { defaultConfig }
