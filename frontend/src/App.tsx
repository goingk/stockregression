import { Component, lazy, Suspense, useCallback, useState } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Activity, BarChart3, Database, FlaskConical, Layers3, Menu, TrendingUp } from 'lucide-react'
import { defaultConfig, Loading } from './components'
import { TaskButton, TaskProvider } from './TaskCenter'
import { WorkbenchProvider, useWorkspaces } from './WorkbenchState'
import { isResearchJob } from './taskState'
import type { BacktestConfig, Job } from './types'

const DataCenterPage = lazy(() => import('./pages/DataCenterPage'))
const BacktestPage = lazy(() => import('./pages/BacktestPage'))
const ResultsPage = lazy(() => import('./pages/ResultsPage'))
const ResearchPage = lazy(() => import('./pages/ResearchPage'))

export type PageKey = 'data' | 'backtest' | 'research' | 'results'
const nav: Array<{ id: PageKey; label: string; hint: string; icon: typeof Database }> = [
  { id: 'data', label: '行情', hint: '股票 · ETF · 自选', icon: Database },
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
  const [page, setPage] = useState<PageKey>('data')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [selectedResult, setSelectedResult] = useState<string | null>(null)
  const [selectedResearch, setSelectedResearch] = useState<string | null>(null)
  const [selectedReport, setSelectedReport] = useState<string | null>(null)
  const [transferNotice, setTransferNotice] = useState<{ page: PageKey; message: string } | null>(null)
  const go = useCallback((next: PageKey) => { setPage(next); setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'instant' }) }, [])
  const openJobResult = useCallback((job: Job) => {
    if (isResearchJob(job)) { setSelectedResearch(job.id); go('research') }
    else if (['backtest', 'batch_backtest'].includes(job.kind)) { setSelectedResult(job.result_id ?? null); go('results') }
    else { setSelectedReport(job.result_id ?? null); go('data') }
  }, [go])
  const clearRequestedResult = useCallback(() => setSelectedResult(null), [])
  const clearRequestedReport = useCallback(() => setSelectedReport(null), [])
  const researchFromConfig = (config: BacktestConfig) => {
    workspace.transfer('research', config)
    setTransferNotice({ page: 'research', message: '已复制回测配置；可调整本页的评估区间与搜索设置后启动研究。' })
    go('research')
  }
  const backtestFromResearch = (config: BacktestConfig) => {
    workspace.transfer('backtest', config)
    setTransferNotice({ page: 'backtest', message: `研究参数已载入回测。${config.selection_asof ? '选参截止日：' + config.selection_asof + '。' : ''}预览将自动更新。` })
    go('backtest')
  }
  return <TaskProvider onOpenResult={openJobResult}><div className="app-shell">
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
      <header className="topbar"><button className="mobile-menu" aria-label="打开导航" onClick={() => setMobileOpen(!mobileOpen)}><Menu size={20} /></button><div className="breadcrumb"><span>研究工作台</span><i>/</i><strong>{nav.find((n) => n.id === page)?.label}</strong></div><div className="topbar-right"><TaskButton /><span className="local-pill"><span />本地模式</span><div className="avatar">SR</div></div></header>
      <div className={`content-area page-${page}`}>
        {transferNotice?.page === page && <div className="notice success"><span>{transferNotice.message}</span><button className="text-link" onClick={() => setTransferNotice(null)}>收起</button></div>}
        <PageErrorBoundary>
          <Suspense fallback={<div className="route-loading">载入工作区…</div>}>
            {page === 'data' && <DataCenterPage requestedReport={selectedReport} onReportHandled={clearRequestedReport} />}
            {page === 'backtest' && <BacktestPage onResearch={researchFromConfig} />}
            {page === 'research' && (workspace.research && workspace.researchSettings ? <ResearchPage base={workspace.research} settings={workspace.researchSettings} onBaseChange={workspace.updateResearch} onSettingsChange={workspace.updateResearchSettings} onLoadConfig={backtestFromResearch} requestedJobId={selectedResearch} onSelectedJobChange={setSelectedResearch} onFeeDefaultsEdit={workspace.rememberFees} onDefaultsEdit={workspace.rememberStrategy} onRestore={() => workspace.restore('research')} /> : <Loading label="读取本机研究设置…" />)}
            {page === 'results' && <ResultsPage requestedResult={selectedResult} clearRequestedResult={clearRequestedResult} />}
          </Suspense>
        </PageErrorBoundary>
      </div>
    </main>
  </div></TaskProvider>
}

export { defaultConfig }
