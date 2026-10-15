import { Component, lazy, Suspense, useState } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Activity, BarChart3, Database, FlaskConical, Layers3, Menu, TrendingUp } from 'lucide-react'
import type { BacktestConfig } from './types'
import { defaultConfig } from './components'

const DataCenterPage = lazy(() => import('./pages/DataCenterPage'))
const BacktestPage = lazy(() => import('./pages/BacktestPage'))
const ResultsPage = lazy(() => import('./pages/ResultsPage'))
const ResearchPage = lazy(() => import('./pages/ResearchPage'))

export type PageKey = 'data' | 'backtest' | 'results' | 'research'
const nav: Array<{ id: PageKey; label: string; hint: string; icon: typeof Database }> = [
  { id: 'data', label: '数据中心', hint: '数据与覆盖', icon: Database },
  { id: 'backtest', label: '回测工作台', hint: '策略配置', icon: Layers3 },
  { id: 'results', label: '结果分析', hint: '绩效与交易', icon: BarChart3 },
  { id: 'research', label: '参数研究', hint: '稳健性搜索', icon: FlaskConical },
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
  const [page, setPage] = useState<PageKey>('data')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [pendingConfig, setPendingConfig] = useState<BacktestConfig | null>(null)
  const [selectedResult, setSelectedResult] = useState<string | null>(null)
  const go = (next: PageKey) => { setPage(next); setMobileOpen(false) }
  return <div className="app-shell">
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
      <header className="topbar"><button className="mobile-menu" aria-label="打开导航" onClick={() => setMobileOpen(!mobileOpen)}><Menu size={20} /></button><div className="breadcrumb"><span>研究工作台</span><i>/</i><strong>{nav.find((n) => n.id === page)?.label}</strong></div><div className="topbar-right"><span className="local-pill"><span />本地模式</span><div className="avatar">SR</div></div></header>
      <div className="content-area">
        <PageErrorBoundary>
          <Suspense fallback={<div className="route-loading">载入工作区…</div>}>
            {page === 'data' && <DataCenterPage />}
            {page === 'backtest' && <BacktestPage initialConfig={pendingConfig} clearInitialConfig={() => setPendingConfig(null)} onOpenResult={(id) => { setSelectedResult(id); go('results') }} />}
            {page === 'results' && <ResultsPage requestedResult={selectedResult} clearRequestedResult={() => setSelectedResult(null)} />}
            {page === 'research' && <ResearchPage onLoadConfig={(config) => { setPendingConfig(config); go('backtest') }} />}
          </Suspense>
        </PageErrorBoundary>
      </div>
    </main>
  </div>
}

export { defaultConfig }
