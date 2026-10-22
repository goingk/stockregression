/** URL and History API state for the single-page workbench. */

export type AppPage = 'data' | 'data-admin' | 'backtest' | 'research' | 'results'
export type ResultView = 'year' | 'month'

export interface NavigationState {
  page: AppPage
  result: string | null
  compare: string | null
  research: string | null
  report: string | null
  resultView: ResultView
  researchDecision: string | null
  taskFilter: string
  taskKind: string
}

export const NAVIGATION_STATE_KEY = 'stockregression'
export const APP_PAGES: readonly AppPage[] = ['data', 'data-admin', 'backtest', 'research', 'results']

export function defaultNavigation(): NavigationState {
  return {
    page: 'data', result: null, compare: null, research: null, report: null,
    resultView: 'year', researchDecision: null, taskFilter: 'all', taskKind: '',
  }
}

function cleanId(value: string | null): string | null {
  const text = (value ?? '').trim()
  return /^[A-Za-z0-9_-]{1,128}$/.test(text) ? text : null
}

function pageValue(value: string | null): AppPage {
  return APP_PAGES.includes(value as AppPage) ? value as AppPage : 'data'
}

/** Restore a page from a copied URL. Unknown or malformed ids become null. */
export function navigationFromUrl(search: string): NavigationState {
  const params = new URLSearchParams(search)
  const resultView = params.get('result_view') === 'month' ? 'month' : 'year'
  const taskFilter = params.get('task_filter') ?? 'all'
  const allowedFilters = new Set(['all', 'active', 'completed', 'partial', 'failed', 'cancelled'])
  return {
    page: pageValue(params.get('page')),
    result: cleanId(params.get('result')),
    compare: cleanId(params.get('compare')),
    research: cleanId(params.get('research')),
    report: cleanId(params.get('report')),
    resultView,
    researchDecision: cleanId(params.get('research_decision')),
    taskFilter: allowedFilters.has(taskFilter) ? taskFilter : 'all',
    taskKind: /^[a-z_]{0,64}$/.test(params.get('task_kind') ?? '') ? params.get('task_kind') ?? '' : '',
  }
}

/** Build a copyable URL. Selection ids are kept only on their owning page. */
export function navigationUrl(search: string, state: NavigationState): string {
  const params = new URLSearchParams(search)
  for (const key of [
    'page', 'result', 'compare', 'research', 'report', 'result_view',
    'research_decision', 'task_filter', 'task_kind',
  ]) params.delete(key)
  params.set('page', state.page)
  if (state.page === 'results') {
    if (state.result) params.set('result', state.result)
    if (state.compare) params.set('compare', state.compare)
    if (state.resultView === 'month') params.set('result_view', 'month')
  }
  if (state.page === 'research') {
    if (state.research) params.set('research', state.research)
    if (state.researchDecision) params.set('research_decision', state.researchDecision)
  }
  if (state.page === 'data' || state.page === 'data-admin') {
    if (state.report) params.set('report', state.report)
  }
  if (state.taskFilter !== 'all') params.set('task_filter', state.taskFilter)
  if (state.taskKind) params.set('task_kind', state.taskKind)
  const query = params.toString()
  return query ? `?${query}` : ''
}

export function navigationHistoryState(state: NavigationState): { [NAVIGATION_STATE_KEY]: NavigationState } {
  return { [NAVIGATION_STATE_KEY]: state }
}

export function navigationFromHistory(value: unknown): NavigationState | null {
  if (!value || typeof value !== 'object') return null
  const candidate = (value as Record<string, unknown>)[NAVIGATION_STATE_KEY]
  if (!candidate || typeof candidate !== 'object') return null
  const item = candidate as Record<string, unknown>
  // State is only trusted after the same validation used for URL values.
  const params = new URLSearchParams(navigationUrl('', {
    ...defaultNavigation(),
    page: pageValue(typeof item.page === 'string' ? item.page : null),
    result: typeof item.result === 'string' ? item.result : null,
    compare: typeof item.compare === 'string' ? item.compare : null,
    research: typeof item.research === 'string' ? item.research : null,
    report: typeof item.report === 'string' ? item.report : null,
    resultView: item.resultView === 'month' ? 'month' : 'year',
    researchDecision: typeof item.researchDecision === 'string' ? item.researchDecision : null,
    taskFilter: typeof item.taskFilter === 'string' ? item.taskFilter : 'all',
    taskKind: typeof item.taskKind === 'string' ? item.taskKind : '',
  }))
  return navigationFromUrl(params.toString())
}
