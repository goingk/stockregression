import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Download, FlaskConical, RefreshCw, SearchCheck, ShieldCheck } from 'lucide-react'
import { download, get, post } from '../api'
import { LatestRequest } from '../requestState'
import { resultsPageQuery } from '../resultsReview'
import Disclosure from '../Disclosure'
import { useTasks } from '../TaskCenter'
import { useWorkspaces } from '../WorkbenchState'
import { flushConfiguration } from '../useWorkbenchDefaults'
import { isEtfResearchJob } from '../taskState'
import {
  Badge, Button, Card, Empty, ErrorBox, Field, Loading, NumField, PageHeader, SelectInput, Stat,
  TextInput, dateText, money, percent, useErrorMessage,
} from '../components'
import type { BacktestConfig, FeeConfig, ResultSummary } from '../types'
import {
  buildEtfResearchRequest, dataStatusLabel, dataStatusTone, etfDataStatusCounts, etfMetricRows, etfMetricText,
  etfRequestKey, etfSymbolValidationPath, executionFromBacktest, filterReasonsText, gapText, itemsPageCount,
  parseSelectedEtfSymbols, resultStatusLabel, resultStatusTone, safeItemsOffset, scoreDisplay, settingsError,
  stopReasonLabel, validationStatusLabel,
} from '../etfResearch.ts'
import type {
  EtfCandidate, EtfExecution, EtfItemsResponse, EtfPreflight, EtfResearchRequest,
  EtfResearchSettings, EtfResult, EtfRow, EtfValidationRow,
} from '../etfResearch.ts'
import { isEtfResearchResult, isEtfValidationResult } from '../etfResearch.ts'

type Props = {
  requestedResult?: string | null
  onSelectedResultChange?: (resultId: string) => void
  onLoadBacktest?: (config: BacktestConfig, message?: string) => void
}

type PreflightState = {
  key: string
  value: EtfPreflight
} | null

const defaultSettings: EtfResearchSettings = {
  scope: 'watchlist',
  symbolsText: '',
  as_of: '',
  window_days: 120,
  validation_top_k: 10,
  min_trades: 10,
  max_principal_drawdown: .2,
  max_runtime_seconds: 3600,
}

const terminalStatuses = new Set(['completed', 'partial', 'failed', 'cancelled'])

function scopeLabel(scope: string) {
  return ({ watchlist: '自选 ETF', local: '本地 ETF', selected: '指定 ETF' } as Record<string, string>)[scope] ?? scope
}

function compactGapText(gaps: EtfPreflight['rows'][number]['gaps']) {
  if (!gaps.length) return '—'
  const labels = gaps.slice(0, 2).map(gapText)
  return labels.join('；') + (gaps.length > 2 ? `；另 ${gaps.length - 2} 项` : '')
}

function jsonText(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2)
}

function ValidationBadge({ status }: { status: string }) {
  const tone = status === 'completed' ? 'green' : status === 'no_qualified' ? 'amber' : status === 'partial' ? 'amber' : status === 'cancelled' ? 'neutral' : status === 'unavailable' ? 'red' : 'neutral'
  return <Badge tone={tone}>{validationStatusLabel(status)}</Badge>
}

export default function EtfResearchPage({ requestedResult, onSelectedResultChange, onLoadBacktest }: Props) {
  const workspace = useWorkspaces()
  const tasks = useTasks()
  const err = useErrorMessage()
  const backtest = workspace.backtest

  const [settings, setSettings] = useState<EtfResearchSettings>(defaultSettings)
  const [execution, setExecution] = useState<EtfExecution | null>(null)
  const [preflight, setPreflight] = useState<PreflightState>(null)
  const [preflightBusy, setPreflightBusy] = useState(false)
  const [preflightError, setPreflightError] = useState('')
  const [researchBusy, setResearchBusy] = useState(false)
  const [validationBusy, setValidationBusy] = useState(false)
  const [validationJobId, setValidationJobId] = useState('')
  const [rowBusy, setRowBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const [history, setHistory] = useState<ResultSummary[]>([])
  const [historyOffset, setHistoryOffset] = useState(0)
  const [historyMore, setHistoryMore] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [selectedResultId, setSelectedResultId] = useState('')
  const [result, setResult] = useState<EtfResult | null>(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [refreshToken, setRefreshToken] = useState(0)
  const [items, setItems] = useState<EtfRow[]>([])
  const [itemsTotal, setItemsTotal] = useState(0)
  const [itemsOffset, setItemsOffset] = useState(0)
  const [itemsLoading, setItemsLoading] = useState(false)
  const [validationItems, setValidationItems] = useState<EtfValidationRow[]>([])
  const [validationItemsTotal, setValidationItemsTotal] = useState(0)
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [detail, setDetail] = useState<EtfRow | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const preflightLatest = useRef(new LatestRequest<string>())
  const historyLatest = useRef(new LatestRequest<string>())
  const itemLatest = useRef(new LatestRequest<string>())
  const detailLatest = useRef(new LatestRequest<string>())
  const taskSnapshotReady = useRef(false)
  const seenTerminalEtfJobs = useRef('')
  const seenTerminalEtfResearchJobs = useRef('')

  useEffect(() => {
    if (!backtest) return
    setExecution(executionFromBacktest(backtest))
  }, [backtest])

  const updateSetting = <K extends keyof EtfResearchSettings>(key: K, value: EtfResearchSettings[K]) => {
    setSettings(current => ({ ...current, [key]: value }))
    setPreflightError('')
  }
  const updateExecution = <K extends keyof EtfExecution>(key: K, value: EtfExecution[K]) => {
    setExecution(current => current ? { ...current, [key]: value } : current)
  }
  const updateFee = <K extends keyof FeeConfig>(key: K, value: FeeConfig[K]) => {
    setExecution(current => current ? { ...current, fees: { ...current.fees, [key]: value } } : current)
  }

  const requestKey = useMemo(() => execution ? etfRequestKey(settings, execution) : '', [settings, execution])
  const currentPreflight = preflight?.key === requestKey ? preflight.value : null
  const requestError = settingsError(settings, execution)
  const selectedSymbols = parseSelectedEtfSymbols(settings.symbolsText)

  const preflightFor = useCallback(async (targetSettings: EtfResearchSettings, targetExecution: EtfExecution) => {
    const validation = settingsError(targetSettings, targetExecution)
    if (validation) {
      setPreflightError(validation)
      throw new Error(validation)
    }
    const request = buildEtfResearchRequest(targetSettings, targetExecution)
    const key = JSON.stringify(request)
    const ticket = preflightLatest.current.begin(key)
    setPreflightBusy(true); setPreflightError('')
    try {
      const value = await post<EtfPreflight>('/etf-research/preflight', request)
      if (preflightLatest.current.isCurrent(ticket)) setPreflight({ key, value })
      return value
    } catch (cause) {
      if (preflightLatest.current.isCurrent(ticket)) setPreflightError(err(cause))
      throw cause
    } finally {
      if (preflightLatest.current.isCurrent(ticket)) setPreflightBusy(false)
    }
  }, [err])

  const runPreflight = async () => {
    if (!execution) { setPreflightError('当前回测草稿尚未载入。'); return }
    setError('')
    try { await preflightFor(settings, execution) } catch { /* The inline preflight error owns the message. */ }
  }

  const loadHistory = useCallback(async (offset = 0, append = false) => {
    const ticket = historyLatest.current.begin(`${offset}:${append}`)
    setHistoryLoading(true)
    try {
      const page = await get<ResultSummary[]>(resultsPageQuery(offset, 50, 'etf_research'))
      if (!historyLatest.current.isCurrent(ticket)) return
      setHistory(current => append ? [...current, ...page.filter(item => !current.some(existing => existing.id === item.id))] : page)
      setHistoryOffset(offset + page.length)
      setHistoryMore(page.length >= 50)
    } catch (cause) {
      if (historyLatest.current.isCurrent(ticket)) setError('ETF 研究历史读取失败：' + err(cause))
    } finally { if (historyLatest.current.isCurrent(ticket)) setHistoryLoading(false) }
  }, [err])

  useEffect(() => { void loadHistory(0, false) }, [loadHistory])

  const selectResult = useCallback((resultId: string) => {
    if (resultId !== selectedResultId) {
      itemLatest.current.invalidate()
      detailLatest.current.invalidate()
      setResult(null)
      setItems([])
      setItemsTotal(0)
      setValidationItems([])
      setValidationItemsTotal(0)
      setItemsLoading(false)
      setSelectedSymbol('')
      setDetail(null)
      setDetailLoading(false)
    }
    setSelectedResultId(resultId)
    setItemsOffset(0)
    onSelectedResultChange?.(resultId)
  }, [onSelectedResultChange, selectedResultId])

  useEffect(() => {
    if (requestedResult && requestedResult !== selectedResultId) selectResult(requestedResult)
  }, [requestedResult, selectedResultId, selectResult])

  useEffect(() => {
    if (!requestedResult && !selectedResultId && history[0]) selectResult(history[0].id)
  }, [requestedResult, selectedResultId, history, selectResult])

  useEffect(() => {
    if (!selectedResultId) { setResult(null); return }
    let active = true
    setResultLoading(true); setError('')
    get<EtfResult>('/results/' + encodeURIComponent(selectedResultId))
      .then(value => {
        if (!active) return
        if (!isEtfResearchResult(value) && !isEtfValidationResult(value)) {
          setResult(null)
          setError('链接结果不是 ETF 研究或本地校验报告。')
          return
        }
        setResult(value)
      })
      .catch(cause => { if (active) setError('ETF 结果读取失败：' + err(cause)) })
      .finally(() => { if (active) setResultLoading(false) })
    return () => { active = false }
  }, [selectedResultId, refreshToken, err])

  useEffect(() => {
    if (!selectedResultId) {
      setItems([]); setItemsTotal(0); setValidationItems([]); setValidationItemsTotal(0); return
    }
    if (!result || result.id !== selectedResultId) return
    const key = selectedResultId + ':' + result.kind + ':' + itemsOffset + ':' + refreshToken
    const ticket = itemLatest.current.begin(key)
    setItemsLoading(true)
    const path = `/etf-research/${encodeURIComponent(selectedResultId)}/items?offset=${itemsOffset}&limit=50`
    const request = isEtfValidationResult(result)
      ? get<EtfItemsResponse<EtfValidationRow>>(path)
      : get<EtfItemsResponse<EtfRow>>(path)
    request
      .then(value => {
        if (!itemLatest.current.isCurrent(ticket)) return
        if (isEtfValidationResult(result)) {
          setValidationItems((value.items ?? []) as EtfValidationRow[])
          setValidationItemsTotal(Number(value.total ?? 0))
        } else {
          setItems((value.items ?? []) as EtfRow[])
          setItemsTotal(Number(value.total ?? 0))
        }
      })
      .catch(cause => { if (itemLatest.current.isCurrent(ticket)) setError((isEtfValidationResult(result) ? 'ETF 校验报告' : 'ETF 排名') + '读取失败：' + err(cause)) })
      .finally(() => { if (itemLatest.current.isCurrent(ticket)) setItemsLoading(false) })
  }, [selectedResultId, result, itemsOffset, refreshToken, err])

  const openSymbol = async (symbol: string) => {
    if (!selectedResultId) return
    const existing = items.find(item => item.symbol === symbol)
    setSelectedSymbol(symbol)
    setDetail(existing ?? null)
    const key = selectedResultId + ':' + symbol
    const ticket = detailLatest.current.begin(key)
    setDetailLoading(true)
    try {
      const page = await get<EtfItemsResponse<EtfRow>>(`/etf-research/${encodeURIComponent(selectedResultId)}/items?offset=0&limit=1&symbol=${encodeURIComponent(symbol)}`)
      if (detailLatest.current.isCurrent(ticket)) setDetail(page.items?.[0] ?? null)
    } catch (cause) {
      if (detailLatest.current.isCurrent(ticket)) setError('ETF 标的详情读取失败：' + err(cause))
    } finally {
      if (detailLatest.current.isCurrent(ticket)) setDetailLoading(false)
    }
  }

  const etfJobs = useMemo(() => tasks.jobs.filter(isEtfResearchJob), [tasks.jobs])
  const terminalEtfSignature = etfJobs
    .filter(job => terminalStatuses.has(job.status))
    .map(job => `${job.id}:${job.status}:${job.result_id ?? ''}:${job.updated_at ?? ''}`)
    .sort().join(',')
  const terminalEtfResearchSignature = etfJobs
    .filter(job => job.kind === 'etf_research' && terminalStatuses.has(job.status))
    .map(job => `${job.id}:${job.status}:${job.result_id ?? ''}:${job.updated_at ?? ''}`)
    .sort().join(',')

  useEffect(() => {
    if (tasks.loading) return
    if (!taskSnapshotReady.current) {
      taskSnapshotReady.current = true
      seenTerminalEtfJobs.current = terminalEtfSignature
      seenTerminalEtfResearchJobs.current = terminalEtfResearchSignature
      return
    }
    if (terminalEtfSignature === seenTerminalEtfJobs.current) return
    seenTerminalEtfJobs.current = terminalEtfSignature
    const researchChanged = terminalEtfResearchSignature !== seenTerminalEtfResearchJobs.current
    seenTerminalEtfResearchJobs.current = terminalEtfResearchSignature
    void loadHistory(0, false)
    if (!researchChanged) return
    const completed = etfJobs
      .filter(job => job.kind === 'etf_research' && ['completed', 'partial', 'cancelled'].includes(job.status) && job.result_id)
      .sort((left, right) => (right.updated_at ?? right.created_at ?? '').localeCompare(left.updated_at ?? left.created_at ?? ''))[0]
    if (completed?.result_id) selectResult(completed.result_id)
  }, [tasks.loading, terminalEtfSignature, terminalEtfResearchSignature, etfJobs, loadHistory, selectResult])

  useEffect(() => {
    if (!validationJobId) return
    const job = tasks.jobs.find(item => item.id === validationJobId)
    if (!job || !terminalStatuses.has(job.status)) return
    setValidationJobId('')
    if (job.status === 'failed') {
      setError('ETF 数据校验失败：' + (job.error ?? job.message ?? '请查看任务中心'))
      return
    }
    setNotice(job.status === 'completed' ? '数据校验任务已完成，正在重新预检范围。' : '数据校验任务已结束，正在重新预检范围。')
    if (execution) void preflightFor(settings, execution).catch(() => undefined)
  }, [tasks.jobs, validationJobId, execution, settings, preflightFor])

  const submitResearch = async (request: EtfResearchRequest, label: string) => {
    const configurationRevision = await flushConfiguration()
    const response = await tasks.run(label, () => post<{ job_id: string }>('/etf-research', {
      ...request,
      configuration_revision: configurationRevision,
    }))
    setNotice(`${label}任务已提交；任务完成后会自动更新 ETF 研究历史。`)
    return response
  }

  const startResearch = async () => {
    if (!execution) return
    if (requestError) { setError(requestError); return }
    if (!currentPreflight) { setError('研究范围已修改，请先重新预检。'); return }
    if (currentPreflight.counts.ready < 1) { setError('预检范围没有可研究的就绪 ETF。'); return }
    setResearchBusy(true); setError('')
    try {
      await submitResearch(buildEtfResearchRequest(settings, execution), 'ETF研究')
    } catch (cause) { setError(err(cause)) } finally { setResearchBusy(false) }
  }

  const startDataValidation = async () => {
    if (!execution) return
    if (requestError) { setError(requestError); return }
    setValidationBusy(true); setError(''); setNotice('')
    try {
      const response = await tasks.run('ETF数据校验', () => post<{ job_id: string }>(
        '/etf-research/validation', buildEtfResearchRequest(settings, execution),
      ))
      setValidationJobId(response.job_id)
      setNotice('数据校验任务已提交；完成后会自动重新预检。')
    } catch (cause) { setError(err(cause)) } finally { setValidationBusy(false) }
  }

  const validateSingleSymbol = async (row: EtfRow) => {
    if (!selectedResultId) { setError('请先选择一个已保存的 ETF 研究结果。'); return }
    setRowBusy(row.symbol); setError(''); setNotice('')
    try {
      await tasks.run(`ETF单标的核验 · ${row.symbol}`, () => post<{ job_id: string }>(
        etfSymbolValidationPath(selectedResultId, row.symbol),
      ))
      setNotice(`${row.symbol} 已按该历史结果的冻结数据版本、窗口和参数提交核验；不会切换到今天的截止日。该单标的横截面样本为 1，GSS 为空属于正常情况。`)
    } catch (cause) { setError(err(cause)) } finally { setRowBusy('') }
  }

  const replayCandidate = async (row: EtfRow, candidate: EtfCandidate) => {
    setRowBusy('replay:' + String(candidate.id)); setError(''); setNotice('')
    try {
      const version = encodeURIComponent(row.version)
      await tasks.run('ETF候选复现', () => post<{ job_id: string }>(
        '/backtests?version=' + version, candidate.replay_config,
      ))
      setNotice('候选复现已按所选数据版本提交，可在任务中心查看回测结果。')
    } catch (cause) { setError(err(cause)) } finally { setRowBusy('') }
  }

  const loadForwardConfig = (row: EtfRow) => {
    const forward = row.validation.forward_config
    if (!forward) { setError('该标的没有可载入的前瞻配置。'); return }
    const next: BacktestConfig = {
      ...structuredClone(forward),
      selection_asof: forward.selection_asof ?? result?.as_of ?? null,
    }
    onLoadBacktest?.(next, 'ETF 候选参数已载入回测草稿，未自动提交。尚无选参截止日后的本地行情时，仅保留草稿。')
  }

  const exportResult = async () => {
    if (!selectedResultId) return
    setError('')
    try {
      const suffix = isEtfValidationResult(result) ? 'etf-validation' : 'etf-research'
      await download(
        '/results/' + encodeURIComponent(selectedResultId) + '/export?kind=json',
        selectedResultId + '-' + suffix + '.json',
      )
    } catch (cause) { setError(err(cause)) }
  }

  const resetExecution = () => {
    if (!backtest) return
    setExecution(executionFromBacktest(backtest))
    setNotice('已重新读取当前回测草稿的数量和费用设置。')
  }

  if (!backtest || !execution) return <Loading label='读取当前回测草稿与 ETF 研究历史…' />

  const pageSize = 50
  const validationReport = isEtfValidationResult(result) ? result : null
  const researchResult = isEtfResearchResult(result) ? result : null
  const visibleItemsTotal = validationReport ? validationItemsTotal : itemsTotal
  const pageCount = itemsPageCount(visibleItemsTotal, pageSize)
  const currentPage = Math.floor(safeItemsOffset(itemsOffset, visibleItemsTotal, pageSize) / pageSize) + 1
  const metricRows = detail ? etfMetricRows(detail.metrics ?? {}, detail.metric_details ?? {}) : []
  const detailValidation = detail?.validation
  const detailCandidates = detailValidation?.candidates ?? []
  const gridBasis = detail?.grid_basis
  const resultDataStatusCounts = researchResult ? etfDataStatusCounts(researchResult.counts) : []
  const detailFilterReasons = detail?.filter_reasons ?? []
  const detailGaps = detail?.gaps ?? []
  const detailWarnings = detail?.warnings ?? []
  const historyControls = <div className='result-select etf-history-select'>
    <SelectInput aria-label='选择 ETF 研究历史' value={selectedResultId} onChange={event => selectResult(event.target.value)}>
      <option value=''>选择研究历史</option>
      {selectedResultId && !history.some(item => item.id === selectedResultId) && <option value={selectedResultId}>{validationReport ? '当前本地校验报告' : '链接中的 ETF 结果'}</option>}
      {history.map(item => <option key={item.id} value={item.id}>{item.symbol || 'ETF 范围'} · {dateText(item.created_at)}</option>)}
    </SelectInput>
    <Button variant='quiet' busy={historyLoading} onClick={() => void loadHistory(0, false)}><RefreshCw size={14} /></Button>
  </div>

  return <div className='etf-research-page'>
    <PageHeader
      eyebrow='ETF GRID RESEARCH'
      title='ETF 网格研究'
      description='先预检本地分钟数据，再生成横截面排名，并按排名自动核验前 N 只 ETF 的样本内参数。'
      action={<div className='button-row'><Button variant='quiet' busy={historyLoading} onClick={() => { void tasks.refresh(); void loadHistory(0, false) }}><RefreshCw size={14} />刷新历史</Button><Button variant='secondary' disabled={!selectedResultId} onClick={() => void exportResult()}><Download size={14} />导出 JSON</Button></div>}
    />
    <div className='etf-sample-banner'><Badge tone='blue'>样本内</Badge><span>排名与候选回放均为样本内；截止日后的区间可在回测页另行回测。</span></div>
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {notice && <div className='notice success'><span>{notice}</span><button className='text-link' onClick={() => setNotice('')}>收起</button></div>}
    <div className='etf-research-layout'>
      <aside className='etf-research-config'>
        <Card title='研究范围' subtitle='预检只检查本地范围和数据状态；选择数据校验后，待任务完成再重新预检。'>
          <div className='form-grid two'>
            <Field label='ETF 范围'><SelectInput value={settings.scope} onChange={event => updateSetting('scope', event.target.value as EtfResearchSettings['scope'])}><option value='watchlist'>自选 ETF</option><option value='local'>全部本地 ETF</option><option value='selected'>指定 ETF</option></SelectInput></Field>
            <Field label='统计窗口（交易日）'><SelectInput value={String(settings.window_days)} onChange={event => updateSetting('window_days', Number(event.target.value) === 250 ? 250 : 120)}><option value='120'>120 日</option><option value='250'>250 日</option></SelectInput></Field>
          </div>
          {settings.scope === 'selected' && <Field label='ETF 代码' hint='空格、逗号或分号分隔；6 位代码会按首位补齐交易所。'><TextInput value={settings.symbolsText} onChange={event => updateSetting('symbolsText', event.target.value)} placeholder='例如 510210.SH, 159611.SZ' /></Field>}
          {settings.scope === 'selected' && selectedSymbols.invalid.length > 0 && <p className='inline-warning'>无法识别：{selectedSymbols.invalid.join('、')}</p>}
          <div className='form-grid two'>
            <Field label='截止日' hint='留空时使用本地分钟数据最近的已收盘交易日。'><TextInput type='date' value={settings.as_of} onChange={event => updateSetting('as_of', event.target.value)} /></Field>
            <NumField label='自动核验前 N 名' min={0} max={100} step='1' hint='按排名自动回测前 N 只 ETF；填 0 时只生成排名。' value={settings.validation_top_k} onChange={next => next != null && updateSetting('validation_top_k', next)} />
            <NumField label='最少成交笔数' min={0} max={1_000_000} step='1' value={settings.min_trades} onChange={next => next != null && updateSetting('min_trades', next)} />
            <NumField label='最大本金回撤' min={.01} max={100} step='1' percentValue suffix='%' value={settings.max_principal_drawdown} onChange={next => next != null && updateSetting('max_principal_drawdown', next)} />
            <NumField label='最长运行时间' min={1} max={86400} step='1' suffix='秒' value={settings.max_runtime_seconds} onChange={next => next != null && updateSetting('max_runtime_seconds', next)} />
          </div>
          {requestError && <p className='inline-warning'>{requestError}</p>}
          <div className='etf-launch-actions'><Button variant='secondary' busy={preflightBusy} onClick={() => void runPreflight()}><SearchCheck size={15} />预检范围</Button><Button busy={researchBusy} disabled={!currentPreflight || currentPreflight.counts.ready < 1 || Boolean(requestError)} onClick={() => void startResearch()}><FlaskConical size={15} />生成排名与核验</Button></div>
          {!currentPreflight && preflight && !preflightBusy && <p className='inline-warning'>研究范围或执行参数已修改，请重新预检。</p>}
        </Card>

        <Card title='当前回测执行设置' subtitle='从当前回测草稿读取；此处调整只用于本页 ETF 研究，不写入本机默认设置。' action={<Button variant='quiet' disabled={!backtest} onClick={resetExecution}><RefreshCw size={14} />重新读取</Button>}>
          <div className='form-grid two'>
            <NumField label='初始持股' min={0} step='100' suffix='份' value={execution.initial_shares} onChange={next => next != null && updateExecution('initial_shares', next)} />
            <NumField label='最低持股' min={0} step='100' suffix='份' value={execution.min_shares} onChange={next => next != null && updateExecution('min_shares', next)} />
            <NumField label='最高持股' min={0} step='100' suffix='份' hint='留空表示不限制。' value={execution.max_shares} onChange={next => updateExecution('max_shares', next)} />
            <NumField label='每次买入' min={100} step='100' suffix='份' value={execution.buy_shares} onChange={next => next != null && updateExecution('buy_shares', next)} />
            <NumField label='每次卖出' min={100} step='100' suffix='份' value={execution.sell_shares} onChange={next => next != null && updateExecution('sell_shares', next)} />
          </div>
          <Disclosure id='etf-execution-fees' title='费用与滑点' summary={`佣金 ${percent(execution.fees.commission_rate)} · 滑点 ${execution.fees.slippage_bps} 基点`}>
            <div className='form-grid two'>
              <NumField label='佣金费率' min={0} max={5} step='.01' percentValue suffix='%' value={execution.fees.commission_rate} onChange={next => next != null && updateFee('commission_rate', next)} />
              <NumField label='最低佣金' min={0} step='.01' suffix='元' value={execution.fees.min_commission} onChange={next => next != null && updateFee('min_commission', next)} />
              <NumField label='滑点' min={0} max={1000} step='.1' suffix='基点' value={execution.fees.slippage_bps} onChange={next => next != null && updateFee('slippage_bps', next)} />
            </div>
          </Disclosure>
          <p className='field-hint'>数量须为 100 份整数倍。ETF 资金语义沿用当前股数回测：成交时补足所需现金，不在这里设置固定现金预算。</p>
        </Card>
      </aside>

      <main className='etf-research-main'>
        <Card title='范围预检' subtitle='只检查数据就绪状态、窗口与版本；不会下载行情或现场扫描正文。' action={<Badge tone={currentPreflight?.counts.data_not_ready || currentPreflight?.counts.pending_validation ? 'amber' : currentPreflight ? 'green' : 'neutral'}>{currentPreflight ? '预检完成' : preflightBusy ? '预检中' : '等待预检'}</Badge>}>
          {preflightBusy && !currentPreflight ? <Loading label='正在检查 ETF 范围…' /> : preflightError && !currentPreflight ? <ErrorBox message={preflightError} onRetry={() => void runPreflight()} /> : !currentPreflight ? <Empty title='先预检研究范围' detail='预检后会显示数据就绪数量、缺口、窗口与锁定版本。' /> : <>
            {preflightError && <ErrorBox message={preflightError} onRetry={() => void runPreflight()} />}
            <div className='stat-grid five etf-preflight-stats'><Stat label='范围总数' value={currentPreflight.counts.total} /><Stat label='就绪' value={currentPreflight.counts.ready} tone='green' /><Stat label='待校验' value={currentPreflight.counts.pending_validation} tone={currentPreflight.counts.pending_validation ? 'amber' : 'neutral'} /><Stat label='历史不足' value={currentPreflight.counts.insufficient_history} tone={currentPreflight.counts.insufficient_history ? 'red' : 'neutral'} /><Stat label='数据未就绪' value={currentPreflight.counts.data_not_ready} tone={currentPreflight.counts.data_not_ready ? 'red' : 'neutral'} /></div>
            <div className='etf-preflight-meta'><span>范围：{scopeLabel(currentPreflight.scope)}</span><span>实际截止：{dateText(currentPreflight.as_of)}</span><span>请求截止：{currentPreflight.requested_as_of ?? '本地最近已收盘日'}</span><span>窗口：{currentPreflight.window_days} 日</span></div>
            <div className='table-wrap scroll-table etf-preflight-table'><table><thead><tr><th>ETF</th><th>状态</th><th>窗口</th><th>锁定版本</th><th>缺口</th><th>提示</th></tr></thead><tbody>{currentPreflight.rows.map(row => <tr key={row.symbol}><td><strong>{row.symbol_name || row.symbol}</strong><small>{row.symbol}</small></td><td><Badge tone={dataStatusTone(row.status)}>{dataStatusLabel(row.status)}</Badge></td><td>{dateText(row.window_start)}<br />至 {dateText(row.window_end)}</td><td><small>{row.version || '—'}</small></td><td><small>{compactGapText(row.gaps)}</small></td><td><small>{row.warnings.join('；') || '—'}</small></td></tr>)}</tbody></table></div>
            <div className='etf-preflight-actions'><Button variant='quiet' busy={validationBusy} disabled={currentPreflight.counts.pending_validation + currentPreflight.counts.insufficient_history + currentPreflight.counts.data_not_ready === 0} onClick={() => void startDataValidation()}><ShieldCheck size={15} />校验待处理数据</Button><span>{validationJobId ? '数据校验任务运行中，完成后自动重新预检。' : '数据校验不会自动启动研究。'}</span></div>
            <Disclosure id='etf-preflight-universe' title='锁定数据包与版本' summary={`${currentPreflight.universe.package_id || '未命名数据包'} · ${currentPreflight.universe.generation || '—'}`}><pre className='json-preview'>{jsonText(currentPreflight.universe)}</pre></Disclosure>
          </>}
        </Card>

        {validationReport ? <Card title='ETF 本地校验报告' className='etf-history-card etf-validation-report-card' action={historyControls}>
          {historyMore && <div className='button-row etf-history-more'><Button variant='quiet' busy={historyLoading} onClick={() => void loadHistory(historyOffset, true)}>加载更早研究历史</Button></div>}
          <div className='etf-result-heading'><div><Badge tone={resultStatusTone(validationReport.status)}>{resultStatusLabel(validationReport.status)}</Badge><Badge tone='amber'>本地质量校验</Badge></div><span>截止 {dateText(validationReport.as_of)} · {validationReport.rows_count} 只 · 方法 {validationReport.method_version}</span></div>
          {validationReport.stop_reason && <p className='inline-warning'>取消/停止原因：{stopReasonLabel(validationReport.stop_reason)}</p>}
          {validationReport.message && <p className='etf-report-message'>{validationReport.message}</p>}
          {validationReport.warnings.length > 0 && <div className='warning-list'>{validationReport.warnings.map((warning, index) => <div key={index}><span className='warning-dot' />{warning}</div>)}</div>}
          <div className='stat-grid five etf-result-stats'><Stat label='范围总数' value={validationReport.counts.total} /><Stat label='校验完成' value={validationReport.counts.completed ?? 0} tone='green' /><Stat label='部分完成' value={validationReport.counts.partial ?? 0} tone={validationReport.counts.partial ? 'amber' : 'neutral'} /><Stat label='失败' value={validationReport.counts.failed ?? 0} tone={validationReport.counts.failed ? 'red' : 'neutral'} /><Stat label='已取消' value={validationReport.counts.cancelled ?? 0} /></div>
          <div className='etf-preflight-meta etf-validation-report-meta'><span>数据包：{validationReport.universe.package_id || '—'}</span><span>代际：{validationReport.universe.generation || '—'}</span><span>实际截止：{dateText(validationReport.as_of)}</span><span>请求截止：{validationReport.request.as_of ?? '自动选择'}</span></div>
          <div className='etf-result-actions'><Button variant='quiet' busy={historyLoading || resultLoading || itemsLoading} onClick={() => { void tasks.refresh(); setRefreshToken(token => token + 1) }}><RefreshCw size={14} />刷新报告</Button><Button variant='quiet' onClick={() => void exportResult()}><Download size={14} />导出 JSON</Button><span>校验报告与 ETF 排名分开保存；可从右上角切换到已有研究历史。</span></div>
          <Disclosure id={`etf-validation-universe-${validationReport.id}`} title='锁定数据包与版本' summary={`${validationReport.universe.package_id || '未命名数据包'} · ${validationReport.universe.generation || '—'}`}><pre className='json-preview'>{jsonText(validationReport.universe)}</pre></Disclosure>
          <div className='table-wrap scroll-table etf-validation-report-table'><table><thead><tr><th>ETF</th><th>状态</th><th>原因与缺口</th><th>窗口 / 版本</th><th>校验明细</th></tr></thead><tbody>{validationItems.map(row => <tr key={row.symbol}><td><strong>{row.symbol_name || row.symbol}</strong><small>{row.symbol}</small></td><td><Badge tone={dataStatusTone(row.status)}>{resultStatusLabel(row.status)}</Badge></td><td><small>{row.reason || row.warnings?.join('；') || '—'}</small>{(row.gaps ?? []).length > 0 && <ul className='etf-report-gaps'>{(row.gaps ?? []).map((gap, index) => <li key={`${gap.kind}-${index}`}>{gapText(gap)}</li>)}</ul>}</td><td><small>{dateText(row.window_start)} 至 {dateText(row.window_end)}<br />版本 {row.version || '—'}</small></td><td>{(row.reports ?? []).length > 0 ? <Disclosure id={`etf-validation-reports-${validationReport.id}-${row.symbol}`} title='reports JSON' summary={`${(row.reports ?? []).length} 项`}><pre className='json-preview'>{jsonText(row.reports)}</pre></Disclosure> : <small>—</small>}</td></tr>)}</tbody></table></div>
          {itemsLoading && <Loading label='读取校验明细…' />}
          {!itemsLoading && !validationItems.length && <Empty title='没有校验明细' detail='该报告没有可显示的分页标的。' />}
          <div className='pagination-row etf-ranking-pagination'><span>第 {currentPage} / {pageCount} 页 · 共 {validationItemsTotal} 行</span><div><Button variant='quiet' disabled={itemsOffset <= 0 || itemsLoading} onClick={() => setItemsOffset(current => safeItemsOffset(current - pageSize, validationItemsTotal, pageSize))}>上一页</Button><Button variant='quiet' disabled={itemsOffset + pageSize >= validationItemsTotal || itemsLoading} onClick={() => setItemsOffset(current => safeItemsOffset(current + pageSize, validationItemsTotal, pageSize))}>下一页</Button></div></div>
        </Card> : (
        <Card title='ETF 研究结果' className='etf-history-card' action={historyControls}>
          {historyMore && <div className='button-row etf-history-more'><Button variant='quiet' busy={historyLoading} onClick={() => void loadHistory(historyOffset, true)}>加载更早 ETF 研究</Button></div>}
          {resultLoading && !result ? <Loading label='读取 ETF 结果…' /> : !selectedResultId ? <Empty title='尚无 ETF 研究结果' detail='完成范围预检后启动排名与核验，或从任务中心打开已有任务。' /> : !researchResult ? <Empty title='未找到 ETF 研究结果' detail='该结果不是 ETF 研究排名；可从右上角切换到已有研究历史。' /> : <>
            <div className='etf-result-heading'><div><Badge tone={resultStatusTone(researchResult.status)}>{resultStatusLabel(researchResult.status)}</Badge><Badge tone='blue'>样本内</Badge><Badge tone={researchResult.ranking_complete ? 'green' : 'amber'}>{researchResult.ranking_complete ? '排名完整' : '排名范围有限'}</Badge></div><span>截止 {dateText(researchResult.as_of)} · {researchResult.rows_count} 只 · 方法 {researchResult.method_version}</span></div>
            {researchResult.stop_reason && <p className='inline-warning'>停止原因：{stopReasonLabel(researchResult.stop_reason)}</p>}
            {researchResult.warnings.length > 0 && <div className='warning-list'>{researchResult.warnings.map((warning, index) => <div key={index}><span className='warning-dot' />{warning}</div>)}</div>}
            <div className='stat-grid five etf-result-stats'><Stat label='范围总数' value={researchResult.counts.total} /><Stat label='可评分' value={researchResult.counts.scored} /><Stat label='符合硬过滤' value={researchResult.counts.eligible} tone='green' /><Stat label='完成自动核验' value={researchResult.counts.validated} /><Stat label='有推荐参数' value={researchResult.counts.recommended} /></div>
            {resultDataStatusCounts.length > 0 && <div className='stat-grid four etf-data-status-stats'>{resultDataStatusCounts.map(item => <Stat key={item.status} label={item.label} value={item.count} tone={item.tone} />)}</div>}
            <div className='etf-result-actions'><Button variant='quiet' busy={historyLoading || resultLoading} onClick={() => { void tasks.refresh(); setRefreshToken(token => token + 1) }}><RefreshCw size={14} />刷新结果</Button><Button variant='quiet' onClick={() => void exportResult()}><Download size={14} />导出 JSON</Button><span>{scopeLabel(researchResult.request.scope)} · {researchResult.request.window_days} 日窗口 · 自动核验前 {researchResult.request.validation_top_k} 名 · 最少成交 {researchResult.request.min_trades}</span></div>

            <div className='table-wrap scroll-table etf-ranking-table'><table><thead><tr><th>排名</th><th>ETF</th><th>GSS</th><th>可入选</th><th>状态</th><th>RV</th><th>路径冗余</th><th>IOVR</th><th>VR 240m</th><th>跳跃</th><th>核验</th><th>操作</th></tr></thead><tbody>{items.map(row => <tr key={row.symbol} className={selectedSymbol === row.symbol ? 'selected-row' : ''}><td>{row.rank ?? '—'}</td><td><strong>{row.symbol_name || row.symbol}</strong><small>{row.symbol} · 横截面 n={row.score_cohort_size}</small></td><td><strong className='etf-score'>{scoreDisplay(row)}</strong></td><td><Badge tone={row.eligible ? 'green' : 'red'}>{row.eligible ? '符合硬过滤' : '已过滤'}</Badge></td><td><small>{dataStatusLabel(row.data_status || row.status)}</small></td><td>{etfMetricText('rv_annualized', row.metrics.rv_annualized)}</td><td>{etfMetricText('path_redundancy', row.metrics.path_redundancy)}</td><td>{etfMetricText('iovr', row.metrics.iovr)}</td><td>{etfMetricText('variance_ratio_240m', row.metrics.variance_ratio_240m)}</td><td>{etfMetricText('jump_ratio', row.metrics.jump_ratio)}</td><td><ValidationBadge status={row.validation.status} /></td><td><Button variant='quiet' busy={detailLoading && selectedSymbol === row.symbol} onClick={() => void openSymbol(row.symbol)}>诊断</Button></td></tr>)}</tbody></table></div>
            {itemsLoading && <Loading label='读取排名行…' />}
            {!itemsLoading && !items.length && <Empty title='没有排名行' detail='当前研究范围可能全部历史不足或数据未就绪。' />}
            <div className='pagination-row etf-ranking-pagination'><span>第 {currentPage} / {pageCount} 页 · 共 {itemsTotal} 行</span><div><Button variant='quiet' disabled={itemsOffset <= 0 || itemsLoading} onClick={() => setItemsOffset(current => safeItemsOffset(current - pageSize, itemsTotal, pageSize))}>上一页</Button><Button variant='quiet' disabled={itemsOffset + pageSize >= itemsTotal || itemsLoading} onClick={() => setItemsOffset(current => safeItemsOffset(current + pageSize, itemsTotal, pageSize))}>下一页</Button></div></div>
            {detail && <section className='etf-detail-section'>
              <div className='etf-detail-heading'><div><h2>{detail.symbol_name || detail.symbol}</h2><span>{detail.symbol} · {dateText(detail.window_start)} 至 {dateText(detail.window_end)} · 数据版本 {detail.version || '—'}</span></div><div><Badge tone={dataStatusTone(detail.data_status)}>{dataStatusLabel(detail.data_status)}</Badge><Badge tone={detail.eligible ? 'green' : 'red'}>{detail.eligible ? '符合硬过滤' : '未通过硬过滤'}</Badge><Badge tone='blue'>样本内</Badge></div></div>
              {(detailFilterReasons.length > 0 || detailGaps.length > 0 || detailWarnings.length > 0) && <div className='etf-filter-reasons'>{detailFilterReasons.length > 0 && <p><strong>过滤原因：</strong>{filterReasonsText(detailFilterReasons)}</p>}{detailGaps.length > 0 && <div className='etf-gap-list'><strong>数据缺口：</strong><ul>{detailGaps.map((gap, index) => <li key={`${gap.timeframe ?? ''}-${gap.kind}-${index}`}>{gapText(gap)}</li>)}</ul></div>}{detailWarnings.length > 0 && <p><strong>数据提示：</strong>{detailWarnings.join('；')}</p>}</div>}
              <div className='etf-detail-score-row'><div><small>GSS 横截面评分</small><strong>{scoreDisplay(detail)}</strong><span>样本数 {detail.score_cohort_size}{detail.rank != null ? ` · 排名 ${detail.rank}` : ''}</span></div><div><small>网格基准</small><strong>{money(gridBasis?.reference_price, 3)} 元</strong><span>当前 {money(gridBasis?.current_raw_price, 3)} · 下轨 {money(gridBasis?.lower_price, 3)} · 上轨 {money(gridBasis?.upper_price, 3)}</span></div><div><small>网格参数核验</small><ValidationBadge status={detailValidation?.status ?? 'not_selected'} /><span>{detailCandidates.length} 个候选</span></div></div>
              <div className='etf-metric-grid'>{metricRows.map(metric => <div key={metric.key}><small>{metric.label}</small><strong>{metric.value}</strong>{metric.detail?.reason && <span>{metric.detail.reason}</span>}{metric.detail?.sample_count != null && <span>样本 {metric.detail.sample_count}</span>}</div>)}</div>
              <div className='etf-grid-basis-grid'><span><small>ATR30</small><strong>{percent(gridBasis?.atr30_pct)}</strong></span><span><small>SVR</small><strong>{money(gridBasis?.svr, 3)}</strong></span><span><small>已实现偏度</small><strong>{money(gridBasis?.realized_skewness, 3)}</strong></span><span><small>已实现峰度</small><strong>{money(gridBasis?.realized_kurtosis, 3)}</strong></span><span><small>最小报价</small><strong>{money(gridBasis?.price_tick, 3)}</strong></span></div>
              {Object.keys(detail.score_contributions ?? {}).length > 0 && <Disclosure id='etf-score-contributions' title='GSS 分项贡献' summary={`${Object.keys(detail.score_contributions).length} 项`}><div className='etf-contribution-grid'>{Object.entries(detail.score_contributions).map(([key, value]) => <span key={key}><small>{key.replaceAll('_', ' ')}</small><strong>{value.toFixed(4)}</strong></span>)}</div></Disclosure>}
              <div className='etf-validation-section'><div className='etf-validation-title'><div><h3>参数核验与候选</h3><p>候选指标为样本内自动回放；GSS 与硬过滤资格独立展示。</p></div><ValidationBadge status={detailValidation?.status ?? 'not_selected'} /></div>
                {detailValidation?.reason && <p className='inline-warning'>核验原因：{detailValidation.reason}</p>}
                {detailValidation?.stop_reason && <p className='inline-warning'>核验停止原因：{stopReasonLabel(detailValidation.stop_reason)}</p>}
                {detailCandidates.length ? <div className='table-wrap scroll-table etf-candidate-table'><table><thead><tr><th>候选</th><th>基准 / 买入 / 卖出步长</th><th>参考 / 上下轨</th><th>网格上下格数</th><th>峰值本金收益</th><th>同资金持有</th><th>差值超额</th><th>最大回撤</th><th>成交 / 买卖</th><th>费用</th><th>资格</th><th>操作</th></tr></thead><tbody>{detailCandidates.map(candidate => <tr key={String(candidate.id)}><td><strong>#{String(candidate.id)}</strong>{detailValidation?.recommendation?.id === candidate.id && <Badge tone='blue'>推荐</Badge>}{candidate.asymmetric && <Badge tone='amber'>非对称</Badge>}</td><td>{percent(candidate.base_step)} / {percent(candidate.buy_step)} / {percent(candidate.sell_step)}</td><td>{money(candidate.reference_price, 3)} / {money(candidate.lower_price, 3)} / {money(candidate.upper_price, 3)}</td><td>{candidate.grids_down_count} / {candidate.grids_up_count}</td><td>{percent(candidate.metrics.peak_principal_return)}</td><td>{percent(candidate.metrics.matched_funding_hold_return)}</td><td>{percent(candidate.metrics.excess_return)}</td><td>{percent(candidate.metrics.max_principal_drawdown)}</td><td>{candidate.metrics.trade_count ?? '—'} / {candidate.metrics.buy_count ?? '—'} / {candidate.metrics.sell_count ?? '—'}</td><td>{money(candidate.metrics.total_fees)}</td><td><Badge tone={candidate.qualified ? 'green' : 'amber'}>{candidate.qualified ? '合格' : '未合格'}</Badge>{candidate.reasons.length > 0 && <small>{candidate.reasons.join('；')}</small>}</td><td><div className='button-row etf-candidate-actions'><Button variant='quiet' busy={rowBusy === 'replay:' + String(candidate.id)} onClick={() => void replayCandidate(detail, candidate)}>按此复现</Button></div></td></tr>)}</tbody></table></div> : <Empty title='没有参数候选' detail={validationStatusLabel(detailValidation?.status ?? 'not_selected')} />}
                <div className='etf-detail-actions'><Button variant='secondary' busy={rowBusy === detail.symbol} onClick={() => void validateSingleSymbol(detail)}><SearchCheck size={15} />按原数据核验此 ETF</Button><Button variant='quiet' disabled={!detail.validation.forward_config} onClick={() => loadForwardConfig(detail)}>载入截止日后回测草稿</Button>{!detail.validation.forward_config && <span>尚无选参截止日后的回测配置。</span>}</div>
              </div>
              <Disclosure id='etf-detail-diagnostics' title='诊断字段与版本' summary={'版本 ' + (detail.version || '—')}><pre className='json-preview'>{jsonText(detail.diagnostics)}</pre></Disclosure>
            </section>}
          </>}
        </Card>
        )}
      </main>
    </div>
  </div>
}
