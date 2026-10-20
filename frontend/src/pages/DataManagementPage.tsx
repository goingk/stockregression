import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import {
  ArrowRight, Database, Download, FileSearch, FileUp, FolderOpen, FolderUp,
  Archive, Merge, PackageCheck, RefreshCw, ShieldCheck, Upload, X,
} from 'lucide-react'
import { get, post, postForm, put } from '../api'
import { Badge, Button, Card, Empty, ErrorBox, Field, Loading, PageHeader, SelectInput, Stat, TextInput, compact, dateText, useErrorMessage } from '../components'
import { useTasks } from '../TaskCenter'
import type {
  Coverage, DataInventoryItem, DataInventoryResponse, DataPackageInfo,
  DataPreparationPlan, DataPreparationRequest, DataSourcesResponse, DataTimeframe,
} from '../types'

const TIMEFRAMES: Array<[DataTimeframe, string]> = [
  ['1min', '1 分钟'], ['5min', '5 分钟'], ['15min', '15 分钟'], ['30min', '30 分钟'],
  ['60min', '60 分钟'], ['1day', '日线'], ['1week', '周线'], ['1month', '月线'],
]
const DATA_JOB_KINDS = new Set([
  'instruments', 'market_sync', 'market_import',
  'market_import_batch', 'market_import_package', 'market_index_package',
  'market_validate', 'market_aggregate', 'market_prepare',
  'market_source_probe', 'market_archive',
])
const TERMINAL = new Set(['completed', 'partial', 'failed', 'cancelled'])
const SUPPORTED_UPLOAD = /\.(csv|parquet)$/i

type Settings = {
  data_dir?: string
  active_data_dir?: string
  package_dir?: string | null
  active_package_dir?: string
  market_cache_mb?: number
  recent_minute_source?: 'sina' | 'tdx'
  etf_daily_source?: 'baostock' | 'eastmoney'
}
type ScanItem = { relative_path: string; symbol?: string | null; size: number; duplicate?: boolean; error?: string | null }
type ScanPreview = { folder: string; timeframe: string; items: ScanItem[]; target_count: number; file_count: number; invalid_count: number }
type PackagePeriod = { file_count: number; bytes: number; rows: number; symbols: number; label_convention: string; first_time?: string | null; last_time?: string | null; schema_error_count: number }
type PackageScan = { folder: string; format: string; timeframes: Record<string, PackagePeriod>; errors: string[]; bytes: number; rows: number }
type SelectedImport = { file: File; relativePath: string }
type BrowserEntry = {
  isFile: boolean
  isDirectory: boolean
  name: string
  file?: (callback: (file: File) => void, error?: (reason: unknown) => void) => void
  createReader?: () => { readEntries: (callback: (entries: BrowserEntry[]) => void, error?: (reason: unknown) => void) => void }
}
type ReportRecord = Record<string, unknown>
type PeriodCoverage = Coverage & {
  segments?: Array<Record<string, unknown>>
  versions?: Array<Record<string, unknown>>
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(2)} GB`
}

function relativePathOf(file: File) {
  return ((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name).replace(/\\/g, '/')
}

async function readDirectoryEntries(reader: { readEntries: (callback: (entries: BrowserEntry[]) => void, error?: (reason: unknown) => void) => void }) {
  const output: BrowserEntry[] = []
  while (true) {
    const batch = await new Promise<BrowserEntry[]>((resolve, reject) => reader.readEntries(resolve, reject))
    if (!batch.length) break
    output.push(...batch)
  }
  return output
}

async function collectEntry(entry: BrowserEntry, prefix = ''): Promise<SelectedImport[]> {
  if (entry.isFile && entry.file) {
    const file = await new Promise<File>((resolve, reject) => entry.file?.(resolve, reject))
    return SUPPORTED_UPLOAD.test(file.name) ? [{ file, relativePath: `${prefix}${file.name}` }] : []
  }
  if (!entry.isDirectory || !entry.createReader) return []
  const children = await readDirectoryEntries(entry.createReader())
  const nested = await Promise.all(children.map(child => collectEntry(child, `${prefix}${entry.name}/`)))
  return nested.flat()
}

async function collectDrop(event: DragEvent<HTMLElement>): Promise<SelectedImport[]> {
  const items = Array.from(event.dataTransfer.items || [])
  const entries = items.map(item => (item as unknown as { webkitGetAsEntry?: () => unknown }).webkitGetAsEntry?.()).filter((entry): entry is BrowserEntry => Boolean(entry))
  if (entries.length) return (await Promise.all(entries.map(entry => collectEntry(entry)))).flat()
  return Array.from(event.dataTransfer.files).filter(file => SUPPORTED_UPLOAD.test(file.name)).map(file => ({ file, relativePath: relativePathOf(file) }))
}

function statusTone(status: string) {
  return status === 'ready' ? 'green' : status === 'partial' ? 'amber' : status === 'missing' || status === 'failed' ? 'red' : 'neutral'
}

function statusLabel(status: string) {
  return status === 'ready' ? '可用' : status === 'partial' ? '有缺口' : status === 'missing' ? '文件缺失' : status === 'failed' ? '失败' : '待核验'
}

function periodText(item: DataPreparationRequest | null) {
  if (!item) return '—'
  return `${item.symbol || '当前标的'} · ${item.frequency || '当前周期'} · ${item.checked_start || item.start || '最早'} → ${item.checked_end || item.end || '最新'}`
}

function firstAvailableTimeframe(item: DataInventoryItem): DataTimeframe {
  const found = TIMEFRAMES.find(([name]) => item.periods[name])
  return found ? found[0] : '1day'
}

export default function DataManagementPage({
  requestedReport, onReportHandled, preparation, onReturn,
}: {
  requestedReport?: string | null
  onReportHandled?: () => void
  preparation?: DataPreparationRequest | null
  onReturn?: () => void
}) {
  const tasks = useTasks()
  const errorText = useErrorMessage()
  const [inventory, setInventory] = useState<DataInventoryResponse | null>(null)
  const [packageInfo, setPackageInfo] = useState<DataPackageInfo | null>(null)
  const [sources, setSources] = useState<DataSourcesResponse | null>(null)
  const [settings, setSettings] = useState<Settings>({})
  const [query, setQuery] = useState('')
  const [assetType, setAssetType] = useState('')
  const [timeframe, setTimeframe] = useState<DataTimeframe | ''>('')
  const [status, setStatus] = useState('')
  const [page, setPage] = useState(0)
  const pageSize = 50
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [selectedTimeframe, setSelectedTimeframe] = useState<DataTimeframe>('1day')
  const [coverage, setCoverage] = useState<PeriodCoverage | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [report, setReport] = useState<ReportRecord | null>(null)
  const lastDataRefresh = useRef('')

  const [syncSymbols, setSyncSymbols] = useState('')
  const [syncMode, setSyncMode] = useState('incremental')
  const [syncScope, setSyncScope] = useState('selected')
  const [initializeAll, setInitializeAll] = useState(false)
  const [syncStart, setSyncStart] = useState('')
  const [syncEnd, setSyncEnd] = useState('')
  const [aggregate, setAggregate] = useState({ source_timeframe: '1day', target_timeframe: '1week', symbols: '', start_date: '', end_date: '' })
  const [importSymbol, setImportSymbol] = useState('')
  const [importPath, setImportPath] = useState('')
  const [labelConvention, setLabelConvention] = useState('auto')
  const [etfCategory, setEtfCategory] = useState('')
  const [calendarPath, setCalendarPath] = useState('')
  const [actionsPath, setActionsPath] = useState('')
  const [marketFile, setMarketFile] = useState<File | null>(null)
  const [calendarFile, setCalendarFile] = useState<File | null>(null)
  const [actionsFile, setActionsFile] = useState<File | null>(null)
  const [skipInvalidRows, setSkipInvalidRows] = useState(false)
  const [importMode, setImportMode] = useState<'merge' | 'snapshot'>('merge')
  const [batchImportMode, setBatchImportMode] = useState<'merge' | 'snapshot'>('merge')
  const [syncDaily, setSyncDaily] = useState(true)
  const [syncMinute, setSyncMinute] = useState(true)
  const [marketDragging, setMarketDragging] = useState(false)
  const marketInput = useRef<HTMLInputElement>(null)
  const calendarInput = useRef<HTMLInputElement>(null)
  const actionsInput = useRef<HTMLInputElement>(null)
  const [batchFiles, setBatchFiles] = useState<SelectedImport[]>([])
  const [batchCalendar, setBatchCalendar] = useState<File | null>(null)
  const [batchLabel, setBatchLabel] = useState('auto')
  const [folder, setFolder] = useState('')
  const [recursive, setRecursive] = useState(false)
  const [scan, setScan] = useState<ScanPreview | null>(null)
  const [batchDragging, setBatchDragging] = useState(false)
  const folderInput = useRef<HTMLInputElement>(null)
  const filesInput = useRef<HTMLInputElement>(null)
  const batchCalendarInput = useRef<HTMLInputElement>(null)
  const [packageFolder, setPackageFolder] = useState('')
  const [packageScan, setPackageScan] = useState<PackageScan | null>(null)
  const [preparationPlan, setPreparationPlan] = useState<DataPreparationPlan | null>(null)
  const [preparationBusy, setPreparationBusy] = useState(false)

  const loadInventory = useCallback(async () => {
    setError('')
    try {
      const params = new URLSearchParams({ limit: String(pageSize), offset: String(page * pageSize) })
      if (query.trim()) params.set('q', query.trim())
      if (assetType) params.set('asset_type', assetType)
      if (timeframe) params.set('timeframe', timeframe)
      if (status) params.set('status', status)
      const [nextInventory, nextPackage, config, nextSources] = await Promise.all([
        get<DataInventoryResponse>(`/data/inventory?${params}`),
        get<DataPackageInfo>('/data/package'),
        get<Settings>('/settings'),
        get<DataSourcesResponse>('/data/sources'),
      ])
      setInventory(nextInventory)
      setPackageInfo(nextPackage)
      setSettings(config)
      setSources(nextSources)
      setPackageFolder(current => current || config.active_package_dir || config.package_dir || '')
      setSelectedSymbol(current => nextInventory.items.some(item => item.symbol === current) ? current : nextInventory.items[0]?.symbol || '')
    } catch (cause) { setError(errorText(cause)) } finally { setLoading(false) }
  }, [assetType, errorText, page, query, status, timeframe])

  useEffect(() => { void loadInventory() }, [loadInventory])
  useEffect(() => { setPage(0) }, [assetType, query, status, timeframe])
  const selectedItem = useMemo(() => inventory?.items.find(item => item.symbol === selectedSymbol) || null, [inventory, selectedSymbol])
  const selectedPeriod = selectedItem?.periods[selectedTimeframe] || null
  useEffect(() => {
    if (!selectedSymbol || !selectedItem?.periods[selectedTimeframe]) { setCoverage(null); return }
    let active = true
    get<PeriodCoverage>(`/data/coverage/${encodeURIComponent(selectedSymbol)}?timeframe=${selectedTimeframe}`)
      .then(value => { if (active) setCoverage(value) })
      .catch(cause => { if (active) { setCoverage(null); setError(errorText(cause)) } })
    return () => { active = false }
  }, [selectedSymbol, selectedTimeframe, selectedItem, errorText])
  useEffect(() => {
    if (!requestedReport) { setReport(null); return }
    get<ReportRecord>(`/results/${encodeURIComponent(requestedReport)}`)
      .then(value => { setReport(value); onReportHandled?.() })
      .catch(cause => setError(errorText(cause)))
  }, [requestedReport, onReportHandled, errorText])

  const completedDataJobs = tasks.jobs
    .filter(job => DATA_JOB_KINDS.has(job.kind) && TERMINAL.has(job.status))
    .map(job => `${job.id}:${job.status}`).join(',')
  useEffect(() => {
    if (!completedDataJobs || completedDataJobs === lastDataRefresh.current) return
    lastDataRefresh.current = completedDataJobs
    void loadInventory()
  }, [completedDataJobs, loadInventory])

  useEffect(() => {
    if (!selectedItem) return
    const available = TIMEFRAMES.map(([name]) => name).filter(name => selectedItem.periods[name])
    if (available.length && !selectedItem.periods[selectedTimeframe]) setSelectedTimeframe(available[0])
    setSyncDaily(Boolean(selectedItem.periods['1day']))
    setSyncMinute(Boolean(selectedItem.periods['1min']))
  }, [selectedItem, selectedTimeframe])

  useEffect(() => {
    if (selectedTimeframe === '1min' && syncMode === 'history') setSyncMode('incremental')
    if (selectedTimeframe === '1min' && initializeAll) setInitializeAll(false)
    if (syncMode === 'history' && syncMinute) setSyncMinute(false)
  }, [initializeAll, selectedTimeframe, syncMinute, syncMode])

  const loadPlan = useCallback(async () => {
    if (!preparation?.config) { setPreparationPlan(null); return }
    setPreparationBusy(true); setError('')
    try {
      setPreparationPlan(await post<DataPreparationPlan>('/data/preparation/plan', { config: preparation.config }))
    } catch (cause) { setError(errorText(cause)) } finally { setPreparationBusy(false) }
  }, [errorText, preparation?.config])
  useEffect(() => { void loadPlan() }, [loadPlan])
  useEffect(() => {
    if (!preparation?.config || !completedDataJobs) return
    void loadPlan()
  }, [completedDataJobs, loadPlan, preparation?.config])

  const run = async (label: string, action: () => Promise<unknown>, success: string) => {
    setBusy(label); setError(''); setNotice('')
    try {
      await tasks.run(label, action)
      setNotice(success)
      await loadInventory()
    } catch (cause) { setError(errorText(cause)) } finally { setBusy('') }
  }

  const runSync = () => {
    const symbols = syncScope === 'selected' ? syncSymbols.split(/[,，\s]+/).filter(Boolean) : []
    if (syncScope === 'selected' && !symbols.length) { setError('请填写要同步的标的。'); return }
    if (syncMode !== 'incremental' && !syncStart) { setError('历史导入和缺口修补需要填写开始日期。'); return }
    const timeframes = [syncDaily ? '1day' : '', syncMinute ? '1min' : ''].filter(Boolean)
    if (!timeframes.length) { setError('请至少选择一个基础周期。'); return }
    if (syncMode === 'history' && timeframes.includes('1min')) {
      setError('1 分钟历史行情只从本地文件导入；历史初始化只能更新日线。')
      return
    }
    void run('同步行情', () => post('/data/sync', {
      symbols, scope: syncScope, timeframe: timeframes[0], timeframes, mode: syncMode,
      start_date: syncStart || null, end_date: syncEnd || null, resume: true,
      initialize: syncScope === 'market' && initializeAll,
    }), `已提交 ${timeframes.join('、')} 基础行情更新。`)
  }

  const chooseSource = async (capability: string, source: string) => {
    setBusy('切换来源'); setError(''); setNotice('')
    try {
      const field = capability === 'recent_minute' ? 'recent_minute_source' : 'etf_daily_source'
      await put('/settings', { [field]: source })
      setNotice(`数据源已切换为 ${source}。`)
      await loadInventory()
    } catch (cause) { setError(errorText(cause)) } finally { setBusy('') }
  }

  const probeSource = (source: string, capability: string, market = 'all') => {
    void run('探测数据源', () => post('/data/sources/probe', { source, capability, market }), `${source} 探测任务已提交；完成后刷新状态。`)
  }

  const submitImport = () => {
    if (selectedTimeframe !== '1day' && selectedTimeframe !== '1min') { setError('粗周期由日线或 1 分钟按需生成，请选择基础周期导入。'); return }
    if (!marketFile && !importPath.trim()) { setError('请选择行情文件或填写本机路径。'); return }
    if (!importSymbol.trim()) { setError('请填写证券代码。'); return }
    if (marketFile) {
      const body = new FormData()
      body.append('market_file', marketFile)
      body.append('symbol', importSymbol.trim().toUpperCase())
      body.append('timeframe', selectedTimeframe)
      body.append('label_convention', labelConvention)
      body.append('skip_invalid_rows', String(skipInvalidRows))
      body.append('import_mode', importMode)
      if (etfCategory) body.append('etf_category', etfCategory)
      if (calendarFile) body.append('calendar_file', calendarFile)
      if (actionsFile) body.append('actions_file', actionsFile)
      void run('导入行情文件', () => postForm('/data/upload', body), `已提交 ${marketFile.name}，将合并到数据包中的 ${selectedTimeframe}。`)
      return
    }
    void run('导入行情文件', () => post('/data/import', {
      symbol: importSymbol.trim().toUpperCase(), timeframe: selectedTimeframe,
      path: importPath.trim(), label_convention: labelConvention,
      etf_category: etfCategory || null, calendar_path: calendarPath.trim() || null,
      actions_path: actionsPath.trim() || null, skip_invalid_rows: skipInvalidRows,
      import_mode: importMode,
    }), `已提交本地路径 ${importPath.trim()}。`)
  }

  const chooseBatchFiles = (files: FileList | null) => {
    const selected = Array.from(files || []).filter(file => SUPPORTED_UPLOAD.test(file.name)).map(file => ({ file, relativePath: relativePathOf(file) }))
    setBatchFiles(selected); setScan(null)
    setNotice(selected.length ? `已选择 ${selected.length} 个文件。` : '')
  }

  const runBatch = () => {
    if (selectedTimeframe !== '1day' && selectedTimeframe !== '1min') { setError('批量导入只接受日线或 1 分钟。'); return }
    if (batchFiles.length) {
      const body = new FormData()
      batchFiles.forEach(item => body.append('files', item.file, item.relativePath))
      body.append('timeframe', selectedTimeframe)
      body.append('label_convention', batchLabel)
      body.append('skip_invalid_rows', String(skipInvalidRows))
      body.append('import_mode', batchImportMode)
      if (batchCalendar) body.append('calendar_file', batchCalendar)
      void run('批量导入', () => postForm('/data/upload-batch', body), `已上传 ${batchFiles.length} 个文件，将写入数据包。`)
      return
    }
    if (!folder.trim()) { setError('请选择文件或填写本机文件夹。'); return }
    void run('批量导入', () => post('/data/import-batch', {
      folder: folder.trim(), timeframe: selectedTimeframe, recursive,
      label_convention: batchLabel, resume: true, skip_invalid_rows: skipInvalidRows,
      import_mode: batchImportMode,
    }), `已提交文件夹 ${folder.trim()}。`)
  }

  const scanFolder = async () => {
    if (!folder.trim()) { setError('请填写本机文件夹。'); return }
    setBusy('扫描文件夹'); setError('')
    try {
      const value = await post<ScanPreview>('/data/import-batch/scan', { folder: folder.trim(), timeframe: selectedTimeframe, recursive })
      setScan(value); setNotice(`扫描到 ${value.file_count} 个文件，可识别 ${value.target_count} 个标的。`)
    } catch (cause) { setError(errorText(cause)) } finally { setBusy('') }
  }

  const scanPackage = async () => {
    if (!packageFolder.trim()) { setError('请填写数据包目录。'); return }
    setBusy('扫描数据包'); setError('')
    try {
      const value = await post<PackageScan>('/data/import-package/scan', { folder: packageFolder.trim(), format: 'package' })
      setPackageScan(value); setNotice(`数据包包含 ${Object.keys(value.timeframes).length} 个周期、${compact(value.rows)} 行。`)
    } catch (cause) { setError(errorText(cause)) } finally { setBusy('') }
  }

  const registerPackage = () => {
    if (!packageFolder.trim()) { setError('请填写数据包目录。'); return }
    void run('登记数据包', () => post('/data/import-package', {
      folder: packageFolder.trim(), format: 'package', timeframes: null, symbols: null,
      label_convention: 'auto', resume: true, workers: 4, skip_invalid_rows: false,
    }), '已提交当前数据包的幂等登记；现有数据版本和结果会保留。')
  }

  const runAggregate = () => {
    const symbols = aggregate.symbols.split(/[,，\s]+/).filter(Boolean)
    if (!symbols.length) { setError('请填写要聚合的标的。'); return }
    void run('本地聚合', () => post('/data/aggregate', {
      symbols, source_timeframe: aggregate.source_timeframe,
      target_timeframe: aggregate.target_timeframe,
      start_date: aggregate.start_date || null, end_date: aggregate.end_date || null,
    }), '本地聚合任务已提交，结果将写入数据包。')
  }

  const archiveSelected = () => {
    if (!selectedItem) return
    void run('保护历史快照', () => post('/data/archive', {
      symbols: [selectedItem.symbol], timeframe: selectedTimeframe,
    }), '历史快照归档任务已提交。')
  }

  const submitPreparation = async () => {
    if (!preparation?.config) return
    setBusy('准备数据'); setError(''); setNotice('')
    try {
      await tasks.run('准备数据', () => post('/data/preparation', { config: preparation.config }))
      setNotice('数据准备任务已提交；完成后会重新检查本次请求。')
    } catch (cause) { setError(errorText(cause)) } finally { setBusy('') }
  }

  const chooseSymbolPeriod = (item: DataInventoryItem, name: DataTimeframe) => {
    setSelectedSymbol(item.symbol); setSelectedTimeframe(name)
    setSyncSymbols(item.symbol); setSyncScope('selected'); setImportSymbol(item.symbol)
    setAggregate(current => ({ ...current, symbols: item.symbol }))
  }

  const totalPages = Math.max(1, Math.ceil((inventory?.count || 0) / pageSize))
  const reportSteps = report?.steps as Array<Record<string, unknown>> | undefined

  return <>
    <PageHeader eyebrow="DATA PACKAGE" title="数据管理" description="当前数据包是唯一正式行情来源；导入、同步、聚合和公司行动资料都写入这里。" action={<div className="button-row"><Button variant="secondary" busy={busy === '刷新'} onClick={() => void run('刷新', loadInventory, '数据包目录已刷新。')}><RefreshCw size={15} />刷新</Button><Button variant="quiet" onClick={tasks.showHistory}><Database size={15} />任务记录</Button></div>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {notice && <div className="notice success"><span>{notice}</span><button className="text-link" onClick={() => setNotice('')}>收起</button></div>}

    {preparation && <Card title="本次回测的数据准备" subtitle={periodText(preparation)} action={preparationPlan ? <Badge tone={preparationPlan.ready ? 'green' : 'amber'}>{preparationPlan.ready ? '已可用' : `${preparationPlan.gaps.length} 项缺口`}</Badge> : <Badge>检查中</Badge>}>
      {preparationBusy ? <Loading label="读取本次请求所需数据…" /> : preparationPlan ? <>
        {preparationPlan.gaps.length > 0 && <div className="preparation-gaps">{preparationPlan.gaps.map((gap, index) => <p key={`${gap.kind}-${index}`}><Badge tone="amber">{gap.timeframe || gap.kind || '数据'}</Badge><span>{gap.message}</span></p>)}</div>}
        <div className="table-wrap"><table><thead><tr><th>顺序</th><th>处理事项</th><th>范围</th><th>说明</th></tr></thead><tbody>{preparationPlan.steps.map((step, index) => <tr key={`${step.kind}-${index}`}><td>{index + 1}</td><td>{step.kind === 'auxiliary' ? '日历与公司行动' : step.kind === 'validate' ? '校验并发布' : `${step.timeframe} 行情`}</td><td>{step.start ? `${step.start} → ${step.end}` : '当前数据包'}</td><td>{step.reason}</td></tr>)}</tbody></table></div>
        <div className="button-row"><Button busy={busy === '准备数据'} disabled={preparationPlan.ready} onClick={() => void submitPreparation()}><Download size={15} />提交准备任务</Button>{preparationPlan.ready && onReturn && <Button variant="secondary" onClick={onReturn}>返回原页面<ArrowRight size={14} /></Button>}</div>
      </> : <Empty title="未载入准备配置" detail="从回测或研究的缺口提示进入时可查看完整准备步骤。" />}
    </Card>}

    <Card title="本地数据" subtitle="按证券查看各周期覆盖、文件与质量；筛选统计来自完整数据包目录。" action={packageInfo ? <Badge tone={packageInfo.status === 'ready' ? 'green' : packageInfo.status === 'missing' ? 'red' : 'amber'}>{packageInfo.status === 'ready' ? '数据包可用' : packageInfo.status === 'missing' ? '数据包不可读' : packageInfo.status === 'index_only' ? '索引已接入 · 待登记包清单' : packageInfo.status === 'empty' ? '空数据包' : '有待恢复事务'}</Badge> : null}>
      <div className="package-summary"><div><small>数据包</small><strong>{packageInfo?.path || settings.active_package_dir || '未配置'}</strong></div><div><small>身份 / 版本</small><strong>{packageInfo?.package_id || '—'} · v{packageInfo?.package_version ?? 0}</strong></div><div><small>容量 / 文件</small><strong>{formatBytes(packageInfo?.bytes || 0)} · {packageInfo?.file_count ?? 0}</strong></div></div>
      <div className="source-panel-head"><div><strong>在线数据源</strong><span>默认来源直接使用；可选来源检测通过后才能启用，失败会暂停 {sources?.cooldown_hours ?? 24} 小时并单次回退默认来源。</span></div><Button variant="quiet" busy={busy === '刷新来源'} onClick={() => void run('刷新来源', loadInventory, '数据源状态已刷新。')}><RefreshCw size={14} />刷新状态</Button></div>
      {!sources ? <Loading label="读取来源状态…" /> : <div className="source-grid">{sources.capabilities.map(capability => <div className="source-panel" key={capability.capability}>
        <div className="source-panel-head"><div><strong>{capability.label}</strong><span>{capability.description}</span></div><Badge tone={capability.routes.some(route => route.fallback) ? 'amber' : 'blue'}>{capability.routes.some(route => route.fallback) ? `${capability.configured} · 部分回退` : `当前 ${capability.configured}`}</Badge></div>
        <div className="source-options">{capability.sources.map(option => {
          return <div className="source-option" key={option.source}>
            <div><strong>{option.label}</strong><small>{option.role === 'default' ? '默认来源' : '可选来源'}</small></div>
            <div className="source-state-list">{option.states.map(state => <span key={state.market} title={state.error || ''}><b>{state.market === 'etf' ? 'ETF' : '股票'}</b><Badge tone={state.status === 'passed' ? 'green' : state.status === 'failed' ? 'red' : 'neutral'}>{state.status === 'passed' ? '通过' : state.status === 'failed' ? '失败' : '未检测'}</Badge><small>{state.rows ? `${compact(state.rows)} 行` : ''}{state.coverage_end ? ` · ${dateText(state.coverage_end)}` : ''}</small></span>)}</div>
            <div className="button-row"><Button variant="quiet" busy={busy === '探测数据源'} onClick={() => probeSource(option.source, capability.capability)}>检测</Button>{!option.selected && <Button variant="secondary" disabled={!option.selectable || busy === '切换来源'} onClick={() => void chooseSource(capability.capability, option.source)}>{option.selectable ? '启用' : '检测通过后启用'}</Button>}</div>
          </div>
        })}</div>
      </div>)}</div>}
      <div className="form-grid four"><Field label="搜索"><TextInput value={query} onChange={event => setQuery(event.target.value)} placeholder="代码或名称" /></Field><Field label="证券类型"><SelectInput value={assetType} onChange={event => setAssetType(event.target.value)}><option value="">股票与 ETF</option><option value="stock">股票</option><option value="etf">ETF</option></SelectInput></Field><Field label="周期"><SelectInput value={timeframe} onChange={event => setTimeframe(event.target.value as DataTimeframe | '')}><option value="">全部周期</option>{TIMEFRAMES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</SelectInput></Field><Field label="状态"><SelectInput value={status} onChange={event => setStatus(event.target.value)}><option value="">全部状态</option><option value="ready">本次规则下可用</option><option value="partial">有缺口</option><option value="pending_validation">待核验</option><option value="missing">文件缺失</option></SelectInput></Field></div>
      <div className="stat-grid four"><Stat label="证券" value={inventory?.stats.symbols ?? 0} note={`共 ${inventory?.dataset_count ?? 0} 个标的周期`} tone="blue" /><Stat label="可用数据集" value={inventory?.stats.ready ?? 0} note="文件、质量与覆盖均已通过" tone="green" /><Stat label="待处理" value={inventory?.stats.pending ?? 0} note="待核验、缺口或缺失" tone={(inventory?.stats.pending || 0) ? 'amber' : 'neutral'} /><Stat label="行情行数" value={compact(inventory?.stats.rows ?? 0)} note={formatBytes(inventory?.stats.bytes || 0)} /></div>
      {loading ? <Loading label="读取数据包目录…" /> : inventory?.items.length ? <div className="table-wrap inventory-table"><table><thead><tr><th>证券</th>{TIMEFRAMES.map(([name, label]) => <th key={name}>{label}</th>)}<th>最近更新</th></tr></thead><tbody>{inventory.items.map(item => <tr key={item.symbol} className={selectedSymbol === item.symbol ? 'selected-row' : ''} onClick={() => chooseSymbolPeriod(item, selectedItem?.periods[selectedTimeframe] ? selectedTimeframe : firstAvailableTimeframe(item))}><td><strong>{item.name}</strong><small>{item.symbol} · {item.asset_type === 'etf' ? 'ETF' : '股票'}</small></td>{TIMEFRAMES.map(([name]) => { const period = item.periods[name]; return <td key={name}>{period ? <button className="period-cell" onClick={event => { event.stopPropagation(); chooseSymbolPeriod(item, name) }}><Badge tone={statusTone(period.status)}>{statusLabel(period.status)}</Badge><small>{dateText(period.last_time)}</small></button> : <span className="period-empty">—</span>}</td>})}<td>{dateText(item.updated_at)}</td></tr>)}</tbody></table></div> : <Empty title="当前筛选没有本地数据" detail="可以从文件导入、数据包登记或线上初始化加入证券。" />}
      <div className="pagination-row"><span>第 {page + 1} / {totalPages} 页 · 共 {inventory?.count ?? 0} 个证券</span><div><Button variant="quiet" disabled={page <= 0} onClick={() => setPage(current => Math.max(0, current - 1))}>上一页</Button><Button variant="quiet" disabled={page + 1 >= totalPages} onClick={() => setPage(current => current + 1)}>下一页</Button></div></div>
    </Card>

    <Card title={selectedItem ? `${selectedItem.name} · ${selectedTimeframe}` : '数据详情'} subtitle="覆盖区间、来源、质量检查和下一步修复入口。" action={selectedPeriod ? <Badge tone={statusTone(selectedPeriod.status)}>{statusLabel(selectedPeriod.status)}</Badge> : null}>
      {!selectedItem || !selectedPeriod ? <Empty title="请选择证券和周期" detail="点击上表中的周期查看详细状态。" /> : <>
        <div className="stat-grid four"><Stat label="首末时间" value={`${dateText(selectedPeriod.first_time)} → ${dateText(selectedPeriod.last_time)}`} note={`版本 ${selectedPeriod.current_version}`} /><Stat label="覆盖" value={{ complete: '完整', partial: '有缺口', unknown: '未知' }[selectedPeriod.coverage_status] || selectedPeriod.coverage_status} note={`已核验至 ${dateText(selectedPeriod.verified_through)}`} tone={selectedPeriod.coverage_status === 'complete' ? 'green' : 'amber'} /><Stat label="文件 / 质量" value={`${selectedPeriod.file_status === 'readable' ? '可读' : selectedPeriod.file_status === 'changed' ? '已变化' : '缺失'} / ${selectedPeriod.quality_status === 'passed' ? '通过' : selectedPeriod.quality_status === 'failed' ? '失败' : '待核验'}`} note={`${compact(selectedPeriod.row_count)} 行`} /><Stat label="合成记录" value={selectedPeriod.synthetic_rows} note="不能触发交易" tone={selectedPeriod.synthetic_rows ? 'amber' : 'green'} /></div>
        {coverage?.segments?.length ? <div className="table-wrap coverage-table compact-coverage"><table><thead><tr><th>检查区间</th><th>结果</th><th>预期 / 实际</th><th>缺口</th><th>检查时间</th></tr></thead><tbody>{coverage.segments.map((segment, index) => { const row = segment as Record<string, unknown>; return <tr key={String(row.start_time) + index}><td>{dateText(String(row.start_time))} → {dateText(String(row.end_time))}</td><td>{String(row.status ?? '—')}</td><td>{String(row.expected_rows ?? '—')} / {String(row.observed_rows ?? '—')}</td><td>{String(row.gap_count ?? 0)}</td><td>{dateText(String(row.checked_at))}</td></tr> })}</tbody></table></div> : <p className="inline-warning">该版本尚无检查区间报告。登记成功不等于质量已通过。</p>}
        <div className="button-row"><Badge tone={selectedPeriod.archive?.protected ? 'green' : 'amber'}>{selectedPeriod.archive?.protected ? '历史快照已保护' : '历史快照未保护'}</Badge><Badge tone="blue">增量文件 {selectedPeriod.increment_count || 0}</Badge>{selectedItem.periods['1day'] && <Badge>日线合规至 {dateText(selectedItem.periods['1day']?.verified_through)}</Badge>}{selectedItem.periods['1min'] && <Badge>1 分钟合规至 {dateText(selectedItem.periods['1min']?.verified_through)}</Badge>}</div><div className="button-row data-detail-actions"><Button variant="secondary" busy={busy === '校验'} onClick={() => void run('校验', () => post('/data/validate', { symbol: selectedItem.symbol, timeframe: selectedTimeframe }), '增量校验任务已提交。')}><ShieldCheck size={14} />增量校验</Button><Button variant="quiet" onClick={() => { setSyncSymbols(selectedItem.symbol); setSyncScope('selected'); document.getElementById('data-import-update')?.scrollIntoView({ behavior: 'smooth' }) }}>更新该周期</Button><Button variant="quiet" busy={busy === '保护历史快照'} disabled={!selectedPeriod.archive || selectedPeriod.archive.protected} onClick={archiveSelected}><Archive size={14} />保护历史快照</Button></div>
        {coverage?.versions?.length ? <details className="advanced-data"><summary>版本历史（{coverage.versions.length}）</summary><div className="table-wrap compact-coverage"><table><thead><tr><th>版本</th><th>时间</th><th>行数</th><th>状态</th><th>来源</th></tr></thead><tbody>{coverage.versions.map((version, index) => { const row = version as Record<string, unknown>; return <tr key={String(row.version) + index}><td>{String(row.version)}</td><td>{dateText(String(row.created_at))}</td><td>{compact(Number(row.row_count))}</td><td>{String(row.status)}</td><td>{String(row.source ?? '—')}</td></tr> })}</tbody></table></div></details> : null}
      </>}
    </Card>

    <div id="data-import-update" />
    <div className="data-columns">
      <Card title="更新基础行情" subtitle="只更新日线和 1 分钟；其他周期会在查询、回测或研究时按需聚合。" action={<Badge tone="blue">基础周期</Badge>}>
        <div className="form-grid two"><Field label="模式"><SelectInput value={syncMode} onChange={event => setSyncMode(event.target.value)}><option value="incremental">增量更新</option>{selectedTimeframe !== '1min' && <option value="history">历史初始化</option>}<option value="repair">缺口修补</option></SelectInput></Field><Field label="范围"><SelectInput value={syncScope} onChange={event => setSyncScope(event.target.value)}><option value="selected">指定证券</option><option value="watchlist">自选</option><option value="market">数据包内已有证券</option></SelectInput></Field><Field label="证券" className="full-width"><TextInput value={syncSymbols} disabled={syncScope !== 'selected'} onChange={event => setSyncSymbols(event.target.value)} placeholder="600000.SH, 510300.SH" /></Field><Field label="开始日期"><TextInput type="date" value={syncStart} disabled={syncMode === 'incremental'} onChange={event => setSyncStart(event.target.value)} /></Field><Field label="结束日期"><TextInput type="date" value={syncEnd} disabled={syncMode === 'incremental'} onChange={event => setSyncEnd(event.target.value)} /></Field></div>
        <div className="button-row"><label className="check-line"><input type="checkbox" checked={syncDaily} onChange={event => setSyncDaily(event.target.checked)} />更新日线</label><label className="check-line"><input type="checkbox" checked={syncMinute} onChange={event => setSyncMinute(event.target.checked)} />更新 1 分钟</label></div>
        {syncMinute && <p className="inline-warning">1 分钟历史行情只从本地数据包读取；线上增量与缺口修补只覆盖近期窗口。</p>}
        {syncScope === 'market' && selectedTimeframe !== '1min' && <label className="check-line"><input type="checkbox" checked={initializeAll} onChange={event => setInitializeAll(event.target.checked)} />同时初始化全市场尚无该周期的证券；首次任务会扩大下载范围</label>}
        <div className="card-bottom"><Button busy={busy === '同步行情'} onClick={runSync}><Download size={15} />更新选中的基础周期</Button></div>
      </Card>
      <Card title="本地聚合" subtitle="从已有周期生成更粗周期，结果写入数据包；源行情修订后需重新生成。" action={<Badge>不联网</Badge>}>
        <div className="form-grid two"><Field label="来源周期"><SelectInput value={aggregate.source_timeframe} onChange={event => setAggregate({ ...aggregate, source_timeframe: event.target.value })}>{TIMEFRAMES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</SelectInput></Field><Field label="目标周期"><SelectInput value={aggregate.target_timeframe} onChange={event => setAggregate({ ...aggregate, target_timeframe: event.target.value })}>{TIMEFRAMES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</SelectInput></Field><Field label="标的" className="full-width"><TextInput value={aggregate.symbols} onChange={event => setAggregate({ ...aggregate, symbols: event.target.value })} placeholder="510300.SH" /></Field><Field label="开始日期"><TextInput type="date" value={aggregate.start_date} onChange={event => setAggregate({ ...aggregate, start_date: event.target.value })} /></Field><Field label="结束日期"><TextInput type="date" value={aggregate.end_date} onChange={event => setAggregate({ ...aggregate, end_date: event.target.value })} /></Field></div>
        <div className="card-bottom"><Button variant="secondary" busy={busy === '本地聚合'} onClick={runAggregate}><Merge size={15} />生成 {aggregate.target_timeframe}</Button></div>
      </Card>
    </div>

    <Card title="文件导入" subtitle="基础行情支持增量合并或全量替换；修改历史会只撤销相关交易日的合规结论。" action={<Badge>本地文件</Badge>}>
      <div className="form-grid two"><Field label="导入方式"><SelectInput value={importMode} onChange={event => setImportMode(event.target.value as 'merge' | 'snapshot')}><option value="merge">增量合并</option><option value="snapshot">全量替换</option></SelectInput></Field><Field label="适用周期"><TextInput value="1day / 1min" disabled readOnly /></Field></div>
      <div className="form-grid two"><Field label="证券代码"><TextInput value={importSymbol} onChange={event => setImportSymbol(event.target.value.toUpperCase())} placeholder="600000.SH" /></Field><Field label="时间标签"><SelectInput value={labelConvention} onChange={event => setLabelConvention(event.target.value)}><option value="auto">自动 / 待核验</option><option value="end">结束标签</option><option value="start">开始标签</option></SelectInput></Field><Field label="ETF 类型"><SelectInput value={etfCategory} onChange={event => setEtfCategory(event.target.value)}><option value="">按证券资料识别</option><option value="stock">股票 ETF</option><option value="bond">债券 ETF</option><option value="crossborder">跨境 ETF</option><option value="commodity">商品 ETF</option><option value="money">货币 ETF</option></SelectInput></Field><Field label="本机路径（不选文件时使用）"><TextInput value={importPath} onChange={event => setImportPath(event.target.value)} placeholder="D:\imports\600000.parquet" /></Field><Field label="交易日历路径"><TextInput value={calendarPath} onChange={event => setCalendarPath(event.target.value)} placeholder="calendar.csv" /></Field><Field label="公司行动路径"><TextInput value={actionsPath} onChange={event => setActionsPath(event.target.value)} placeholder="actions.csv" /></Field></div>
      <input ref={marketInput} hidden type="file" accept=".csv,.parquet" onChange={event => { setMarketFile(event.target.files?.[0] || null); event.currentTarget.value = '' }} />
      <input ref={calendarInput} hidden type="file" accept=".csv,.parquet" onChange={event => { setCalendarFile(event.target.files?.[0] || null); event.currentTarget.value = '' }} />
      <input ref={actionsInput} hidden type="file" accept=".csv,.parquet" onChange={event => { setActionsFile(event.target.files?.[0] || null); event.currentTarget.value = '' }} />
      <div className={`upload-drop-zone ${marketDragging ? 'drag-active' : ''}`} onDragOver={event => { event.preventDefault(); setMarketDragging(true) }} onDragLeave={() => setMarketDragging(false)} onDrop={event => { event.preventDefault(); setMarketDragging(false); const file = event.dataTransfer.files[0]; if (file && SUPPORTED_UPLOAD.test(file.name)) setMarketFile(file) }}><div className="upload-drop-icon"><FileUp size={22} /></div><div className="upload-drop-copy"><strong>{marketFile?.name || '拖入一个行情文件'}</strong><span>CSV / Parquet；合并到当前数据包</span></div><div className="upload-drop-actions"><Button variant="secondary" onClick={() => marketInput.current?.click()}><FileUp size={14} />选择文件</Button>{marketFile && <Button variant="quiet" onClick={() => setMarketFile(null)}><X size={14} />清除</Button>}</div></div>
      <div className="optional-files"><div><small>交易日历</small><strong>{calendarFile?.name || '未选择'}</strong><Button variant="quiet" onClick={() => calendarInput.current?.click()}>选择</Button></div><div><small>公司行动</small><strong>{actionsFile?.name || '未选择'}</strong><Button variant="quiet" onClick={() => actionsInput.current?.click()}>选择</Button></div></div>
      <label className="check-line"><input type="checkbox" checked={skipInvalidRows} onChange={event => setSkipInvalidRows(event.target.checked)} />跳过非正价格记录，并在覆盖报告中记录缺口</label>
      <div className="card-bottom"><Button busy={busy === '导入行情文件'} onClick={submitImport}><Upload size={15} />导入到 {selectedTimeframe}</Button></div>
      <details className="advanced-data"><summary>文件夹批量导入</summary><input ref={folderInput} hidden type="file" multiple accept=".csv,.parquet" {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} onChange={event => { chooseBatchFiles(event.target.files); event.currentTarget.value = '' }} /><input ref={filesInput} hidden type="file" multiple accept=".csv,.parquet" onChange={event => { chooseBatchFiles(event.target.files); event.currentTarget.value = '' }} /><input ref={batchCalendarInput} hidden type="file" accept=".csv,.parquet" onChange={event => { setBatchCalendar(event.target.files?.[0] || null); event.currentTarget.value = '' }} /><div className={`upload-drop-zone ${batchDragging ? 'drag-active' : ''}`} onDragOver={event => { event.preventDefault(); setBatchDragging(true) }} onDragLeave={() => setBatchDragging(false)} onDrop={event => { event.preventDefault(); setBatchDragging(false); void collectDrop(event).then(value => { setBatchFiles(value); setScan(null) }) }}><div className="upload-drop-icon"><FolderUp size={22} /></div><div className="upload-drop-copy"><strong>{batchFiles.length ? `已选择 ${batchFiles.length} 个文件` : '拖入文件夹或多个文件'}</strong><span>保留相对路径并识别六位证券代码</span></div><div className="upload-drop-actions"><Button variant="secondary" onClick={() => folderInput.current?.click()}><FolderUp size={14} />选择文件夹</Button><Button variant="quiet" onClick={() => filesInput.current?.click()}><FileUp size={14} />选择文件</Button></div></div><div className="form-grid two"><Field label="本机文件夹"><TextInput value={folder} onChange={event => { setFolder(event.target.value); setScan(null) }} placeholder="D:\imports\1day" /></Field><Field label="分钟标签"><SelectInput value={batchLabel} onChange={event => setBatchLabel(event.target.value)}><option value="auto">自动</option><option value="end">结束</option><option value="start">开始</option></SelectInput></Field><label className="check-line"><input type="checkbox" checked={recursive} onChange={event => setRecursive(event.target.checked)} />递归扫描子目录</label></div><div className="optional-files single"><div><small>批次日历（可选）</small><strong>{batchCalendar?.name || '未选择'}</strong><Button variant="quiet" onClick={() => batchCalendarInput.current?.click()}>选择</Button></div></div><div className="button-row"><Field label="批量导入方式"><SelectInput value={batchImportMode} onChange={event => setBatchImportMode(event.target.value as 'merge' | 'snapshot')}><option value="merge">增量合并</option><option value="snapshot">全量替换</option></SelectInput></Field><Button variant="secondary" busy={busy === '扫描文件夹'} onClick={() => void scanFolder()}><FileSearch size={15} />扫描本机路径</Button><Button busy={busy === '批量导入'} onClick={runBatch} disabled={!batchFiles.length && !scan?.target_count}><FolderOpen size={15} />执行批量导入</Button></div>{batchFiles.length > 0 && <p className="muted-copy">已选择 {batchFiles.length} 个文件，合计 {formatBytes(batchFiles.reduce((sum, item) => sum + item.file.size, 0))}。</p>}{scan && <div className="scan-summary"><strong>{scan.file_count} 个文件 · {scan.target_count} 个标的</strong><span>{scan.invalid_count} 个需修正</span></div>}</details>
    </Card>

    <Card title="固定数据包" subtitle="正式行情只读取 ETF数据更新至2026.9.30；登记只更新当前包索引，不切换数据源。" action={<Badge tone="blue">唯一来源</Badge>}>
      <div className="form-grid one"><Field label="当前数据包目录"><TextInput readOnly value={packageFolder} /></Field></div>
      <div className="button-row"><Button variant="secondary" busy={busy === '扫描数据包'} onClick={() => void scanPackage()}><FileSearch size={15} />扫描结构</Button><Button busy={busy === '登记数据包'} disabled={!packageFolder.trim()} onClick={registerPackage}><PackageCheck size={15} />重新登记当前数据包</Button>{preparationPlan?.ready && onReturn && <Button variant="quiet" onClick={onReturn}>返回回测<ArrowRight size={14} /></Button>}</div>
      {packageScan && <div className="scan-summary"><strong>{compact(packageScan.rows)} 行 · {formatBytes(packageScan.bytes)}</strong><span>{packageScan.errors.length} 个结构问题</span><div className="table-wrap scan-table"><table><thead><tr><th>周期</th><th>文件 / 标的</th><th>行数</th><th>标签</th><th>首末时间</th><th>问题</th></tr></thead><tbody>{Object.entries(packageScan.timeframes).map(([name, item]) => <tr key={name}><td>{name}</td><td>{item.file_count} / {item.symbols}</td><td>{compact(item.rows)}</td><td>{item.label_convention}</td><td>{dateText(item.first_time)} → {dateText(item.last_time)}</td><td>{item.schema_error_count || '—'}</td></tr>)}</tbody></table></div></div>}
    </Card>

    {report && <div className="task-drawer-overlay" onMouseDown={event => { if (event.target === event.currentTarget) setReport(null) }}><section className="task-drawer report-drawer" role="dialog" aria-modal="true" aria-label="数据任务报告"><Card title="数据任务报告" subtitle={String(report.message ?? report.status ?? '')} action={<Button variant="quiet" onClick={() => setReport(null)}><X size={14} />关闭</Button>}>
      {reportSteps?.length ? <div className="table-wrap"><table><thead><tr><th>步骤</th><th>状态</th><th>说明</th></tr></thead><tbody>{reportSteps.map((step, index) => <tr key={index}><td>{String(step.kind ?? step.timeframe ?? `步骤 ${index + 1}`)}</td><td>{String(step.status ?? '—')}</td><td>{String(step.error ?? step.reason ?? '')}</td></tr>)}</tbody></table></div> : <p className="muted-copy">新增、修订、未变化、冲突和失败数量可在高级诊断中查看。</p>}
      <details className="advanced-data"><summary>高级诊断 JSON</summary><pre className="json-preview">{JSON.stringify(report, null, 2)}</pre></details>
    </Card></section></div>}
  </>
}
