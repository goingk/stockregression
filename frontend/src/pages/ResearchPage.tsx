import { useCallback, useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { FlaskConical, RefreshCw, RotateCcw } from 'lucide-react'
import { get, post } from '../api'
import ChartView from '../ChartView'
import EquityPreview from '../EquityPreview'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, Field, JobProgress, Loading, PageHeader, SelectInput, TextInput, defaultConfig, configPayload, loadConfig, percent, useErrorMessage } from '../components'
import type { BacktestConfig, Job } from '../types'

type Range = { low: number; high: number; step: number }
type Candidate = Record<string, unknown> & { params?: Record<string, unknown>; metrics?: Record<string, number> }
type SearchConfig = { base?: BacktestConfig; mode?: string; n_trials?: number; seed?: number; ranges?: Record<string, unknown>; [key: string]: unknown }
type TestRun = { strategies?: Record<string, { metrics?: Record<string, number> }> }
type OptimizationResult = {
  mode?: string; seed?: number; n_trials?: number; data_version?: string; label?: string; warnings?: string[]
  config?: SearchConfig; is_demo?: boolean; quality?: Record<string, unknown>
  historical?: { status?: string; best_params?: Record<string, unknown>; candidates?: Candidate[]; best_result?: Record<string, unknown> }
  robust?: { status?: string; reason?: string; selected_params?: Record<string, unknown>; candidates?: Candidate[]; split?: Record<string, unknown>; train_metrics?: Record<string, number>; validation_metrics?: Record<string, number>; test_metrics?: Record<string, number>; test_result?: TestRun & Record<string, unknown> }
}
type Props = { onLoadConfig: (config: BacktestConfig) => void }
const labels: Record<string, string> = { buy_step: '买入步长', sell_step: '卖出步长', grid_step: '网格间距', buy_ratio: '旧版买入比例', sell_ratio: '旧版卖出比例', initial_position: '旧版初始仓位', min_position: '旧版最小仓位', max_position: '旧版最大仓位', buy_shares: '每次买入股数', sell_shares: '每次卖出股数', initial_shares: '初始持股', min_shares: '最低持股', max_shares: '最高持股' }
const defaults: Record<string, Range> = {
  buy_step: { low: 0.005, high: 0.1, step: 0.005 }, sell_step: { low: 0.005, high: 0.1, step: 0.005 }, grid_step: { low: 0.005, high: 0.1, step: 0.005 },
  buy_shares: { low: 100, high: 1000, step: 100 }, sell_shares: { low: 100, high: 1000, step: 100 },
  initial_shares: { low: 0, high: 2000, step: 100 }, min_shares: { low: 0, high: 1000, step: 100 }, max_shares: { low: 1000, high: 5000, step: 100 },
}
const knownRangeNames = Object.keys(defaults)
const pctRange = new Set(['buy_step', 'sell_step', 'grid_step'])
function n(value: unknown, fallback = 0) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : fallback }
function record(value: unknown): Record<string, unknown> | undefined { return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : undefined }
function searchConfigOf(value: unknown): SearchConfig | undefined { return record(value) as SearchConfig | undefined }
function warningsOf(value: unknown): string[] {
  const item = record(value)
  const quality = record(item?.quality)
  return [item?.warnings, quality?.warnings].flatMap((items) => Array.isArray(items) ? items.map(String) : [])
}
function hasDemoFlag(value: unknown) {
  const item = record(value)
  return item?.is_demo === true || record(item?.quality)?.is_demo === true || record(item?.metadata)?.is_demo === true
}
function isOptimizationJob(job: Job) {
  const config = record(job.payload?.config)
  return job.kind === 'optimization' || job.kind === 'optimize' || record(config?.base) != null
}
function jobModeLabel(value: unknown) {
  return value === 'historical' ? '历史最优' : value === 'robust' ? '稳健验证' : value === 'both' ? '历史最优 + 稳健验证' : '研究任务'
}
function jobStatusLabel(value: string) {
  return value === 'queued' ? '排队中' : value === 'running' ? '运行中' : value === 'completed' ? '已完成' : value === 'failed' ? '失败' : value === 'cancelled' ? '已取消' : value
}
function rangeValue(key: string, value: unknown) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '—'
  return pctRange.has(key) ? percent(parsed, 3) : String(Number(parsed.toPrecision(6)))
}
function getMetrics(candidate: Candidate, outOfSample = false) {
  if (candidate.metrics && typeof candidate.metrics === 'object') return candidate.metrics
  if (outOfSample && candidate.validation_metrics && typeof candidate.validation_metrics === 'object') return candidate.validation_metrics as Record<string, number>
  if (candidate.train_metrics && typeof candidate.train_metrics === 'object') return candidate.train_metrics as Record<string, number>
  return candidate as Record<string, number>
}
function paramsOf(candidate: Candidate) { return candidate.params && typeof candidate.params === 'object' ? candidate.params : candidate }

function StrategyComparison({ run }: { run?: TestRun }) {
  const strategies = run?.strategies ?? {}
  const entries = [['grid', '网格策略'], ['dca', '定投基准'], ['buy_hold', '买入持有基准']] as const
  const metrics = [['total_return', '总收益'], ['cagr', '年化收益'], ['max_drawdown', '最大回撤'], ['sharpe', '夏普比率']] as const
  const available = entries.filter(([key]) => strategies[key])
  return <div className='robust-metric-block'><h4>测试段策略对比（未参与筛选）</h4>{available.length ? <div className='table-wrap'><table><thead><tr><th>指标</th>{available.map(([key, label]) => <th key={key}>{label}</th>)}</tr></thead><tbody>{metrics.map(([key, label]) => <tr key={key}><td>{label}</td>{available.map(([strategyKey]) => { const raw = strategies[strategyKey].metrics?.[key]; const value = raw == null ? NaN : Number(raw); return <td key={strategyKey}>{Number.isFinite(value) ? key === 'sharpe' ? value.toFixed(2) : percent(value) : '—'}</td> })}</tr>)}</tbody></table></div> : <Empty title='测试段策略明细暂不可用' detail='可查看上方测试段网格指标。' />}</div>
}

function MetricStrip({ title, metrics }: { title: string; metrics?: Record<string, number> }) {
  if (!metrics) return <div className='robust-metric-block'><h4>{title}</h4><Empty title='暂无指标' /></div>
  return <div className='robust-metric-block'><h4>{title}</h4><div className='metric-chip-grid'>{[['总收益', metrics.total_return], ['年化收益', metrics.cagr], ['最大回撤', metrics.max_drawdown], ['夏普', metrics.sharpe]].map(([name, value]) => <div key={String(name)}><small>{String(name)}</small><strong>{value == null ? '—' : String(name).includes('夏普') ? n(value).toFixed(2) : percent(n(value))}</strong></div>)}</div></div>
}

export default function ResearchPage({ onLoadConfig }: Props) {
  const [base, setBase] = useState<BacktestConfig>(defaultConfig)
  const [ranges, setRanges] = useState<Record<string, Range>>(defaults)
  const [mode, setMode] = useState<'historical' | 'robust' | 'both'>('both')
  const [trials, setTrials] = useState(200)
  const [seed, setSeed] = useState(42)
  const [topCandidates, setTopCandidates] = useState(20)
  const [maxDrawdown, setMaxDrawdown] = useState(0.2)
  const [minTrades, setMinTrades] = useState(10)
  const [minHistoryDays, setMinHistoryDays] = useState(504)
  const [minMinuteRatio, setMinMinuteRatio] = useState(0.95)
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState('')
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [result, setResult] = useState<OptimizationResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const err = useErrorMessage()

  const refreshJobs = useCallback(async () => {
    try {
      const list = await get<Job[]>('/jobs')
      const filtered = list.filter(isOptimizationJob)
      setJobs(filtered)
      setSelectedJobId((current) => current && filtered.some((job) => job.id === current) ? current : filtered[0]?.id ?? '')
    } catch (e) { setError(err(e)) } finally { setLoading(false) }
  }, [err])
  useEffect(() => { void refreshJobs() }, [refreshJobs])
  useEffect(() => { const timer = window.setInterval(() => void refreshJobs(), 3000); return () => window.clearInterval(timer) }, [refreshJobs])

  useEffect(() => {
    if (!selectedJobId) { setSelectedJob(null); setResult(null); return }
    let active = true
    setResult(null)
    const load = async () => {
      try {
        const job = await get<Job>('/jobs/' + encodeURIComponent(selectedJobId))
        if (!active) return
        setSelectedJob(job)
        if (job.result_id && (job.status === 'completed' || job.status === 'cancelled')) {
          const data = await get<OptimizationResult>('/jobs/' + encodeURIComponent(selectedJobId) + '/result')
          if (active) setResult(data)
        }
      } catch (e) { if (active) setError(err(e)) }
    }
    void load()
    const events = new EventSource('/api/v1/jobs/' + encodeURIComponent(selectedJobId) + '/events')
    events.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data) as Job
        if (!active) return
        setSelectedJob(job)
        if (job.status === 'completed' || job.status === 'cancelled') {
          if (!job.result_id) { events.close(); return }
          void get<OptimizationResult>('/jobs/' + encodeURIComponent(selectedJobId) + '/result').then((data) => { if (active) setResult(data) }).catch((e) => { if (active) setError(err(e)) })
          events.close()
        } else if (job.status === 'failed') events.close()
      } catch (e) { setError(err(e)) }
    }
    events.onerror = () => { /* the polling request remains the fallback */ }
    return () => { active = false; events.close() }
  }, [selectedJobId, err])

  const updateRange = (name: string, key: keyof Range, shown: number) => {
    const next = pctRange.has(name) ? shown / 100 : shown
    setRanges((old) => ({ ...old, [name]: { ...old[name], [key]: next } }))
  }
  const submit = async () => {
    setBusy(true); setError(''); setNotice('')
    try {
      const response = await post<{ job_id: string }>('/optimizations', {
        base: configPayload(base), mode, n_trials: Math.min(5000, Math.max(1, trials)), seed, top_candidates: topCandidates,
        max_drawdown: maxDrawdown, min_trades: minTrades, min_history_days: minHistoryDays, min_minute_ratio: minMinuteRatio,
        ranges: Object.fromEntries(Object.entries(ranges).filter(([name]) => !['buy_step', 'sell_step', 'grid_step'].includes(name) || (base.grid_mode === 'trailing' ? name !== 'grid_step' : name === 'grid_step'))),
      })
      setSelectedJobId(response.job_id); setNotice('参数搜索任务已提交，完成后可查看历史最优与样本外推荐。')
      await refreshJobs()
    } catch (e) { setError(err(e)) } finally { setBusy(false) }
  }
  const loadBest = (which: 'historical' | 'robust') => {
    if (!result) return
    const params = which === 'historical' ? result.historical?.best_params : result.robust?.selected_params
    if (!params) { setError(which === 'historical' ? '历史最优参数尚未生成。' : '样本外推荐参数尚未生成。'); return }
    const originalBase = result.config?.base ?? searchConfigOf(selectedJob?.payload?.config)?.base
    if (!originalBase) { setError('这条研究任务未保存原始股票与区间配置，无法安全载入。请重新运行参数研究。'); return }
    const { buy_ratio, sell_ratio, ...configParams } = params
    const patch = configParams as Partial<BacktestConfig>
    const config = { ...originalBase, ...patch, fees: { ...originalBase.fees, ...patch.fees } } as BacktestConfig
    const legacy = config as unknown as Record<string, unknown>
    if (typeof buy_ratio === 'number') legacy.buy_quantity = { mode: 'equity_ratio', value: buy_ratio }
    if (typeof sell_ratio === 'number') legacy.sell_quantity = { mode: 'equity_ratio', value: sell_ratio }
    void loadConfig(config).then(value => onLoadConfig(value.config)).catch(error => setError(err(error)))
  }

  const cancel = async () => {
    if (!selectedJobId) return
    try { await post('/jobs/' + encodeURIComponent(selectedJobId) + '/cancel'); await refreshJobs() }
    catch (e) { setError(err(e)) }
  }
  const chartOption: EChartsOption = useMemo(() => {
    const points: Array<{ value: [number, number]; name: string; params: Record<string, unknown> }> = []
    const sections: Array<[string, Candidate[] | undefined]> = [['历史候选', result?.historical?.candidates], ['样本外候选', result?.robust?.candidates]]
    for (const [section, candidates] of sections) for (const item of candidates ?? []) {
      const metrics = getMetrics(item, section === '样本外候选')
      const ret = n(metrics.total_return ?? metrics.return)
      const dd = Math.abs(n(metrics.max_drawdown ?? metrics.drawdown))
      if (Number.isFinite(ret) && Number.isFinite(dd)) points.push({ value: [ret, dd], name: section, params: paramsOf(item) })
    }
    return {
      tooltip: { trigger: 'item', formatter: (value: unknown) => { const p = value as { data: { name: string; value: [number, number]; params: Record<string, unknown> } }; return p.data.name + '<br/>收益：' + percent(p.data.value[0]) + '<br/>回撤：' + percent(p.data.value[1]) + '<br/>' + JSON.stringify(p.data.params) } },
      legend: { top: 0 }, grid: { left: 55, right: 24, top: 36, bottom: 48 },
      xAxis: { name: '总收益率', nameLocation: 'middle', nameGap: 30, axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#edf0f5' } } },
      yAxis: { name: '最大回撤', axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#edf0f5' } } },
      series: [
        { name: '历史候选', type: 'scatter', data: points.filter((p) => p.name === '历史候选'), symbolSize: 10, itemStyle: { color: '#2864dc', opacity: 0.72 } },
        { name: '样本外候选', type: 'scatter', data: points.filter((p) => p.name === '样本外候选'), symbolSize: 11, itemStyle: { color: '#e85c63', opacity: 0.8 } },
      ],
    }
  }, [result])
  const historyCandidates = result?.historical?.candidates ?? []
  const robustCandidates = result?.robust?.candidates ?? []
  const robust = result?.robust
  const resultConfig = result?.config
  const taskConfig = searchConfigOf(selectedJob?.payload?.config)
  const originalBase = resultConfig?.base ?? taskConfig?.base
  const actualRanges = record(resultConfig?.ranges) ?? record(taskConfig?.ranges)
  const actualMode = result?.mode ?? resultConfig?.mode ?? taskConfig?.mode
  const actualTrials = result?.n_trials ?? resultConfig?.n_trials ?? taskConfig?.n_trials
  const actualSeed = result?.seed ?? resultConfig?.seed ?? taskConfig?.seed
  const nestedRuns = [result?.historical?.best_result, robust?.test_result]
  const researchWarnings = Array.from(new Set([result, ...nestedRuns].flatMap(warningsOf)))
  const isDemoResearch = [result, result?.quality, ...nestedRuns].some(hasDemoFlag) || researchWarnings.some((warning) => /演示|demo/i.test(warning))

  return <>
    <PageHeader eyebrow='PARAMETER RESEARCH' title='参数研究' description='分别评估历史收益最优参数与分段验证后的样本外推荐；每种模式最多 5000 次试验。' action={<Button variant='secondary' onClick={() => { setRanges(defaults); setNotice('已恢复默认搜索范围。') }}><RotateCcw size={15} />恢复默认范围</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}{notice && <div className='notice success'>{notice}</div>}
    <div className='research-layout research-share-layout'>
      <div className='main-column'><Card title='基准回测配置' subtitle='优化会在此配置和数据覆盖基础上展开。'><div className="parameter-scroll"><ConfigForm value={base} onChange={setBase} /></div></Card></div>
      <aside className='side-column'><EquityPreview config={base} />
        <Card title='搜索任务' subtitle='选择研究模式和筛选条件。'>
          <div className='form-grid two'><Field label='研究模式'><SelectInput value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}><option value='both'>历史最优 + 稳健验证</option><option value='historical'>历史最优</option><option value='robust'>稳健验证</option></SelectInput></Field><Field label='候选数上限'><TextInput type='number' min={1} max={100} value={topCandidates} onChange={(e) => setTopCandidates(Number(e.target.value))} /></Field><Field label='每种搜索试验次数（最多 5000）'><TextInput type='number' min={1} max={5000} value={trials} onChange={(e) => setTrials(Number(e.target.value))} /><small>“历史最优 + 稳健验证”会分别搜索，各执行 N 次；合计最多 2×N 次试验。</small></Field><Field label='随机种子'><TextInput type='number' value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field><Field label='最大回撤限制'><TextInput type='number' min={1} max={100} step='1' value={Number((maxDrawdown * 100).toPrecision(6))} onChange={(e) => setMaxDrawdown(Number(e.target.value) / 100)} /><small>按百分数设置</small></Field><Field label='最少成交笔数'><TextInput type='number' min={0} value={minTrades} onChange={(e) => setMinTrades(Number(e.target.value))} /></Field></div>
          <p className='inline-warning'>一年默认区间通常不足稳健研究的 504 个交易日门槛；不会自动降低样本要求。</p><Button variant='quiet' onClick={() => { get<{ first_date?: string }>(`/data/coverage/${base.symbol}`).then(value => { if (value.first_date) setBase(previous => ({ ...previous, start_date: new Date(Date.parse(value.first_date! + 'T00:00:00Z') + 86400000).toISOString().slice(0, 10), reference_price_mode: 'auto', reference_price: null })) }).catch(error => setError(err(error))) }}>使用全部本地历史</Button><details className='advanced-disclosure'><summary>高级样本约束</summary><div className='form-grid two'><Field label='最少历史交易日'><TextInput type='number' min={20} value={minHistoryDays} onChange={(e) => setMinHistoryDays(Number(e.target.value))} /></Field><Field label='最小分钟覆盖率'><TextInput type='number' min={0} max={100} step='1' value={Number((minMinuteRatio * 100).toPrecision(6))} onChange={(e) => setMinMinuteRatio(Number(e.target.value) / 100)} /><small>按百分数设置</small></Field></div></details>
          <div className='card-bottom'><Button busy={busy} onClick={() => void submit()}><FlaskConical size={15} />启动参数搜索</Button></div>
        </Card>
        <Card title='研究进度' subtitle={selectedJob ? selectedJob.message || selectedJob.status : '提交任务后查看实时进度。'} action={<Button variant='quiet' onClick={() => { setError(''); void refreshJobs() }}><RefreshCw size={14} /></Button>}>
          {loading ? <Loading /> : jobs.length === 0 ? <Empty title='暂无研究任务' /> : <><Field label='任务记录'><SelectInput value={selectedJobId} onChange={(e) => setSelectedJobId(e.target.value)}>{jobs.map((job) => { const config = searchConfigOf(job.payload?.config); const symbol = config?.base?.symbol ?? '未识别标的'; return <option key={job.id} value={job.id}>{symbol} · {jobModeLabel(config?.mode)} · {jobStatusLabel(job.status)}</option> })}</SelectInput></Field>{selectedJob && <div className='job-card active-job'><JobProgress progress={selectedJob.progress} status={selectedJob.status} /><p>{selectedJob.message}</p>{selectedJob.error != null ? <pre className='error-detail'>{typeof selectedJob.error === 'string' ? selectedJob.error : JSON.stringify(selectedJob.error, null, 2)}</pre> : selectedJob.status === 'failed' && <pre className='error-detail'>任务失败，但没有返回具体错误原因。请重试或检查任务日志。</pre>}{['running', 'queued'].includes(selectedJob.status) && <Button variant='quiet' onClick={() => void cancel()}>取消搜索</Button>}</div>}</>}
        </Card>
      </aside>
    </div>
    <Card title='可调搜索范围' subtitle='持股和交易量以 100 股 / 份为单位；仅搜索当前模式生效的参数。'>
      <div className='range-grid'>{knownRangeNames.filter(name => !['buy_step', 'sell_step', 'grid_step'].includes(name) || (base.grid_mode === 'trailing' ? name !== 'grid_step' : name === 'grid_step')).map((name) => <div className='range-row' key={name}><div><strong>{labels[name]}</strong><small>{pctRange.has(name) ? '步长（%）' : '股 / 份'}</small></div>{(['low', 'high', 'step'] as const).map((key) => { const value = ranges[name][key]; const shown = pctRange.has(name) ? Number((value * 100).toPrecision(8)) : value; return <Field key={key} label={key === 'low' ? '下限' : key === 'high' ? '上限' : '步长'}><TextInput type='number' step={pctRange.has(name) ? 'any' : '100'} min={0} value={shown} onChange={(e) => updateRange(name, key, Number(e.target.value))} /></Field> })}</div>)}</div>
    </Card>
    {result && <><Card title='本次研究样本' subtitle='以下标的、区间和参数范围来自已完成任务的原始配置。'><div className='result-meta'><div><strong>{originalBase?.symbol ?? '原始标的不详'}</strong><span>{originalBase ? `${originalBase.frequency} · ${originalBase.start_date} 至 ${originalBase.end_date ?? '最新可用日期'}` : '任务未保存原始股票与日期配置'}</span></div><div>{isDemoResearch && <Badge tone='amber'>演示数据</Badge>}<Badge tone='blue'>{jobModeLabel(actualMode)}</Badge></div></div>{isDemoResearch && <div className='demo-callout'>此研究使用演示行情生成，不能代表真实历史数据或投资结论。</div>}<div className='result-meta'><span>每种搜索试验次数：{actualTrials ?? '—'}</span><span>随机种子：{actualSeed ?? '—'}</span><span>数据版本：{result.data_version ?? '—'}</span></div>{researchWarnings.length > 0 && <div className='warning-list'>{researchWarnings.map((warning, index) => <div key={index}><span className='warning-dot' />{warning}</div>)}</div>}<div className='param-block'><strong>本次实际搜索范围</strong>{actualRanges && Object.keys(actualRanges).length ? <div className='table-wrap'><table><thead><tr><th>参数</th><th>下限</th><th>上限</th><th>步长</th></tr></thead><tbody>{Object.entries(actualRanges).map(([key, raw]) => { const range = record(raw); return <tr key={key}><td>{labels[key] ?? key}</td><td>{range ? rangeValue(key, range.low) : '—'}</td><td>{range ? rangeValue(key, range.high) : '—'}</td><td>{range ? rangeValue(key, range.step) : '—'}</td></tr> })}</tbody></table></div> : <Empty title='任务中未保存搜索范围' />}</div></Card><div className='research-summary'><Card title='已搜索候选中的历史最优' subtitle='在历史样本内已搜索候选中得分最高的配置，不代表样本外表现。' action={<Button variant='secondary' onClick={() => loadBest('historical')} disabled={!result.historical?.best_params || !originalBase}>载入历史最优配置</Button>}>
      {!result.historical ? <Empty title='本次任务未运行历史优化' /> : <><div className='result-meta'><Badge tone='blue'>{result.historical.status ?? '已完成'}</Badge><span>版本 {result.data_version ?? '—'}</span></div><pre className='json-preview'>{JSON.stringify({ best_params: result.historical.best_params, best_result: result.historical.best_result }, null, 2)}</pre></>}
    </Card><Card title='样本外推荐' subtitle='展示训练、验证、测试切分和测试段策略对比。' action={<Button variant='secondary' onClick={() => loadBest('robust')} disabled={!robust?.selected_params || !originalBase}>载入样本外配置</Button>}>
      {!robust ? <Empty title='本次任务未运行稳健验证' /> : <><div className='result-meta'><Badge tone={robust.status === 'selected' || robust.selected_params ? 'green' : 'amber'}>{robust.status ?? '验证结果'}</Badge>{robust.reason && <span>{robust.reason}</span>}</div>{robust.split && <pre className='json-preview split-preview'>{JSON.stringify(robust.split, null, 2)}</pre>}<MetricStrip title='训练段（用于候选搜索）' metrics={robust.train_metrics} /><MetricStrip title='验证段（用于候选筛选）' metrics={robust.validation_metrics} /><MetricStrip title='测试段网格策略（最终样本外评估）' metrics={robust.test_metrics} /><StrategyComparison run={robust.test_result} /><div className='param-block'><strong>样本外推荐参数</strong><pre className='json-preview'>{JSON.stringify(robust.selected_params ?? {}, null, 2)}</pre></div></>}
    </Card></div>
    <div className='chart-grid'><Card title='候选收益与回撤' subtitle='比较每组候选参数的历史收益与最大回撤。'><ChartView option={chartOption} height={350} /></Card><Card title='候选列表' subtitle={'历史候选 ' + historyCandidates.length + ' · 样本外候选 ' + robustCandidates.length}><div className='table-wrap scroll-table'><table><thead><tr><th>类别</th><th>总收益</th><th>最大回撤</th><th>夏普</th><th>参数</th></tr></thead><tbody>{[...historyCandidates.map((x) => ['历史', x] as const), ...robustCandidates.map((x) => ['样本外', x] as const)].map(([group, candidate], i) => { const metrics = getMetrics(candidate, group === '样本外'); return <tr key={i}><td>{group}</td><td>{percent(n(metrics.total_return ?? metrics.return))}</td><td>{percent(n(metrics.max_drawdown ?? metrics.drawdown))}</td><td>{n(metrics.sharpe).toFixed(2)}</td><td><code>{JSON.stringify(paramsOf(candidate))}</code></td></tr> })}</tbody></table></div></Card></div>
    </>}
  </>
}
