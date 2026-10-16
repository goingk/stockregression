import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Copy, FlaskConical, Play, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'
import { del, get, post, put } from '../api'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, JobProgress, Loading, PageHeader, dateText, defaultConfig, configError, configPayload, loadConfig, markFeeDefaultsPending, readFeeDefaults, readPendingFeeDefaults, clearPendingFeeDefaults, validFeeDefaults, writeFeeDefaults, useErrorMessage } from '../components'
import EquityPreview from '../EquityPreview'
import ResearchPage, { type ResearchPanelHandle } from './ResearchPage'
import type { BacktestConfig, ConfigRecord, Job } from '../types'
import { rememberWorkbenchGroup, useWorkbenchDefaults } from '../useWorkbenchDefaults'
import { strategyDefaults, workbenchConfig } from '../workbenchDefaults'
import type { ResearchDefaults, StrategyDefaultsGroup } from '../workbenchDefaults'

type SettingsResponse = { fee_defaults?: unknown }

export default function BacktestPage({ onOpenResult }: { onOpenResult: (id: string) => void }) {
  const preferences = useWorkbenchDefaults()
  const [config, setConfig] = useState<BacktestConfig | null>(null)
  const [researchSettings, setResearchSettings] = useState<ResearchDefaults | null>(null)
  const [formRevision, setFormRevision] = useState(0)
  const [feeDefaults, setFeeDefaults] = useState(readFeeDefaults)
  const [feeDefaultsDirty, setFeeDefaultsDirty] = useState(false)
  const [records, setRecords] = useState<ConfigRecord[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [researchJob, setResearchJob] = useState<Job | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [researchBusy, setResearchBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [settingsLoading, setSettingsLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [saveDialog, setSaveDialog] = useState<{ copy: boolean; name: string } | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<ConfigRecord | null>(null)
  const [dialogError, setDialogError] = useState('')
  const [taskBusy, setTaskBusy] = useState<string | null>(null)
  const researchRef = useRef<ResearchPanelHandle>(null)
  const feeRevision = useRef(0)
  const feeSaveChain = useRef<Promise<void>>(Promise.resolve())
  const feeDefaultsRef = useRef(feeDefaults)
  const feeDefaultsDirtyRef = useRef(false)
  const feeKeepaliveRevision = useRef<number | null>(null)
  const err = useErrorMessage()

  const flushFeeDefaults = useCallback(() => {
    if (!feeDefaultsDirtyRef.current) return
    const revision = feeRevision.current
    if (feeKeepaliveRevision.current === revision) return
    feeKeepaliveRevision.current = revision
    const body = JSON.stringify({ fee_defaults: { ...feeDefaultsRef.current } })
    void fetch('/api/v1/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body, keepalive: true }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      if (revision === feeRevision.current) {
        feeDefaultsDirtyRef.current = false
      }
    }).catch(() => {
      if (revision === feeRevision.current) feeKeepaliveRevision.current = null
    })
  }, [])

  useEffect(() => {
    const handlePageHide = () => flushFeeDefaults()
    window.addEventListener('pagehide', handlePageHide)
    return () => {
      window.removeEventListener('pagehide', handlePageHide)
      flushFeeDefaults()
    }
  }, [flushFeeDefaults])

  const updateConfig = useCallback((next: BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)) => {
    setConfig((previous) => previous ? typeof next === 'function' ? next(previous) : next : previous)
  }, [])

  useEffect(() => {
    let active = true
    get<SettingsResponse>('/settings').then((settings) => {
      if (!active) return
      const pendingDefaults = readPendingFeeDefaults()
      const defaults = pendingDefaults ?? (validFeeDefaults(settings.fee_defaults) ? settings.fee_defaults : readFeeDefaults())
      feeDefaultsRef.current = defaults
      feeDefaultsDirtyRef.current = pendingDefaults != null
      writeFeeDefaults(defaults)
      setFeeDefaults(defaults)
      if (pendingDefaults) {
        feeRevision.current += 1
        setFeeDefaultsDirty(true)
      }
    }).catch(() => {
      if (active) {
        const pendingDefaults = readPendingFeeDefaults()
        const defaults = pendingDefaults ?? readFeeDefaults()
        feeDefaultsRef.current = defaults
        feeDefaultsDirtyRef.current = pendingDefaults != null
        setFeeDefaults(defaults)
        if (pendingDefaults) {
          feeRevision.current += 1
          setFeeDefaultsDirty(true)
        }
      }
    }).finally(() => { if (active) setSettingsLoading(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (settingsLoading || preferences.loading) return
    setConfig(current => current ?? workbenchConfig(defaultConfig(feeDefaultsRef.current), preferences.value))
    setResearchSettings(current => current ?? structuredClone(preferences.value.research))
  }, [settingsLoading, preferences.loading, preferences.value])

  useEffect(() => {
    if (!feeDefaultsDirty) return
    const revision = feeRevision.current
    const snapshot = { ...feeDefaults }
    feeSaveChain.current = feeSaveChain.current.catch(() => undefined).then(async () => {
      if (revision !== feeRevision.current) return
      await put('/settings', { fee_defaults: snapshot })
      if (revision === feeRevision.current) {
        feeDefaultsDirtyRef.current = false
        setFeeDefaultsDirty(false)
        clearPendingFeeDefaults()
      }
    }).catch((e) => { if (revision === feeRevision.current) setError(`默认佣金设置未能保存：${err(e)}`) })
  }, [feeDefaults, feeDefaultsDirty, err])

  const refresh = useCallback(async () => {
    try {
      const [saved, allJobs] = await Promise.all([get<ConfigRecord[]>('/configs'), get<Job[]>('/jobs')])
      setRecords(saved)
      setJobs(allJobs.filter((job) => ['backtest', 'batch_backtest'].includes(job.kind) || job.payload?.configs != null).slice(0, 8))
    } catch (e) { setError(err(e)) } finally { setLoading(false) }
  }, [err])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    const timer = window.setInterval(() => { void refresh() }, 2500)
    return () => window.clearInterval(timer)
  }, [refresh])

  const activeBacktest = useMemo(() => jobs.find((job) => ['queued', 'running'].includes(job.status)) ?? null, [jobs])
  const displayedBacktest = activeBacktest ?? jobs[0] ?? null
  const taskStatuses = [displayedBacktest, researchJob].filter((job): job is Job => Boolean(job))

  const rememberFeeDefaults = (next: typeof feeDefaults) => {
    if (!validFeeDefaults(next)) return
    feeRevision.current += 1
    feeDefaultsRef.current = next
    feeDefaultsDirtyRef.current = true
    setFeeDefaults(next)
    setFeeDefaultsDirty(true)
    writeFeeDefaults(next)
    markFeeDefaultsPending()
  }

  const rememberStrategyDefaults = (group: StrategyDefaultsGroup, next: BacktestConfig) => {
    const defaults = strategyDefaults(next)
    rememberWorkbenchGroup(group, defaults[group])
  }

  const updateResearchSettings = useCallback((next: ResearchDefaults) => {
    setResearchSettings(next)
    rememberWorkbenchGroup('research', next)
  }, [])

  const restoreParameters = () => {
    setConfig(workbenchConfig(defaultConfig(feeDefaults), preferences.value))
    setResearchSettings(structuredClone(preferences.value.research))
    setFormRevision(current => current + 1)
    setNotice('已恢复上次有效参数，标的已清空。')
  }

  const loadExplicitConfig = async (value: BacktestConfig, copied = false) => {
    const loaded = await loadConfig(value)
    setConfig(loaded.config)
    setNotice((copied ? '已复制配置草稿。' : '') + loaded.notice)
  }

  const saveConfig = (copy = false) => {
    if (!config) return
    const validation = configError(config)
    if (validation) { setError(validation); return }
    setError(''); setNotice(''); setDialogError('')
    setSaveDialog({ copy, name: copy ? `${config.symbol} 配置副本` : `${config.symbol} 网格配置` })
  }

  const confirmSaveConfig = async () => {
    const name = saveDialog?.name.trim()
    if (!saveDialog || !name || !config) { setDialogError('请填写配置名称。'); return }
    const validation = configError(config)
    if (validation) { setDialogError(validation); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const response = await post<{ id: string }>('/configs', { name: name.trim(), config: configPayload(config) })
      const saved: ConfigRecord = { id: response.id, name: name.trim(), config }
      setRecords((list) => [saved, ...list.filter((item) => item.id !== saved.id)])
      setNotice(`配置“${saved.name}”已保存。`)
      setSaveDialog(null)
    } catch (e) { setDialogError(err(e)) } finally { setBusy(false) }
  }

  const runOne = async () => {
    if (!config) return
    const validation = configError(config)
    if (validation) { setError(validation); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const result = await post<{ job_id: string }>('/backtests', configPayload(config))
      setNotice(`回测任务 ${result.job_id} 已提交。`)
      await refresh()
    } catch (e) { setError(err(e)) } finally { setBusy(false) }
  }

  const runResearch = async () => {
    setResearchBusy(true); setError('')
    try { await researchRef.current?.submit() } finally { setResearchBusy(false) }
  }

  const runBatch = async () => {
    if (!config) return
    const chosen = records.filter((item) => selected.includes(item.id))
    if (chosen.length < 2) { setError('批量回测至少选择两份已保存配置。'); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const batchConfigs = await Promise.all(chosen.map(async item => configPayload((await loadConfig(item.config)).config)))
      const result = await post<{ job_ids: string[] }>('/backtests/batch', { configs: batchConfigs })
      setNotice(`已独立提交 ${result.job_ids.length} 个回测任务。`)
      await refresh()
    } catch (e) { setError(err(e)) } finally { setBusy(false) }
  }

  const removeConfig = (record: ConfigRecord) => { setDialogError(''); setDeleteTarget(record) }
  const confirmRemoveConfig = async () => {
    if (!deleteTarget) return
    setBusy(true); setDialogError('')
    try {
      await del(`/configs/${encodeURIComponent(deleteTarget.id)}`)
      setRecords((list) => list.filter((item) => item.id !== deleteTarget.id))
      setSelected((list) => list.filter((id) => id !== deleteTarget.id))
      setNotice(`配置“${deleteTarget.name}”已删除。`)
      setDeleteTarget(null)
    } catch (e) { setDialogError(err(e)) } finally { setBusy(false) }
  }

  const cancelJob = async (jobId: string) => {
    setTaskBusy(jobId)
    try { await post(`/jobs/${encodeURIComponent(jobId)}/cancel`); await refresh() } catch (e) { setError(err(e)) } finally { setTaskBusy(null) }
  }

  const cancelTask = async (job: Job) => {
    if (job !== researchJob) { await cancelJob(job.id); return }
    setTaskBusy(job.id)
    try { await researchRef.current?.cancel() } finally { setTaskBusy(null) }
  }

  const acceptPreviewFunding = useCallback((funding: NonNullable<import('../types').BacktestResult['funding']>, context: { symbol: string; start_date: string }) => {
    const reference = funding.reference_price > 0 ? funding.reference_price : funding.valuation_price > 0 ? funding.valuation_price : null
    if (reference == null) return
    setConfig((previous) => {
      if (!previous || previous.symbol !== context.symbol || previous.start_date !== context.start_date || previous.reference_price_mode !== 'auto' || previous.reference_price === reference) return previous
      return { ...previous, reference_price: reference }
    })
  }, [])

  const loadResearchConfig = (value: BacktestConfig) => {
    setConfig(value)
    setNotice('研究候选配置已载入，预览已更新。')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const renderJobStatus = (job: Job) => <div className='task-status-row' key={job.id}>
    <div className='task-status-heading'><strong>{job.kind === 'optimization' || job.kind === 'optimize' ? '参数研究' : job.kind === 'batch_backtest' || job.payload?.configs ? '批量回测' : '策略回测'}</strong><span>{job.message || (job.status === 'completed' ? '任务完成' : job.status)}</span><Badge tone={job.status === 'completed' ? 'green' : job.status === 'failed' ? 'red' : job.status === 'cancelled' ? 'neutral' : 'blue'}>{job.status}</Badge></div>
    <JobProgress progress={job.progress} status={job.status} />
    <div className='task-status-actions'>{job.result_id && (job.kind === 'optimization' || job.kind === 'optimize' ? <button className='text-link' onClick={() => researchRef.current?.focusResult()}>查看研究结果</button> : <button className='text-link' onClick={() => onOpenResult(job.result_id!)}>查看结果</button>)}{['queued', 'running'].includes(job.status) && <button className='text-link danger-text' disabled={taskBusy === job.id} onClick={() => void cancelTask(job)}>{taskBusy === job.id ? '取消中…' : '取消'}</button>}</div>
  </div>

  if (!config || !researchSettings) return <><PageHeader eyebrow='BACKTEST & RESEARCH' title='回测与研究' description='共用一份策略配置，预览网格、全仓持有和定投的收益率曲线。' /><div className='route-loading'><Loading label='读取本机参数设置…' /></div></>
  const invalidConfig = Boolean(configError(config))

  return <>
    <PageHeader eyebrow='BACKTEST & RESEARCH' title='回测与研究' description='共用策略配置与实时预览；运行回测，或搜索历史最优与样本外稳健参数。' action={<Button variant='secondary' onClick={restoreParameters}><RefreshCw size={15} />恢复上次参数</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {preferences.error && <ErrorBox message={preferences.error} />}
    {notice && <div className='notice success'><span>{notice}</span></div>}
    <div className='workbench-layout'>
      <div className='workbench-config-column'><Card title='策略配置' subtitle='同一组参数用于实时预览、正式回测与参数研究。'>
        <div className='parameter-scroll'><ConfigForm key={formRevision} value={config} onChange={updateConfig} onFeeDefaultsEdit={rememberFeeDefaults} onDefaultsEdit={rememberStrategyDefaults} /></div>
        <div className='form-footer'><span className='form-footnote'>有效修改会自动记忆；运行前校验数据覆盖与仓位约束。</span><div className='button-row'><Button variant='secondary' busy={busy} disabled={invalidConfig} onClick={() => saveConfig(true)}><Copy size={15} />复制配置</Button><Button variant='secondary' busy={busy} disabled={invalidConfig} onClick={() => saveConfig()}><Save size={15} />保存配置</Button></div></div>
      </Card></div>
      <aside className='workbench-preview-column'>
        <EquityPreview config={config} onFunding={acceptPreviewFunding} />
        <div className='workbench-run-actions'><Button variant='secondary' busy={researchBusy} disabled={invalidConfig || busy || researchBusy} onClick={() => void runResearch()}><FlaskConical size={16} />搜索参数</Button><Button busy={busy} disabled={invalidConfig || researchBusy} onClick={() => void runOne()}><Play size={16} />运行回测</Button></div>
        <div className='task-status-stack' aria-live='polite'>{taskStatuses.length ? taskStatuses.map(renderJobStatus) : <div className='task-status-empty'><span className='online-dot' />没有正在运行的任务</div>}</div>
      </aside>
    </div>

    <div className='workbench-secondary'><ResearchPage ref={researchRef} base={config} settings={researchSettings} onSettingsChange={updateResearchSettings} onBaseChange={updateConfig} onLoadConfig={loadResearchConfig} onJobState={setResearchJob} />
      <details className='workspace-details'><summary>已保存配置与批量回测 <span>{records.length} 份配置</span></summary>
        <Card title='已保存配置' subtitle='可载入或复制配置，也可选择多份配置独立运行比较。' action={<Badge tone='blue'>{records.length} 份</Badge>}>
          {loading ? <Loading /> : records.length === 0 ? <Empty title='还没有保存的配置' detail='保存当前参数后即可复用或批量比较。' /> : <div className='saved-list'>{records.map((record) => <div className='saved-row' key={record.id}><label className='check-label'><input type='checkbox' checked={selected.includes(record.id)} onChange={(e) => setSelected((old) => e.target.checked ? [...old, record.id] : old.filter((id) => id !== record.id))} /><span className='saved-copy'><strong>{record.name}</strong><small>{record.config.symbol} · {record.config.frequency} · {record.config.grid_mode}</small></span></label><div className='saved-actions'><button title='载入' onClick={() => { void loadExplicitConfig(record.config).catch(e => setError(err(e))) }}><RefreshCw size={14} /></button><button title='复制' onClick={() => { void loadExplicitConfig(record.config, true).catch(e => setError(err(e))) }}><Copy size={14} /></button><button title='删除' className='danger-icon' onClick={() => removeConfig(record)}><Trash2 size={14} /></button></div></div>)}</div>}
          <div className='saved-footer'><Button variant='secondary' disabled={selected.length < 2 || busy} busy={busy} onClick={() => void runBatch()}><Plus size={15} />批量运行所选（{selected.length}）</Button></div>
        </Card>
      </details>
      <details className='workspace-details'><summary>回测任务历史 <span>{jobs.length} 条</span></summary>
        {loading ? <Loading /> : jobs.length === 0 ? <Empty title='暂无回测任务' /> : <div className='job-list'>{jobs.map((job) => <div className='job-card' key={job.id}><div className='job-title'><div><strong>{job.kind === 'batch_backtest' ? '批量回测' : '策略回测'}</strong><small>{dateText(job.created_at)} · {job.id.slice(0, 12)}</small></div><Badge tone={job.status === 'completed' ? 'green' : job.status === 'failed' ? 'red' : 'blue'}>{job.status}</Badge></div><JobProgress progress={job.progress} status={job.status} /><p>{job.message || '等待处理'}</p>{job.error != null && <pre className='error-detail'>{typeof job.error === 'string' ? job.error : JSON.stringify(job.error, null, 2)}</pre>}<div className='job-bottom'>{job.result_id && <button className='text-link' onClick={() => onOpenResult(job.result_id!)}>查看结果</button>}{['queued', 'running'].includes(job.status) && <button className='text-link danger-text' onClick={() => void cancelJob(job.id)}>取消任务</button>}</div></div>)}</div>}
      </details>
    </div>

    {saveDialog && <div className='backtest-dialog-overlay' role='presentation' onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setSaveDialog(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setSaveDialog(null) }}>
      <section className='backtest-dialog' role='dialog' aria-modal='true' aria-labelledby='save-config-title'>
        <h2 id='save-config-title'>{saveDialog.copy ? '复制配置' : '保存配置'}</h2><p>为当前回测参数填写一个便于识别的名称。</p>
        <form onSubmit={(event) => { event.preventDefault(); void confirmSaveConfig() }}><label className='backtest-dialog-field'><span>配置名称</span><input autoFocus className='input' value={saveDialog.name} maxLength={80} onChange={(event) => { setSaveDialog({ ...saveDialog, name: event.target.value }); setDialogError('') }} placeholder='例如：浦发银行月度网格' /></label>{dialogError && <div className='backtest-dialog-error' role='alert'>{dialogError}</div>}<div className='backtest-dialog-actions'><Button type='button' variant='quiet' disabled={busy} onClick={() => setSaveDialog(null)}>取消</Button><Button type='submit' disabled={busy}>{busy ? '保存中…' : '确认保存'}</Button></div></form>
      </section>
    </div>}
    {deleteTarget && <div className='backtest-dialog-overlay' role='presentation' onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setDeleteTarget(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setDeleteTarget(null) }}>
      <section className='backtest-dialog' role='dialog' aria-modal='true' aria-labelledby='delete-config-title'><h2 id='delete-config-title'>确认删除配置</h2><p>确定删除“<strong>{deleteTarget.name}</strong>”吗？此操作无法撤销。</p>{dialogError && <div className='backtest-dialog-error' role='alert'>{dialogError}</div>}<div className='backtest-dialog-actions'><Button variant='quiet' disabled={busy} onClick={() => setDeleteTarget(null)}>取消</Button><Button variant='danger' disabled={busy} onClick={() => void confirmRemoveConfig()}>{busy ? '删除中…' : '确认删除'}</Button></div></section>
    </div>}
  </>
}
