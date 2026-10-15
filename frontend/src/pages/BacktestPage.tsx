import { useCallback, useEffect, useMemo, useState } from 'react'
import { Copy, Play, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'
import { del, get, post } from '../api'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, JobProgress, Loading, PageHeader, dateText, defaultConfig, configPayload, loadConfig, useErrorMessage } from '../components'
import EquityPreview from '../EquityPreview'
import type { BacktestConfig, ConfigRecord, Job } from '../types'

type Props = { initialConfig: BacktestConfig | null; clearInitialConfig: () => void; onOpenResult: (id: string) => void }

export default function BacktestPage({ initialConfig, clearInitialConfig, onOpenResult }: Props) {
  const [config, setConfig] = useState<BacktestConfig>(defaultConfig)
  const [records, setRecords] = useState<ConfigRecord[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [saveDialog, setSaveDialog] = useState<{ copy: boolean; name: string } | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<ConfigRecord | null>(null)
  const [dialogError, setDialogError] = useState('')
  const err = useErrorMessage()

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
  useEffect(() => {
    if (initialConfig) { void loadConfig(initialConfig).then(value => { setConfig(value.config); setNotice(value.notice) }).catch(e => setError(err(e))); clearInitialConfig() }
  }, [initialConfig, clearInitialConfig])

  const activeJobs = useMemo(() => jobs.filter((job) => ['queued', 'running'].includes(job.status)), [jobs])

  const saveConfig = (copy = false) => {
    setError(''); setNotice(''); setDialogError('')
    setSaveDialog({ copy, name: copy ? `${config.symbol} 配置副本` : `${config.symbol} 网格配置` })
  }

  const confirmSaveConfig = async () => {
    const name = saveDialog?.name.trim()
    if (!saveDialog || !name) { setDialogError('请填写配置名称。'); return }
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
    setBusy(true); setError(''); setNotice('')
    try {
      const result = await post<{ job_id: string }>('/backtests', configPayload(config))
      setNotice(`回测任务 ${result.job_id} 已提交。`)
      await refresh()
    } catch (e) { setError(err(e)) } finally { setBusy(false) }
  }

  const runBatch = async () => {
    const chosen = records.filter((item) => selected.includes(item.id))
    if (chosen.length < 2) { setError('批量回测至少选择两份已保存配置。'); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const result = await post<{ job_ids: string[] }>('/backtests/batch', { configs: await Promise.all(chosen.map(async item => configPayload((await loadConfig(item.config)).config))) })
      setNotice(`已独立提交 ${result.job_ids.length} 个回测任务。`)
      await refresh()
    } catch (e) { setError(err(e)) } finally { setBusy(false) }
  }

  const removeConfig = (record: ConfigRecord) => {
    setDialogError('')
    setDeleteTarget(record)
  }

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
    try { await post(`/jobs/${encodeURIComponent(jobId)}/cancel`); await refresh() } catch (e) { setError(err(e)) }
  }

  return <>
    <PageHeader eyebrow="BACKTEST WORKSPACE" title="回测工作台" description="按股数配置网格，实时查看净值与定投获利；正式回测保存完整分析。" action={<Button variant="secondary" onClick={() => setConfig(defaultConfig())}><RefreshCw size={15} />重置表单</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {notice && <div className="notice success"><span>{notice}</span></div>}
    <div className="backtest-layout">
      <div className="main-column"><Card title="回测参数" subtitle="按股数设置持仓与交易量，初始持股在开始前已持有。">
        <div className="parameter-scroll"><ConfigForm value={config} onChange={setConfig} /></div>
        <div className="form-footer"><span className="form-footnote">运行前会校验数据覆盖、日期区间和仓位约束。</span><div className="button-row"><Button variant="secondary" busy={busy} onClick={() => void saveConfig(true)}><Copy size={15} />复制配置</Button><Button variant="secondary" busy={busy} onClick={() => void saveConfig()}><Save size={15} />保存配置</Button><Button busy={busy} onClick={() => void runOne()}><Play size={15} />运行回测</Button></div></div>
      </Card></div>
      <aside className="side-column"><EquityPreview config={config} /><div className="workspace-support">
        <Card title="已保存配置" subtitle="选择多份配置，可同时独立运行并分别生成结果。" action={<Badge tone="blue">{records.length} 份</Badge>}>
          {loading ? <Loading /> : records.length === 0 ? <Empty title="还没有保存的配置" detail="保存后即可复用或批量比较。" /> : <div className="saved-list">{records.map((record) => <div className="saved-row" key={record.id}><label className="check-label"><input type="checkbox" checked={selected.includes(record.id)} onChange={(e) => setSelected((old) => e.target.checked ? [...old, record.id] : old.filter((id) => id !== record.id))} /><span className="saved-copy"><strong>{record.name}</strong><small>{record.config.symbol} · {record.config.frequency} · {record.config.grid_mode}</small></span></label><div className="saved-actions"><button title="载入" onClick={() => { void loadConfig(record.config).then(value => { setConfig(value.config); setNotice(value.notice) }).catch(e => setError(err(e))) }}><RefreshCw size={14} /></button><button title="复制" onClick={() => { void loadConfig(record.config).then(value => { setConfig(value.config); setNotice('已复制配置草稿。' + value.notice) }).catch(e => setError(err(e))) }}><Copy size={14} /></button><button title="删除" className="danger-icon" onClick={() => removeConfig(record)}><Trash2 size={14} /></button></div></div>)}</div>}
          <div className="saved-footer"><Button variant="secondary" disabled={selected.length < 2 || busy} busy={busy} onClick={() => void runBatch()}><Plus size={15} />批量运行所选（{selected.length}）</Button></div>
        </Card>
        <Card title="任务进度" subtitle={activeJobs.length ? `${activeJobs.length} 个任务正在处理` : '提交后的任务会显示在这里。'} action={<button className="icon-button" onClick={() => { setError(''); void refresh() }} aria-label="刷新任务"><RefreshCw size={15} /></button>}>
          {loading ? <Loading /> : jobs.length === 0 ? <Empty title="暂无回测任务" /> : <div className="job-list">{jobs.map((job) => <div className="job-card" key={job.id}><div className="job-title"><div><strong>{job.kind === 'batch_backtest' ? '批量回测' : '策略回测'}</strong><small>{dateText(job.created_at)} · {job.id.slice(0, 12)}</small></div><Badge tone={job.status === 'completed' ? 'green' : job.status === 'failed' ? 'red' : 'blue'}>{job.status}</Badge></div><JobProgress progress={job.progress} status={job.status} /><p>{job.message || '等待处理'}</p>{job.error != null && <pre className="error-detail">{typeof job.error === 'string' ? job.error : JSON.stringify(job.error, null, 2)}</pre>}<div className="job-bottom">{job.result_id && <button className="text-link" onClick={() => onOpenResult(job.result_id!)}>查看结果</button>}{['queued', 'running'].includes(job.status) && <button className="text-link danger-text" onClick={() => void cancelJob(job.id)}>取消任务</button>}</div></div>)}</div>}
        </Card>
      </div></aside>
    </div>
    {saveDialog && <div className="backtest-dialog-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setSaveDialog(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setSaveDialog(null) }}>
      <section className="backtest-dialog" role="dialog" aria-modal="true" aria-labelledby="save-config-title">
        <h2 id="save-config-title">{saveDialog.copy ? '复制配置' : '保存配置'}</h2>
        <p>为当前回测参数填写一个便于识别的名称。</p>
        <form onSubmit={(event) => { event.preventDefault(); void confirmSaveConfig() }}>
          <label className="backtest-dialog-field"><span>配置名称</span><input autoFocus className="input" value={saveDialog.name} maxLength={80} onChange={(event) => { setSaveDialog({ ...saveDialog, name: event.target.value }); setDialogError('') }} placeholder="例如：浦发银行月度网格" /></label>
          {dialogError && <div className="backtest-dialog-error" role="alert">{dialogError}</div>}
          <div className="backtest-dialog-actions"><Button type="button" variant="quiet" disabled={busy} onClick={() => setSaveDialog(null)}>取消</Button><Button type="submit" disabled={busy}>{busy ? '保存中…' : '确认保存'}</Button></div>
        </form>
      </section>
    </div>}
    {deleteTarget && <div className="backtest-dialog-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setDeleteTarget(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setDeleteTarget(null) }}>
      <section className="backtest-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-config-title">
        <h2 id="delete-config-title">确认删除配置</h2>
        <p>确定删除“<strong>{deleteTarget.name}</strong>”吗？此操作无法撤销。</p>
        {dialogError && <div className="backtest-dialog-error" role="alert">{dialogError}</div>}
        <div className="backtest-dialog-actions"><Button variant="quiet" disabled={busy} onClick={() => setDeleteTarget(null)}>取消</Button><Button variant="danger" disabled={busy} onClick={() => void confirmRemoveConfig()}>{busy ? '删除中…' : '确认删除'}</Button></div>
      </section>
    </div>}
  </>
}
