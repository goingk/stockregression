import { useCallback, useEffect, useState } from 'react'
import { Copy, FlaskConical, Play, Plus, RefreshCw, Save, Trash2 } from 'lucide-react'
import { ApiError, del, get, post } from '../api'
import { Badge, Button, Card, ConfigForm, Empty, ErrorBox, Loading, PageHeader, configError, configPayload, loadConfig, revealConfigError, useErrorMessage } from '../components'
import Disclosure from '../Disclosure'
import EquityPreview from '../EquityPreview'
import { useTasks } from '../TaskCenter'
import { useWorkspaces } from '../WorkbenchState'
import type { BacktestConfig, ConfigRecord, DataPreparationRequest } from '../types'

export default function BacktestPage({ onResearch, onPrepareData }: { onResearch: (config: BacktestConfig) => void; onPrepareData?: (request: DataPreparationRequest) => void }) {
  const workspace = useWorkspaces()
  const { backtest: config, updateBacktest: updateConfig } = workspace
  const tasks = useTasks()
  const [records, setRecords] = useState<ConfigRecord[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [formRevision, setFormRevision] = useState(0)
  const [saveDialog, setSaveDialog] = useState<{ copy: boolean; name: string } | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<ConfigRecord | null>(null)
  const [dialogError, setDialogError] = useState('')
  const [dataIssue, setDataIssue] = useState<DataPreparationRequest | null>(null)
  const err = useErrorMessage()
  const refresh = useCallback(async () => {
    try { setRecords(await get<ConfigRecord[]>('/configs')) }
    catch (cause) { setError(err(cause)) } finally { setLoading(false) }
  }, [err])
  useEffect(() => { void refresh() }, [refresh])
  const validate = () => {
    const message = config ? configError(config) : '参数尚未载入'
    if (message) { setError(message); revealConfigError(message); return false }
    return true
  }
  const saveConfig = (copy = false) => {
    if (!config || !validate()) return
    setError(''); setNotice(''); setDialogError('')
    setSaveDialog({ copy, name: copy ? `${config.symbol} 配置副本` : `${config.symbol} 网格配置` })
  }
  const confirmSaveConfig = async () => {
    const name = saveDialog?.name.trim()
    if (!saveDialog || !name || !config) { setDialogError('请填写配置名称。'); return }
    setBusy(true); setDialogError('')
    try {
      const response = await post<{ id: string }>('/configs', { name, config: configPayload(config) })
      setRecords(list => [{ id: response.id, name, config: structuredClone(config) }, ...list])
      setNotice(`配置“${name}”已保存。`); setSaveDialog(null)
    } catch (cause) { setDialogError(err(cause)) } finally { setBusy(false) }
  }
  const runOne = async () => {
    if (!config || !validate()) return
    setBusy(true); setError(''); setNotice(''); setDataIssue(null)
    try { await tasks.run('策略回测', () => post<{ job_id: string }>('/backtests', configPayload(config))) }
    catch (cause) {
      setError(err(cause))
      if (cause instanceof ApiError && cause.status === 409 && cause.detail && typeof cause.detail === 'object' && (cause.detail as Record<string, unknown>).code === 'DATA_NOT_READY') setDataIssue(cause.detail as DataPreparationRequest)
    } finally { setBusy(false) }
  }
  const runBatch = async () => {
    const chosen = records.filter(item => selected.includes(item.id))
    if (chosen.length < 2) { setError('批量回测至少选择两份已保存配置。'); return }
    setBusy(true); setError('')
    try {
      await tasks.run('批量回测', async () => {
        const configs = chosen.map(item => configPayload(loadConfig(item.config).config))
        return post<{ job_ids: string[] }>('/backtests/batch', { configs })
      })
    } catch (cause) { setError(err(cause)) } finally { setBusy(false) }
  }
  const loadExplicitConfig = (value: BacktestConfig, copied = false) => {
    const loaded = loadConfig(value)
    updateConfig(structuredClone(loaded.config)); setFormRevision(current => current + 1)
    setNotice((copied ? '已复制配置草稿。' : '已载入配置。') + loaded.notice)
  }
  const removeConfig = (record: ConfigRecord) => { setDialogError(''); setDeleteTarget(record) }
  const confirmRemoveConfig = async () => {
    if (!deleteTarget) return
    setBusy(true); setDialogError('')
    try {
      await del(`/configs/${encodeURIComponent(deleteTarget.id)}`)
      setRecords(list => list.filter(item => item.id !== deleteTarget.id))
      setSelected(list => list.filter(id => id !== deleteTarget.id))
      setNotice(`配置“${deleteTarget.name}”已删除。`); setDeleteTarget(null)
    } catch (cause) { setDialogError(err(cause)) } finally { setBusy(false) }
  }
  const acceptPreviewFunding = useCallback((funding: NonNullable<import('../types').BacktestResult['funding']>, context: { symbol: string; start_date: string }) => {
    const reference = funding.reference_price > 0 ? funding.reference_price : funding.valuation_price > 0 ? funding.valuation_price : null
    if (reference == null) return
    updateConfig(previous => previous.symbol === context.symbol && previous.start_date === context.start_date && previous.reference_price_mode === 'auto' && previous.reference_price !== reference ? { ...previous, reference_price: reference } : previous)
  }, [updateConfig])
  if (!config) return <Loading label="读取本机参数设置…" />
  return <div className="backtest-page">
    <PageHeader eyebrow="STRATEGY BACKTEST" title="策略回测" description="调整网格与账户，实时比较网格、全仓持有和定投的收益率。" action={<Button variant="secondary" onClick={() => { workspace.restore('backtest'); setFormRevision(current => current + 1); setNotice('已恢复上次有效参数，标的已清空。') }}><RefreshCw size={15} />恢复上次参数</Button>} />
    {error && <ErrorBox message={error} onClose={() => setError('')} />}
    {dataIssue && <div className="notice"><span>本次区间缺少本地数据。可在数据管理中显式准备该标的和周期；系统不会在后台自动下载。</span><Button variant="secondary" onClick={() => onPrepareData?.(dataIssue)}>去准备数据</Button></div>}
    {workspace.error && <ErrorBox message={workspace.error} />}
    {notice && <div className="notice success"><span>{notice}</span><button className="text-link" onClick={() => setNotice('')}>收起</button></div>}
    {config.selection_asof && <div className="selection-context"><Badge tone="amber">研究参数</Badge><span>选参截止 {config.selection_asof}{config.start_date < config.selection_asof ? ' · 当前区间为样本内历史回看' : ''}</span></div>}
    <div className="workbench-layout">
      <section className="workbench-config-column"><Card title="策略配置" subtitle="常用参数直接调整，其他设置可展开。">
        <div className="parameter-scroll"><ConfigForm key={formRevision} value={config} onChange={updateConfig} onFeeDefaultsEdit={workspace.rememberFees} onDefaultsEdit={workspace.rememberStrategy} /></div>
        <div className="form-footer"><Button variant="secondary" busy={busy} onClick={() => saveConfig(true)}><Copy size={14} />复制</Button><Button variant="secondary" busy={busy} onClick={() => saveConfig()}><Save size={14} />保存配置</Button><Button busy={busy} onClick={() => void runOne()}><Play size={14} />运行回测</Button></div>
      </Card></section>
      <section className="workbench-preview-column"><EquityPreview config={config} onFunding={acceptPreviewFunding} /><div className="workbench-run-actions"><span>有效修改后自动重算预览</span><Button variant="secondary" disabled={busy} onClick={() => { if (validate()) onResearch(config) }}><FlaskConical size={15} />用当前配置研究</Button></div></section>
    </div>
    <Disclosure id="backtest-saved-configs" title="已保存配置与批量回测" summary={`${records.length} 份配置`}>
      {loading ? <Loading /> : !records.length ? <Empty title="还没有保存的配置" detail="保存当前参数后即可复用或批量比较。" /> : <div className="saved-list">{records.map(record => <div className="saved-row" key={record.id}><label className="check-label"><input type="checkbox" checked={selected.includes(record.id)} onChange={event => setSelected(old => event.target.checked ? [...old, record.id] : old.filter(id => id !== record.id))} /><span className="saved-copy"><strong>{record.name}</strong><small>{record.config.symbol} · {record.config.frequency} · {record.config.grid_mode}</small></span></label><div className="saved-actions"><button title="载入" onClick={() => { try { loadExplicitConfig(record.config) } catch (cause) { setError(err(cause)) } }}><RefreshCw size={14} /></button><button title="复制" onClick={() => { try { loadExplicitConfig(record.config, true) } catch (cause) { setError(err(cause)) } }}><Copy size={14} /></button><button title="删除" className="danger-icon" onClick={() => removeConfig(record)}><Trash2 size={14} /></button></div></div>)}</div>}
      <div className="saved-footer"><Button variant="secondary" disabled={selected.length < 2 || busy} busy={busy} onClick={() => void runBatch()}><Plus size={15} />批量运行所选（{selected.length}）</Button></div>
    </Disclosure>
    {saveDialog && <div className='backtest-dialog-overlay' role='presentation' onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setSaveDialog(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setSaveDialog(null) }}>
      <section className='backtest-dialog' role='dialog' aria-modal='true' aria-labelledby='save-config-title'>
        <h2 id='save-config-title'>{saveDialog.copy ? '复制配置' : '保存配置'}</h2><p>为当前回测参数填写一个便于识别的名称。</p>
        <form onSubmit={(event) => { event.preventDefault(); void confirmSaveConfig() }}><label className='backtest-dialog-field'><span>配置名称</span><input autoFocus className='input' value={saveDialog.name} maxLength={80} onChange={(event) => { setSaveDialog({ ...saveDialog, name: event.target.value }); setDialogError('') }} placeholder='例如：浦发银行月度网格' /></label>{dialogError && <div className='backtest-dialog-error' role='alert'>{dialogError}</div>}<div className='backtest-dialog-actions'><Button type='button' variant='quiet' disabled={busy} onClick={() => setSaveDialog(null)}>取消</Button><Button type='submit' disabled={busy}>{busy ? '保存中…' : '确认保存'}</Button></div></form>
      </section>
    </div>}
    {deleteTarget && <div className='backtest-dialog-overlay' role='presentation' onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setDeleteTarget(null) }} onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setDeleteTarget(null) }}>
      <section className='backtest-dialog' role='dialog' aria-modal='true' aria-labelledby='delete-config-title'><h2 id='delete-config-title'>确认删除配置</h2><p>确定删除“<strong>{deleteTarget.name}</strong>”吗？此操作无法撤销。</p>{dialogError && <div className='backtest-dialog-error' role='alert'>{dialogError}</div>}<div className='backtest-dialog-actions'><Button variant='quiet' disabled={busy} onClick={() => setDeleteTarget(null)}>取消</Button><Button variant='danger' disabled={busy} onClick={() => void confirmRemoveConfig()}>{busy ? '删除中…' : '确认删除'}</Button></div></section>
    </div>}
  </div>
}
