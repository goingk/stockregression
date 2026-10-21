import { useEffect, useRef, useState } from 'react'
import { del, get, post } from './api'
import EquityCurve from './EquityCurve'
import DataDiagnosticsPanel from './DataDiagnosticsPanel'
import Disclosure from './Disclosure'
import { Badge, Card, Empty, configError, configPayload, money, percent } from './components'
import type { BacktestConfig, BacktestResult } from './types'
import { useWorkbenchDefaults } from './useWorkbenchDefaults'

type Preview = { revision: number; status: string; progress?: number; message?: string; error?: string; result?: BacktestResult }

export default function EquityPreview({ config, onFunding }: { config: BacktestConfig; onFunding?: (funding: NonNullable<BacktestResult['funding']>, context: { symbol: string; start_date: string }) => void }) {
  const configuration = useWorkbenchDefaults()
  const ui = (configuration.document.ui ?? {}) as Record<string, unknown>
  const previewDebounceMs = Math.max(50, Number(ui.preview_debounce_ms) || 300)
  const pollingMs = Math.max(500, Number(ui.polling_interval_ms) || 2000)
  const session = useRef(`preview_${Math.random().toString(36).slice(2)}`)
  const revision = useRef(0)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [status, setStatus] = useState('等待参数')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const payload = JSON.stringify(configPayload(config))
  const shareUnit = /^[15]/.test(config.symbol) ? '份' : '股'
  const quality = result?.quality ?? {}
  const minuteRatio = Number(result?.diagnostics?.minute_ratio ?? quality.minute_ratio)
  const fallbackDays = Number(result?.diagnostics?.daily_days ?? quality.daily_days)
  const pendingCalibrationDays = Number(result?.diagnostics?.pending_calibration_days ?? quality.pending_calibration_days ?? 0)
  useEffect(() => {
    const current = ++revision.current
    let active = true
    let poll: number | undefined
    const validation = configError(config)
    if (!config.symbol) {
      setResult(null); setError(''); setStatus('等待选择标的'); setMessage('')
      void del(`/backtests/preview/${session.current}`).catch(() => undefined)
      return () => { active = false }
    }
    setError(validation); setStatus(validation ? '参数待完善' : '等待计算'); setMessage('')
    const accept = (value: Preview) => {
      if (!active || current !== revision.current || value.revision !== current) return
      if (value.status === 'completed' && value.result) {
        setResult(value.result); setError(''); setStatus('已更新'); setMessage(value.message ?? '')
        if (value.result.funding) onFunding?.(value.result.funding, { symbol: config.symbol, start_date: config.start_date })
        return
      }
      if (value.status === 'failed') { setError(value.error ?? '预览计算失败'); setStatus('计算失败'); setMessage(value.message ?? ''); return }
      setStatus(value.status === 'queued' ? '排队中' : '计算中'); setMessage(value.message ?? '')
      poll = window.setTimeout(() => { get<Preview>(`/backtests/preview/${session.current}`).then(accept).catch(fail) }, pollingMs)
    }
    const fail = (value: unknown) => { if (active) { setStatus('无法预览'); setError(value instanceof Error ? value.message : String(value)) } }
    const timer = validation ? undefined : window.setTimeout(() => {
      setStatus('计算中')
      post<Preview>('/backtests/preview', { session_id: session.current, revision: current, config: JSON.parse(payload) }).then(accept).catch(fail)
    }, previewDebounceMs)
    return () => { active = false; clearTimeout(timer); clearTimeout(poll) }
  }, [payload, onFunding, pollingMs, previewDebounceMs])
  useEffect(() => () => { void del(`/backtests/preview/${session.current}`).catch(() => undefined) }, [])
  return <Card title="收益率预览" subtitle="峰值本金收益 · 修改参数后自动重算" className="preview-card" action={<span title={message}><Badge tone={status === '已更新' ? 'green' : 'amber'}>{status}</Badge></span>}>
    {result && <div className="preview-summary-strip"><span>{result.config.frequency === '1d' ? '日线模式' : `分钟覆盖 ${Number.isFinite(minuteRatio) ? `${(minuteRatio * 100).toFixed(1)}%` : '—'}`} · 日线降级 {result.config.frequency === '1d' ? 0 : Number.isFinite(fallbackDays) ? Math.max(0, fallbackDays) : '—'} 天</span>{result.funding && <span>峰值本金 <strong>{money(result.funding.peak_invested ?? result.funding.initial_equity)}</strong> 元 · 外部补款 <strong>{money(result.funding.external_topups ?? result.funding.topups)}</strong> 元 · 起始资产 <strong>{money(result.funding.initial_equity)}</strong> 元</span>}</div>}
    {result && result.config.frequency !== '1d' && pendingCalibrationDays > 0 && <p className="inline-warning" role="status">{pendingCalibrationDays} 个交易日的分钟数据待校准，当前采用日线；展开账户与数据详情查看。</p>}
    {error && <p className="inline-warning" role="status">{error}</p>}
    <div className={`preview-plot ${result && status !== '已更新' ? 'preview-stale' : ''}`}>{result ? <EquityCurve result={result} height={370} stretch /> : <Empty title={config.symbol ? '填写参数后显示收益率预览' : '请选择回测标的'} detail={config.symbol ? '预览会自动读取并补充可用行情。' : '点击标的输入框查看最近提交的标的，或输入代码和名称搜索。'} />}</div>
    {result && <div className="preview-metrics"><span>峰值本金收益 <strong>{percent(result.strategies.grid.metrics.peak_principal_return)}</strong></span><span>最大本金回撤 <strong>{percent(result.strategies.grid.metrics.max_principal_drawdown)}</strong></span><span>同资金被动持有 <strong>{percent(result.strategies.grid.metrics.matched_funding_hold_return)}</strong></span><span>成交 <strong>{result.strategies.grid.metrics.trade_count ?? 0} 笔</strong></span></div>}
    {result && <Disclosure id="backtest-preview-details" title="账户与数据详情" summary={`实际截至 ${result.effective_end_date ?? '—'}`}>
      {result.funding && <div className="preview-funding"><span>估值日 {result.funding.reference_date}</span><span>期初持有：网格 {result.config.initial_shares} {shareUnit} · 全仓 {result.funding.buy_hold_initial_shares ?? 1000} {shareUnit} · 定投 {result.config.dca.initial_shares} {shareUnit}</span></div>}
      <DataDiagnosticsPanel diagnostics={result.diagnostics} sourceEvents={result.source_events} warnings={result.warnings} syntheticBars={quality.synthetic_bars} />
    </Disclosure>}
  </Card>
}
