import { useEffect, useRef, useState } from 'react'
import { del, get, post } from './api'
import EquityCurve from './EquityCurve'
import { Badge, Card, Empty, configError, configPayload, money, percent } from './components'
import type { BacktestConfig, BacktestResult } from './types'

type Preview = { revision: number; status: string; progress?: number; message?: string; error?: string; result?: BacktestResult }

export default function EquityPreview({ config, onFunding }: { config: BacktestConfig; onFunding?: (funding: NonNullable<BacktestResult['funding']>, context: { symbol: string; start_date: string }) => void }) {
  const session = useRef(`preview_${Math.random().toString(36).slice(2)}`)
  const revision = useRef(0)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [status, setStatus] = useState('等待参数')
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')
  const payload = JSON.stringify(configPayload(config))
  const shareUnit = /^[15]/.test(config.symbol) ? '份' : '股'
  const quality = result?.quality ?? {}
  const minuteRatio = Number(quality.minute_ratio)
  const fallbackDays = Number(quality.daily_days)
  useEffect(() => {
    const current = ++revision.current
    let active = true
    let poll: number | undefined
    const validation = configError(config)
    if (!config.symbol) {
      setResult(null); setError(''); setStatus('等待选择标的'); setProgress(0); setMessage('')
      void del(`/backtests/preview/${session.current}`).catch(() => undefined)
      return () => { active = false }
    }
    setError(validation); setStatus(validation ? '参数待完善' : '等待计算'); setProgress(0); setMessage('')
    const accept = (value: Preview) => {
      if (!active || current !== revision.current || value.revision !== current) return
      if (value.status === 'completed' && value.result) {
        setResult(value.result); setError(''); setStatus('已更新'); setProgress(1); setMessage(value.message ?? '')
        if (value.result.funding) onFunding?.(value.result.funding, { symbol: config.symbol, start_date: config.start_date })
        return
      }
      if (value.status === 'failed') { setError(value.error ?? '预览计算失败'); setStatus('计算失败'); setProgress(0); setMessage(value.message ?? ''); return }
      setStatus(value.status === 'queued' ? '排队中' : '计算中'); setProgress(Math.max(0, Math.min(1, value.progress ?? 0))); setMessage(value.message ?? '')
      poll = window.setTimeout(() => { get<Preview>(`/backtests/preview/${session.current}`).then(accept).catch(fail) }, 400)
    }
    const fail = (value: unknown) => { if (active) { setStatus('无法预览'); setError(value instanceof Error ? value.message : String(value)) } }
    const timer = validation ? undefined : window.setTimeout(() => {
      setStatus('计算中')
      post<Preview>('/backtests/preview', { session_id: session.current, revision: current, config: JSON.parse(payload) }).then(accept).catch(fail)
    }, 600)
    return () => { active = false; clearTimeout(timer); clearTimeout(poll) }
  }, [payload, onFunding])
  useEffect(() => () => { void del(`/backtests/preview/${session.current}`).catch(() => undefined) }, [])
  return <Card title="收益率曲线" subtitle="资产 ÷ 累计投入 − 1 · 修改参数后自动重算" className="preview-card" action={<Badge tone={status === '已更新' ? 'green' : 'amber'}>{status}</Badge>}>
    {result && <div className="preview-quality">{result.config.frequency === '1d' ? '分钟覆盖不适用' : `分钟覆盖 ${Number.isFinite(minuteRatio) ? `${(minuteRatio * 100).toFixed(1)}%` : '—'}`}<span>·</span>日线降级 {result.config.frequency === '1d' ? 0 : Number.isFinite(fallbackDays) ? Math.max(0, fallbackDays) : '—'} 天</div>}
    {result?.funding && <div className="preview-funding"><span>网格预留现金 <strong>{money(result.funding.reserve_cash)} 元</strong></span><span>网格起始资产 <strong>{money(result.funding.initial_equity)} 元</strong></span><span>估值日 {result.funding.reference_date}</span><span>实际截至 {result.effective_end_date}</span><span>期初持有：网格 {result.config.initial_shares} {shareUnit} · 全仓 {result.funding.buy_hold_initial_shares ?? 1000} {shareUnit} · 定投 {result.config.dca.initial_shares} {shareUnit}</span></div>}
    {error && <p className="inline-warning" role="status">{error}</p>}
    {message && <div className="preview-progress"><span>{message}</span><strong>{Math.round(progress * 100)}%</strong><i><b style={{ width: `${Math.round(progress * 100)}%` }} /></i></div>}
    <div className={result && status !== '已更新' ? 'preview-stale' : ''}>{result ? <EquityCurve result={result} height={325} /> : <Empty title={config.symbol ? '填写参数后显示收益率预览' : '请选择回测标的'} detail={config.symbol ? '预览会自动读取并补充可用行情；暂时没有结果时请稍等片刻。' : '点击标的输入框查看最近提交的标的，也可以输入代码或名称搜索。'} />}</div>
    {result && <><div className="preview-metrics"><span>网格收益 <strong>{percent(result.strategies.grid.metrics.total_return)}</strong></span><span>最大回撤 <strong>{percent(result.strategies.grid.metrics.max_drawdown)}</strong></span><span>成交 <strong>{result.strategies.grid.metrics.trade_count ?? 0} 笔</strong></span><span>定投收益 <strong>{percent(result.strategies.dca.metrics.total_return)}</strong></span></div>{Boolean(result.warnings?.length) && <details className="preview-warnings"><summary>数据说明与警告（{result.warnings!.length}）</summary>{result.warnings!.map((warning, index) => <p key={index}>{warning}</p>)}</details>}</>}
  </Card>
}
