import { useCallback, useState } from 'react'
import type { ButtonHTMLAttributes, ChangeEvent, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'
import { AlertCircle, Check, ChevronRight, LoaderCircle, X } from 'lucide-react'
import type { BacktestConfig, FeeConfig, QuantityConfig } from './types'

export function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>
}

export function Card({ title, subtitle, action, children, className = '' }: { title?: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>
    {(title || subtitle || action) && <div className="card-head"><div>{title && <h2>{title}</h2>}{subtitle && <p>{subtitle}</p>}</div>{action && <div className="card-action">{action}</div>}</div>}
    {children}
  </section>
}

export function ErrorBox({ message, onClose }: { message: string; onClose?: () => void }) {
  return <div className="notice error"><AlertCircle size={17} /><span>{message}</span>{onClose && <button className="icon-button" onClick={onClose} aria-label="关闭错误"><X size={16} /></button>}</div>
}

export function SuccessBox({ children }: { children: ReactNode }) {
  return <div className="notice success"><Check size={16} /><span>{children}</span></div>
}

export function Loading({ label = '正在读取…' }: { label?: string }) {
  return <div className="loading"><LoaderCircle className="spin" size={18} />{label}</div>
}

export function Empty({ title, detail }: { title: string; detail?: string }) {
  return <div className="empty-state"><div className="empty-mark"><ChevronRight size={18} /></div><strong>{title}</strong>{detail && <span>{detail}</span>}</div>
}

export function Field({ label, hint, children, className = '' }: { label: string; hint?: string; children: ReactNode; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <small>{hint}</small>}</label>
}

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`input ${props.className ?? ''}`} />
}

export function SelectInput(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`input select ${props.className ?? ''}`} />
}

export function Button({ children, variant = 'primary', busy = false, className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'quiet' | 'danger'; busy?: boolean }) {
  return <button {...props} className={`button ${variant} ${className}`} disabled={props.disabled || busy}><span>{busy && <LoaderCircle className="spin" size={15} />}{children}</span></button>
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'blue' | 'green' | 'red' | 'amber' }) {
  return <span className={`badge ${tone}`}>{children}</span>
}

export function Stat({ label, value, note, tone = 'neutral' }: { label: string; value: ReactNode; note?: string; tone?: 'neutral' | 'blue' | 'green' | 'red' | 'amber' }) {
  return <div className={`stat ${tone}`}><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>
}

export function JobProgress({ progress, status }: { progress: number; status: string }) {
  const value = Math.max(0, Math.min(100, Math.round((progress || 0) * 100)))
  return <div className="progress-wrap"><div className="progress"><i style={{ width: `${status === 'completed' ? 100 : value}%` }} /></div><span>{status === 'completed' ? '完成' : `${value}%`}</span></div>
}

export function useErrorMessage() {
  return useCallback((error: unknown) => error instanceof Error ? error.message : String(error), [])
}

export const percent = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`
export const money = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)
export const compact = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat('zh-CN', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
export const dateText = (value?: string | null) => value ? value.slice(0, 10) : '—'

export function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export const today = () => new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString().slice(0, 10)

export function defaultConfig(): BacktestConfig {
  const start = new Date(today() + 'T00:00:00Z')
  start.setUTCFullYear(start.getUTCFullYear() - 3)
  return {
    symbol: '600000.SH', start_date: start.toISOString().slice(0, 10), end_date: null,
    frequency: '1m', initial_cash: 100000, initial_position: 0.5, min_position: 0, max_position: 1,
    grid_mode: 'trailing', buy_step: 0.03, sell_step: 0.03, grid_step: 0.03,
    reference_price: null, lower_price: null, upper_price: null,
    buy_quantity: { mode: 'equity_ratio', value: 0.1 }, sell_quantity: { mode: 'equity_ratio', value: 0.1 },
    fees: { commission_rate: 0.0003, min_commission: 5, slippage_bps: 5, share_face_value: 1, legacy_sh_transfer_rate: 0.001, legacy_sh_min_transfer: 1, legacy_sz_transfer_rate: 0.0000255 }, dca_frequency: 'monthly',
  }
}

function NumField({ label, value, onChange, suffix, percentValue = false, min, max, step = 'any', hint }: { label: string; value: number | null; onChange: (value: number | null) => void; suffix?: string; percentValue?: boolean; min?: number; max?: number; step?: string; hint?: string }) {
  const shown = value == null ? '' : percentValue ? String(Number((value * 100).toPrecision(8))) : String(value)
  const handle = (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.value) return onChange(null)
    const parsed = Number(event.target.value)
    onChange(Number.isFinite(parsed) ? percentValue ? parsed / 100 : parsed : null)
  }
  return <Field label={label} hint={hint}><span className="input-suffix"><TextInput type="number" value={shown} onChange={handle} min={min} max={max} step={step} />{suffix && <i>{suffix}</i>}</span></Field>
}

function QuantityEditor({ label, value, onChange }: { label: string; value: QuantityConfig; onChange: (next: QuantityConfig) => void }) {
  return <div className="quantity-editor"><div className="quantity-title">{label}</div><div className="form-grid two">
    <Field label="计量方式"><SelectInput value={value.mode} onChange={(e) => onChange({ ...value, mode: e.target.value as QuantityConfig['mode'] })}><option value="equity_ratio">净资产比例</option><option value="shares">股数</option><option value="cash">现金金额</option></SelectInput></Field>
    <NumField label={value.mode === 'equity_ratio' ? '下单比例' : value.mode === 'shares' ? '股数' : '现金金额'} value={value.mode === 'equity_ratio' ? value.value : value.value} onChange={(next) => next != null && onChange({ ...value, value: value.mode === 'shares' ? Math.round(next) : next })} percentValue={value.mode === 'equity_ratio'} suffix={value.mode === 'equity_ratio' ? '%' : value.mode === 'shares' ? '股' : '元'} step={value.mode === 'shares' ? '100' : 'any'} min={0} />
  </div></div>
}

export function ConfigForm({ value, onChange }: { value: BacktestConfig; onChange: (next: BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)) => void }) {
  const set = <K extends keyof BacktestConfig>(key: K, next: BacktestConfig[K]) => onChange((previous) => ({ ...previous, [key]: next }))
  const fees: Required<FeeConfig> = { share_face_value: 1, legacy_sh_transfer_rate: 0.001, legacy_sh_min_transfer: 1, legacy_sz_transfer_rate: 0.0000255, ...value.fees }
  const setFees = (next: Partial<FeeConfig>) => onChange((previous) => ({ ...previous, fees: { ...previous.fees, ...next } }))
  return <div className="config-form">
    <div className="form-section"><div className="section-title"><span>01</span><div><strong>标的与区间</strong><small>选择股票、频率和回测起止日期</small></div></div><div className="form-grid three">
      <Field label="股票代码"><TextInput value={value.symbol} onChange={(e) => set('symbol', e.target.value.toUpperCase())} placeholder="600000.SH" /></Field>
      <Field label="数据频率"><SelectInput value={value.frequency} onChange={(e) => set('frequency', e.target.value as BacktestConfig['frequency'])}>{['1m', '5m', '15m', '30m', '60m', '1d'].map((f) => <option key={f} value={f}>{f}</option>)}</SelectInput></Field>
      <Field label="开始日期"><TextInput type="date" value={value.start_date} onInput={(e) => set('start_date', e.currentTarget.value)} onChange={(e) => set('start_date', e.target.value)} /></Field>
      <Field label="结束日期" hint="留空表示截至最新可用数据"><TextInput type="date" value={value.end_date ?? ''} onInput={(e) => set('end_date', e.currentTarget.value || null)} onChange={(e) => set('end_date', e.target.value || null)} /></Field>
      <NumField label="初始资金" value={value.initial_cash} onChange={(v) => v != null && set('initial_cash', v)} suffix="元" min={1} />
      <Field label="定投频率"><SelectInput value={value.dca_frequency} onChange={(e) => set('dca_frequency', e.target.value as BacktestConfig['dca_frequency'])}><option value="weekly">每周</option><option value="monthly">每月</option></SelectInput></Field>
    </div></div>

    <div className="form-section"><div className="section-title"><span>02</span><div><strong>网格规则</strong><small>收益和仓位按比例输入，例如 3 表示 3%</small></div></div><div className="form-grid three">
      <Field label="网格模式"><SelectInput value={value.grid_mode} onChange={(e) => set('grid_mode', e.target.value as BacktestConfig['grid_mode'])}><option value="trailing">移动网格</option><option value="arithmetic">等差网格</option><option value="geometric">等比网格</option></SelectInput></Field>
      <NumField label="买入步长" value={value.buy_step} onChange={(v) => v != null && set('buy_step', v)} percentValue suffix="%" min={0.01} max={100} />
      <NumField label="卖出步长" value={value.sell_step} onChange={(v) => v != null && set('sell_step', v)} percentValue suffix="%" min={0.01} max={100} />
      <NumField label="网格间距" value={value.grid_step} onChange={(v) => v != null && set('grid_step', v)} percentValue suffix="%" min={0.01} max={100} />
      <NumField label="参考价格" value={value.reference_price} onChange={(v) => set('reference_price', v)} suffix="元" min={0} />
      <NumField label="价格下限" value={value.lower_price} onChange={(v) => set('lower_price', v)} suffix="元" min={0} />
      <NumField label="价格上限" value={value.upper_price} onChange={(v) => set('upper_price', v)} suffix="元" min={0} />
    </div></div>

    <div className="form-section"><div className="section-title"><span>03</span><div><strong>仓位与交易量</strong><small>仓位按百分数设置</small></div></div><div className="form-grid three">
      <NumField label="初始仓位" value={value.initial_position} onChange={(v) => v != null && set('initial_position', v)} percentValue suffix="%" min={0} max={100} />
      <NumField label="最小仓位" value={value.min_position} onChange={(v) => v != null && set('min_position', v)} percentValue suffix="%" min={0} max={100} />
      <NumField label="最大仓位" value={value.max_position} onChange={(v) => v != null && set('max_position', v)} percentValue suffix="%" min={0} max={100} />
    </div><div className="quantity-grid"><QuantityEditor label="买入数量" value={value.buy_quantity} onChange={(q) => set('buy_quantity', q)} /><QuantityEditor label="卖出数量" value={value.sell_quantity} onChange={(q) => set('sell_quantity', q)} /></div></div>

    <div className="form-section last"><div className="section-title"><span>04</span><div><strong>费用设置</strong><small>按真实下单成本估算手续费与滑点</small></div></div><div className="form-grid three">
      <NumField label="佣金费率" value={fees.commission_rate} onChange={(v) => v != null && setFees({ commission_rate: v })} percentValue suffix="%" min={0} max={5} />
      <NumField label="最低佣金" value={fees.min_commission} onChange={(v) => v != null && setFees({ min_commission: v })} suffix="元" min={0} />
      <NumField label="滑点" value={fees.slippage_bps} onChange={(v) => v != null && setFees({ slippage_bps: v })} suffix="基点" min={0} max={1000} />
    </div><details className="advanced-disclosure fee-settings"><summary>早期过户费设置</summary><p className="fee-model-note">2012 年 6 月以前的收费沿革不完整，模型使用估算并在结果说明中标注；可根据本人券商资料调整。2012 年 6 月至 2015 年按中国结算收费的可核实部分估算，佣金按券商合约费用计取，不重复计入单列过户费。</p><div className="form-grid three">
      <NumField label="股票面值" value={fees.share_face_value} onChange={(v) => v != null && setFees({ share_face_value: v })} suffix="元/股" min={0.0001} />
      <NumField label="沪市历史过户费率" value={fees.legacy_sh_transfer_rate} onChange={(v) => v != null && setFees({ legacy_sh_transfer_rate: v })} percentValue suffix="%" min={0} max={5} hint="默认 0.1%" />
      <NumField label="沪市历史最低过户费" value={fees.legacy_sh_min_transfer} onChange={(v) => v != null && setFees({ legacy_sh_min_transfer: v })} suffix="元" min={0} />
      <NumField label="深市历史过户费率" value={fees.legacy_sz_transfer_rate} onChange={(v) => v != null && setFees({ legacy_sz_transfer_rate: v })} percentValue suffix="%" min={0} max={5} hint="默认 0.00255%" />
    </div></details></div>
  </div>
}

export function useConfirm() {
  const [value, setValue] = useState<string | null>(null)
  return { confirm: (message: string) => { setValue(message); return new Promise<boolean>((resolve) => { (window as Window & { __confirmResolve?: (value: boolean) => void }).__confirmResolve = resolve }) }, node: value && <div className="modal-scrim"><div className="confirm-modal"><h3>请确认</h3><p>{value}</p><div className="button-row"><Button variant="quiet" onClick={() => { setValue(null); (window as Window & { __confirmResolve?: (v: boolean) => void }).__confirmResolve?.(false) }}>取消</Button><Button variant="danger" onClick={() => { setValue(null); (window as Window & { __confirmResolve?: (v: boolean) => void }).__confirmResolve?.(true) }}>确认删除</Button></div></div></div> }
}
