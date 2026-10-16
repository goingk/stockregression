import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { get, post } from './api'
import type { ButtonHTMLAttributes, ChangeEvent, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'
import { AlertCircle, Check, ChevronRight, LoaderCircle, X } from 'lucide-react'
import type { BacktestConfig, FeeConfig, Instrument } from './types'
import { strategyDefaults, validWorkbenchGroup } from './workbenchDefaults'
import type { StrategyDefaultsGroup } from './workbenchDefaults'

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

const FEE_DEFAULTS_KEY = 'stockregression-fee-defaults-v1'
const FEE_DEFAULTS_PENDING_KEY = 'stockregression-fee-defaults-pending-v1'
export type FeeDefaults = Pick<FeeConfig, 'commission_rate' | 'min_commission'>
const baseFeeDefaults: FeeDefaults = { commission_rate: .0005, min_commission: .1 }

export function validFeeDefaults(value: unknown): value is FeeDefaults {
  if (!value || typeof value !== 'object') return false
  const item = value as Record<string, unknown>
  return typeof item.commission_rate === 'number' && Number.isFinite(item.commission_rate) && item.commission_rate >= 0 && item.commission_rate <= .05
    && typeof item.min_commission === 'number' && Number.isFinite(item.min_commission) && item.min_commission >= 0
}

export function readFeeDefaults(): FeeDefaults {
  try {
    const raw = localStorage.getItem(FEE_DEFAULTS_KEY)
    if (raw) {
      const value: unknown = JSON.parse(raw)
      if (validFeeDefaults(value)) return value
    }
  } catch { /* local preferences are optional */ }
  return baseFeeDefaults
}

export function writeFeeDefaults(value: FeeDefaults) {
  try { localStorage.setItem(FEE_DEFAULTS_KEY, JSON.stringify(value)) } catch { /* settings API remains the durable copy */ }
}

export function readPendingFeeDefaults(): FeeDefaults | null {
  try {
    if (localStorage.getItem(FEE_DEFAULTS_PENDING_KEY) !== '1') return null
    const raw = localStorage.getItem(FEE_DEFAULTS_KEY)
    if (!raw) return null
    const value: unknown = JSON.parse(raw)
    return validFeeDefaults(value) ? value : null
  } catch { return null }
}

export function markFeeDefaultsPending() {
  try { localStorage.setItem(FEE_DEFAULTS_PENDING_KEY, '1') } catch { /* settings API remains the durable copy */ }
}

export function clearPendingFeeDefaults() {
  try { localStorage.removeItem(FEE_DEFAULTS_PENDING_KEY) } catch { /* local preferences are optional */ }
}

export function defaultConfig(feeDefaults: FeeDefaults = readFeeDefaults()): BacktestConfig {
  const current = today()
  const year = Number(current.slice(0, 4)) - 1
  const month = Number(current.slice(5, 7))
  const day = Math.min(Number(current.slice(8, 10)), new Date(Date.UTC(year, month, 0)).getUTCDate())
  return {
    schema_version: 2, symbol: '600000.SH', start_date: `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`, end_date: current,
    frequency: '1m', initial_shares: 1000, min_shares: 0, max_shares: 2000, buy_shares: 100, sell_shares: 100,
    grid_mode: 'trailing', buy_step: .03, sell_step: .03, grid_step: .03,
    reference_price: null, reference_price_mode: 'auto', lower_price: null, upper_price: null,
    fees: { ...feeDefaults, slippage_bps: 5 }, dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
  }
}

export function configPayload(value: BacktestConfig) {
  const result: Record<string, unknown> = { ...value }
  for (const key of ['initial_cash', 'initial_position', 'min_position', 'max_position', 'buy_quantity', 'sell_quantity', 'dca_frequency']) delete result[key]
  for (const key of value.grid_mode === 'trailing' ? ['grid_step'] : ['buy_step', 'sell_step']) delete result[key]
  return result
}

export async function loadConfig(value: BacktestConfig): Promise<{ config: BacktestConfig; notice: string }> {
  const result = await post<{ config: BacktestConfig; notice: string }>('/configs/convert', value)
  const defaults = defaultConfig()
  return { ...result, config: {
    ...defaults,
    ...result.config,
    fees: { ...defaults.fees, ...result.config.fees },
    dca: { ...defaults.dca, ...result.config.dca },
  } }
}

export function configError(value: BacktestConfig): string {
  if (!/^(?:[56]\d{5}\.SH|[013]\d{5}\.SZ)$/.test(value.symbol)) return '请选择有效的沪深股票或 ETF'
  if (!value.start_date || !value.end_date || value.start_date > value.end_date) return '请填写有效起止日期'
  const isStarBoard = value.symbol.startsWith('688') || value.symbol.startsWith('689')
  const lotSize = 100
  for (const name of ['initial_shares', 'min_shares', 'max_shares', 'buy_shares', 'sell_shares'] as const) {
    const minShares = name === 'buy_shares' ? isStarBoard ? 200 : lotSize : name === 'sell_shares' ? lotSize : 0
    if (!Number.isInteger(value[name]) || value[name] % lotSize !== 0 || value[name] < minShares) return `持股和交易数量应以 ${lotSize} 为单位${name === 'buy_shares' || name === 'sell_shares' ? `，该交易量至少 ${minShares}` : ''}`
  }
  if (value.min_shares > value.initial_shares || value.initial_shares > value.max_shares) return '最低持股 ≤ 初始持股 ≤ 最高持股'
  const minDcaShares = isStarBoard ? 200 : lotSize
  if (!Number.isInteger(value.dca.shares) || value.dca.shares < minDcaShares || value.dca.shares % lotSize !== 0) return `定投数量应为 ${lotSize} 的正整数倍，至少 ${minDcaShares}`
  if (!Number.isInteger(value.dca.initial_shares) || value.dca.initial_shares < 0 || value.dca.initial_shares % lotSize !== 0) return `定投初始持仓应为 ${lotSize} 的整数倍`
  if (!validWorkbenchGroup('grid', strategyDefaults(value).grid)) return '请填写有效网格步长，价格下限必须小于上限'
  if (!validFeeDefaults(value.fees) || !Number.isFinite(value.fees.slippage_bps) || value.fees.slippage_bps < 0 || value.fees.slippage_bps > 1000) return '请填写有效的费用与滑点'
  return ''
}

export function NumField({ label, value, onChange, suffix, percentValue = false, min, max, step = 'any', hint }: { label: string; value: number | null; onChange: (value: number | null) => void; suffix?: string; percentValue?: boolean; min?: number; max?: number; step?: string; hint?: string }) {
  const format = (next: number | null) => next == null ? '' : String(percentValue ? Number((next * 100).toPrecision(10)) : next)
  const [draft, setDraft] = useState(() => format(value))
  const focused = useRef(false)
  useEffect(() => { if (!focused.current) setDraft(format(value)) }, [value, percentValue])
  const handle = (event: ChangeEvent<HTMLInputElement>) => {
    const text = event.target.value
    setDraft(text)
    if (!text) { onChange(null); return }
    if (/^-?\d+(?:\.\d+)?$/.test(text)) {
      const parsed = Number(text)
      if (Number.isFinite(parsed)) onChange(percentValue ? parsed / 100 : parsed)
    }
  }
  return <Field label={label} hint={hint}><span className="input-suffix"><TextInput type="number" value={draft} onChange={handle} onFocus={() => { focused.current = true }} onBlur={() => { focused.current = false; setDraft(format(value)) }} min={min} max={max} step={step} />{suffix && <i>{suffix}</i>}</span></Field>
}

export function SymbolPicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [query, setQuery] = useState(value)
  const [options, setOptions] = useState<Instrument[]>([])
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'history' | 'search'>('history')
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const emitted = useRef<string | null>(null)
  const optionsId = useId()
  useEffect(() => {
    if (emitted.current === value) { emitted.current = null; return }
    setQuery(value)
  }, [value])
  useEffect(() => {
    if (!open) return
    let active = true
    setOptions([]); setHighlight(0); setLoading(true); setFailed(false)
    const path = mode === 'history' ? '/workbench/recent-symbols' : `/instruments?q=${encodeURIComponent(query)}&limit=12`
    const timer = window.setTimeout(() => {
      get<Instrument[]>(path).then(list => { if (active) setOptions(list) }).catch(() => { if (active) setFailed(true) }).finally(() => { if (active) setLoading(false) })
    }, mode === 'history' ? 0 : 250)
    return () => { active = false; clearTimeout(timer) }
  }, [query, mode, open])
  const emit = (symbol: string) => { emitted.current = symbol; onChange(symbol) }
  const choose = (item: Instrument) => { emit(item.symbol); setQuery(item.symbol); setOpen(false) }
  return <div className="symbol-picker"><TextInput value={query} role="combobox" aria-label="股票 / ETF" aria-expanded={open} aria-controls={optionsId} aria-activedescendant={open && options[highlight] ? optionsId + '-' + highlight : undefined} autoComplete="off" placeholder="选择最近标的，或输入代码 / 名称" onFocus={() => { setMode('history'); setOpen(true) }} onBlur={() => window.setTimeout(() => setOpen(false), 150)} onKeyDown={event => {
    if (event.key === 'Escape') { setOpen(false); return }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setOpen(true); setHighlight(current => Math.max(0, Math.min(options.length - 1, current + (event.key === 'ArrowDown' ? 1 : -1)))) }
    if (event.key === 'Enter' && open && options[highlight]) { event.preventDefault(); choose(options[highlight]) }
  }} onChange={event => {
    const next = event.target.value
    setQuery(next); setMode(next.trim() ? 'search' : 'history'); setOpen(true)
    const symbol = /^\d{6}(?:\.(?:SH|SZ))?$/i.test(next) ? next.includes('.') ? next.toUpperCase() : next + (['5', '6'].includes(next[0]) ? '.SH' : '.SZ') : ''
    if (symbol !== value) emit(symbol)
  }} />{open && <div className="symbol-options" id={optionsId} role="listbox" aria-label={mode === 'history' ? '最近提交的标的' : '标的搜索结果'}><div className="symbol-options-caption">{mode === 'history' ? '最近提交的标的' : '搜索结果'}</div>{loading ? <div className="symbol-options-message">正在读取…</div> : failed ? <div className="symbol-options-message">{mode === 'history' ? '历史暂不可用，可输入代码或名称搜索' : '搜索暂不可用，请稍后重试'}</div> : options.length ? options.map((item, index) => <button key={item.symbol} id={optionsId + '-' + index} role="option" aria-selected={highlight === index} type="button" onMouseEnter={() => setHighlight(index)} onMouseDown={event => event.preventDefault()} onClick={() => choose(item)}><strong>{item.name || item.symbol}</strong><small>{item.symbol} · {item.asset_type === 'etf' ? 'ETF' : '股票'}</small></button>) : <div className="symbol-options-message">{mode === 'history' ? '暂无回测或研究历史，可输入代码或名称搜索' : '没有找到匹配标的'}</div>}</div>}</div>
}

export function ConfigForm({ value, onChange, onFeeDefaultsEdit, onDefaultsEdit }: { value: BacktestConfig; onChange: (next: BacktestConfig | ((previous: BacktestConfig) => BacktestConfig)) => void; onFeeDefaultsEdit?: (next: FeeDefaults) => void; onDefaultsEdit?: (group: StrategyDefaultsGroup, next: BacktestConfig) => void }) {
  const [reference, setReference] = useState<{ reference_date: string; reference_price: number } | null>(null)
  const [referenceError, setReferenceError] = useState('')
  const set = <K extends keyof BacktestConfig>(key: K, next: BacktestConfig[K]) => {
    onChange(previous => ({ ...previous, [key]: next }))
    const group = key === 'dca' ? 'dca' : ['grid_mode', 'buy_step', 'sell_step', 'grid_step', 'lower_price', 'upper_price'].includes(key) ? 'grid' : ['initial_shares', 'min_shares', 'max_shares', 'buy_shares', 'sell_shares'].includes(key) ? 'position' : null
    if (group) onDefaultsEdit?.(group, { ...value, [key]: next })
  }
  const unit = /^[15]/.test(value.symbol) ? '份' : '股'
  const fees: Required<FeeConfig> = { share_face_value: 1, legacy_sh_transfer_rate: .001, legacy_sh_min_transfer: 1, legacy_sz_transfer_rate: .0000255, ...value.fees }
  const setFees = (next: Partial<FeeConfig>) => {
    onChange(previous => ({ ...previous, fees: { ...previous.fees, ...next } }))
    if (onFeeDefaultsEdit && ('commission_rate' in next || 'min_commission' in next)) {
      const preferences = { commission_rate: next.commission_rate ?? value.fees.commission_rate, min_commission: next.min_commission ?? value.fees.min_commission }
      if (validFeeDefaults(preferences)) onFeeDefaultsEdit(preferences)
    }
  }
  useEffect(() => {
    let active = true
    setReference(null); setReferenceError('')
    if (!value.symbol || !value.start_date) return
    get<{ reference_date: string; reference_price: number }>(`/data/reference-price/${encodeURIComponent(value.symbol)}?start_date=${value.start_date}`).then(context => {
      if (!active) return
      setReference(context)
      onChange(previous => previous.reference_price_mode === 'auto' ? { ...previous, reference_price: context.reference_price } : previous)
    }).catch(error => { if (active) setReferenceError(error instanceof Error ? error.message : String(error)) })
    return () => { active = false }
  }, [value.symbol, value.start_date, onChange])
  const isStarBoard = value.symbol.startsWith('688') || value.symbol.startsWith('689')
  const share = (key: 'initial_shares' | 'min_shares' | 'max_shares' | 'buy_shares' | 'sell_shares', label: string) => <NumField key={key} label={label} value={value[key]} onChange={next => next != null && set(key, next)} suffix={unit} min={key === 'buy_shares' ? isStarBoard ? 200 : 100 : key === 'sell_shares' ? 100 : 0} step="100" />
  return <div className="config-form share-config-form">
    <div className="form-section"><div className="section-title"><span>01</span><strong>标的与区间</strong></div><div className="form-grid two">
      <Field label="股票 / ETF" className="full-width"><SymbolPicker value={value.symbol} onChange={symbol => onChange(previous => ({ ...previous, symbol, reference_price_mode: 'auto', reference_price: null }))} /></Field>
      <Field label="开始日期"><TextInput type="date" value={value.start_date} onChange={event => onChange(previous => ({ ...previous, start_date: event.target.value, reference_price_mode: 'auto', reference_price: null }))} /></Field>
      <Field label="结束日期"><TextInput type="date" value={value.end_date ?? ''} onChange={event => set('end_date', event.target.value || null)} /></Field>
      <Field label="数据频率"><SelectInput value={value.frequency} onChange={event => set('frequency', event.target.value as BacktestConfig['frequency'])}>{['1m', '5m', '15m', '30m', '60m', '1d'].map(item => <option key={item}>{item}</option>)}</SelectInput></Field>
      <NumField label="参考价格" value={value.reference_price} onChange={next => onChange(previous => ({ ...previous, reference_price: next, reference_price_mode: next == null ? 'auto' : 'manual' }))} suffix="元" min={.001} hint={reference ? `${reference.reference_date} 收盘 ${reference.reference_price} 元 · ${value.reference_price_mode === 'auto' ? '自动' : '手动'}` : '开始日期前最近交易日收盘价'} />
    </div>{referenceError && value.reference_price == null && value.reference_price_mode === 'auto' && <p className="inline-warning">{referenceError}</p>}</div>
    <div className="form-section"><div className="section-title"><span>02</span><strong>网格规则</strong></div><div className="form-grid two">
      <Field label="网格模式"><SelectInput value={value.grid_mode} onChange={event => set('grid_mode', event.target.value as BacktestConfig['grid_mode'])}><option value="trailing">移动网格</option><option value="arithmetic">等差网格</option><option value="geometric">等比网格</option></SelectInput></Field>
      {value.grid_mode === 'trailing' ? <><NumField label="买入步长" value={value.buy_step ?? .03} onChange={next => next != null && set('buy_step', next)} percentValue suffix="%" min={.01} max={100} /><NumField label="卖出步长" value={value.sell_step ?? .03} onChange={next => next != null && set('sell_step', next)} percentValue suffix="%" min={.01} max={100} /></> : <NumField label="网格间距" value={value.grid_step ?? .03} onChange={next => next != null && set('grid_step', next)} percentValue suffix="%" min={.01} max={100} hint={value.grid_mode === 'arithmetic' && value.reference_price ? `固定每格约 ${(value.reference_price * (value.grid_step ?? .03)).toFixed(3)} 元` : undefined} />}
      <NumField label="价格下限" value={value.lower_price} onChange={next => set('lower_price', next)} suffix="元" min={.001} />
      <NumField label="价格上限" value={value.upper_price} onChange={next => set('upper_price', next)} suffix="元" min={.001} />
    </div></div>
    <div className="form-section"><div className="section-title"><span>03</span><div><strong>仓位与交易量</strong><small>初始持股在开始前已持有，首日可卖；现金自动预留</small></div></div><div className="form-grid two">
      {share('initial_shares', '初始持股')}{share('min_shares', '最低持股')}{share('max_shares', '最高持股')}{share('buy_shares', '每次买入')}{share('sell_shares', '每次卖出')}
    </div></div>
    <div className="form-section"><div className="section-title"><span>04</span><div><strong>定投设置</strong><small>定投账户独立持仓；获利按累计实际投入计算</small></div></div><div className="form-grid two">
      <Field label="定投频率"><SelectInput value={value.dca.frequency} onChange={event => set('dca', { ...value.dca, frequency: event.target.value as BacktestConfig['dca']['frequency'] })}><option value="daily">每天</option><option value="weekly">每周</option><option value="monthly">每月</option></SelectInput></Field>
      <NumField label="每期定投" value={value.dca.shares} onChange={next => next != null && set('dca', { ...value.dca, shares: next })} suffix={unit} min={isStarBoard ? 200 : 100} step="100" />
      <NumField label="定投初始持仓" value={value.dca.initial_shares} onChange={next => next != null && set('dca', { ...value.dca, initial_shares: next })} suffix={unit} min={0} step="100" hint="回测开始前已持有，不收取开仓手续费" />
    </div></div>
    <div className="form-section last"><div className="section-title"><span>05</span><strong>费用设置</strong></div><div className="form-grid two">
      <NumField label="佣金费率" value={fees.commission_rate} onChange={next => next != null && setFees({ commission_rate: next })} percentValue suffix="%" min={0} max={5} hint="0.05 表示 0.05%" />
      <NumField label="最低佣金" value={fees.min_commission} onChange={next => next != null && setFees({ min_commission: next })} suffix="元" min={0} />
      <NumField label="滑点" value={fees.slippage_bps} onChange={next => next != null && setFees({ slippage_bps: next })} suffix="基点" min={0} max={1000} />
    </div>{unit === '股' && <details className="advanced-disclosure"><summary>股票早期过户费设置</summary><div className="form-grid two">
      <NumField label="股票面值" value={fees.share_face_value} onChange={next => next != null && setFees({ share_face_value: next })} suffix="元" min={.0001} />
      <NumField label="沪市历史过户费率" value={fees.legacy_sh_transfer_rate} onChange={next => next != null && setFees({ legacy_sh_transfer_rate: next })} percentValue suffix="%" min={0} />
      <NumField label="沪市历史最低过户费" value={fees.legacy_sh_min_transfer} onChange={next => next != null && setFees({ legacy_sh_min_transfer: next })} suffix="元" min={0} />
      <NumField label="深市历史过户费率" value={fees.legacy_sz_transfer_rate} onChange={next => next != null && setFees({ legacy_sz_transfer_rate: next })} percentValue suffix="%" min={0} />
    </div></details>}</div>
  </div>
}

export function useConfirm() {
  const [value, setValue] = useState<string | null>(null)
  return { confirm: (message: string) => { setValue(message); return new Promise<boolean>((resolve) => { (window as Window & { __confirmResolve?: (value: boolean) => void }).__confirmResolve = resolve }) }, node: value && <div className="modal-scrim"><div className="confirm-modal"><h3>请确认</h3><p>{value}</p><div className="button-row"><Button variant="quiet" onClick={() => { setValue(null); (window as Window & { __confirmResolve?: (v: boolean) => void }).__confirmResolve?.(false) }}>取消</Button><Button variant="danger" onClick={() => { setValue(null); (window as Window & { __confirmResolve?: (v: boolean) => void }).__confirmResolve?.(true) }}>确认删除</Button></div></div></div> }
}
