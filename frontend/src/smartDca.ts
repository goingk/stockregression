import type { DcaConfig } from './types.ts'

export const dcaIndexes: Record<string, string> = {
  '000001.SH': '上证指数', '000300.SH': '沪深300', '000905.SH': '中证500',
  '000016.SH': '上证50', '399001.SZ': '深证成指', '399006.SZ': '创业板指',
}

export function defaultDca(): Required<DcaConfig> {
  return { frequency: 'monthly', shares: 100, initial_shares: 0, mode: 'fixed_shares',
    base_amount: 1000, reference: 'target', reference_symbol: null, reference_version: null, ma_window: 250,
    change_window: 1, sensitivity: 5, min_multiplier: .5, max_multiplier: 2 }
}

export function normalizeDca(value: DcaConfig): Required<DcaConfig> {
  return { ...defaultDca(), ...value }
}

export function dcaError(value: DcaConfig): string {
  const d = normalizeDca(value)
  const number = (v: number, low: number, high: number) => typeof v === 'number' && Number.isFinite(v) && v >= low && v <= high
  if (!['fixed_shares', 'ma', 'price_change'].includes(d.mode)) return '请选择有效的定投模式'
  if (!['target', 'security', 'index'].includes(d.reference)) return '请选择有效的定投参考类型'
  if (!number(d.base_amount, .01, 1e9)) return '智能定投基准金额应大于 0 且不超过 10 亿元'
  if (!Number.isInteger(d.ma_window) || !number(d.ma_window, 2, 1000)) return '定投均线窗口应为 2–1000 个交易日'
  if (!Number.isInteger(d.change_window) || !number(d.change_window, 1, 1000)) return '定投涨跌幅窗口应为 1–1000 个交易日'
  if (!number(d.sensitivity, .000001, 100)) return '定投调整力度应大于 0 且不超过 100'
  if (!number(d.min_multiplier, 0, 1) || !number(d.max_multiplier, 1, 20)) return '定投倍数下限应在 0–1 之间，上限应在 1–20 之间'
  if (d.mode !== 'fixed_shares' && d.reference !== 'target' && !d.reference_symbol?.trim()) return '请选择定投参考证券或指数'
  if (d.reference === 'index' && d.reference_symbol && !dcaIndexes[d.reference_symbol]) return '请选择列表中的定投参考指数'
  if (d.reference === 'security' && d.reference_symbol && !/^(?:[56]\d{5}\.SH|[013]\d{5}\.SZ)$/i.test(d.reference_symbol.trim())) return '定投参考证券应填写完整代码，例如 510300.SH'
  return ''
}

export function dcaSummary(value: DcaConfig, unit = '股'): string {
  const d = normalizeDca(value)
  const frequency = { daily: '每天', weekly: '每周', monthly: '每月' }[d.frequency]
  return d.mode === 'fixed_shares' ? `${frequency} ${d.shares} ${unit}`
    : `${frequency} · ${d.mode === 'ma' ? '均线智能' : '涨跌幅智能'} · 基准 ${d.base_amount} 元`
}
