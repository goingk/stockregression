import { useRef, useState } from 'react'
import { post } from './api'
import { Button, Field, NumField, SelectInput, TextInput, configPayload } from './components'
import { dcaIndexes, normalizeDca } from './smartDca'
import type { BacktestConfig, DcaConfig } from './types'

export default function SmartDcaFields({ config, onChange }: { config: BacktestConfig; onChange: (value: DcaConfig) => void }) {
  const dca = normalizeDca(config.dca)
  const latest = useRef(dca)
  latest.current = dca
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const set = <K extends keyof DcaConfig>(name: K, value: DcaConfig[K]) => onChange({ ...dca, [name]: value, ...(name === 'reference_symbol' ? { reference_version: null } : {}) })
  async function prepare() {
    setBusy(true); setNotice('')
    try {
      const result = await post<{ row_count: number; first_date: string; last_date: string; version: string }>('/dca/reference/prepare', configPayload(config))
      if (latest.current.reference === 'index' && latest.current.reference_symbol === dca.reference_symbol) onChange({ ...latest.current, reference_version: result.version })
      setNotice(`已准备 ${result.row_count} 条指数日线（${result.first_date} 至 ${result.last_date}）。可运行回测；窗口不足的日期会跳过。`)
    } catch (error) { setNotice(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }
  return <>
    <NumField label="每期基准金额" value={dca.base_amount} onChange={v => v != null && set('base_amount', v)} suffix="元" min={.01} max={1e9} hint="金额乘投入倍数后，按交易单位向下取整；手续费另计，余款不累积" />
    <Field label="信号参考"><SelectInput value={dca.reference} onChange={e => {
      const reference = e.target.value as DcaConfig['reference']
      setNotice('')
      onChange({ ...dca, reference, reference_symbol: reference === 'index' ? '000300.SH' : null, reference_version: null })
    }}><option value="target">当前交易标的</option><option value="security">其他证券 / ETF</option><option value="index">市场指数</option></SelectInput></Field>
    {dca.reference === 'security' && <Field label="参考证券代码" hint="例如 510300.SH；须先在数据管理中同步该证券的日线和公司行动"><TextInput value={dca.reference_symbol ?? ''} placeholder="510300.SH" onChange={e => set('reference_symbol', e.target.value.trim().toUpperCase())} /></Field>}
    {dca.reference === 'index' && <Field label="参考指数"><SelectInput value={dca.reference_symbol ?? ''} onChange={e => { setNotice(''); set('reference_symbol', e.target.value) }}>{Object.entries(dcaIndexes).map(([symbol, name]) => <option key={symbol} value={symbol}>{name} · {symbol}</option>)}</SelectInput></Field>}
    {dca.mode === 'ma'
      ? <NumField label="均线窗口" value={dca.ma_window} onChange={v => v != null && set('ma_window', v)} min={2} max={1000} step="1" suffix="交易日" hint="执行日前最近 N 个已知收盘价的算术平均" />
      : <NumField label="涨跌幅窗口" value={dca.change_window} onChange={v => v != null && set('change_window', v)} min={1} max={1000} step="1" suffix="交易日" hint="用此前 N+1 个收盘价，计算最近 N 个交易日的累计涨跌幅" />}
    <NumField label="调整力度" value={dca.sensitivity} onChange={v => v != null && set('sensitivity', v)} min={.000001} max={100} hint="例如 5：信号下跌 10% 时投入 1.5 倍，上涨 10% 时投入 0.5 倍" />
    <NumField label="最低投入倍数" value={dca.min_multiplier} onChange={v => v != null && set('min_multiplier', v)} min={0} max={1} suffix="倍" />
    <NumField label="最高投入倍数" value={dca.max_multiplier} onChange={v => v != null && set('max_multiplier', v)} min={1} max={20} suffix="倍" />
    <div className="field" style={{ gridColumn: '1 / -1' }}>
      <small>本项目规则：投入倍数 = 1 − 调整力度 × {dca.mode === 'ma' ? '（此前收盘价 ÷ 均线 − 1）' : '此前累计涨跌幅'}，限制在上下限内。跌多增投、涨多减投。仅使用执行日前的数据；参考缺失、停牌或历史不足时跳过并记录，不换用其他标的。</small>
      <small>参考智能定投理念，非支付宝未公开算法的精确复刻。公司行动按当时已发生的除权口径处理。</small>
      {dca.reference === 'index' && <Button type="button" variant="secondary" disabled={busy || !dca.reference_symbol} onClick={prepare}>{busy ? '正在准备指数日线…' : '准备指数参考数据'}</Button>}
      {notice && <small role="status">{notice}</small>}
    </div>
  </>
}
