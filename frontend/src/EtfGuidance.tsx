import type { EtfParameterGuidance } from './etfResearch'
import { Badge, Button, percent } from './components'
import Disclosure from './Disclosure'

export default function EtfGuidance({ guidance, onApply }: { guidance?: EtfParameterGuidance | null; onApply: () => void }) {
  const usable = guidance && ['ready', 'limited'].includes(guidance.status) && guidance.tau
  return <section className='etf-guidance' aria-label='历史参数研究建议'>
    <div className='etf-guidance-heading'><div><h3>历史参数研究建议</h3><p>为后续研究确定探索范围与安全垫，建议值可在研究草稿中继续调整。</p></div>
      <Badge tone={usable ? guidance.status === 'limited' ? 'amber' : 'blue' : 'neutral'}>{usable ? '历史启发式建议' : '暂无建议'}</Badge></div>
    {usable ? <>
      <div className='etf-guidance-values'>
        {(['buy_step', 'sell_step'] as const).map(name => <div key={name}><small>{name === 'buy_step' ? '买入' : '卖出'}网格间距上下限</small><strong>{percent(guidance.ranges[name]?.low)} – {percent(guidance.ranges[name]?.high)}</strong><span>细粒度 {percent(guidance.granularity[name], 2)} · 约 {guidance.evidence.spacing_points} 个采样点</span></div>)}
        <div><small>安全垫阈值 τ</small><strong>{percent(guidance.tau!.value)}</strong><span>敏感性对照 {guidance.tau!.sensitivity.map(value => percent(value)).join(' / ')}</span></div>
        <div><small>买卖数量研究细粒度</small><strong>{guidance.granularity.buy_shares} / {guidance.granularity.sell_shares} 份</strong><span>保留当前数量范围与研究预算</span></div>
      </div>
      <p className='etf-guidance-evidence'>截止 {guidance.as_of} · {guidance.evidence.daily_return_samples} 个日收益样本 · 日波动 {percent(guidance.evidence.daily_volatility)} · NATR14 {percent(guidance.evidence.natr14)} · 100 份往返费用约 {percent(guidance.evidence.round_trip_cost)}</p>
      <Disclosure id='etf-parameter-guidance-method' title='建议依据与适用条件' summary={`τ 以 ${guidance.tau!.reference_days} 个交易日为参考跨度`}>
        <ul>{guidance.notes.map(note => <li key={note}>{note}</li>)}</ul><p>{guidance.tau!.reason}</p>
      </Disclosure>
      <div className='etf-guidance-action'><Button variant='secondary' onClick={onApply}>按建议建立后续研究</Button><span>带入间距上下限、细粒度与 τ，研究起点设为截止日之后。</span></div>
    </> : <p>{guidance?.reason || '此历史结果未记录参数建议，请重新运行 ETF 研究。'}</p>}
  </section>
}
