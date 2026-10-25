import { Badge, Button, Empty } from './components'
import { formatLandscapeParameter, landscapeLabels, type LandscapePoint } from './landscapeData'

type LoadPoint = (params: Record<string, unknown>) => void
const fields = ['buy_step', 'sell_step', 'buy_shares', 'sell_shares']
const pct = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`
/** Red-up / green-down tone shared with the rest of the workbench. */
const toneClass = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '' : value > 0 ? 'gain' : value < 0 ? 'loss' : ''

function Parameters({ point }: { point: LandscapePoint }) {
  return <div className='research-parameter-badges'>
    {fields.filter(name => point.params[name] != null).map(name => <span key={name}>
      <small>{landscapeLabels[name]}</small><strong>{formatLandscapeParameter(name, point.params[name])}</strong>
    </span>)}
  </div>
}

export function RobustPointSummary({ point, kind, onLoad }: { point: LandscapePoint | null; kind: 'excess' | 'j'; onLoad?: LoadPoint }) {
  const title = kind === 'excess' ? '超额收益最高点' : 'J 最高点'
  const headline = kind === 'excess' ? point?.excess_return : point?.j_score
  return <article className='research-representative'>
    <div className='research-representative-head'>
      <div><small>{title}{point ? ` · #${point.candidate_id}` : ''}</small><strong className={toneClass(headline)}>{kind === 'excess' ? '超额收益 ' : '稳健 J '}{pct(headline)}</strong></div>
      {point && onLoad && <Button variant='quiet' aria-label={`载入${title}参数`} onClick={() => onLoad(point.params)}>载入</Button>}
    </div>
    {point ? <>
      <Parameters point={point} />
      <div className='research-representative-metrics'>
        <span><small>{kind === 'excess' ? '稳健 J' : '中心超额'}</small><strong className={toneClass(kind === 'excess' ? point.j_score : point.excess_return)}>{pct(kind === 'excess' ? point.j_score : point.excess_return)}</strong></span>
        <span><small>邻域均值</small><strong>{pct(point.neighborhood_mean)}</strong></span>
        <span><small>下行风险</small><strong>{pct(point.downside_risk_tau)}</strong></span>
        <span><small>正收益占比</small><strong>{pct(point.pos_win_rate)}</strong></span>
        <span><small>邻域点数</small><strong>{point.neighborhood_support_size ?? point.neighborhood_size ?? '—'}</strong></span>
        <span><small>评分完整</small><strong>{point.score_complete === true ? '是' : point.score_complete === false ? '否' : '未计算'}</strong></span>
        <span><small>底仓惩罚</small><strong>{pct(point.floor_penalty, 4)}</strong></span>
      </div>
    </> : <p className='field-hint'>尚无完整 J 评分。</p>}
  </article>
}

export function RobustJRanking({ points, total, count, coverage, onLoad }: { points: LandscapePoint[]; total: number; count: number; coverage?: string | null; onLoad?: LoadPoint }) {
  return <section className='research-j-ranking'>
    <div className='research-peak-regions-head'>
      <div><h3>稳健 J 参数排名</h3><span>训练候选按 J 从高到低推荐前 {count} 组，当前显示 {points.length} / {total.toLocaleString()} 组；约束状态随参数展示。训练候选不是样本外结果。</span></div>
      <Badge tone='neutral'>{coverage === 'global_exhaustive' ? '全局穷尽' : '预算搜索'}</Badge>
    </div>
    {points.length ? <div className='research-peak-regions'>{points.map((point, index) => <article className='research-peak-region' key={point.candidate_id}>
      <div className='research-peak-region-main'>
        <span className='research-peak-rank'>#{index + 1}</span>
        <div className='research-peak-params'><Parameters point={point} /></div>
        <div className={`research-peak-score${(point.j_score ?? 0) < 0 ? ' negative' : ''}`}><small>稳健 J</small><strong className={toneClass(point.j_score)}>{pct(point.j_score, 4)}</strong></div>
        {onLoad && <Button variant='quiet' onClick={() => onLoad(point.params)}>载入</Button>}
      </div>
      <div className='research-peak-metrics'>
        <span><small>中心 R</small><strong className={toneClass(point.excess_return)}>{pct(point.excess_return, 4)}</strong></span>
        <span><small>邻域均值</small><strong>{pct(point.neighborhood_mean, 4)}</strong></span>
        <span><small>下行风险</small><strong>{pct(point.downside_risk_tau, 4)}</strong></span>
        <span><small>持仓惩罚</small><strong>{pct(point.pos_penalty, 4)}</strong></span>
        <span><small>底仓段数</small><strong>{point.floor_episode_count ?? '—'}</strong></span>
        <span><small>触底后上涨格数</small><strong>{point.floor_episode_grids?.length ? point.floor_episode_grids.join(' / ') : point.floor_episode_count === 0 ? '无' : '—'}</strong></span>
      </div>
      <div className='research-peak-status'>
        <Badge tone={point.constraint_eligible ? 'green' : 'amber'}>{point.constraint_eligible ? '满足约束' : '未满足约束'}</Badge>
        <span>邻域 {point.neighborhood_support_size ?? point.neighborhood_size ?? '—'} 点 · {point.score_complete === true ? '评分完整' : '评分未完成'}</span>
        {point.local_peak_verified && <span>峰值已核验 · 平台 {point.platform_size ?? 1} 点</span>}
        {point.stress?.status && <span>压力{point.stress.status === 'completed' ? '已完成' : point.stress.status === 'failed' ? '失败' : '部分完成'} · 均值 {pct(point.stress.mean_excess_return)}</span>}
        {!!point.constraint_reasons?.length && <span>{point.constraint_reasons.join('；')}</span>}
      </div>
      <details className='research-peak-details'>
        <summary>持仓与成交诊断</summary>
        <div className='research-peak-detail-grid'>
          <span><small>实际持仓范围</small><strong>{point.actual_min_pos_ratio?.toFixed(3) ?? '—'} – {point.actual_max_pos_ratio?.toFixed(3) ?? '—'}</strong></span>
          <span><small>底仓惩罚 / 段数</small><strong>{pct(point.floor_penalty, 4)} / {point.floor_episode_count ?? '—'}</strong></span>
          <span><small>各段上涨格数</small><strong>{point.floor_episode_grids?.length ? point.floor_episode_grids.join(' / ') : point.floor_episode_count === 0 ? '无' : '未记录'}</strong></span>
          <span><small>邻域最高持仓</small><strong>{point.neighborhood_max_eta?.toFixed(3) ?? '—'}</strong></span>
          <span><small>正收益 / 安全垫</small><strong>{pct(point.pos_win_rate)} / {pct(point.safe_win_rate)}</strong></span>
          <span><small>成交笔数</small><strong>{point.trade_count ?? '—'}</strong></span>
          <span><small>库存不足 / T+1</small><strong>{point.stockout_count ?? '—'} / {point.tplus1_block_count ?? '—'}</strong></span>
        </div>
      </details>
    </article>)}</div> : <Empty title='暂无完整 J 评分' detail='完成研究后，按推荐参数个数展示 J 排名。' />}
  </section>
}
