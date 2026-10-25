import { Badge } from './components'
import TableViewport from './TableViewport'
import { reportFailures, reportReason, reportRows, syncReportRuns } from './dataReportModel'
import { taskStatusLabel } from './taskState'

const gapKindLabels: Record<string, string> = {
  quality: '质量核验', range: '覆盖范围', reference_price: '前置收盘价',
  bars: '行情缺失', fields: '字段缺失', calendar: '交易日历',
  actions: '公司行动', price_quality: '价格质量', version: '数据版本',
}

const interruptedSections = [
  { key: 'published_quality_pending', title: '行情已发布，质量待核验' },
  { key: 'downloaded_uncommitted', title: '已下载，尚未提交' },
  { key: 'unprocessed', title: '未处理' },
] as const

function qualityLabel(item: Record<string, unknown>) {
  const quality = String(item.quality_status ?? item.quality_compliant ?? '')
  if (quality === 'partial') return '部分合规'
  if (quality === 'failed' || quality === 'false') return '不合规'
  if (quality === 'passed' || quality === 'complete' || quality === 'ready' || quality === 'true') return '合规'
  return quality || '待核验'
}

function qualityTone(item: Record<string, unknown>): 'green' | 'amber' | 'red' {
  const quality = qualityLabel(item)
  return quality === '合规' ? 'green' : quality === '不合规' ? 'red' : 'amber'
}

export function DataTaskReport({ report }: { report: Record<string, unknown> }) {
  const runs = syncReportRuns(report)
  const failures = reportFailures(report)
  const steps = reportRows(report.steps)
  const gaps = reportRows(report.remaining_gaps)
  const partial = report.status === 'cancelled' || report.partial === true
  const failureCategories = report.failure_categories && typeof report.failure_categories === 'object'
    ? Object.entries(report.failure_categories as Record<string, unknown>).filter(([, count]) => typeof count === 'number')
    : []
  const interrupted = interruptedSections.map(section => ({
    ...section,
    items: reportRows(report[section.key]),
  })).filter(section => section.items.length > 0)
  const failureContext = (item: Record<string, unknown>) => {
    const parts = [
      item.error_category ?? item.category,
      item.product_type,
      item.fallback_reason,
      item.requested_start || item.requested_end || item.request_start || item.start_date || item.request_end || item.end_date
        ? `${String(item.requested_start ?? item.request_start ?? item.start_date ?? '起始未记录')} 至 ${String(item.requested_end ?? item.request_end ?? item.end_date ?? '结束未记录')}` : undefined,
      item.source_attempts != null ? `来源尝试：${typeof item.source_attempts === 'string' ? item.source_attempts : JSON.stringify(item.source_attempts)}` : undefined,
    ].filter(value => value != null && String(value).length > 0).map(String)
    return parts.join(' · ')
  }
  return <>
    {partial && <p className="inline-warning">此报告保留了取消或中断时已处理的部分结果；未处理目标不会被计为成功。</p>}
    {partial && interrupted.length > 0 && <section aria-label="取消时的部分结果">
      <h3>取消时的处理状态</h3>
      <div className="stat-grid three">
        {interrupted.map(section => <div className="stat amber" key={section.key}>
          <span>{section.title}</span><strong>{section.items.length}</strong>
        </div>)}
      </div>
      {interrupted.map(section => <div key={section.key}>
        <h4>{section.title}（{section.items.length}）</h4>
        <ul className="compact-list">{section.items.map((item, index) => <li key={`${String(item.symbol ?? 'item')}-${index}`}>
          <strong>{String(item.symbol ?? '—')}</strong> · {String(item.state ?? '—')}
        </li>)}</ul>
      </div>)}
    </section>}
    {gaps.length > 0 && <div role="status">
      <h3>数据缺口（{gaps.length}）</h3>
      <TableViewport owner="parent" label="数据缺口"><table><thead><tr><th>类型</th><th>周期</th><th>原因</th></tr></thead><tbody>
        {gaps.map((gap, index) => <tr key={index}>
          <td>{gapKindLabels[String(gap.kind ?? '')] ?? String(gap.kind ?? '—')}</td>
          <td>{String(gap.timeframe ?? '—')}</td>
          <td className="task-report-reason">{reportReason(gap)}</td>
        </tr>)}
      </tbody></table></TableViewport>
      <p className="muted-copy">准备步骤结束不等于数据可用；上述缺口补齐前，研究或回测仍会被拒绝。</p>
    </div>}
    {runs.length > 0 && <>
      <p className="muted-copy">任务处理结束后，各周期分别列出成功、跳过和失败情况；跳过项未执行更新。取消或中断任务仍以其部分报告为准。</p>
      <TableViewport owner="parent" label="周期任务结果"><table><thead><tr><th>周期</th><th>状态</th><th>成功</th><th>跳过</th><th>失败</th></tr></thead><tbody>
        {runs.map((run, index) => <tr key={index}><td>{run.timeframe}</td><td><Badge tone={run.status === 'failed' ? 'red' : run.status === 'partial' || run.status === 'cancelled' ? 'amber' : 'green'}>{taskStatusLabel(String(run.status ?? ''))}</Badge></td><td>{run.successes.length}</td><td>{run.skipped.length}</td><td>{run.failures.length}</td></tr>)}
      </tbody></table></TableViewport>
    </>}
    {failureCategories.length > 0 && <div role="status"><h3>失败分类汇总</h3><ul className="compact-list">
      {failureCategories.map(([category, count]) => <li key={category}>{category}：{String(count)}</li>)}
    </ul></div>}
    {failures.length > 0 && <div role="status"><h3>失败明细（{failures.length}）</h3><TableViewport owner="parent" stickyKeyColumn={1} label="失败明细"><table><thead><tr><th>标的</th><th>周期</th><th>失败原因</th><th>来源与请求范围</th></tr></thead><tbody>
      {failures.map((item, index) => <tr key={index}><td>{String(item.symbol ?? '—')}</td><td>{String(item.timeframe ?? report.timeframe ?? '—')}</td><td className="task-report-reason">{reportReason(item)}</td><td className="task-report-reason">{failureContext(item) || '—'}</td></tr>)}
    </tbody></table></TableViewport></div>}
    {runs.map((run, index) => <details className="advanced-data" key={index}><summary>{run.timeframe} · 操作状态与质量结果</summary><p className="muted-copy">操作是否完成与质量是否合规分别显示；任务成功结束不代表质量核验一定通过。</p><TableViewport owner="parent" stickyKeyColumn={1} label="操作状态与质量结果"><table><thead><tr><th>标的</th><th>操作</th><th>质量合规</th><th>说明</th></tr></thead><tbody>
      {[...run.successes.map(item => ({ item, status: '已完成' })), ...run.skipped.map(item => ({ item, status: '已跳过' }))].map(({ item, status }, row) => <tr key={row}><td>{String(item.symbol ?? '—')}</td><td><Badge tone="green">{status}</Badge></td><td><Badge tone={qualityTone(item)}>{qualityLabel(item)}</Badge></td><td className="task-report-reason">{reportReason(item)}</td></tr>)}
    </tbody></table></TableViewport></details>)}
    {steps.length > 0 && <TableViewport owner="parent" stickyKeyColumn={1} label="任务步骤"><table><thead><tr><th>步骤</th><th>状态</th><th>说明</th></tr></thead><tbody>{steps.map((step, index) => <tr key={index}><td>{String(step.kind ?? step.timeframe ?? `步骤 ${index + 1}`)}</td><td>{taskStatusLabel(String(step.status ?? '—'))}</td><td className="task-report-reason">{reportReason(step)}</td></tr>)}</tbody></table></TableViewport>}
    {!runs.length && !steps.length && !failures.length && <p className="muted-copy">{String(report.message ?? '任务已结束，完整记录可在高级诊断中查看。')}</p>}
  </>
}
