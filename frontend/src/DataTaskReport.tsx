import { Badge } from './components'
import { reportFailures, reportReason, reportRows, syncReportRuns } from './dataReportModel'
import { taskStatusLabel } from './taskState'

export function DataTaskReport({ report }: { report: Record<string, unknown> }) {
  const runs = syncReportRuns(report)
  const failures = reportFailures(report)
  const steps = reportRows(report.steps)
  return <>
    {runs.length > 0 && <>
      <p className="muted-copy">进度 100% 表示任务已处理结束。各周期的成功、跳过和失败情况如下；跳过项未执行更新。</p>
      <div className="table-wrap"><table><thead><tr><th>周期</th><th>状态</th><th>成功</th><th>跳过</th><th>失败</th></tr></thead><tbody>
        {runs.map((run, index) => <tr key={index}><td>{run.timeframe}</td><td><Badge tone={run.status === 'failed' ? 'red' : run.status === 'partial' ? 'amber' : 'green'}>{taskStatusLabel(String(run.status ?? ''))}</Badge></td><td>{run.successes.length}</td><td>{run.skipped.length}</td><td>{run.failures.length}</td></tr>)}
      </tbody></table></div>
    </>}
    {failures.length > 0 && <div role="status"><h3>失败明细（{failures.length}）</h3><div className="table-wrap"><table><thead><tr><th>标的</th><th>周期</th><th>失败原因</th></tr></thead><tbody>
      {failures.map((item, index) => <tr key={index}><td>{String(item.symbol ?? '—')}</td><td>{String(item.timeframe ?? report.timeframe ?? '—')}</td><td className="task-report-reason">{reportReason(item)}</td></tr>)}
    </tbody></table></div></div>}
    {runs.map((run, index) => <details className="advanced-data" key={index}><summary>{run.timeframe} · 成功与跳过明细</summary><div className="table-wrap"><table><thead><tr><th>标的</th><th>状态</th><th>说明</th></tr></thead><tbody>
      {[...run.successes.map(item => ({ item, status: '成功' })), ...run.skipped.map(item => ({ item, status: '跳过' }))].map(({ item, status }, row) => <tr key={row}><td>{String(item.symbol ?? '—')}</td><td>{status}</td><td className="task-report-reason">{reportReason(item)}</td></tr>)}
    </tbody></table></div></details>)}
    {steps.length > 0 && <div className="table-wrap"><table><thead><tr><th>步骤</th><th>状态</th><th>说明</th></tr></thead><tbody>{steps.map((step, index) => <tr key={index}><td>{String(step.kind ?? step.timeframe ?? `步骤 ${index + 1}`)}</td><td>{taskStatusLabel(String(step.status ?? '—'))}</td><td className="task-report-reason">{reportReason(step)}</td></tr>)}</tbody></table></div>}
    {!runs.length && !steps.length && !failures.length && <p className="muted-copy">{String(report.message ?? '任务已结束，完整记录可在高级诊断中查看。')}</p>}
  </>
}
