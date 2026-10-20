type RecordValue = Record<string, unknown>

export function reportRows(value: unknown): RecordValue[] {
  return Array.isArray(value) ? value.filter((row): row is RecordValue => Boolean(row) && typeof row === 'object' && !Array.isArray(row)) : []
}

/** Older reports store failures both at the top level and inside each run. */
export function syncReportRuns(report: RecordValue) {
  return reportRows(report.runs).map(run => ({
    ...run,
    timeframe: String(run.timeframe ?? '—'),
    status: String(run.status ?? ''),
    successes: reportRows(run.symbols),
    skipped: reportRows(run.skipped),
    failures: reportRows(run.failures),
  }))
}

export function reportFailures(report: RecordValue): RecordValue[] {
  const nested = syncReportRuns(report).flatMap(run => run.failures.map(item => ({ ...item, timeframe: run.timeframe })))
  return nested.length ? nested : reportRows(report.failures)
}

export const reportReason = (row: RecordValue) => String(row.reason ?? row.error ?? row.message ?? '—')
