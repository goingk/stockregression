import { useEffect, useMemo, useRef, useState } from 'react'
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams, EChartsOption } from 'echarts'
import ChartView from './ChartView'
import { get } from './api'
import { Badge, Button, Card, Empty, ErrorBox, Field, Loading, TextInput } from './components'
import { amplitudeBucketCount, buildAmplitudeHistogram } from './amplitudeMath'
import type { AmplitudeAdjustment, AmplitudeMetricKey, AmplitudeResponse } from './types'
import { LatestRequest } from './requestState'

type AppliedRange =
  | { kind: 'lookback'; lookback: number }
  | { kind: 'dates'; startDate: string; endDate: string }

const QUICK_LOOKBACKS = [20, 60, 120, 250] as const
const METRIC_VIEWS: Array<{ key: AmplitudeMetricKey; label: string; shortLabel: string; color: string; className: string }> = [
  { key: 'amplitude', label: '日均总振幅', shortLabel: '总振幅', color: '#3b82f6', className: 'total' },
  { key: 'upside', label: '日均向上振幅', shortLabel: '向上振幅', color: '#e11d48', className: 'upside' },
  { key: 'downside', label: '日均向下振幅', shortLabel: '向下振幅', color: '#0d9488', className: 'downside' },
]
const TITLE_TOPS = [3, 193, 383]
const GRID_TOPS = [38, 228, 418]

function precisePercent(value: number): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—'
}

function validIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const [year, month, day] = value.split('-').map(Number)
  const parsed = new Date(Date.UTC(year, month - 1, day))
  return parsed.getUTCFullYear() === year && parsed.getUTCMonth() === month - 1 && parsed.getUTCDate() === day
}

function dateRangeError(startDate: string, endDate: string): string {
  if (!startDate || !endDate) return '请完整填写开始日期和结束日期。'
  if (!validIsoDate(startDate) || !validIsoDate(endDate)) return '日期格式须为 YYYY-MM-DD。'
  if (startDate > endDate) return '开始日期不能晚于结束日期。'
  return ''
}

function localErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export default function AmplitudePanel({
  symbol,
  adjustment,
  taskRevision,
}: {
  symbol: string | null
  adjustment: AmplitudeAdjustment
  taskRevision: string
}) {
  const [appliedRange, setAppliedRange] = useState<AppliedRange>({ kind: 'lookback', lookback: 60 })
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [dateError, setDateError] = useState('')
  const [queryRevision, setQueryRevision] = useState(0)
  const [result, setResult] = useState<AmplitudeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const latestRequest = useRef(new LatestRequest<string>())

  const appliedKey = appliedRange.kind === 'lookback'
    ? `lookback:${appliedRange.lookback}`
    : `dates:${appliedRange.startDate}:${appliedRange.endDate}`

  useEffect(() => {
    if (!symbol) {
      latestRequest.current.invalidate()
      setResult(null)
      setLoading(false)
      setError('')
      return
    }

    const key = `${symbol}|${adjustment}|${appliedKey}|${taskRevision}|${queryRevision}`
    const ticket = latestRequest.current.begin(key)
    const params = new URLSearchParams({ adjustment })
    if (appliedRange.kind === 'lookback') params.set('lookback', String(appliedRange.lookback))
    else {
      params.set('start_date', appliedRange.startDate)
      params.set('end_date', appliedRange.endDate)
    }

    setLoading(true)
    setError('')
    setResult(null)
    get<AmplitudeResponse>(`/market/amplitude/${encodeURIComponent(symbol)}?${params.toString()}`)
      .then(value => {
        if (latestRequest.current.isCurrent(ticket)) setResult(value)
      })
      .catch(requestError => {
        if (latestRequest.current.isCurrent(ticket)) setError(localErrorMessage(requestError))
      })
      .finally(() => {
        if (latestRequest.current.isCurrent(ticket)) setLoading(false)
      })

    return () => latestRequest.current.invalidate(ticket.key)
  }, [symbol, adjustment, appliedKey, taskRevision, queryRevision, appliedRange])

  const histogram = useMemo(() => {
    if (!result) return null
    const sampleCount = Number.isFinite(result.sample_count) ? result.sample_count : result.samples.length
    return buildAmplitudeHistogram(result.samples, amplitudeBucketCount(sampleCount))
  }, [result])

  const chart = useMemo<EChartsOption | null>(() => {
    if (!result || !histogram || !result.samples.length) return null
    const upperPercent = histogram.upperBound * 100
    const axisMaximum = upperPercent > 0 ? upperPercent : 1

    return {
      animation: false,
      title: METRIC_VIEWS.map((view, index) => ({
        text: view.shortLabel,
        subtext: `均值 ${precisePercent(result.metrics?.[view.key]?.mean)} · 中位数 ${precisePercent(result.metrics?.[view.key]?.median)}`,
        left: 8,
        top: TITLE_TOPS[index],
        itemGap: 3,
        textStyle: { fontSize: 12, fontWeight: 600, color: '#334155' },
        subtextStyle: { fontSize: 10, color: '#64748b' },
      })),
      grid: METRIC_VIEWS.map((_, index) => ({ left: 64, right: 26, top: GRID_TOPS[index], height: 122 })),
      tooltip: {
        trigger: 'item',
        confine: true,
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.28); padding: 10px 12px;',
      },
      xAxis: METRIC_VIEWS.map((_, index) => ({
        type: 'value' as const,
        gridIndex: index,
        min: 0,
        max: axisMaximum,
        splitNumber: 4,
        axisLine: { lineStyle: { color: '#cbd5e1' } },
        axisTick: { show: false },
        axisLabel: { color: '#64748b', fontSize: 10, hideOverlap: true, formatter: (value: number) => `${value.toFixed(2)}%` },
        splitLine: { show: false },
      })),
      yAxis: METRIC_VIEWS.map((_, index) => ({
        type: 'value' as const,
        gridIndex: index,
        min: 0,
        minInterval: 1,
        splitNumber: 2,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#64748b', fontSize: 10 },
        splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' as const } },
      })),
      series: METRIC_VIEWS.map((view, index) => {
        const bins = histogram.bins[view.key]
        const summary = result.metrics?.[view.key]
        return {
          name: view.shortLabel,
          type: 'custom' as const,
          xAxisIndex: index,
          yAxisIndex: index,
          dimensions: ['start', 'end', 'count', 'share'],
          encode: { x: [0, 1], y: 2 },
          clip: true,
          data: bins.map(bin => ({
            value: [bin.start * 100, bin.end * 100, bin.count, bin.share],
          })),
          renderItem: (_params: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI) => {
            const start = api.value(0) as number
            const end = api.value(1) as number
            const count = api.value(2) as number
            const left = api.coord([start, 0])[0]
            const right = api.coord([end, 0])[0]
            const top = api.coord([start, count])[1]
            const baseline = api.coord([start, 0])[1]
            const span = right - left
            const width = Math.max(2, span - 1)
            return {
              type: 'rect',
              shape: {
                x: span < 1 ? left + 1 : left + (span - width) / 2,
                y: top,
                width,
                height: Math.max(0, baseline - top),
              },
              style: { fill: view.color, opacity: 0.86 },
            }
          },
          tooltip: {
            formatter: (params: unknown) => {
              const dataIndex = (params as { dataIndex?: number }).dataIndex ?? -1
              const bin = bins[dataIndex]
              if (!bin) return ''
              return `<strong style="display:block;margin-bottom:5px">${view.shortLabel}</strong>区间 [${precisePercent(bin.start)}, ${precisePercent(bin.end)}${bin.includeUpper ? ']' : ')'}<br/>频数 ${bin.count} 天 · 占比 ${(bin.share * 100).toFixed(1)}%<br/>均值 ${precisePercent(summary?.mean)} · 中位数 ${precisePercent(summary?.median)}`
            },
          },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              ...(Number.isFinite(summary?.mean) ? [{
                xAxis: (summary?.mean as number) * 100,
                lineStyle: { color: '#475569', type: 'dashed' as const, width: 1 },
                label: {
                  show: true,
                  position: 'insideEndTop' as const,
                  formatter: `均值 ${precisePercent(summary?.mean)}`,
                  color: '#334155',
                  fontSize: 9,
                  rotate: 0,
                  backgroundColor: 'rgba(255,255,255,.9)',
                  padding: [2, 4],
                  borderRadius: 3,
                },
              }] : []),
              ...(Number.isFinite(summary?.median) ? [{
                xAxis: (summary?.median as number) * 100,
                lineStyle: { color: '#b47719', type: 'dotted' as const, width: 1 },
                label: {
                  show: true,
                  position: 'insideEndBottom' as const,
                  formatter: `中位数 ${precisePercent(summary?.median)}`,
                  color: '#93671d',
                  fontSize: 9,
                  rotate: 0,
                  backgroundColor: 'rgba(255,255,255,.9)',
                  padding: [2, 4],
                  borderRadius: 3,
                },
              }] : []),
            ],
          },
        }
      }),
    }
  }, [histogram, result])

  const queryNote = appliedRange.kind === 'lookback'
    ? `当前按最近 ${appliedRange.lookback} 个交易日查询`
    : `当前按 ${appliedRange.startDate} 至 ${appliedRange.endDate} 查询`
  const warnings = result ? [...result.warnings] : []
  if (result && result.sample_count !== result.samples.length) {
    warnings.push(`接口返回样本记录 ${result.samples.length} 条，样本计数为 ${result.sample_count} 条。`)
  }
  if (result && histogram && METRIC_VIEWS.some(view => histogram.totals[view.key] !== result.sample_count)) {
    warnings.push('部分样本缺少有效的振幅数值，直方图仅统计有效值。')
  }

  const selectLookback = (lookback: number) => {
    if (!symbol) return
    setDateError('')
    setAppliedRange({ kind: 'lookback', lookback })
    setQueryRevision(value => value + 1)
  }
  const computeCustomRange = () => {
    if (!symbol) return
    const message = dateRangeError(startDate, endDate)
    if (message) {
      setDateError(message)
      return
    }
    setDateError('')
    setAppliedRange({ kind: 'dates', startDate, endDate })
    setQueryRevision(value => value + 1)
  }

  return <Card
    title="区间日均振幅"
    subtitle="固定日线 · 以前一交易日收盘为基准 · 与主图复权保持一致"
    className="amplitude-card"
    action={<Badge tone="blue">{adjustment === 'qfq' ? '前复权' : '不复权'} · 日线</Badge>}
  >
    <div className="amplitude-toolbar">
      <div className="amplitude-quick">
        <span className="amplitude-toolbar-label">快捷区间</span>
        {QUICK_LOOKBACKS.map(lookback => {
          const selected = appliedRange.kind === 'lookback' && appliedRange.lookback === lookback
          return <button
            key={lookback}
            type="button"
            className={`amplitude-range-button ${selected ? 'active' : ''}`}
            aria-pressed={selected}
            disabled={!symbol}
            onClick={() => selectLookback(lookback)}
          >近 {lookback} 个交易日</button>
        })}
      </div>
      <div className="amplitude-custom-range">
        <Field label="开始日期"><TextInput type="date" value={startDate} disabled={!symbol} onChange={event => { setStartDate(event.target.value); setDateError('') }} /></Field>
        <span className="amplitude-date-separator">至</span>
        <Field label="结束日期"><TextInput type="date" value={endDate} disabled={!symbol} onChange={event => { setEndDate(event.target.value); setDateError('') }} /></Field>
        <Button variant="quiet" className="amplitude-compute" disabled={!symbol} onClick={computeCustomRange}>计算</Button>
      </div>
    </div>
    <div className="amplitude-query-note" aria-live="polite"><span>{symbol ? queryNote : '选择证券后自动查询日线振幅'}</span>{dateError && <strong role="alert">{dateError}</strong>}</div>

    {!symbol && <Empty title="尚未选择证券" detail="从左侧自选或搜索列表选择标的，振幅统计会随主图自动更新。" />}
    {symbol && loading && <div className="amplitude-loading"><Loading label="正在计算区间日均振幅…" /></div>}
    {symbol && error && <ErrorBox message={error} onClose={() => setError('')} onRetry={() => setQueryRevision(value => value + 1)} />}

    {symbol && result && <>
      <div className="amplitude-range-summary">
        <div><small>实际日期范围</small><strong>{result.actual.start_date} 至 {result.actual.end_date}</strong></div>
        <div><small>样本数</small><strong>{result.sample_count} 个交易日</strong></div>
        <div><small>排除数</small><strong>{result.excluded_count} 个交易日</strong></div>
        <div><small>复权口径</small><strong>{result.adjustment === 'qfq' ? '前复权' : '不复权'}</strong></div>
      </div>

      <div className="amplitude-stat-grid">
        {METRIC_VIEWS.map(view => {
          const summary = result.metrics?.[view.key]
          return <section className={`amplitude-stat ${view.className}`} key={view.key}>
            <span className="amplitude-stat-label"><i style={{ background: view.color }} />{view.label}</span>
            <div className="amplitude-stat-values">
              <div className="mean"><small>均值</small><strong>{precisePercent(summary?.mean)}</strong></div>
              <div className="median"><small>中位数</small><strong>{precisePercent(summary?.median)}</strong></div>
            </div>
          </section>
        })}
      </div>

      {warnings.length > 0 && <div className="amplitude-warnings" role="status">
        <strong>数据完整性提示</strong>
        <ul>{warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul>
      </div>}

      {!result.samples.length
        ? <Empty title="所选区间没有可用日线样本" detail="请检查日期范围，或先更新该标的日线行情后重试。" />
        : chart && <div className="amplitude-chart"><ChartView option={chart} height={590} /></div>}
    </>}
  </Card>
}
