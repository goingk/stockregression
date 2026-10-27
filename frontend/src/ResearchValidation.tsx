import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { CheckCircle2, Download, Plus, RefreshCw, ShieldCheck, Trash2 } from 'lucide-react'
import { download, post } from './api'
import { Badge, Button, Card, Empty, Field, NumField, TextInput, money, percent, useErrorMessage } from './components'
import Disclosure from './Disclosure'
import { useTasks } from './TaskCenter'
import { flushConfiguration } from './useWorkbenchDefaults'
import {
  buildValidationRequest, canSubmitValidationPlan, parseValidationSymbols,
  researchResultHistoryLabel, researchV5FromDraft, requestedVersionsFromPlan, validationCellSummary,
  validationAsyncTicketCurrent, validationDraftKey, validationRequestKey,
  validationBlocks, validationEvidenceGaps, validationEvidenceRecorded, validationOverviewItems, validationPlanOf,
  validationPlanSymbols, validationPrimaryArm, validationPrimaryMetrics, validationSchemeEntries,
  validationAccountRows, validationSegmentGroups,
} from './researchValidationModel.ts'
import type {
  ValidationAccountInput, ValidationAsyncTicket, ValidationContributionInput, ValidationEvidenceSection,
  ValidationHistoryPolicy, ValidationSegmentGroup, ValidationTone,
} from './researchValidationModel.ts'

type JsonObject = Record<string, unknown>
type AccountDraft = ValidationAccountInput & { contributions: ValidationContributionInput[] }

export type ResearchValidationLauncherProps = {
  buildResearch: (configurationRevision?: string, requestedDataVersion?: string) => unknown
  initialSymbol: string
  initialShares: number
  datesExplicit?: boolean
  enabled: boolean
  disabledReason?: string
  onSubmitted: (jobId: string) => void
}

const objectValue = (value: unknown): JsonObject | null => value != null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : null
const textValue = (value: unknown): string => {
  if (value == null || value === '') return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value)) return value.length ? value.map(item => typeof item === 'object' ? compactSummary(item) : String(item)).join('、') : '—'
  if (typeof value === 'object') return compactSummary(value)
  return String(value)
}
const compactSummary = (value: unknown, limit = 4): string => {
  const item = objectValue(value)
  if (!item) return textValue(value)
  const entries = Object.entries(item).filter(([, raw]) => raw == null || typeof raw !== 'object').slice(0, limit)
  return entries.length ? entries.map(([key, raw]) => labelValue(key) + ' ' + metricValue(key, raw)).join(' · ') : Object.keys(item).length + ' 项'
}
const firstValue = (source: JsonObject | null, keys: string[]): unknown => {
  if (!source) return undefined
  for (const key of keys) if (source[key] != null && source[key] !== '') return source[key]
  return undefined
}
const labels: Record<string, string> = {
  start: '开始', start_date: '开始', end: '结束', end_date: '结束', evaluation_start: '评价开始',
  evaluation_end: '评价结束', training_start: '训练开始', training_end: '训练结束', training_days: '训练交易日',
  days: '交易日', paired_days: '配对交易日',
  blocks: '回放区块', gaps: '数据缺口', initial_cash: '初始现金', estimated_work: '预计评价工作',
  data_version: '数据版本', status: '状态', primary_arm: '主方案', plan_hash: '方案校验码',
  completion_status: '完成情况', candidate_status: '候选结果', qualification_status: '预算资格', evidence: '证据',
  peak_principal_return: '峰值本金收益', trade_count: '成交笔数',
  return_on_committed_budget: '总预算收益', total_budget: '总预算', max_shares: '最高持股',
  initial_shares: '初始持股', extra_delay_bars: '额外延迟',
  actual_traded: '实际成交', filled_ratio: '成交比例', fill_ratio: '成交比例',
  funding_rejections: '资金拒单', min_cash: '最低现金', fees: '费用', utilization: '预算占用',
  window_mode: '训练窗', prior_policy: '先验', branch: '参数方案', selection_target: '参数方案',
  symbol: '标的', stress_case: '压力情形', scenario: '压力情形',
  rejected_orders: '拒单', equity_points: '净值点数',
  selection_runs: '选参运行', arms: '方案数', candidate_evaluations_upper_bound: '候选评价上限',
  qualification_runs: '预算复测运行', execution_pressure_scenarios_per_arm: '每个方案执行压力情形',
  price_pressure_scenarios_per_decision: '每个决策价格压力情形', estimated_memory_mb: '预计内存（MiB）',
  total_evaluation_budget: '总评价预算', workspace_budget_mb: '工作区上限（MiB）', max_runtime_seconds: '最长运行（秒）',
  human_untouched_verified: '人工查看独立性核验', historical_universe_verified: '历史证券池已核验',
  significance_tests_used: '使用显著性检验', descriptive_only: '仅描述性证据', gss_status: '评分对照状态',
  gss_reasons: '评分对照限制', qualification_problems: '资格复测问题',
  initial_holdings_value: '初始持仓估值', scheduled_contributions: '计划补款', external_topups: '实际到账补款',
  total_invested: '累计资金投入', peak_invested: '峰值本金', final_cash: '期末现金',
  final_equity: '期末资产', final_position: '期末持股', trade_fill_count: '独立成交笔数',
  fill_count: '成交笔数', rejection_count: '拒单笔数', insufficient_cash_rejections: '资金不足拒单',
  cash_shortfall_action: '资金短缺处理', day_end_cancel: '日内撤单', cancelled_orders: '撤单',
  starting_equity: '期初资产', ending_equity: '期末资产', starting_invested: '期初投入',
  ending_invested: '期末投入', capital_added: '期间新增资金', ending_profit: '期末盈利',
  profit_delta: '阶段盈利变化', segment_peak_principal: '阶段峰值本金', principal_return: '阶段本金收益',
  max_principal_drawdown: '阶段峰值本金回撤', initial_source: '分段起点依据',
  arm_group: '方案组', scheme: '参数方案', ttl_bars: '待成交时限', participation_rate: '参与率',
}
function labelValue(key: string): string {
  return labels[key] ?? key.replaceAll('_', ' ')
}
function enumValueLabel(key: string, value: string): string {
  const statuses: Record<string, string> = {
    complete: '完成', completed: '已完成', partial: '部分完成', cancelled: '已取消', failed: '失败',
    ready: '就绪', data_issue: '数据未就绪', budget_exhausted: '预算耗尽', timeout: '超时',
    no_candidate: '没有可评价候选', no_qualified: '没有合格候选', not_qualified: '未满足预设条件',
    qualified: '合格', execution_limited: '执行受限', unevaluable: '不可评价', not_applicable: '不适用',
    running: '运行中', queued: '排队中', limited: '有限证据', incomplete: '未完成', mixed: '部分合格',
  }
  if (/^(status|state|completion_status|candidate_status|qualification_status|evidence_status)$/.test(key)) return statuses[value] ?? value
  if (key === 'prior_policy') return ({ off: '先验关闭', daily_state_v1: '日线状态先验' } as Record<string, string>)[value] ?? value
  if (key === 'window_mode' || key === 'training_window_mode') return ({ same_price: '同价训练窗', fixed: '固定训练窗' } as Record<string, string>)[value] ?? value
  if (key === 'scheme' || key === 'branch' || key === 'selection_target') return ({ fixed: '固定参数', adaptive: '自适应参数', both: '固定与自适应' } as Record<string, string>)[value] ?? value
  if (key === 'kind') return ({ qualification: '资格复测', optimization: '参数选择', stress: '压力测试', synthetic: '人造价格压力', gss: '评分对照', ledger: '账本' } as Record<string, string>)[value] ?? value
  if (key === 'strategy') return ({ grid: '网格', buy_hold: '被动持有', scheduled_funding_hold: '同资金被动持有' } as Record<string, string>)[value] ?? value
  if (key === 'cash_shortfall_action') return ({ cancel_remainder: '撤销剩余委托，不追加资金', cancel_remainder_without_cash_resize: '撤销剩余委托，不擅自改预算' } as Record<string, string>)[value] ?? value
  if (key === 'initial_source') return ({ prior_equity_point: '前一时点资产', explicit_initial_equity: '预检初始资产', first_observed_equity: '首个观测资产' } as Record<string, string>)[value] ?? value
  return value
}
function metricValue(key: string, value: unknown): string {
  if (typeof value === 'string') return enumValueLabel(key, value)
  if (typeof value !== 'number' || !Number.isFinite(value)) return textValue(value)
  if (/return|drawdown|ratio|rate/i.test(key) && Math.abs(value) <= 2) return percent(value)
  if (/amount|budget|cash|fee|profit|equity|principal|invest|capital/i.test(key)) return money(value)
  return value.toLocaleString('zh-CN', { maximumFractionDigits: 3 })
}

function statusTone(value: unknown): ValidationTone {
  const text = String(value ?? '').toLowerCase()
  if (/(complete|completed|ready|qualified|success|passed)/.test(text)) return 'green'
  if (/(failed|error|blocked|invalid|risk_exceeded)/.test(text)) return 'red'
  if (/(partial|incomplete|limited|warning|insufficient|budget|data_issue)/.test(text)) return 'amber'
  if (/(running|queued|planned|pending)/.test(text)) return 'blue'
  return 'neutral'
}

function planStatusLabel(value: unknown) {
  const text = String(value ?? '').toLowerCase()
  return ({
    ready: '方案可提交', planned: '方案已生成', valid: '方案可提交', frozen: '方案已冻结',
    blocked: '存在阻断项', invalid: '方案无效', error: '方案检查失败', data_not_ready: '数据不足',
    data_issue: '数据未就绪', budget_exceeded: '预算或内存不足', insufficient: '历史不足', rejected: '方案被拒绝',
  } as Record<string, string>)[text] ?? (value ? String(value) : '方案待检查')
}

function dateRange(row: JsonObject): string {
  const dates = objectValue(row.dates)
  const start = firstValue(row, ['evaluation_start', 'evaluation_start_date', 'oos_start', 'start_date'])
    ?? firstValue(dates, ['evaluation_start', 'start', 'start_date'])
  const end = firstValue(row, ['evaluation_end', 'evaluation_end_date', 'oos_end', 'end_date'])
    ?? firstValue(dates, ['evaluation_end', 'end', 'end_date'])
  return start || end ? `${textValue(start)} 至 ${textValue(end)}` : '—'
}

function trainingRange(row: JsonObject): string {
  const training = objectValue(row.training)
  const start = firstValue(row, ['training_start', 'train_start']) ?? firstValue(training, ['start', 'start_date'])
  const end = firstValue(row, ['training_end', 'train_end']) ?? firstValue(training, ['end', 'end_date'])
  return start || end ? `${textValue(start)} 至 ${textValue(end)}` : '—'
}

function blockSummary(blocks: unknown): string {
  if (Array.isArray(blocks)) return blocks.length ? `${blocks.length} 段 · ${blocks.map(item => {
    if (typeof item === 'string') return item
    const row = objectValue(item)
    const start = firstValue(row, ['start', 'start_date', 'evaluation_start'])
    const end = firstValue(row, ['end', 'end_date', 'evaluation_end'])
    return start || end ? `${textValue(start)}–${textValue(end)}` : compactSummary(item)
  }).join('；')}` : '无分段证据'
  if (objectValue(blocks)) return compactSummary(blocks)
  return textValue(blocks)
}

function primarySegmentSummary(groups: ValidationSegmentGroup[], fallback: unknown): string {
  const summaries = groups.map(group => {
    const start = firstValue(group.segments[0] ?? null, ['start', 'start_date'])
    const end = firstValue(group.segments.at(-1) ?? null, ['end', 'end_date'])
    const range = start || end ? ` · ${textValue(start)}–${textValue(end)}` : ''
    return `${group.symbol} ${group.segments.length} 段${range}`
  })
  return summaries.length ? summaries.join('；') : blockSummary(fallback)
}

function gapDescription(value: unknown): string {
  const row = objectValue(value)
  const message = firstValue(row, ['message', 'reason', 'detail', 'summary'])
  const facts: string[] = []
  if (typeof row?.available_days === 'number') facts.push(`连续可用 ${row.available_days} 日`)
  if (typeof row?.additional_verified_days === 'number') facts.push(`尚需额外 ${row.additional_verified_days} 日`)
  if (typeof row?.required_days === 'number') facts.push(`至少需要 ${row.required_days} 日`)
  if (row?.start || row?.end) facts.push(`${textValue(row.start)} 至 ${textValue(row.end)}`)
  if (typeof message === 'string' && message.trim()) return message.trim() + (facts.length ? `（${facts.join('；')}）` : '')
  return facts.length ? facts.join('；') : blockSummary(value)
}

function ValidationPlanCard({ plan }: { plan: unknown }) {
  const record = validationPlanOf(plan)
  const rows = validationPlanSymbols(record)
  const gaps = validationEvidenceGaps({ plan: record })
  const estimatedWork = objectValue(record.estimated_work)
  return <section className='research-validation-plan' aria-label='完整验证方案'>
    <div className='research-validation-plan-head'>
      <div><small>方案状态</small><strong>{planStatusLabel(record.status)}</strong></div>
      <div><small>主方案</small><strong>{validationPrimaryArm({ plan: record })}</strong></div>
      <div><small>标的数</small><strong>{rows.length}</strong></div>
      <div><small>方案校验码</small><code title={String(record.plan_hash ?? '')}>{String(record.plan_hash ?? '—').slice(0, 18) || '—'}</code></div>
    </div>
    {estimatedWork && <div className='research-validation-plan-work'>{Object.entries(estimatedWork).filter(([, value]) => value == null || typeof value !== 'object').map(([key, value]) => <span key={key}><small>{labelValue(key)}</small><strong>{metricValue(key, value)}</strong></span>)}</div>}
    {rows.length ? <div className='research-validation-plan-list'>{rows.map(row => <article key={row.symbol}>
      <div className='research-validation-plan-title'><strong>{row.symbol}</strong><Badge tone={statusTone(row.status)}>{planStatusLabel(row.status)}</Badge></div>
      <dl>
        <div><dt>评价区间</dt><dd>{dateRange(row)}</dd></div>
        <div><dt>训练区间</dt><dd>{trainingRange(row)}</dd></div>
        <div><dt>训练交易日</dt><dd>{textValue(row.training_days ?? objectValue(row.training)?.days)}</dd></div>
        <div><dt>初始现金</dt><dd>{typeof row.initial_cash === 'number' ? money(row.initial_cash) + ' 元' : textValue(row.initial_cash)}</dd></div>
        <div><dt>预计评价工作</dt><dd title={textValue(row.estimated_work)}>{compactSummary(row.estimated_work)}</dd></div>
        <div><dt>数据版本</dt><dd className='source-version'>{textValue(row.data_version ?? row.requested_data_version)}</dd></div>
      </dl>
      <p className='research-validation-blocks'><span>分段回放</span>{blockSummary(row.blocks)}</p>
      {Array.isArray(row.gaps) && row.gaps.length > 0 && <div className='inline-warning research-validation-gap-list'><strong>数据缺口</strong><ul>{row.gaps.map((gap, index) => <li key={index}>{gapDescription(gap)}</li>)}</ul></div>}
    </article>)}</div> : <p className='field-hint'>方案没有返回逐标的信息；请重新检查。</p>}
    {gaps.length > 0 && <div className='research-validation-gaps'><strong>需要知悉的数据限制</strong><ul>{gaps.slice(0, 8).map((gap, index) => <li key={index}>{gap}</li>)}</ul></div>}
  </section>
}

export function ResearchValidationLauncher({ buildResearch, initialSymbol, initialShares, datesExplicit = false, enabled, disabledReason, onSubmitted }: ResearchValidationLauncherProps) {
  const tasks = useTasks()
  const errorText = useErrorMessage()
  const previousInitial = useRef('')
  const previousDatesExplicit = useRef(Boolean(datesExplicit))
  const previousResearchDate = useRef('')
  const actionRevision = useRef(0)
  const draftKeyRef = useRef('')
  const previewDraftKeyRef = useRef('')
  const [symbolText, setSymbolText] = useState(initialSymbol)
  const [historyPolicy, setHistoryPolicy] = useState<ValidationHistoryPolicy>(datesExplicit ? 'explicit' : 'auto')
  const [familyId, setFamilyId] = useState('')
  const [maxRuntimeSeconds, setMaxRuntimeSeconds] = useState(86400)
  const [accounts, setAccounts] = useState<Record<string, AccountDraft>>({})
  const [preview, setPreview] = useState<unknown>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [submitBusy, setSubmitBusy] = useState(false)
  const [error, setError] = useState('')

  const research = researchV5FromDraft(buildResearch())
  const researchContext = objectValue(research.research_context)
  const researchDateKey = String(researchContext?.start_date ?? '') + '\u0000' + String(researchContext?.end_date ?? '')
  const parsed = parseValidationSymbols(symbolText)
  const symbolKey = parsed.symbols.join('|')
  const minimumShares = Math.max(100, Math.ceil(Math.max(0, initialShares) / 100) * 100)
  const draftKey = validationDraftKey({
    research,
    symbols: parsed.symbols,
    invalidSymbols: parsed.invalid,
    historyPolicy,
    accounts,
    familyId,
    maxRuntimeSeconds,
  })
  draftKeyRef.current = draftKey
  const plan = validationPlanOf(preview)
  const maySubmit = canSubmitValidationPlan(preview)

  useEffect(() => {
    if (initialSymbol && (!previousInitial.current || symbolText === previousInitial.current || !symbolText.trim())) setSymbolText(initialSymbol)
    previousInitial.current = initialSymbol
  }, [initialSymbol])
  useEffect(() => {
    const dateChanged = Boolean(previousResearchDate.current) && previousResearchDate.current !== researchDateKey
    const becameExplicit = datesExplicit && !previousDatesExplicit.current
    previousResearchDate.current = researchDateKey
    previousDatesExplicit.current = Boolean(datesExplicit)
    if (datesExplicit && (dateChanged || becameExplicit)) setHistoryPolicy('explicit')
  }, [datesExplicit, researchDateKey])
  useEffect(() => {
    setAccounts(current => {
      const next: Record<string, AccountDraft> = {}
      for (const symbol of parsed.symbols) {
        next[symbol] = current[symbol] ?? {
          total_budget: null,
          max_shares: null,
          contributions: [],
        }
      }
      return next
    })
  }, [symbolKey, minimumShares])
  useEffect(() => {
    if (!previewDraftKeyRef.current || previewDraftKeyRef.current === draftKey) return
    previewDraftKeyRef.current = ''
    setPreview(null)
    setError('')
  }, [draftKey])

  const updateAccount = (symbol: string, patch: Partial<AccountDraft>) => {
    setAccounts(current => ({ ...current, [symbol]: { ...current[symbol], ...patch, contributions: patch.contributions ?? current[symbol]?.contributions ?? [] } }))
  }
  const addContribution = (symbol: string) => {
    const current = accounts[symbol]
    if (!current) return
    updateAccount(symbol, { contributions: [...current.contributions, { available_at: '', amount: null }] })
  }
  const updateContribution = (symbol: string, index: number, patch: Partial<ValidationContributionInput>) => {
    const current = accounts[symbol]
    if (!current) return
    updateAccount(symbol, { contributions: current.contributions.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item) })
  }
  const removeContribution = (symbol: string, index: number) => {
    const current = accounts[symbol]
    if (!current) return
    updateAccount(symbol, { contributions: current.contributions.filter((_, itemIndex) => itemIndex !== index) })
  }
  const buildRequest = (researchValue: Record<string, unknown>, planHash?: string) => buildValidationRequest({
    research: researchValue,
    symbols: parsed.symbols,
    historyPolicy,
    accounts,
    initialShares,
    familyId,
    maxRuntimeSeconds,
    planHash,
    requirePlanHash: Boolean(planHash),
    requestedDataVersions: requestedVersionsFromPlan(plan),
  })
  const checkPlan = async () => {
    setError('')
    if (!enabled) { setError(disabledReason || '当前研究草稿还不能开始完整验证。'); return }
    if (parsed.invalid.length) { setError('以下标的代码无效：' + parsed.invalid.join('、')); return }
    const built = buildRequest(research)
    if (built.error || !built.request) { setError(built.error || '验证方案不完整。'); return }
    const revision = ++actionRevision.current
    const ticket: ValidationAsyncTicket = { revision, phase: 'preview', draftKey: draftKeyRef.current, requestKey: '' }
    const current = () => validationAsyncTicketCurrent(ticket, actionRevision.current, draftKeyRef.current)
    setPreviewBusy(true)
    try {
      const configurationRevision = await flushConfiguration()
      if (!current()) return
      const freshResearch = researchV5FromDraft(buildResearch(configurationRevision))
      const request = buildRequest(freshResearch)
      if (request.error || !request.request) throw new Error(request.error || '验证方案不完整。')
      ticket.requestKey = validationRequestKey(request.request, false)
      if (!current()) return
      const response = await post<unknown>('/research-validations/preview', request.request)
      if (!current()) return
      previewDraftKeyRef.current = ticket.draftKey
      setPreview(response)
    } catch (cause) {
      if (current()) {
        previewDraftKeyRef.current = ''
        setPreview(null)
        setError(errorText(cause))
      }
    } finally {
      if (revision === actionRevision.current) setPreviewBusy(false)
    }
  }
  const submit = async () => {
    setError('')
    if (!maySubmit || previewDraftKeyRef.current !== draftKeyRef.current) { setError('请先按当前研究设置检查并冻结验证方案。'); return }
    const hash = String(plan.plan_hash ?? '').trim()
    const built = buildRequest(research, hash)
    if (built.error || !built.request) { setError(built.error || '验证方案不完整。'); return }
    const revision = ++actionRevision.current
    const ticket: ValidationAsyncTicket = { revision, phase: 'submit', draftKey: draftKeyRef.current, requestKey: '' }
    const current = () => validationAsyncTicketCurrent(ticket, actionRevision.current, draftKeyRef.current)
    setSubmitBusy(true)
    try {
      const configurationRevision = await flushConfiguration()
      if (!current()) return
      const freshResearch = researchV5FromDraft(buildResearch(configurationRevision))
      const request = buildRequest(freshResearch, hash)
      if (request.error || !request.request) throw new Error(request.error || '验证方案不完整。')
      ticket.requestKey = validationRequestKey(request.request)
      if (!current()) return
      const response = await tasks.run('完整研究验证', () => post<{ job_id: string }>('/research-validations', request.request))
      if (!current()) {
        setError('任务已经提交，但提交期间的验证设置已改变。请从任务中心打开该任务；当前界面不会自动切换结果。')
        void tasks.refresh()
        return
      }
      onSubmitted(response.job_id)
    } catch (cause) {
      if (current()) setError(errorText(cause))
    } finally {
      if (revision === actionRevision.current) setSubmitBusy(false)
    }
  }

  return <Card className='research-validation-launch-card' title='完整研究验证' subtitle='正式入口：冻结实际数据区间后，检验八种训练窗、先验与参数方案，再用逐标的真实预算检查执行资格。' action={<Badge tone='blue'>正式入口</Badge>}>
    <div className='research-validation-form-grid'>
      <Field label='标的池' hint='用逗号、空格或换行分隔；自动补全沪深市场后缀。'>
        <TextInput value={symbolText} onChange={event => setSymbolText(event.target.value.toUpperCase())} placeholder='600000.SH, 510300.SH' />
      </Field>
      <Field label='历史区间' hint={historyPolicy === 'auto' ? '按每只标的的可用连续历史自动决定训练与评价区间；手动修改上方研究日期会自动切换为使用当前日期。' : '严格使用上方研究草稿中的开始和结束日期。'}>
        <div className='research-validation-toggle' role='group' aria-label='历史区间策略'>
          <button type='button' className={historyPolicy === 'auto' ? 'selected' : ''} aria-pressed={historyPolicy === 'auto'} onClick={() => setHistoryPolicy('auto')}>按数据自动规划</button>
          <button type='button' className={historyPolicy === 'explicit' ? 'selected' : ''} aria-pressed={historyPolicy === 'explicit'} onClick={() => setHistoryPolicy('explicit')}>使用当前日期</button>
        </div>
      </Field>
    </div>
    {parsed.invalid.length > 0 && <p className='inline-warning'>无法识别：{parsed.invalid.join('、')}</p>}
    <div className='research-validation-account-list'>
      {parsed.symbols.map(symbol => {
        const account = accounts[symbol]
        return <article className='research-validation-account' key={symbol}>
          <div className='research-validation-account-head'><strong>{symbol}</strong><span>独立账户 · 初始持股 {initialShares.toLocaleString()} 股</span></div>
          <div className='form-grid two'>
            <NumField label='总预算（必填）' value={account?.total_budget ?? null} min={.01} step='100' suffix='元' onChange={value => updateAccount(symbol, { total_budget: value })} hint='包括初始持仓估值、初始现金和全部计划补款。' />
        <NumField label='最高持股（必填）' value={account?.max_shares ?? null} min={minimumShares} step='100' suffix='股' onChange={value => updateAccount(symbol, { max_shares: value })} hint={`请填写不低于初始持股的有限上限，并按 100 股对齐；系统不代为设定。`} />
          </div>
          <div className='research-validation-contributions'>
            <div><strong>计划补款</strong><span>可选；补款只会在指定到账时间进入账户，不随成交自动追加。</span></div>
            {account?.contributions.map((contribution, index) => <div className='research-validation-contribution-row' key={index}>
              <Field label='到账时间'><TextInput type='datetime-local' value={contribution.available_at} onChange={event => updateContribution(symbol, index, { available_at: event.target.value })} /></Field>
              <NumField label='金额' value={contribution.amount} min={.01} step='100' suffix='元' onChange={value => updateContribution(symbol, index, { amount: value })} />
              <Button type='button' variant='quiet' aria-label={`删除 ${symbol} 的计划补款`} onClick={() => removeContribution(symbol, index)}><Trash2 size={14} /></Button>
            </div>)}
            <Button type='button' variant='quiet' onClick={() => addContribution(symbol)}><Plus size={14} />添加计划补款</Button>
          </div>
        </article>
      })}
    </div>
    <Disclosure id='research-validation-options' title='任务与证据选项' summary={`最长 ${Math.round(maxRuntimeSeconds / 3600)} 小时 · ${familyId ? '加入研究族' : '不指定研究族'}`}>
      <div className='form-grid two'>
        <Field label='研究族（可选）' hint='同一研究族会保留父研究、方法版本与累计试验记录。'><TextInput value={familyId} onChange={event => setFamilyId(event.target.value)} placeholder='例如 quality-study-01' /></Field>
        <NumField label='最长运行时间（秒）' value={maxRuntimeSeconds} min={60} max={86400} step='60' onChange={value => setMaxRuntimeSeconds(value ?? 86400)} />
      </div>
    </Disclosure>
    {!enabled && <p className='field-hint'>{disabledReason || '请先完成当前研究草稿的日期和参数检查。'}</p>}
    {error && <div className='notice error' role='alert'><span>{error}</span></div>}
    <div className='research-validation-actions'>
      <Button variant='secondary' busy={previewBusy} disabled={!enabled || submitBusy} onClick={() => void checkPlan()}><ShieldCheck size={15} />检查验证方案</Button>
      <Button busy={submitBusy} disabled={!enabled || !maySubmit || previewBusy} onClick={() => void submit()}><CheckCircle2 size={15} />启动完整验证</Button>
      <span>{maySubmit ? '方案已冻结；修改草稿或预算后需重新检查。' : preview ? planStatusLabel(plan.status) + '；请处理后重新检查。' : '请先检查方案，确认数据缺口和预计工作量。'}</span>
    </div>
    {Boolean(preview) && <ValidationPlanCard plan={preview} />}
  </Card>
}

function EvidenceTable({ value, columns }: { value: unknown; columns?: string[] }) {
  const object = objectValue(value)
  const rows: unknown[] = Array.isArray(value)
    ? value
    : object
      ? Object.entries(object).map(([key, item]) => {
          const row = objectValue(item)
          return row ? { section: key, ...row } : { section: key, value: item }
        })
      : []
  if (!rows.length) return <p className='field-hint'>暂无可展开的明细。</p>
  const normalizedRows = rows.map(raw => {
    const original = objectValue(raw) ?? { value: raw }
    const result = objectValue(original.result)
    return {
      ...(objectValue(original.strategy_metrics) ?? objectValue(result?.strategy_metrics) ?? {}),
      ...(objectValue(original.bounded_ledger) ?? objectValue(result?.bounded_ledger) ?? {}),
      ...original,
    }
  })
  const preferred = ['symbol', 'arm_group', 'scheme', 'status', 'stress_case', 'scenario', 'peak_principal_return', 'max_principal_drawdown', 'return_on_committed_budget', 'trade_count', 'rejected_orders', 'rejection_count', 'start', 'end']
  const available = [...new Set(normalizedRows.flatMap(row => Object.keys(row)))]
  const keys = columns
    ? columns.filter(key => available.includes(key))
    : [...preferred.filter(key => available.includes(key)), ...available.filter(key => !preferred.includes(key))].slice(0, 9)
  return <div className='table-wrap scroll-table'><table>
    <thead><tr>{keys.map(key => <th key={key}>{labelValue(key)}</th>)}</tr></thead>
    <tbody>{normalizedRows.slice(0, 200).map((row, index) => {
      return <tr key={String(row.id ?? row.block_id ?? index)}>{keys.map(key => <td key={key} title={textValue(row[key])}>{metricValue(key, row[key])}</td>)}</tr>
    })}</tbody>
  </table>{rows.length > 200 && <p className='field-hint'>另有 {rows.length - 200} 项，请导出完整结果查看。</p>}</div>
}

function SegmentEvidenceTable({ groups, label }: { groups: ValidationSegmentGroup[]; label: string }) {
  const rows = groups.flatMap(group => group.segments.map((row, index) => ({ key: group.key + ':' + index, row })))
  if (!rows.length) return <p className='field-hint'>暂无可展开的分段明细。</p>
  return <div className='table-wrap scroll-table research-validation-segment-table-wrap'>
    <table className='research-validation-segment-table' aria-label={label}>
      <thead><tr>{segmentColumns.map(key => <th key={key}>{labelValue(key)}</th>)}</tr></thead>
      <tbody>{rows.slice(0, 200).map(({ key, row }) => <tr key={key}>
        {segmentColumns.map(field => <td key={field} data-field={field} title={textValue(row[field])}>{metricValue(field, row[field])}</td>)}
      </tr>)}</tbody>
    </table>
    {rows.length > 200 && <p className='field-hint'>另有 {rows.length - 200} 段，请导出完整结果查看。</p>}
  </div>
}

function EvidenceObject({ value }: { value: unknown }) {
  const item = objectValue(value)
  if (!item) return <p className='field-hint'>{textValue(value)}</p>
  const entries = Object.entries(item)
  return <div className='research-validation-evidence-grid'>{entries.map(([key, raw]) => <div key={key}>
    <small>{labelValue(key)}</small>
    {Array.isArray(raw) || objectValue(raw) ? <details><summary>{Array.isArray(raw) ? `${raw.length} 项` : `${Object.keys(objectValue(raw) ?? {}).length} 项`}</summary>
      {Array.isArray(raw) ? <EvidenceTable value={raw} /> : <pre className='json-preview'>{JSON.stringify(raw, null, 2)}</pre>}
    </details> : <strong>{metricValue(key, raw)}</strong>}
  </div>)}</div>
}

function primaryMetrics(result: JsonObject): Array<[string, unknown]> {
  const metrics = validationPrimaryMetrics(result)
  return Object.entries(metrics).filter(([key, value]) => /return|drawdown|trade|budget|cash|fee|profit|equity|ratio|rejection|utilization/i.test(key) && (typeof value === 'number' || typeof value === 'string')).slice(0, 12)
}

type EvidenceViewState = { status: 'idle' | 'saving' | 'recorded' | 'error'; error?: string }
const evidenceViewRequests = new Map<string, Promise<void>>()
const evidenceSections: ValidationEvidenceSection[] = ['overview', 'cells', 'blocks', 'execution', 'pressure', 'scoring', 'audit']
const segmentColumns = ['symbol', 'start', 'end', 'days', 'principal_return', 'max_principal_drawdown', 'capital_added', 'starting_equity', 'ending_equity', 'status']
const accountColumns = ['symbol', 'total_budget', 'total_invested', 'initial_cash', 'final_cash', 'final_equity', 'fill_ratio', 'total_fees', 'trade_count', 'rejection_count', 'insufficient_cash_rejections', 'cancelled_orders', 'cash_shortfall_action']

function TrackedDisclosure({ id, section, viewState, onOpen, onRetry, children, ...props }: {
  id: string
  section: ValidationEvidenceSection
  viewState?: EvidenceViewState
  onOpen: () => void
  onRetry: () => void
  children: ReactNode
  title: string
  summary?: ReactNode
}) {
  const openIfClosed = (target: EventTarget | null) => {
    if (!(target instanceof Element)) return
    const summary = target.closest('summary')
    const details = summary?.closest('details') as HTMLDetailsElement | null
    if (details?.dataset.disclosure === id && !details.open) onOpen()
  }
  return <div className='research-validation-tracked-section' data-evidence-section={section} onClickCapture={event => openIfClosed(event.target)}
    onKeyDownCapture={event => {
      if ((event.key === 'Enter' || event.key === ' ') && (event.target as Element | null)?.closest('summary')) openIfClosed(event.target)
    }}>
    <Disclosure id={id} {...props}>
      {viewState?.status === 'saving' && <div className='research-validation-section-view' role='status'>正在记录“{props.title}”首次展开…</div>}
      {viewState?.status === 'error' && <div className='research-validation-section-view error' role='alert'><span>首次展开记录失败：{viewState.error}</span><button type='button' className='text-link' onClick={onRetry}>重试记录</button></div>}
      {children}
    </Disclosure>
  </div>
}

export function ResearchValidationReport({ result, resultId, onRefresh }: { result: unknown; resultId?: string; onRefresh?: () => void }) {
  const item = objectValue(result) ?? {}
  const id = String(resultId ?? item.id ?? '')
  const states = validationOverviewItems(item)
  const gaps = validationEvidenceGaps(item)
  const schemes = validationSchemeEntries(item)
  const primary = primaryMetrics(item)
  const audit = objectValue(item.audit)
  const plan = item.plan ?? item.validation_plan ?? audit?.plan ?? item
  const planRows = validationPlanSymbols(validationPlanOf(plan))
  const blocks = validationBlocks(item)
  const segmentGroups = validationSegmentGroups(item)
  const accountRows = validationAccountRows(item)
  const initialViewStates = Object.fromEntries(evidenceSections.map(section => [section,
    validationEvidenceRecorded(item, section) ? { status: 'recorded' } : { status: 'idle' },
  ])) as Record<ValidationEvidenceSection, EvidenceViewState>
  const [viewStates, setViewStates] = useState<Record<ValidationEvidenceSection, EvidenceViewState>>(initialViewStates)
  const viewStatesRef = useRef(viewStates)
  const errorText = useErrorMessage()

  const setViewState = useCallback((section: ValidationEvidenceSection, state: EvidenceViewState) => {
    const next = { ...viewStatesRef.current, [section]: state }
    viewStatesRef.current = next
    setViewStates(next)
  }, [])

  const recordSection = useCallback((section: ValidationEvidenceSection, force = false) => {
    if (!id) return
    const key = id + ':' + section
    if (!force && (validationEvidenceRecorded(item, section)
      || viewStatesRef.current[section]?.status === 'recorded'
      || evidenceViewRequests.has(key))) return
    setViewState(section, { status: 'saving' })
    const request = post('/results/' + encodeURIComponent(id) + '/evidence-views', { section }).then(() => undefined)
    evidenceViewRequests.set(key, request)
    void request.then(() => setViewState(section, { status: 'recorded' })).catch(cause => {
      evidenceViewRequests.delete(key)
      setViewState(section, { status: 'error', error: errorText(cause) })
    })
  }, [errorText, id, item, setViewState])

  useEffect(() => {
    const now = { ...viewStatesRef.current }
    let changed = false
    for (const section of evidenceSections) {
      if (validationEvidenceRecorded(item, section) && now[section]?.status !== 'recorded') {
        now[section] = { status: 'recorded' }
        changed = true
      }
    }
    if (changed) {
      viewStatesRef.current = now
      setViewStates(now)
    }
  }, [item])

  useEffect(() => {
    if (!id) return
    recordSection('overview')
  }, [id, recordSection])

  const [exportBusy, setExportBusy] = useState('')
  const [exportError, setExportError] = useState('')
  const exportResult = async (kind: 'validation' | 'equity' | 'trades' | 'json') => {
    if (!id || exportBusy) return
    setExportBusy(kind)
    setExportError('')
    try {
      const extension = kind === 'json' ? 'json' : 'csv'
      await download('/results/' + encodeURIComponent(id) + '/export?kind=' + encodeURIComponent(kind), id + '-' + kind + '.' + extension)
    } catch (cause) {
      setExportError(errorText(cause))
    } finally {
      setExportBusy('')
    }
  }

  const title = objectValue(item.plan)?.title ?? item.title
  return <>
    <Card className='research-validation-report' title={typeof title === 'string' && title ? title : '完整研究验证'} subtitle={`${planRows.map(row => row.symbol).join('、') || textValue(item.symbol)} · ${textValue(item.created_at)}`} action={<div className='research-validation-report-actions'><Badge tone='blue'>统一完整验证</Badge>{onRefresh && <Button variant='quiet' onClick={onRefresh}><RefreshCw size={14} />刷新</Button>}</div>}>
      <div className='research-validation-primary-arm'><ShieldCheck size={16} /><span>{validationPrimaryArm(item)}</span><small>所有方案在检查时冻结，测试后不会换用表现更好的方案作为主方案。</small></div>
      <div className='research-validation-status-grid'>
        {(['completion', 'candidate', 'qualification', 'evidence'] as const).map(kind => {
          const state = states[kind]
          const labels: Record<typeof kind, string> = { completion: '完成情况', candidate: '候选结果', qualification: '预算资格', evidence: '证据' }
          return <div className='research-validation-status' key={kind}>
            <small>{labels[kind]}</small><Badge tone={state.tone}>{state.label}</Badge>{state.note && <span>{state.note}</span>}
          </div>
        })}
      </div>
      {gaps.length > 0 && <div className='research-validation-gaps'><strong>证据限制与数据缺口</strong><ul>{gaps.slice(0, 10).map((gap, index) => <li key={index}>{gap}</li>)}</ul></div>}
      <div className='research-validation-view-status' role='status'>
        {viewStates.overview?.status === 'recorded' ? <><CheckCircle2 size={13} />首次查看已记录</> : viewStates.overview?.status === 'saving' ? '正在记录首次查看…' : viewStates.overview?.status === 'error' ? <>查看记录失败：{viewStates.overview.error}<button className='text-link' onClick={() => recordSection('overview', true)}>重试</button></> : '报告已打开，正在记录首次查看…'}
      </div>
    </Card>
    {primary.length > 0 && <Card title='主方案账户摘要' subtitle='各标的独立记账；这里列出首个标的主方案的收益、回撤与预算占用，完整逐标的账户见方案对照和审计。'>
      <div className='research-validation-primary-metrics'>{primary.map(([key, value]) => <div key={key}><small>{labelValue(key)}</small><strong>{metricValue(key, value)}</strong></div>)}</div>
    </Card>}
    <Card title='实际数据方案' subtitle='每只标的独立使用可验证区间；自动规划只读取数据可用性，不按收益挑选区间。'>
      {planRows.length ? <div className='research-validation-plan-compact'>{planRows.map(row => <div key={row.symbol}>
        <strong>{row.symbol}</strong><span>{dateRange(row)}</span><small>{textValue(row.training_days ?? objectValue(row.training)?.days)} 个训练交易日 · {Array.isArray(row.blocks) ? row.blocks.length : 0} 个回放区块 · 初始现金 {typeof row.initial_cash === 'number' ? money(row.initial_cash) + ' 元' : textValue(row.initial_cash)}</small>
        {Array.isArray(row.gaps) && row.gaps.length > 0 && <div className='inline-warning research-validation-gap-list'><strong>数据缺口</strong><ul>{row.gaps.map((gap, index) => <li key={index}>{gapDescription(gap)}</li>)}</ul></div>}
      </div>)}</div> : <Empty title='方案未返回逐标的信息' />}
    </Card>
    <Card title='导出核验证据' subtitle='页面只加载轻量摘要。选择导出后才展开该结果保存时的冻结账本，不会套用之后更新的报告状态。'>
      <div className='button-row research-validation-export-actions'>
        <Button variant='quiet' busy={exportBusy === 'validation'} disabled={!id || Boolean(exportBusy)} onClick={() => void exportResult('validation')}><Download size={14} />验证明细 CSV</Button>
        <Button variant='quiet' busy={exportBusy === 'equity'} disabled={!id || Boolean(exportBusy)} onClick={() => void exportResult('equity')}><Download size={14} />资金曲线 CSV</Button>
        <Button variant='quiet' busy={exportBusy === 'trades'} disabled={!id || Boolean(exportBusy)} onClick={() => void exportResult('trades')}><Download size={14} />成交流水 CSV</Button>
        <Button variant='quiet' busy={exportBusy === 'json'} disabled={!id || Boolean(exportBusy)} onClick={() => void exportResult('json')}><Download size={14} />完整 JSON（较慢）</Button>
      </div>
      <p className='field-hint'>导出文件会按历史检查点还原当时的结果；JSON 适合完整归档，CSV 适合逐项核验验证状态、资金曲线和成交记录。</p>
      {exportError && <div className='notice error' role='alert'><span>导出失败：{exportError}</span></div>}
    </Card>
    <TrackedDisclosure id={`research-validation-cells-${id}`} section='cells' title='八方案对照' summary={`${schemes.length} 个方案 · ${states.candidate.label}`}
      viewState={viewStates.cells} onOpen={() => recordSection('cells')} onRetry={() => recordSection('cells', true)}>
      {schemes.length ? <div className='research-validation-scheme-grid'>{schemes.map(scheme => {
        const summary = validationCellSummary(scheme.value)
        const metrics = Object.entries(summary.metrics).filter(([, value]) => typeof value === 'number' || typeof value === 'string').slice(0, 8)
        return <article key={scheme.key}>
          <div><strong>{scheme.label}</strong><Badge tone={summary.completion.tone}>{summary.completion.label}</Badge></div>
          <div className='research-validation-scheme-status'><Badge tone={summary.qualification.tone}>{summary.qualification.label}</Badge>{summary.qualification.note && <small>{summary.qualification.note}</small>}</div>
          {metrics.length ? <dl>{metrics.map(([key, value]) => <div key={key}><dt>{labelValue(key)}</dt><dd>{metricValue(key, value)}</dd></div>)}</dl> : <p className='field-hint'>方案明细尚未完成。</p>}
          <details><summary>查看方案原始证据</summary><EvidenceObject value={scheme.value} /></details>
        </article>
      })}</div> : <Empty title='方案结果尚未生成' detail='任务完成或产生部分结果后，这里会显示冻结的八方案明细。' />}
      {Array.isArray(item.cells) && item.cells.length > 0 && <details className='research-validation-cells-raw'><summary>查看全部验证单元（{item.cells.length} 项）</summary><EvidenceTable value={item.cells} /></details>}
    </TrackedDisclosure>
    <TrackedDisclosure id={`research-validation-blocks-${id}`} section='blocks' title='分段回放' summary={primarySegmentSummary(segmentGroups.primary, blocks)}
      viewState={viewStates.blocks} onOpen={() => recordSection('blocks')} onRetry={() => recordSection('blocks', true)}>
      <p className='field-hint'>区块收益来自同一连续账户，资金流按实际到账时间计入，不拼接重置账户后的收益率。</p>
      {segmentGroups.primary.length > 0 ? <>
        <h4>主方案实际分段</h4>
        <p className='field-hint'>每个标的显示主自适应参数方案在连续账户中的实际收益、回撤与资金流。</p>
        <SegmentEvidenceTable groups={segmentGroups.primary} label='主方案实际分段明细' />
      </> : <>
        <p className='inline-warning'>主方案实际分段收益尚未生成；以下仅是冻结计划的回放日期，不能作为收益证据。</p>
        <EvidenceTable value={blocks} columns={['symbol', 'start', 'end', 'days']} />
      </>}
      {segmentGroups.alternatives.length > 0 && <details className='research-validation-alternative-segments'>
        <summary>查看其他方案实际分段（{segmentGroups.alternatives.length} 组）</summary>
        {segmentGroups.alternatives.map(group => <section key={group.key}>
          <h4>{group.label}</h4>
          <SegmentEvidenceTable groups={[group]} label={group.label + '实际分段明细'} />
        </section>)}
      </details>}
    </TrackedDisclosure>
    <TrackedDisclosure id={`research-validation-execution-${id}`} section='execution' title='预算与成交限制' summary={accountRows.length ? `${accountRows.length} 个标的主方案 · ${accountRows.reduce((sum, row) => sum + (typeof row.rejection_count === 'number' ? row.rejection_count : 0), 0)} 次拒单` : compactSummary(item.funding ?? item.execution ?? item.qualification)}
      viewState={viewStates.execution} onOpen={() => recordSection('execution')} onRetry={() => recordSection('execution', true)}>
      <p className='field-hint'>这里分别记录预算占用、成交能力、拒单与撤单。资格复测不会重新挑选参数。</p>
      {accountRows.length > 0 ? <>
        <EvidenceTable value={accountRows} columns={accountColumns} />
        <p className='field-hint'>成交比例按独立成交笔数除以成交与拒单总次数计算；撤单优先读取资格诊断，若只记录日内撤单规则则显示该规则；资金不足拒单单独列出，不并入一般拒单。</p>
      </> : item.funding || item.execution || item.qualification ? <EvidenceObject value={item.funding ?? item.execution ?? item.qualification} /> : <Empty title='预算与成交诊断尚未生成' detail='任务完成主方案资格复测后，这里会显示逐标的预算、资金流、成交、拒单和资金短缺证据。' />}
    </TrackedDisclosure>
    <TrackedDisclosure id={`research-validation-pressure-${id}`} section='pressure' title='压力测试' summary={compactSummary(item.pressure ?? item.stress)}
      viewState={viewStates.pressure} onOpen={() => recordSection('pressure')} onRetry={() => recordSection('pressure', true)}>
      <h4>历史执行压力</h4>
      <EvidenceTable value={item.pressure ?? item.stress} />
      {Array.isArray(item.synthetic) && item.synthetic.length > 0 && <><h4>固定参数人造价格压力</h4><p className='field-hint'>人造路径只检验该决策已选参数，不读取后续真实换参表，也不写入行情库。</p><EvidenceTable value={item.synthetic} /></>}
    </TrackedDisclosure>
    <TrackedDisclosure id={`research-validation-scoring-${id}`} section='scoring' title='评分配对与适用边界' summary={compactSummary(item.scoring ?? item.gss ?? item.etf_comparison)}
      viewState={viewStates.scoring} onOpen={() => recordSection('scoring')} onRetry={() => recordSection('scoring', true)}>
      <p className='field-hint'>评分、过滤与后续表现分开报告；重叠区间和相关标的不按独立样本累加证据。</p>
      <EvidenceObject value={item.scoring ?? item.gss ?? item.etf_comparison} />
    </TrackedDisclosure>
    <TrackedDisclosure id={`research-validation-audit-${id}`} section='audit' title='研究审计' summary={compactSummary(item.audit)}
      viewState={viewStates.audit} onOpen={() => recordSection('audit')} onRetry={() => recordSection('audit', true)}>
      <EvidenceObject value={item.audit} />
    </TrackedDisclosure>
  </>
}

export function LegacyValidationStatus() {
  return <div className='research-legacy-validation'><Badge tone='amber'>未完整验证</Badge><span>这是原有参数研究结果，没有执行完整套件、真实预算资格复测和证据审计。</span></div>
}

export function ResearchResultTitle({ item }: { item: { kind?: unknown; title?: unknown } }) {
  return <>{researchResultHistoryLabel(item)}</>
}
