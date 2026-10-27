export type ValidationTone = 'neutral' | 'blue' | 'green' | 'red' | 'amber'
export type ValidationStatusKind = 'completion' | 'candidate' | 'qualification' | 'evidence'
export type ValidationHistoryPolicy = 'auto' | 'explicit'
export type ValidationEvidenceSection = 'overview' | 'cells' | 'blocks' | 'execution' | 'pressure' | 'scoring' | 'audit'

type JsonObject = Record<string, unknown>

export interface ValidationContributionInput {
  available_at: string
  amount: number | null
}

export interface ValidationAccountInput {
  total_budget: number | null
  max_shares: number | null
  contributions?: ValidationContributionInput[]
}

export interface ValidationRequestBuildInput {
  research: JsonObject
  symbols: string[]
  historyPolicy: ValidationHistoryPolicy
  accounts: Record<string, ValidationAccountInput>
  requestedDataVersions?: Record<string, string>
  initialShares: number
  familyId?: string
  planHash?: string
  maxRuntimeSeconds?: number
  requirePlanHash?: boolean
}

export interface ValidationDraftKeyInput {
  research: JsonObject
  symbols: string[]
  historyPolicy: ValidationHistoryPolicy
  accounts: Record<string, ValidationAccountInput>
  familyId?: string
  maxRuntimeSeconds?: number
  invalidSymbols?: string[]
}

export interface ValidationAsyncTicket {
  revision: number
  phase: 'preview' | 'submit'
  draftKey: string
  requestKey: string
}

export interface ValidationStatusView {
  raw: unknown
  label: string
  tone: ValidationTone
  note: string
}

export interface ValidationPlanSymbol extends JsonObject {
  symbol: string
}

export interface ValidationSegmentGroup {
  key: string
  symbol: string
  armGroup: string
  scheme: 'fixed' | 'adaptive'
  label: string
  primary: boolean
  segments: JsonObject[]
}

const empty = (value: unknown) => value == null || (typeof value === 'string' && !value.trim())

function objectValue(value: unknown): JsonObject | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : null
}

function compactObject(value: JsonObject): JsonObject {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined))
}

function canonicalValidationValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValidationValue)
  const item = objectValue(value)
  if (!item) return value
  return Object.fromEntries(Object.keys(item).sort().flatMap(key => {
    const entry = item[key]
    return entry === undefined ? [] : [[key, canonicalValidationValue(entry)]]
  }))
}

function canonicalValidationJson(value: unknown): string {
  return JSON.stringify(canonicalValidationValue(value)) ?? 'null'
}

function canonicalContributionKey(value: ValidationContributionInput): string {
  return canonicalValidationJson({
    available_at: String(value.available_at ?? ''),
    amount: value.amount ?? null,
  })
}

/**
 * Fingerprint only inputs that make a rendered preflight stale. Presentation
 * recalculation, plan hashes and server configuration revisions are excluded.
 */
export function validationDraftKey(input: ValidationDraftKeyInput): string {
  const research = { ...input.research }
  delete research.configuration_revision
  const symbols = [...new Set(input.symbols.map(normalizeValidationSymbol).filter(Boolean))].sort()
  const accounts = Object.entries(input.accounts)
    .map(([symbol, account]) => ({
      symbol: normalizeValidationSymbol(symbol) || symbol,
      total_budget: account?.total_budget ?? null,
      max_shares: account?.max_shares ?? null,
      contributions: [...(account?.contributions ?? [])]
        .sort((left, right) => canonicalContributionKey(left).localeCompare(canonicalContributionKey(right))),
    }))
    .sort((left, right) => left.symbol.localeCompare(right.symbol))
  return canonicalValidationJson({
    research,
    symbols,
    invalid_symbols: [...new Set(input.invalidSymbols ?? [])].sort(),
    history_policy: input.historyPolicy,
    accounts,
    family_id: input.familyId?.trim() || null,
    max_runtime_seconds: input.maxRuntimeSeconds ?? 86400,
  })
}

export function validationRequestKey(value: unknown, includePlanHash = true): string {
  const request = objectValue(value)
  if (!request) return canonicalValidationJson(value)
  const normalized = { ...request }
  if (Array.isArray(normalized.symbols)) {
    normalized.symbols = [...new Set(normalized.symbols.map(normalizeValidationSymbol).filter(Boolean))].sort()
  }
  const accounts = objectValue(normalized.accounts)
  if (accounts) {
    normalized.accounts = Object.fromEntries(Object.entries(accounts).map(([symbol, account]) => {
      const row = objectValue(account)
      return [normalizeValidationSymbol(symbol) || symbol, row ? {
        ...row,
        contributions: Array.isArray(row.contributions)
          ? [...row.contributions].sort((left, right) => canonicalValidationJson(left).localeCompare(canonicalValidationJson(right)))
          : row.contributions,
      } : account]
    }))
  }
  if (!includePlanHash) delete normalized.plan_hash
  return canonicalValidationJson(normalized)
}

export function validationAsyncTicketCurrent(ticket: ValidationAsyncTicket, revision: number, draftKey: string): boolean {
  return ticket.revision === revision && ticket.draftKey === draftKey
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function integer(value: unknown, minimum = Number.MIN_SAFE_INTEGER): value is number {
  return finite(value) && Number.isSafeInteger(value) && value >= minimum
}

function firstText(source: JsonObject | null, keys: string[]): string {
  if (!source) return ''
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

/** Canonicalise one A-share or ETF code using the same rules as the API. */
export function normalizeValidationSymbol(value: unknown): string {
  let symbol = String(value ?? '').trim().toUpperCase()
  if (!symbol) return ''
  symbol = symbol.replace(/^(SH|SZ)\./, '')
  if (/^(SH|SZ)\d{6}$/.test(symbol)) symbol = symbol.slice(2) + '.' + symbol.slice(0, 2)
  if (/^\d{6}$/.test(symbol)) symbol += /^[56]/.test(symbol) ? '.SH' : '.SZ'
  if (!/^\d{6}\.(SH|SZ)$/.test(symbol)) return ''
  const code = symbol.slice(0, 6)
  if (!/^[01356]/.test(code)) return ''
  if ((/^[56]/.test(code)) !== symbol.endsWith('.SH')) return ''
  return symbol
}

export function parseValidationSymbols(value: unknown): { symbols: string[]; invalid: string[] } {
  const entries = String(value ?? '')
    .split(/[\s,，;；]+/)
    .map(item => item.trim())
    .filter(Boolean)
  const symbols: string[] = []
  const invalid: string[] = []
  for (const entry of entries) {
    const symbol = normalizeValidationSymbol(entry)
    if (!symbol) invalid.push(entry)
    else if (!symbols.includes(symbol)) symbols.push(symbol)
  }
  return { symbols, invalid: [...new Set(invalid)] }
}

/**
 * Turn the existing OptimizationConfig-shaped research draft into the public
 * ResearchRequestV5 accepted inside a complete-validation request. The draft
 * itself remains the single source of all v5 research settings.
 */
export function researchV5FromDraft(value: unknown): JsonObject {
  const draft = objectValue(value)
  if (!draft) return {}
  const existing = objectValue(draft.research_context)
  if (existing) {
    const keys = [
      'research_context', 'ranges', 'strategy', 'selection_target', 'rebalance_days', 'seed', 'top_candidates',
      'max_principal_drawdown', 'min_trades', 'min_history_days', 'min_minute_ratio',
      'configuration_revision', 'prior_policy', 'source_etf', 'requested_data_version', 'research_origin',
    ]
    return compactObject(Object.fromEntries(keys.filter(key => key in draft).map(key => [key, draft[key]])))
  }

  const base = objectValue(draft.base)
  if (!base) return {}
  const contextKeys = [
    'symbol', 'start_date', 'end_date', 'frequency', 'initial_shares', 'grid_mode', 'fees',
    'reference_price_mode', 'reference_price', 'lower_price', 'upper_price', 'dca', 'parameter_links',
  ]
  const researchContext = compactObject(Object.fromEntries(contextKeys
    .filter(key => key in base)
    .map(key => [key, base[key]])))
  const researchKeys = [
    'ranges', 'strategy', 'selection_target', 'rebalance_days', 'seed', 'top_candidates',
    'max_principal_drawdown', 'min_trades', 'min_history_days', 'min_minute_ratio',
    'configuration_revision', 'prior_policy', 'source_etf', 'requested_data_version', 'research_origin',
  ]
  return compactObject({
    research_context: researchContext,
    ...Object.fromEntries(researchKeys.filter(key => key in draft).map(key => [key, draft[key]])),
  })
}

export function validationResearchWithRevision(research: JsonObject, revision?: string): JsonObject {
  return compactObject({ ...research, configuration_revision: revision ?? research.configuration_revision })
}

function validationAccountError(symbol: string, account: ValidationAccountInput | undefined, initialShares: number): string {
  if (!account) return symbol + ' 缺少预算设置。'
  if (!finite(account.total_budget) || account.total_budget <= 0) return symbol + ' 请填写大于零的总预算。'
  const minimumShares = Math.max(100, Math.ceil(initialShares / 100) * 100)
  if (!integer(account.max_shares, minimumShares) || account.max_shares % 100 !== 0) {
    return symbol + ' 的最高持股必须是不低于初始持股的有限整数，并按 100 股对齐。'
  }
  let contributionTotal = 0
  for (const contribution of account.contributions ?? []) {
    if (!contribution.available_at) return symbol + ' 的计划补款需要填写到账时间。'
    if (!finite(contribution.amount) || contribution.amount <= 0) return symbol + ' 的计划补款金额必须大于零。'
    contributionTotal += contribution.amount
  }
  if (contributionTotal > account.total_budget) return symbol + ' 的计划补款总额不能超过总预算。'
  return ''
}

/** Build the exact public validation payload shared by preflight and submission. */
export function buildValidationRequest(input: ValidationRequestBuildInput): { request?: JsonObject; error?: string } {
  const symbols = [...new Set(input.symbols.map(normalizeValidationSymbol).filter(Boolean))]
  if (!symbols.length) return { error: '请填写至少一个有效的标的代码。' }
  const researchContext = objectValue(input.research.research_context)
  if (!researchContext) return { error: '当前研究草稿缺少研究上下文，无法建立完整验证。' }
  if (input.historyPolicy === 'explicit' && empty(researchContext.end_date)) {
    return { error: '严格使用当前日期时需要填写结束日期。' }
  }
  const accounts: Record<string, JsonObject> = {}
  for (const symbol of symbols) {
    const account = input.accounts[symbol]
    const error = validationAccountError(symbol, account, input.initialShares)
    if (error) return { error }
    const contributions = (account?.contributions ?? [])
      .filter(item => item.available_at || item.amount != null)
      .map(item => ({ available_at: item.available_at, amount: item.amount }))
      .sort((left, right) => left.available_at.localeCompare(right.available_at))
    accounts[symbol] = {
      total_budget: account!.total_budget,
      max_shares: account!.max_shares,
      contributions,
      participation_rate: .01,
      ttl_bars: 3,
      extra_delay_bars: 0,
    }
  }
  const planHash = String(input.planHash ?? '').trim()
  if (input.requirePlanHash && !planHash) return { error: '请先完成方案检查，再启动完整验证。' }
  const runtime = input.maxRuntimeSeconds ?? 86400
  if (!integer(runtime, 1) || runtime > 86400) return { error: '任务运行时限必须在 1 秒至 24 小时之间。' }
  const requestedDataVersions = Object.fromEntries(Object.entries(input.requestedDataVersions ?? {})
    .map(([symbol, version]) => [normalizeValidationSymbol(symbol), String(version)])
    .filter(([symbol, version]) => Boolean(symbol && version)))
  return {
    request: compactObject({
      research: input.research,
      symbols,
      history_policy: input.historyPolicy,
      accounts,
      requested_data_versions: requestedDataVersions,
      suite_version: 'research_validation_v1',
      primary_branch: 'adaptive',
      ...(input.familyId?.trim() ? { family_id: input.familyId.trim() } : {}),
      ...(planHash ? { plan_hash: planHash } : {}),
      max_runtime_seconds: runtime,
    }),
  }
}

export function validationPlanOf(value: unknown): JsonObject {
  const response = objectValue(value)
  return objectValue(response?.plan) ?? response ?? {}
}

/** Accept the symbol-list variants a preflight may use while it is being stabilised. */
export function validationPlanSymbols(planValue: unknown): ValidationPlanSymbol[] {
  const plan = validationPlanOf(planValue)
  const raw = plan.symbols
  if (Array.isArray(raw)) {
    return raw.flatMap((item, index) => {
      if (typeof item === 'string') return [{ symbol: normalizeValidationSymbol(item) || item, _index: index }]
      const row = objectValue(item)
      if (!row) return []
      const symbol = normalizeValidationSymbol(row.symbol ?? row.code ?? row.ticker)
      return [{ ...row, symbol: symbol || String(row.symbol ?? '') }]
    }).filter(item => Boolean(item.symbol))
  }
  const mapped = objectValue(raw)
  if (mapped) {
    return Object.entries(mapped).flatMap(([symbol, value]) => {
      const normalized = normalizeValidationSymbol(symbol)
      if (!normalized) return []
      const row = objectValue(value)
      return [{ ...(row ?? {}), symbol: normalized }]
    })
  }
  const symbol = normalizeValidationSymbol(plan.symbol)
  return symbol ? [{ ...plan, symbol }] : []
}

export function requestedVersionsFromPlan(planValue: unknown): Record<string, string> {
  const plan = validationPlanOf(planValue)
  const versions: Record<string, string> = {}
  const mapped = objectValue(plan.data_versions) ?? objectValue(plan.requested_data_versions)
  if (mapped) {
    for (const [symbol, version] of Object.entries(mapped)) {
      const normalized = normalizeValidationSymbol(symbol)
      if (normalized && typeof version === 'string' && version) versions[normalized] = version
    }
  }
  for (const row of validationPlanSymbols(plan)) {
    const version = firstText(row, ['data_version', 'requested_data_version', 'target_data_version'])
    if (version) versions[row.symbol] = version
  }
  return versions
}

export function canSubmitValidationPlan(value: unknown): boolean {
  const plan = validationPlanOf(value)
  const hash = plan.plan_hash
  if (typeof hash !== 'string' || !hash.trim()) return false
  const status = firstText(plan, ['status', 'state']).toLowerCase()
  return status === 'ready' || status === 'valid'
}

export function isResearchValidationResult(value: unknown): boolean {
  const result = objectValue(value)
  if (!result) return false
  return result.kind === 'research_validation'
    || result.suite_version === 'research_validation_v1'
    || objectValue(result.validation_plan) != null && objectValue(result.validation_plan)?.suite_version === 'research_validation_v1'
}

export function isParameterResearchResult(value: unknown): boolean {
  const result = objectValue(value)
  return Boolean(result) && !isResearchValidationResult(result)
}

export function researchResultHistoryLabel(item: { kind?: unknown; title?: unknown } | null | undefined): string {
  if (item?.kind === 'research_validation') return '完整研究验证'
  const title = typeof item?.title === 'string' && item.title.trim() ? item.title.trim() : '参数研究 v5'
  return title + ' · 未完整验证'
}

function nestedStatus(value: unknown): unknown {
  if (typeof value === 'string' || typeof value === 'boolean') return value
  const item = objectValue(value)
  return item?.status ?? item?.state ?? item?.code ?? item?.outcome ?? value
}

function statusCode(value: unknown): string {
  const nested = nestedStatus(value)
  return typeof nested === 'string' ? nested.trim().toLowerCase().replaceAll('-', '_').replaceAll(' ', '_') : ''
}

function statusNote(value: unknown): string {
  const item = objectValue(value)
  if (!item) return ''
  const text = firstText(item, ['message', 'reason', 'summary', 'detail', 'limitation', 'scope_note'])
  if (text) return text
  const reasons = Array.isArray(item.reasons) ? item.reasons.map(String).filter(Boolean) : []
  return reasons.join('；')
}

export function validationStatusView(value: unknown, kind: ValidationStatusKind): ValidationStatusView {
  const code = statusCode(value)
  const note = statusNote(value)
  if (kind === 'completion') {
    if (['complete', 'completed', 'success', 'succeeded', 'done'].includes(code)) return { raw: value, label: '验证完整', tone: 'green', note }
    if (['queued', 'running', 'pending', 'in_progress'].includes(code)) return { raw: value, label: '正在验证', tone: 'blue', note }
    if (['cancelled', 'canceled'].includes(code)) return { raw: value, label: '验证已取消', tone: 'neutral', note }
    if (['failed', 'error'].includes(code)) return { raw: value, label: '验证失败', tone: 'red', note }
    if (['partial', 'incomplete', 'budget_exhausted', 'timeout'].includes(code)) return { raw: value, label: '验证未完成', tone: 'amber', note }
    return { raw: value, label: '完成情况待确认', tone: 'neutral', note }
  }
  if (kind === 'candidate') {
    if (['qualified', 'available', 'present', 'has_candidate', 'candidate_available'].includes(code)) return { raw: value, label: '存在合格候选', tone: 'green', note }
    if (['none', 'no_qualified', 'no_candidate', 'not_qualified', 'unqualified', 'empty', 'not_available', 'unavailable'].includes(code)) return { raw: value, label: '没有合格候选', tone: 'amber', note }
    if (code === 'mixed') return { raw: value, label: '部分标的有合格候选', tone: 'amber', note }
    if (['incomplete', 'partial', 'pending'].includes(code)) return { raw: value, label: '候选状态未完成', tone: 'neutral', note }
    return { raw: value, label: '候选状态待确认', tone: 'neutral', note }
  }
  if (kind === 'qualification') {
    if (['qualified', 'passed', 'eligible', 'met'].includes(code)) return { raw: value, label: '满足预设条件', tone: 'green', note }
    if (['risk_exceeded', 'risk_breached', 'drawdown_exceeded', 'failed'].includes(code)) return { raw: value, label: '风险超限', tone: 'red', note }
    if (['execution_limited', 'limited', 'constrained', 'partial'].includes(code)) return { raw: value, label: '执行受限', tone: 'amber', note }
    if (['no_candidate', 'not_qualified', 'unqualified'].includes(code)) return { raw: value, label: '没有可评价候选', tone: 'amber', note }
    if (['incomplete', 'pending'].includes(code)) return { raw: value, label: '资格复测未完成', tone: 'amber', note }
    if (['unevaluable', 'unavailable', 'not_applicable'].includes(code)) return { raw: value, label: '不可评价', tone: 'neutral', note }
    return { raw: value, label: '资格状态待确认', tone: 'neutral', note }
  }
  if (['complete', 'completed', 'sufficient', 'available', 'recorded'].includes(code)) return { raw: value, label: '证据已记录', tone: 'green', note }
  if (['limited', 'partial', 'warning', 'incomplete'].includes(code)) return { raw: value, label: '证据有限', tone: 'amber', note }
  if (['insufficient', 'missing', 'failed', 'error'].includes(code)) return { raw: value, label: '证据不足', tone: 'red', note }
  if (['pending', 'unknown', 'not_available'].includes(code)) return { raw: value, label: '等待证据', tone: 'neutral', note }
  if (objectValue(value) && Object.keys(objectValue(value) ?? {}).length) return { raw: value, label: '证据已记录', tone: 'blue', note }
  return { raw: value, label: '等待证据', tone: 'neutral', note }
}

export function validationOverviewItems(result: unknown): Record<ValidationStatusKind, ValidationStatusView> {
  const item = objectValue(result) ?? {}
  const evidence = objectValue(item.evidence)
  const audit = objectValue(item.audit)
  const cells = Array.isArray(item.cells) ? item.cells.map(objectValue).filter((cell): cell is JsonObject => Boolean(cell)) : []
  const qualificationCells = cells.filter(cell => cell.kind === 'qualification')
  const arms = Array.isArray(item.symbols)
    ? item.symbols.flatMap(symbol => {
        const row = objectValue(symbol)
        return Array.isArray(row?.arms) ? row.arms.map(objectValue).filter((arm): arm is JsonObject => Boolean(arm)) : []
      })
    : []
  const candidatesAvailable = arms.some(arm => {
    const schemes = objectValue(arm.schemes)
    if (!schemes) return false
    return ['fixed', 'adaptive'].some(name => {
      const scheme = objectValue(schemes[name])
      const schedule = Array.isArray(scheme?.schedule) ? scheme.schedule.map(objectValue) : []
      return objectValue(scheme?.selected) != null || schedule.some(row => row?.paused === false)
    })
  })
  const completionCode = statusCode(item.completion_status)
  let candidateValue = item.candidate_status
  if (empty(candidateValue) && (completionCode === 'complete' || completionCode === 'incomplete')) {
    candidateValue = candidatesAvailable ? 'qualified' : arms.length || qualificationCells.length ? 'no_qualified' : 'pending'
  }
  let qualificationValue = item.qualification_status
  if (empty(qualificationValue) && qualificationCells.length) {
    const qualificationStatuses = qualificationCells.flatMap(cell => {
      const raw = objectValue(cell.result)
      return [cell.status, raw?.qualification_status, raw?.status].filter(value => typeof value === 'string') as string[]
    })
    if (qualificationStatuses.includes('unevaluable')) qualificationValue = 'unevaluable'
    else if (qualificationStatuses.some(value => !['completed', 'complete', 'qualified'].includes(value))) qualificationValue = 'incomplete'
    else if (qualificationStatuses.includes('qualified')) qualificationValue = 'qualified'
  }
  let evidenceValue = item.evidence_status ?? (typeof item.evidence === 'string' ? item.evidence : evidence)
  if (empty(evidenceValue) && audit) {
    const warnings = Array.isArray(item.warnings) ? item.warnings : []
    const gss = objectValue(item.gss)
    evidenceValue = warnings.length || gss?.status === 'incomplete' || audit.human_untouched_verified !== true
      ? { status: 'limited', reason: '记录可追溯，但人工未查看、历史样本覆盖或部分对照仍未得到独立验证。' }
      : { status: 'recorded' }
  }
  return {
    completion: validationStatusView(item.completion_status, 'completion'),
    candidate: validationStatusView(candidateValue, 'candidate'),
    qualification: validationStatusView(qualificationValue, 'qualification'),
    evidence: validationStatusView(evidenceValue, 'evidence'),
  }
}

export function validationPrimaryArm(value: unknown): string {
  const result = objectValue(value) ?? {}
  const plan = validationPlanOf(result.plan ?? result.validation_plan)
  const arm = firstText(plan, ['primary_arm', 'primary_branch']) || firstText(result, ['primary_arm', 'primary_branch'])
  if (arm === 'adaptive') return '自适应参数主方案'
  if (arm === 'fixed') return '固定参数主方案'
  if (arm.includes(':')) {
    const [window, prior, branch] = arm.split(':')
    return [
      window === 'same_price' ? '同价训练窗' : window === 'fixed' ? '固定训练窗' : '',
      prior === 'daily_state_v1' ? '日线先验开启' : prior === 'off' ? '先验关闭' : '',
      branch === 'adaptive' ? '自适应参数主方案' : branch === 'fixed' ? '固定参数主方案' : '',
    ].filter(Boolean).join(' · ') || arm
  }
  return arm || '主方案已冻结'
}

export function validationEvidenceRecorded(result: unknown, section = 'overview'): boolean {
  const item = objectValue(result) ?? {}
  if (section === 'overview' && (!empty(item.overview_first_viewed_at) || item.overview_viewed === true)) return true
  const audit = objectValue(item.audit)
  const sources = [item.view_events, item.evidence_views, audit?.view_events, audit?.views, audit?.evidence_views]
  return sources.some(raw => Array.isArray(raw) && raw.some(event => {
    const row = objectValue(event)
    return row?.section === section || row?.kind === section
  }))
}

export function validationEvidenceGaps(result: unknown): string[] {
  const item = objectValue(result) ?? {}
  const audit = objectValue(item.audit)
  const plan = validationPlanOf(item.plan ?? item.validation_plan ?? audit?.plan)
  const gaps: string[] = []
  const add = (value: unknown) => {
    if (Array.isArray(value)) {
      for (const entry of value) {
        if (typeof entry === 'string' && entry.trim()) gaps.push(entry.trim())
        else {
          const row = objectValue(entry)
          const text = firstText(row, ['message', 'reason', 'detail', 'summary'])
          if (text) gaps.push(text)
        }
      }
    } else if (typeof value === 'string' && value.trim()) gaps.push(value.trim())
  }
  add(plan.gaps)
  for (const row of validationPlanSymbols(plan)) add(row.gaps)
  const evidence = objectValue(item.evidence)
  add(evidence?.gaps)
  add(evidence?.limitations)
  add(evidence?.warnings)
  add(item.warnings)
  if (audit && audit.human_untouched_verified !== true) gaps.push('查看记录不能证明历史上的研究从未被人工查看；独立性保持保守标记。')
  const gss = objectValue(item.gss)
  if (gss?.status === 'incomplete') gaps.push('ETF评分对照未覆盖全部冻结的共同决策日。')
  if (gss?.historical_universe_verified === false) gaps.push('当前本地证券池不能证明包含历史退市标的。')
  return [...new Set(gaps)]
}

export function validationSchemeEntries(result: unknown): Array<{ key: string; label: string; value: unknown }> {
  const item = objectValue(result) ?? {}
  const symbolRows = Array.isArray(item.symbols)
    ? item.symbols.map(objectValue).filter((row): row is JsonObject => Boolean(row))
    : []
  const fromSymbols = symbolRows.flatMap(row => {
    const symbol = String(row.symbol ?? '')
    return Array.isArray(row.arms) ? row.arms.flatMap((rawArm, armIndex) => {
      const arm = objectValue(rawArm) ?? {}
      const schemes = objectValue(arm.schemes) ?? {}
      return ['fixed', 'adaptive'].flatMap(name => {
        const scheme = objectValue(schemes[name])
        if (!scheme) return []
        const window = firstText(arm, ['window_mode'])
        const prior = firstText(arm, ['prior_policy'])
        const label = [symbol,
          window === 'same_price' ? '同价训练窗' : window === 'fixed' ? '固定训练窗' : '',
          prior === 'daily_state_v1' ? '日线先验开启' : prior === 'off' ? '先验关闭' : '',
          name === 'adaptive' ? '自适应参数' : '固定参数',
        ].filter(Boolean).join(' · ')
        return [{ key: `${symbol}:${armIndex}:${name}`, label, value: { ...scheme, window_mode: window, prior_policy: prior, branch: name } }]
      })
    }) : []
  })
  if (fromSymbols.length) return fromSymbols
  const raw = item.cells ?? item.schemes ?? item.arms
  const entries: Array<[string, unknown]> = Array.isArray(raw)
    ? raw.map((value, index) => [String(index), value])
    : Object.entries(objectValue(raw) ?? {})
  return entries.map(([key, value]) => {
    const row = objectValue(value) ?? {}
    const parts = [
      firstText(row, ['window_mode', 'training_window_mode', 'window']) === 'same_price' ? '同价训练窗'
        : firstText(row, ['window_mode', 'training_window_mode', 'window']) === 'fixed' ? '固定训练窗' : '',
      firstText(row, ['prior_policy', 'prior']) === 'daily_state_v1' ? '日线先验开启'
        : firstText(row, ['prior_policy', 'prior']) === 'off' ? '先验关闭' : '',
      firstText(row, ['branch', 'selection_target', 'arm']) === 'adaptive' ? '自适应参数'
        : firstText(row, ['branch', 'selection_target', 'arm']) === 'fixed' ? '固定参数' : '',
    ].filter(Boolean)
    const label = firstText(row, ['label', 'name', 'title']) || parts.join(' · ')
    return { key, label: label || '方案 ' + (Number(key) + 1 || key), value }
  })
}

export function validationCellSummary(value: unknown): { completion: ValidationStatusView; qualification: ValidationStatusView; metrics: JsonObject } {
  const cell = objectValue(value) ?? {}
  const qualification = objectValue(cell.qualification)
  const selected = objectValue(cell.selected)
  const metrics = objectValue(cell.metrics)
    ?? objectValue(qualification?.strategy_metrics)
    ?? objectValue(qualification?.result)?.strategy_metrics
    ?? objectValue(cell.result)?.metrics
    ?? objectValue(cell.primary)?.metrics
    ?? objectValue(selected?.metrics)
    ?? {}
  return {
    completion: validationStatusView(cell.completion_status ?? qualification?.status ?? cell.status, 'completion'),
    qualification: validationStatusView(cell.qualification_status ?? qualification?.qualification_status ?? qualification?.status, 'qualification'),
    metrics: objectValue(metrics) ?? {},
  }
}

export function validationPrimaryMetrics(result: unknown): JsonObject {
  const item = objectValue(result) ?? {}
  const explicit = objectValue(item.primary) ?? objectValue(item.primary_result) ?? objectValue(item.primary_arm_result)
  if (explicit) return objectValue(explicit.metrics) ?? explicit
  const audit = objectValue(item.audit)
  const primary = firstText(validationPlanOf(item.plan ?? item.validation_plan ?? audit?.plan), ['primary_arm'])
    || firstText(item, ['primary_arm'])
  const [window = '', prior = '', branch = 'adaptive'] = primary.split(':')
  const row = (Array.isArray(item.symbols) ? item.symbols : []).map(objectValue)
    .find(value => value?.arms != null)
  const arm = (Array.isArray(row?.arms) ? row!.arms : []).map(objectValue)
    .find(value => value?.window_mode === window && value?.prior_policy === prior)
  const scheme = objectValue(objectValue(arm?.schemes)?.[branch])
  return objectValue(objectValue(scheme?.qualification)?.strategy_metrics) ?? {}
}

export function validationBlocks(result: unknown): JsonObject[] {
  const item = objectValue(result) ?? {}
  const root = Array.isArray(item.blocks) ? item.blocks : Array.isArray(item.segments) ? item.segments : []
  if (root.length) return root.map(objectValue).filter((row): row is JsonObject => Boolean(row))
  return (Array.isArray(item.symbols) ? item.symbols : []).flatMap(symbol => {
    const row = objectValue(symbol)
    return Array.isArray(row?.blocks)
      ? row.blocks.map(block => ({ symbol: row?.symbol, ...(objectValue(block) ?? { value: block }) }))
      : []
  })
}

function primarySchemeRecords(result: unknown): Array<{
  symbol: string
  armIndex: number
  arm: JsonObject
  schemeName: 'fixed' | 'adaptive'
  scheme: JsonObject
  qualification: JsonObject
}> {
  const item = objectValue(result) ?? {}
  const audit = objectValue(item.audit)
  const plan = validationPlanOf(item.plan ?? item.validation_plan ?? audit?.plan)
  const primary = firstText(plan, ['primary_arm']) || firstText(item, ['primary_arm'])
  const [primaryWindow, primaryPrior] = primary.split(':')
  const records: ReturnType<typeof primarySchemeRecords> = []
  for (const symbolRow of Array.isArray(item.symbols) ? item.symbols.map(objectValue) : []) {
    if (!symbolRow) continue
    const symbol = String(symbolRow.symbol ?? '')
    const arms = Array.isArray(symbolRow.arms) ? symbolRow.arms.map(objectValue).filter((arm): arm is JsonObject => Boolean(arm)) : []
    let arm = arms.find(raw => raw.primary === true)
      ?? arms.find(raw => raw.window_mode === primaryWindow && raw.prior_policy === primaryPrior)
      ?? arms.find(raw => raw.arm_group === `${primaryWindow}:${primaryPrior}`)
      ?? arms.find(raw => objectValue(raw.schemes)?.adaptive != null)
    if (!arm) continue
    const armIndex = arms.indexOf(arm)
    const scheme = objectValue(objectValue(arm.schemes)?.adaptive)
    if (!scheme) continue
    const qualification = objectValue(scheme.qualification) ?? {}
    records.push({ symbol, armIndex, arm, schemeName: 'adaptive', scheme, qualification })
  }
  return records
}

export function validationSegmentGroups(result: unknown): { primary: ValidationSegmentGroup[]; alternatives: ValidationSegmentGroup[] } {
  const item = objectValue(result) ?? {}
  const primaryRecords = primarySchemeRecords(item)
  const primaryKeys = new Set(primaryRecords.map(row => `${row.symbol}:${row.armIndex}:${row.schemeName}`))
  const primary: ValidationSegmentGroup[] = []
  const alternatives: ValidationSegmentGroup[] = []
  for (const symbolRow of Array.isArray(item.symbols) ? item.symbols.map(objectValue) : []) {
    if (!symbolRow) continue
    const symbol = String(symbolRow.symbol ?? '')
    const arms = Array.isArray(symbolRow.arms) ? symbolRow.arms.map(objectValue).filter((arm): arm is JsonObject => Boolean(arm)) : []
    arms.forEach((arm, armIndex) => ['fixed', 'adaptive'].forEach(schemeName => {
      const scheme = objectValue(objectValue(arm.schemes)?.[schemeName])
      const qualification = objectValue(scheme?.qualification)
      const rawSegments = objectValue(qualification?.segment_returns)?.segments
      if (!Array.isArray(rawSegments)) return
      const segments = rawSegments.map(objectValue).filter((row): row is JsonObject => Boolean(row))
      if (!segments.length) return
      const windowMode = String(arm.window_mode ?? '')
      const priorPolicy = String(arm.prior_policy ?? '')
      const label = [symbol,
        windowMode === 'same_price' ? '同价训练窗' : windowMode === 'fixed' ? '固定训练窗' : '',
        priorPolicy === 'daily_state_v1' ? '日线先验开启' : priorPolicy === 'off' ? '先验关闭' : '',
        schemeName === 'adaptive' ? '自适应参数' : '固定参数',
      ].filter(Boolean).join(' · ')
      const key = `${symbol}:${armIndex}:${schemeName}`
      const group = {
        key, symbol, armGroup: String(arm.arm_group ?? `${windowMode}:${priorPolicy}`), scheme: schemeName as 'fixed' | 'adaptive',
        label, primary: primaryKeys.has(key),
        segments: segments.map(row => ({ symbol, arm_group: arm.arm_group, scheme: schemeName, ...row })),
      }
      if (group.primary) primary.push(group)
      else alternatives.push(group)
    }))
  }
  return { primary, alternatives }
}

export function validationAccountRows(result: unknown): JsonObject[] {
  const item = objectValue(result) ?? {}
  const planSymbols = new Map(validationPlanSymbols(item.plan ?? item.validation_plan ?? objectValue(item.audit)?.plan ?? item).map(row => [row.symbol, row]))
  const rows: JsonObject[] = []
  for (const record of primarySchemeRecords(item)) {
    const metrics = objectValue(record.qualification.strategy_metrics) ?? {}
    const bounded = objectValue(record.qualification.bounded_ledger) ?? {}
    const funding = objectValue(metrics.funding_diagnostics) ?? {}
    const execution = objectValue(metrics.execution_diagnostics) ?? {}
    const reasons = objectValue(execution.rejections_by_reason) ?? {}
    const planRow: JsonObject = planSymbols.get(record.symbol) ?? {}
    const fillCount = finite(execution.trade_fill_count) ? execution.trade_fill_count
      : finite(metrics.trade_count) ? metrics.trade_count : undefined
    const rejectionCount = finite(execution.rejection_count) ? execution.rejection_count
      : finite(bounded.rejection_count) ? bounded.rejection_count : undefined
    const attempts = fillCount != null && rejectionCount != null ? fillCount + rejectionCount : undefined
    const cancelled = finite(execution.cancelled_orders) ? execution.cancelled_orders
      : finite(execution.cancelled_order_count) ? execution.cancelled_order_count
        : execution.day_end_cancel === true ? '日内剩余委托撤销' : undefined
    rows.push({
      symbol: record.symbol,
      arm_group: record.arm.arm_group,
      scheme: record.schemeName,
      total_budget: metrics.total_budget,
      initial_equity: metrics.initial_equity,
      initial_holdings_value: metrics.initial_holdings_value,
      initial_cash: metrics.initial_cash ?? planRow.initial_cash,
      scheduled_contributions: metrics.scheduled_contributions,
      external_topups: metrics.external_topups,
      total_invested: metrics.total_invested ?? bounded.ending_principal,
      peak_invested: metrics.peak_invested,
      final_cash: metrics.final_cash ?? funding.unspent_cash ?? bounded.ending_cash,
      final_equity: metrics.final_equity ?? bounded.ending_equity,
      final_position: metrics.final_position ?? bounded.ending_position,
      total_fees: metrics.total_fees,
      trade_count: metrics.trade_count ?? bounded.trade_count,
      fill_count: fillCount,
      fill_ratio: attempts && attempts > 0 ? Number(fillCount) / attempts : undefined,
      rejection_count: rejectionCount,
      insufficient_cash_rejections: reasons.insufficient_cash,
      cancelled_orders: cancelled,
      cash_shortfall_action: funding.cash_shortfall_action,
      participation_rate: execution.participation_rate,
      ttl_bars: execution.ttl_bars,
    })
  }
  return rows
}
