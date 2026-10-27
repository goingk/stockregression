import { expect, test, type Page } from '@playwright/test'

const configuration = {
  revision: 'fixture-revision',
  config: { ui: { task_polling_interval_ms: 60_000, polling_interval_ms: 60_000 } },
  pending_restart: [],
}

async function installTaskRoutes(page: Page, onPreview: (body: Record<string, any>) => Promise<Record<string, unknown>>) {
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname === '/api/v1/configuration') return route.fulfill({ json: configuration })
    if (url.pathname === '/api/v1/jobs/active') return route.fulfill({ json: [] })
    if (url.pathname === '/api/v1/jobs/history') return route.fulfill({ json: { items: [], total: 0, offset: 0, limit: 50, kinds: [] } })
    if (url.pathname === '/api/v1/research-validations/preview' && request.method() === 'POST') {
      return route.fulfill({ json: await onPreview(request.postDataJSON()) })
    }
    return route.fulfill({ status: 404, json: { detail: `unexpected fixture request: ${request.method()} ${url.pathname}` } })
  })
}

function readyPlan(startDate: string) {
  return {
    plan: {
      plan_hash: 'frozen-plan-' + startDate,
      status: 'ready',
      primary_arm: 'same_price:off:adaptive',
      symbols: [{
        symbol: '600000.SH', status: 'ready', training_days: 250, initial_cash: 170000,
        evaluation_start: '2024-03-01', evaluation_end: '2025-06-30',
        blocks: [{ start: '2024-03-01', end: '2025-06-30', days: 310 }], gaps: [],
      }],
    },
  }
}

async function fillAccounts(page: Page) {
  const fields = page.locator('.research-validation-account input[type=number]')
  await expect(fields).toHaveCount(2)
  await expect(fields.nth(1)).toHaveValue('')
  await fields.nth(0).fill('200000')
  await fields.nth(1).fill('3000')
}

test('equal parent rerenders retain preflight while a date edit switches to explicit', async ({ page }) => {
  const previews: Array<Record<string, any>> = []
  await installTaskRoutes(page, async body => {
    previews.push(body)
    if (body.research.research_context.start_date === '2024-02-01') {
      return {
        plan: {
          plan_hash: 'strict-history-gap', status: 'data_issue', primary_arm: 'same_price:off:adaptive',
          symbols: [{
            symbol: '600000.SH', status: 'data_issue', blocks: [], training_days: 250, initial_cash: null,
            available_days: 250,
            gaps: [{ kind: 'history', available_days: 250, additional_verified_days: 10, start: '2020-01-02', end: '2021-01-04', message: '连续可用历史250日，训练与样本外仍不足；未知缺口不拼接' }],
          }],
        },
      }
    }
    return readyPlan(String(body.research.research_context.start_date))
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=launcher')
  await fillAccounts(page)
  await page.getByRole('button', { name: '检查验证方案' }).click()
  await expect(page.locator('code[title="frozen-plan-2024-01-02"]')).toBeVisible()

  await page.getByRole('button', { name: /同值重渲染/ }).click()
  await expect(page.locator('code[title="frozen-plan-2024-01-02"]')).toBeVisible()
  expect(previews).toHaveLength(1)

  await page.getByRole('button', { name: '修改研究日期', exact: true }).click()
  await expect(page.locator('.research-validation-plan')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '使用当前日期' })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: '检查验证方案' }).click()
  await expect(page.getByText(/连续可用历史250日/).first()).toBeVisible()
  await expect(page.getByText(/连续可用 250 日；尚需额外 10 日/)).toBeVisible()
  expect(previews.at(-1)?.history_policy).toBe('explicit')
  expect(previews.at(-1)?.research.research_context.start_date).toBe('2024-02-01')
  const policyButtons = page.locator('[aria-label="历史区间策略"] button')
  await policyButtons.first().click()
  await expect(policyButtons.first()).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: '再次修改研究日期' }).click()
  await expect(policyButtons.nth(1)).toHaveAttribute('aria-pressed', 'true')
})

test('a preflight response arriving after a date edit cannot replace the current state', async ({ page }) => {
  let release: () => void = () => undefined
  const gate = new Promise<void>(resolve => { release = resolve })
  const previews: Array<Record<string, any>> = []
  await installTaskRoutes(page, async body => {
    previews.push(body)
    if (previews.length === 1) await gate
    return readyPlan(String(body.research.research_context.start_date))
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=launcher')
  await fillAccounts(page)
  await page.getByRole('button', { name: '检查验证方案' }).click()
  await expect.poll(() => previews.length).toBe(1)
  await page.getByRole('button', { name: '修改研究日期', exact: true }).click()
  release()
  await expect.poll(() => previews.length).toBe(1)
  await expect(page.locator('.research-validation-plan')).toHaveCount(0)
  await page.getByRole('button', { name: '检查验证方案' }).click()
  await expect(page.locator('code[title="frozen-plan-2024-02-01"]')).toBeVisible()
  expect(previews.at(-1)?.history_policy).toBe('explicit')
})

test('report exposes independent root states, detailed gaps and records each expanded section once', async ({ page }) => {
  const views: string[] = []
  const exports: string[] = []
  const ordinaryFullReads: string[] = []
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname === '/api/v1/results/validation-report-1/evidence-views') {
      views.push(String(request.postDataJSON().section))
      return route.fulfill({ json: { section: request.postDataJSON().section } })
    }
    if (url.pathname === '/api/v1/results/validation-report-1/export') {
      exports.push(String(url.searchParams.get('kind')))
      return route.fulfill({ status: 200, contentType: url.searchParams.get('kind') === 'json' ? 'application/json' : 'text/csv', body: url.searchParams.get('kind') === 'json' ? '{"kind":"research_validation"}' : '\ufeffunit_id\ncell-1\n' })
    }
    if (url.pathname === '/api/v1/results/validation-report-1') ordinaryFullReads.push(url.toString())
    return route.fulfill({ status: 404, json: { detail: `unexpected fixture request: ${url.pathname}` } })
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=report')
  await expect(page.locator('.research-validation-status').filter({ hasText: '候选结果' })).toContainText('没有合格候选')
  await expect(page.locator('.research-validation-status').filter({ hasText: '预算资格' })).toContainText('没有可评价候选')
  await expect(page.locator('.research-validation-status').filter({ hasText: '完成情况' })).toContainText('验证完整')
  await expect(page.locator('.research-validation-status').filter({ hasText: '证据' })).toContainText('证据有限')
  await expect(page.getByText(/连续可用历史250日/).first()).toBeVisible()
  await expect(page.getByText(/连续可用 250 日；尚需额外 10 日/)).toBeVisible()
  await expect.poll(() => views).toContain('overview')

  for (const section of ['cells', 'blocks', 'execution', 'pressure', 'scoring', 'audit']) {
    const summary = page.locator(`[data-evidence-section="${section}"] > details > summary`)
    await summary.click()
    await expect.poll(() => views.includes(section)).toBe(true)
  }
  const blocksSummary = page.locator('[data-evidence-section="blocks"] > details > summary')
  await blocksSummary.click()
  await blocksSummary.click()
  await expect.poll(() => views.filter(section => section === 'blocks')).toHaveLength(1)

  const segmentBody = page.locator('[data-evidence-section="blocks"] > details .disclosure-body')
  await expect(page.locator('[data-evidence-section="blocks"] > details > summary')).toContainText('600000.SH 1 段 · 2025-01-02–2025-03-31')
  const primarySegments = segmentBody.getByRole('table', { name: '主方案实际分段明细' })
  await expect(primarySegments).toBeVisible()
  await expect(primarySegments.locator('th')).toHaveText([
    '标的', '开始', '结束', '交易日', '阶段本金收益', '阶段峰值本金回撤',
    '期间新增资金', '期初资产', '期末资产', '状态',
  ])
  await expect(primarySegments.locator('[data-field="principal_return"]')).toHaveText('12.00%')
  await expect(primarySegments.locator('[data-field="capital_added"]')).toHaveText('10,000.00')
  await expect(primarySegments.locator('[data-field="ending_equity"]')).toHaveText('224,000.00')
  await segmentBody.locator('.research-validation-alternative-segments > summary').click()
  await expect(segmentBody.getByRole('heading', { name: /同价训练窗 · 先验关闭 · 固定参数/ }).first()).toBeVisible()
  const executionBody = page.locator('[data-evidence-section="execution"] > details .disclosure-body')
  await expect(executionBody.getByText('200,000.00', { exact: true }).first()).toBeVisible()
  await expect(executionBody.getByText('12,345.67', { exact: true }).first()).toBeVisible()
  await expect(executionBody.getByText('81.82%', { exact: true })).toBeVisible()
  await expect(executionBody.getByText('撤销剩余委托，不追加资金', { exact: true })).toBeVisible()
  await expect(executionBody.getByText('日内剩余委托撤销', { exact: true })).toBeVisible()

  for (const [label, kind] of [
    ['验证明细 CSV', 'validation'], ['资金曲线 CSV', 'equity'], ['成交流水 CSV', 'trades'], ['完整 JSON（较慢）', 'json'],
  ]) {
    const pending = page.waitForEvent('download')
    await page.getByRole('button', { name: label }).click()
    await pending
    await expect.poll(() => exports.includes(kind)).toBe(true)
  }
  expect(ordinaryFullReads).toEqual([])
})

test('validation resume requires an explicit increase after budget exhaustion and opens the new task', async ({ page }) => {
  const resumeBodies: unknown[] = []
  const original = {
    id: 'validation-exhausted', kind: 'research_validation', status: 'partial', progress: .7,
    message: '评价预算耗尽，等待恢复', result_id: 'validation-result-exhausted', created_at: '2026-10-27T01:00:00Z',
    payload: {
      config: { max_runtime_seconds: 3600, total_evaluation_budget: 1000 },
      plan: { estimated_work: { max_runtime_seconds: 3600, total_evaluation_budget: 1000 } },
      research: { research_context: { symbol: '600000.SH' } },
    },
  }
  await page.route('**/api/v1/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (url.pathname === '/api/v1/configuration') return route.fulfill({ json: configuration })
    if (url.pathname === '/api/v1/jobs/active') return route.fulfill({ json: [] })
    if (url.pathname === '/api/v1/jobs/history') return route.fulfill({ json: { items: [original], total: 1, offset: 0, limit: 50, kinds: [{ kind: 'research_validation', count: 1 }] } })
    if (url.pathname === '/api/v1/jobs/validation-exhausted/result') return route.fulfill({ json: { status: 'partial', stop_reason: 'evaluation_budget', computation: { total_evaluations: 1000 } } })
    if (url.pathname === '/api/v1/jobs/validation-exhausted/resume' && request.method() === 'POST') {
      resumeBodies.push(request.postDataJSON())
      return route.fulfill({ status: 202, json: { job_id: 'validation-resumed', resumed_from: 'validation-exhausted' } })
    }
    if (url.pathname === '/api/v1/jobs/validation-resumed') return route.fulfill({ json: { ...original, id: 'validation-resumed', status: 'queued', result_id: null, message: '继续任务已创建' } })
    return route.fulfill({ status: 404, json: { detail: `unexpected fixture request: ${url.pathname}` } })
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=task')
  await page.getByRole('button', { name: /任务/ }).click()
  await page.getByRole('button', { name: '查看详情' }).click()
  const limits = page.locator('.task-resume-limits input[type=number]')
  await expect(limits.nth(0)).toHaveValue('3600')
  await expect(limits.nth(1)).toHaveValue('1000')
  await page.getByRole('button', { name: '提高上限并继续' }).click()
  await expect(page.locator('.task-resume-validation .notice.error')).toContainText('至少提高一项上限')
  expect(resumeBodies).toEqual([])

  await limits.nth(0).fill('4800')
  await page.getByRole('button', { name: '提高上限并继续' }).click()
  await expect.poll(() => resumeBodies.length).toBe(1)
  expect(resumeBodies[0]).toEqual({ max_runtime_seconds: 4800 })
  await expect(page.getByText('继续任务已创建').first()).toBeVisible()
})

test('capture desktop validation report and launcher acceptance screenshots', async ({ page }) => {
  await page.setViewportSize({ width: 1560, height: 1040 })
  const previews: Array<Record<string, any>> = []
  await installTaskRoutes(page, async body => {
    previews.push(body)
    return readyPlan(String(body.research.research_context.start_date))
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=launcher')
  await fillAccounts(page)
  await page.getByRole('button', { name: '检查验证方案' }).click()
  await expect(page.locator('code[title="frozen-plan-2024-01-02"]')).toBeVisible()
  await page.locator('.research-validation-launch-card').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'test-results/validation-launcher-desktop.png' })
  expect(previews).toHaveLength(1)

  await page.unrouteAll({ behavior: 'wait' })
  await page.route('**/api/v1/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/v1/results/validation-report-1/evidence-views') return route.fulfill({ json: { section: route.request().postDataJSON().section } })
    return route.fulfill({ status: 404, json: { detail: `unexpected fixture request: ${url.pathname}` } })
  })
  await page.goto('/tests/fixtures/research-validation.html?mode=report')
  await page.locator('[data-evidence-section="blocks"] > details > summary').click()
  await page.locator('[data-evidence-section="pressure"] > details > summary').click()
  const blockSection = page.locator('[data-evidence-section="blocks"]')
  const pressureSection = page.locator('[data-evidence-section="pressure"]')
  const blockBox = await blockSection.boundingBox()
  const pressureBox = await pressureSection.boundingBox()
  const currentScroll = await page.evaluate(() => window.scrollY)
  const targetTop = Math.max(0, currentScroll + Math.min(blockBox?.y ?? 0, pressureBox?.y ?? 0) - 90)
  await page.evaluate(top => window.scrollTo({ top }), targetTop)
  await page.screenshot({ path: 'test-results/validation-report-desktop.png' })
  const blockDisclosure = blockSection.locator(':scope > details')
  const pressureDisclosure = pressureSection.locator(':scope > details')
  await expect(blockDisclosure).toHaveAttribute('open', '')
  await expect(pressureDisclosure).toHaveAttribute('open', '')
  const primaryReportSegments = blockSection.getByRole('table', { name: '主方案实际分段明细' })
  await expect(primaryReportSegments).toBeVisible()
  await expect(primaryReportSegments.locator('[data-field="capital_added"]')).toHaveText('10,000.00')
  await expect(primaryReportSegments.locator('[data-field="ending_equity"]')).toHaveText('224,000.00')
  await expect(pressureSection.getByRole('heading', { name: '历史执行压力' })).toBeVisible()
  await expect(pressureSection.getByText('低流动性', { exact: true })).toBeVisible()
})
