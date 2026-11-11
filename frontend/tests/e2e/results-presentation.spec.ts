import { expect, test } from '@playwright/test'

test('results show excess returns and side-by-side metrics while keeping the internal baseline hidden', async ({ page }) => {
  await page.setViewportSize({ width: 1560, height: 1040 })
  const makeStrategy = (reason: string, hidden = false) => ({
    metrics: { peak_principal_return: .1, max_principal_drawdown: .02, trade_count: 1, total_fees: 1, matched_funding_hold_return: .05 },
    equity: [
      { timestamp: hidden ? '2023-01-02' : '2024-01-02', equity: 100, total_invested: 100, principal_curve: 1, principal_drawdown: 0, position: 10 },
      { timestamp: '2024-02-02', equity: 110, total_invested: 100, principal_curve: 1.1, principal_drawdown: .02, position: 10 },
    ],
    trades: [{ timestamp: '2024-01-02T10:00:00', side: 'buy', quantity: 10, price: 10, fees: 1, cash: 0, shares: 10, reason }],
    rejections: [{ timestamp: '2024-01-02T10:05:00', side: 'buy', reason: hidden ? 'matched_funding_hold_rejection' : 'insufficient_cash', requested_quantity: 10 }],
  })
  const current = {
    symbol: '600000.SH', schema_version: 3, result_schema_version: 2, accounting_basis: 'peak_principal_v2',
    config: { schema_version: 3, frequency: '1d', start_date: '2024-01-01', end_date: '2024-02-02', initial_shares: 10 },
    funding: { reference_date: '2024-01-01', peak_invested: 100, external_topups: 0, matched_funding_hold_invested: 100 },
    strategies: {
      grid: makeStrategy('grid_buy'),
      matched_funding_hold: makeStrategy('matched_funding_hold_trade', true),
      same_account_passive: makeStrategy('initial_position'),
      dca: makeStrategy('dca'),
      buy_hold: makeStrategy('buy_hold'),
    },
  }
  const legacy = {
    symbol: '600000.SH', config: { initial_cash: 100 },
    strategies: {
      grid: { metrics: {}, equity: [{ timestamp: '2024-01-02', equity: 110, drawdown: .03 }], trades: [], rejections: [] },
      same_account_passive: makeStrategy('initial_position'),
    },
  }
  current.strategies.buy_hold.metrics.peak_principal_return = .04
  current.strategies.dca.metrics.peak_principal_return = .07
  const comparison = structuredClone(current)
  comparison.strategies.buy_hold.metrics.peak_principal_return = .05
  const summaries = ['current', 'comparison', 'legacy'].map(id => ({ id, kind: 'backtest', title: id, symbol: '600000.SH', created_at: '2024-02-03' }))
  await page.route('**/api/v1/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = path === '/api/v1/results' ? summaries
      : path === '/api/v1/results/legacy' ? legacy
      : path === '/api/v1/results/comparison' ? comparison
      : path.startsWith('/api/v1/results/') ? current
      : path === '/api/v1/jobs/history' ? { items: [], has_more: false }
      : []
    await route.fulfill({ json: body })
  })
  const pageErrors: string[] = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await page.goto('/?page=results&result=current&compare=comparison')
  const results = page.locator('.results-page')
  await expect(results.locator('.strategy-card')).toHaveCount(4)
  await expect(results.locator('.results-compare-metrics tbody tr')).toHaveCount(8)
  await expect(results.locator('.results-compare-metrics tbody tr').filter({ hasText: '超额收益' }).locator('td')).toHaveText(['超额收益（较买入持有）', '+6.00 个百分点', '+5.00 个百分点', '+1.00 个百分点'])
  for (const summary of await results.locator('[data-disclosure] > summary').all()) await summary.click()
  expect(await results.textContent()).not.toMatch(/同资金|matched_funding_hold/)

  const expectedHeaders = ['指标', '网格策略', '同账户被动持有', '定期定额', '买入并持有']
  const metrics = results.locator('.results-full-metrics')
  await expect(metrics.getByRole('heading', { name: '完整收益指标' })).toBeVisible()
  await expect(metrics.locator('thead th')).toHaveText(expectedHeaders)
  const excessRow = metrics.locator('tbody tr').filter({ hasText: '超额收益' })
  await expect(excessRow.locator('td')).toHaveText(['超额收益（较买入持有）', '+6.00 个百分点', '+6.00 个百分点', '+3.00 个百分点', '0.00 个百分点'])
  await expect(results.locator('.strategy-card.grid')).toContainText('+6.00 个百分点')

  for (const width of [1560, 1100, 720, 390]) {
    await page.setViewportSize({ width, height: 1040 })
    const chartBox = await results.locator('.results-curves-card').boundingBox()
    const metricsBox = await metrics.boundingBox()
    const parametersBox = await results.locator('.results-performance-details .result-parameters').boundingBox()
    const layoutBox = await results.locator('.results-performance-grid').boundingBox()
    expect(chartBox).not.toBeNull()
    expect(metricsBox).not.toBeNull()
    expect(parametersBox).not.toBeNull()
    expect(parametersBox!.y).toBeGreaterThanOrEqual(metricsBox!.y + metricsBox!.height)
    expect(Math.abs(parametersBox!.x - metricsBox!.x)).toBeLessThan(2)
    // Narrow only the horizontal axis: the chart keeps its full vertical size.
    const plotBox = await results.locator('.results-curves [role="img"]').boundingBox()
    expect(plotBox!.height).toBeGreaterThanOrEqual(620)
    expect(chartBox!.y + chartBox!.height - plotBox!.y - plotBox!.height).toBeLessThanOrEqual(16)
    const chartFont = await results.locator('.results-curves [role="img"]').evaluate(el => getComputedStyle(el).transform)
    expect(chartFont).toBe('none')
    if (width > 1100) {
      expect(Math.abs(chartBox!.y - metricsBox!.y)).toBeLessThan(2)
      expect(metricsBox!.x).toBeGreaterThan(chartBox!.x + chartBox!.width)
      expect(chartBox!.width / layoutBox!.width).toBeGreaterThan(.42)
      expect(chartBox!.width / layoutBox!.width).toBeLessThan(.45)
      expect(Math.abs(parametersBox!.y + parametersBox!.height - chartBox!.y - chartBox!.height)).toBeLessThan(2)
    } else {
      expect(metricsBox!.y).toBeGreaterThanOrEqual(chartBox!.y + chartBox!.height)
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
    const tableSize = await metrics.locator('.table-viewport').evaluate(el => ({ h: el.clientHeight, sh: el.scrollHeight, w: el.clientWidth, sw: el.scrollWidth }))
    expect(tableSize.sh).toBeLessThanOrEqual(tableSize.h + 1)
    expect(tableSize.sw).toBeLessThanOrEqual(tableSize.w + 1)
  }
  await page.setViewportSize({ width: 1560, height: 1040 })
  await results.locator('.results-performance-grid').screenshot({ path: '../.local/results-analysis-layout-1560.png' })
  const periods = results.locator('[data-disclosure="results-period-returns"]')
  await expect(periods.locator('thead th')).toHaveText(['区间', ...expectedHeaders.slice(1)])
  await expect(periods.locator('tbody tr')).toHaveCount(1)
  await periods.getByRole('button', { name: '月度', exact: true }).click()
  await expect(periods.locator('tbody tr')).toHaveCount(2)
  await expect(periods.locator('thead th')).toHaveText(['区间', ...expectedHeaders.slice(1)])

  for (const label of ['查看策略成交', '查看策略拒单']) {
    const selector = page.getByLabel(label)
    await expect(selector.locator('option')).toHaveText(expectedHeaders.slice(1))
    await selector.selectOption('buy_hold')
    await expect(selector).toHaveValue('buy_hold')
  }
  await expect(results.locator('[data-disclosure="results-trades"] tbody')).toContainText('初始全仓')
  await expect(results.locator('[data-disclosure="results-rejections"] tbody')).toContainText('现金不足')
  await expect(results.locator('[data-disclosure="results-funding"]')).toContainText('peak_invested')

  const chart = results.locator('.results-curves [role="img"]')
  await expect(chart.locator('canvas')).toBeVisible()
  const chartData = await chart.evaluate(async element => {
    const modulePath = '/src/chartSetup.ts'
    const { default: echarts } = await import(modulePath)
    const chart = echarts.getInstanceByDom(element)
    const option = chart.getOption()
    return {
      legend: option.legend[0].data,
      series: option.series.map((series: { id: string }) => series.id),
      tooltip: option.tooltip[0].formatter([{ value: [Date.parse('2024-01-02'), 0] }]),
      firstDate: option.xAxis[0].min,
      grids: [0, 1, 2].map(index => {
        const rect = chart.getModel().getComponent('grid', index).coordinateSystem.getRect()
        return { y: rect.y, height: rect.height }
      }),
    }
  })
  expect(chartData.legend).toEqual(['网格策略', '同账户被动持有', '定投', '买入并持有'])
  expect(chartData.series).toHaveLength(9)
  expect(JSON.stringify(chartData)).not.toMatch(/同资金|matched_funding_hold/)
  expect(chartData.tooltip).toContain('同账户被动持有')
  expect(chartData.firstDate).toBe(Date.parse('2024-01-02'))
  expect(Math.abs(chartData.grids[0].height - chartData.grids[1].height)).toBeLessThan(1)
  expect(Math.abs(chartData.grids[1].height - chartData.grids[2].height)).toBeLessThan(1)
  const gaps = [0, 1].map(index => chartData.grids[index + 1].y - chartData.grids[index].y - chartData.grids[index].height)
  expect(Math.abs(gaps[0] - gaps[1])).toBeLessThan(1)
  expect(gaps[0]).toBeGreaterThan(20)

  await page.getByLabel('选择对照结果').selectOption('')
  await page.getByLabel('选择回测结果').selectOption('legacy')
  await expect(results.locator('.strategy-card')).toHaveCount(2)
  await expect(metrics.locator('tbody tr').filter({ hasText: '超额收益' }).locator('td')).toHaveText(['超额收益（较买入持有）', '—', '—'])
  await expect(page.getByLabel('查看策略成交')).toHaveValue('grid')
  await expect(page.getByLabel('查看策略成交').locator('option')).toHaveText(['网格策略', '同账户被动持有'])
  await expect(results.locator('[data-disclosure="results-trades"]')).toContainText('没有成交记录')
  await expect(results.locator('[data-disclosure="results-funding"]')).toContainText('历史回测口径')
  expect(await results.textContent()).not.toMatch(/同资金|matched_funding_hold/)
  expect(pageErrors).toEqual([])
})
