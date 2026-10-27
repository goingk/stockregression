import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ResearchValidationLauncher, ResearchValidationReport } from '../../src/ResearchValidation'
import { TaskButton, TaskProvider } from '../../src/TaskCenter'
import '../../src/styles.css'

function researchFor(symbol: string, startDate: string) {
  return {
    research_context: {
      symbol,
      start_date: startDate,
      end_date: '2025-06-30',
      frequency: '1m',
      initial_shares: 1000,
      grid_mode: 'geometric',
      fees: { commission_rate: .0003, min_commission: .1, slippage_bps: 2 },
      reference_price_mode: 'auto',
      reference_price: null,
      lower_price: null,
      upper_price: null,
      dca: { frequency: 'monthly', shares: 100, initial_shares: 0 },
      parameter_links: { spacing: false, shares: false },
    },
    selection_target: 'both',
    rebalance_days: 63,
    seed: 42,
    ranges: { buy_step: { low: .01, high: .08 }, sell_step: { low: .01, high: .08 } },
    strategy: { strategy_id: 'geometric_robust_j', strategy_version: 5 },
    prior_policy: 'off',
  }
}

function LauncherFixture() {
  const [tick, setTick] = useState(0)
  const [startDate, setStartDate] = useState('2024-01-02')
  const [datesExplicit, setDatesExplicit] = useState(false)
  return <TaskProvider onOpenResult={() => undefined} filter='active' kind='' onFilterChange={() => undefined}>
    <main style={{ padding: 16 }}>
      <button id='rerender-parent' onClick={() => setTick(value => value + 1)}>同值重渲染 {tick}</button>
      <button id='change-research-date' onClick={() => { setStartDate('2024-02-01'); setDatesExplicit(true) }}>修改研究日期</button>
      <button id='change-research-date-again' onClick={() => { setStartDate('2024-03-01'); setDatesExplicit(true) }}>再次修改研究日期</button>
      <ResearchValidationLauncher
        buildResearch={() => researchFor('600000.SH', startDate)}
        initialSymbol='600000.SH'
        initialShares={1000}
        datesExplicit={datesExplicit}
        enabled
        onSubmitted={() => undefined}
      />
    </main>
  </TaskProvider>
}

const arms = ['same_price:off', 'same_price:daily_state_v1', 'fixed:off', 'fixed:daily_state_v1'].map((group, index) => {
  const [window_mode, prior_policy] = group.split(':')
  return {
    arm_group: group, window_mode, prior_policy, primary: index === 0,
    schemes: {
      fixed: {
        completion_status: 'complete',
        qualification: {
          status: 'completed', qualification_status: 'qualified',
          segment_returns: { segments: [{ block: 1, start: '2025-01-02', end: '2025-03-31', days: 58, principal_return: .06 + index / 100, max_principal_drawdown: .05, capital_added: 10000, starting_equity: 200000, ending_equity: 221000, status: 'completed' }] },
          strategy_metrics: { peak_principal_return: .02 + index / 100, max_principal_drawdown: .1 },
        },
      },
      adaptive: {
        completion_status: 'complete',
        qualification: {
          status: 'completed', qualification_status: 'qualified',
          segment_returns: { segments: [{ block: 1, start: '2025-01-02', end: '2025-03-31', days: 58, principal_return: index === 0 ? .12 : .07 + index / 100, max_principal_drawdown: .04, capital_added: 10000, starting_equity: 200000, ending_equity: 224000, status: 'completed' }] },
          strategy_metrics: {
            peak_principal_return: .03 + index / 100, return_on_committed_budget: .04,
            total_budget: 200000, initial_equity: 200000, initial_holdings_value: 30000, initial_cash: 170000,
            scheduled_contributions: 0, external_topups: 10000, total_invested: 210000, peak_invested: 210000,
            final_cash: 12345.67, final_equity: 234000, final_position: 2600, total_fees: 345.67, trade_count: 18,
            funding_diagnostics: { unspent_cash: 12345.67, cash_shortfall_action: 'cancel_remainder' },
            execution_diagnostics: {
              trade_fill_count: 18, rejection_count: 4, rejections_by_reason: { insufficient_cash: 2, limit_up_locked: 2 },
              day_end_cancel: true, participation_rate: .01, ttl_bars: 3,
            },
          },
          bounded_ledger: { trade_count: 18, rejection_count: 4, ending_cash: 12345.67, ending_equity: 234000, ending_position: 2600, ending_principal: 210000 },
        },
      },
    },
  }
})

const report = {
  id: 'validation-report-1',
  kind: 'research_validation',
  suite_version: 'research_validation_v1',
  created_at: '2026-10-27T10:00:00',
  completion_status: 'complete',
  candidate_status: 'not_qualified',
  qualification_status: 'no_candidate',
  evidence: { status: 'limited', reason: 'ETF 共同后续区间不足' },
  primary_arm: 'same_price:off:adaptive',
  plan: {
    plan_hash: 'frozen-validation-plan',
    status: 'ready',
    primary_arm: 'same_price:off:adaptive',
    symbols: [{
      symbol: '600000.SH', status: 'data_issue', available_days: 250,
      gaps: [{ kind: 'history', available_days: 250, additional_verified_days: 10, start: '2020-01-02', end: '2021-01-04', message: '连续可用历史250日，训练与样本外仍不足；未知缺口不拼接' }],
      blocks: [], training_days: 250, initial_cash: 0,
    }],
  },
  symbols: [{ symbol: '600000.SH', arms, blocks: [{ start: '2025-01-02', end: '2025-06-30', days: 120 }] }],
  cells: [{ kind: 'qualification', status: 'completed', result: { status: 'completed' } }],
  blocks: [{ start: '2025-01-02', end: '2025-06-30', days: 120 }],
  execution: { status: 'no_candidate', rejected_orders: 0 },
  pressure: [{ scenario: '低流动性', status: 'completed' }],
  scoring: { status: 'limited', paired_days: 120 },
  audit: { human_untouched_verified: false, view_events: [] },
}

function ReportFixture() {
  return <main style={{ padding: 16 }}><ResearchValidationReport result={report} resultId={report.id} /></main>
}

function TaskFixture() {
  return <TaskProvider onOpenResult={() => undefined} filter='all' kind='research_validation' onFilterChange={() => undefined}>
    <main style={{ padding: 16 }}><TaskButton /></main>
  </TaskProvider>
}

const mode = new URLSearchParams(location.search).get('mode')
createRoot(document.getElementById('root')!).render(mode === 'launcher' ? <LauncherFixture /> : mode === 'task' ? <TaskFixture /> : <ReportFixture />)
