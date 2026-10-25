import React from 'react'
import { createRoot } from 'react-dom/client'
import TableViewport from '../../src/TableViewport'
import '../../src/styles.css'

function Fixture() {
  const rows = Array.from({ length: 42 }, (_, index) => index)
  const longText = '数据来源与请求范围说明：'.repeat(18)
  return <main style={{ padding: 16, width: '100%', boxSizing: 'border-box' }}>
    <h1>滚动布局检查</h1>
    <TableViewport owner="local" stickyKeyColumn={1} label="长表格本地滚动">
      <table style={{ minWidth: 1800 }}><thead><tr>{['标的', '周期', '状态', '数据缺口说明', '来源', '更新时间'].map(x => <th key={x}>{x}</th>)}</tr></thead><tbody>{rows.map(i => <tr key={i} className={i === 0 ? "selected-row" : ""}><td>标的-{i}</td><td>日线</td><td>待核验</td><td>{longText}</td><td>来源-{i}</td><td>2026-10-25</td></tr>)}</tbody></table>
    </TableViewport>
    <section className="task-detail-panel" aria-label="任务报告滚动面板" style={{ height: 480, flex: "none", overflow: "auto" }}>
      <h2>任务报告长文本</h2>
      <TableViewport owner="parent" stickyKeyColumn={1} label="父面板滚动报告表格">
        <table><thead><tr><th>标的</th><th>周期</th><th>失败原因和请求上下文</th></tr></thead><tbody>{rows.map(i => <tr key={i}><td>股票-{i}</td><td>5 分钟</td><td>{longText}</td></tr>)}</tbody></table>
      </TableViewport>
      <p>{longText.repeat(12)}</p>
    </section>
    <section className="task-drawer report-drawer" aria-label="市场报告滚动面板" style={{ height: 320, maxWidth: '100%', flex: 'none', marginTop: 20 }}>
      <h2>市场报告</h2>
      <TableViewport owner="parent" label="市场任务报告"><table><thead><tr><th>ETF</th><th>说明</th></tr></thead><tbody>{rows.map(i => <tr key={i}><td>ETF-{i}</td><td>{longText}</td></tr>)}</tbody></table></TableViewport>
    </section>
  </main>
}

createRoot(document.getElementById('root')!).render(<Fixture />)
