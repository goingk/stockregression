# 系统问题排查记录（2026-10-28）

本次针对任务生命周期与持久化、行情同步和版本读取、研究与验证计算、前端请求与导航状态做了代码审计和确定性故障注入。真实数据仅做只读检查；复现使用临时数据或内存夹具。没有重新下载行情、修改 Windows 计划或重启运行中的服务。

## 已修复并回归的两项问题

### S01：省略同步范围导致自动更新失败

- 位置：`src/stockregression/cli.py`、`src/stockregression/jobs.py`。
- 本机证据：2026-10-26、2026-10-27 18:30 的 `market_sync` 任务均失败，错误为“同步必须指定标的”。Windows Daily/Weekly 计划仍分别调用旧命令 `sync --mode daily`、`sync --mode repair`，没有传 scope。worker 将缺省范围解析为空目标。
- 修复：CLI 与 worker 共用范围规范化；有代码时默认为 selected，无代码时默认为 watchlist。显式 selected 空目标继续报错；空自选或没有目标的基础周期正常跳过。已有逐周期冻结目标继续复用，不随当前自选变化而扩大。
- 验证：新增 17 项回归，包括非空自选、排除其他本地证券、恢复冻结范围、市场范围带冻结标的、空冻结周期、损坏和不完整冻结映射。显式存在但损坏的冻结映射会失败，不改用当前范围；完整市场冻结任务保留其原始范围。
- 部署边界：旧计划下一次调用新代码会更新自选。要与 README 所述“数据包已有证券”范围一致，需要显式重新安装系统计划；现有安装命令为 `uv run --frozen stockregression schedule install`。本次没有修改计划范围。

### S02：任务配置文档与实际执行设置可能来自不同 revision

- 位置：`src/stockregression/configuration.py:310`。
- 触发：`task_configuration_snapshot()` 读取 document 后，再次从文件读取 settings；两次读取之间发生自动保存。
- 复现：document 的 UI 轮询配置为 2000，settings 却变为后续保存的 9876，任务记录的 revision 仍为旧值。
- 修复：从同一 document 派生 settings，按项目根目录解析相对路径，再覆盖运行中实际生效的数据目录、包目录、监听主机和端口。
- 验证：确定性模拟读取间自动保存，并检查研究参数、UI、revision、相对路径和 active 覆盖；已通过独立 Reviewer 审查。

## 剩余发现：尚未修改实现

### S03 / P1：评价异常可能被当作完整的无候选结论

- 位置：`research/robust.py:753`、`research/robust.py:1410`、`research_v5.py:191`、`validation_runner.py:115`。
- 触发：批量训练评价器抛异常，坐标变为 failed，但搜索状态仍可 completed；无推荐参数时分支为 no_qualified。
- 故障注入结果：3 个评价全部失败，`score_complete=false`、`completion_status=available`，业务完整性门禁仍返回 `(True, None)`。
- 影响：内部计算失败可能被解释为有证据支持的“没有合格候选”，污染研究完整性与证据结论。
- 修复方向：将异常评价失败显式传递到搜索/验证完整性状态；保留正常完成、但确实无合格候选的合法结论，不能仅用通用 score_complete 一刀切。

### S04 / P1：中断的验证单元漏记父任务预算

- 位置：`validation_runner.py:938`、`validation_runner.py:982`、`validation_runner.py:1008`。
- 触发：执行前只检查预算，成功返回才记账；资格/压力等单元启动后抛出异常或 InterruptedError，异常分支保存单元却不登记已启动工作。
- 故障注入结果：总预算 8、候选已消费 7；资格回测中断后恢复，实际启动 2 次，最终只记录 1 次 ledger evaluation。
- 影响：反复中断和恢复可以超过冻结的评价预算。
- 修复方向：执行前持久化预留，并按尝试标识保证扣费与恢复幂等；明确失败工作计费和旧检查点迁移策略。

### S05 / P1：行情读取期间并发发布，响应可能混用两个版本

- 位置：`market_data/service.py:573`、`market_data/service.py:604`，对应 repository 的当前版本读取。
- 触发：服务层先取得 dataset v1，仓储层未传固定版本而再次读取 current；此间发布 v2。
- 确定性复现：响应 `version=1`、`last_time=2026-01-05`，行数据却含 v2 新增的 `2026-01-06`。
- 影响：版本标识、行情行、范围及质量信息不一致；图表分页与后续版本引用可能据此作出错误判断。本次未据此推断冻结回测必然被污染。
- 修复方向：固定一次读视图，行数据与元数据均使用同一版本；`last_bars`、聚合及复权依赖也要遵守同一读视图。

### S06 / P2 口径待明确：末日收盘后的补款被接受，但永不到账且仍可合格

- 位置：`validation_planner.py:192`、`share_engine.py:639`、`share_engine.py:647`、`share_engine.py:848`。
- 触发：预检仅按自然日检查；评价末日 16:00 的补款晚于最后 15:00 bar，但仍在允许的日期范围内。
- 复现：预算 11000、1000 元计划补款，得到 `received_contributions=0`、`unreceived_contributions=1000`、`total_invested=10000`，同时 `completed/qualified`。
- 影响：未到账补款已从初始现金预扣，该金额在整个回测内不可使用。结果确实暴露 outside_backtest 和未到账证据；总预算又被定义为包含全部计划补款，因此这不应直接定性为算术错误。需要明确产品是否允许评价期内未能到账的计划金额，并据此决定预检和资格口径。
- 修复方向：按冻结行情可执行时间轴检查计划补款，或把 outside_backtest 补款传递为不完整/不可评价状态；盘中补款和压力情景的既有语义应保持一致。

### S07 / P2：正式行情预览忽略取消和准备缓存

- 位置：`data.py:197-208`、`preview.py:145`、`market_data/consumer.py:97`。
- 触发：market_data 分支无条件 readiness + load_backtest_bundle，未使用 cancelled/cached_bundle；后续旧缓存与取消逻辑不可达。
- 复现：传入立即返回真的取消回调和匹配版本的缓存，取消回调调用数为 0；两次请求仍做两次完整加载，未复用缓存。取消页面状态后，已阻塞的底层加载仍未停止。
- 影响：大范围分钟预览取消后继续占用资源；同区间调参会重复准备行情。
- 修复方向：恢复有效缓存并保留完整请求身份与冻结版本校验；在准备阶段边界检查取消，大数据读取按分片/月份提供可取消边界。

### S08 / P2：行情展示周期被当作网络同步周期

- 位置：`frontend/src/MarketPanel.tsx:91`、`api.py:1673`。
- 触发：切换到 5m、周线等派生图表周期，再点击任一行情更新按钮。
- 结果：请求携带派生 frequency，而后端同步只支持 1day/1min，返回 422。
- 注意：默认 1d 会被后端规范化为 1day，默认日线不属于此问题。
- 修复方向：更新基础日线/分钟，再按需派生图表周期；增加非基础周期下更新的浏览器测试。

### S09 / P2：补算界面显示决策与实际提交决策不一致

- 位置：`frontend/src/pages/ResearchPage.tsx:693`、`:785`。
- 触发：导航状态残留 adaptive:2，当前结果没有该决策；界面按校验值回退 current:0，提交函数却仍读取原始导航值。
- 影响：用户看到 current:0，却为当前结果提交 adaptive:2，可能创建失败的后台任务。
- 修复方向：显示、分支判定和提交统一使用同一个经过当前结果验证的决策值；后端同步校验决策是否存在。

### S10 / P1：同步与研究补算的恢复接口不幂等

- 位置：`api.py:2562`、`api.py:2573`。
- 复现：对同一可恢复来源连续调用两次 resume，创建两个不同 queued 任务。
- 影响：重复执行同一冻结工作；研究补算的任务还共享 checkpoint 身份。重复预算消耗或修订冲突的具体后果需进一步并发测试，不能仅凭重复建任务断言数据已损坏。
- 修复方向：为恢复谱系设置稳定 dedupe key，并与活动任务检查/创建放在同一原子边界内。
- 另一个未修复的恢复边界：`api.py` 构造恢复任务时，如果旧冻结映射缺少一个周期，会用恢复出的标的并集补齐该周期。虽然 worker 现已拒绝原始不完整映射，HTTP 恢复入口预先补齐后仍可能扩大原逐周期范围。需要在入口也拒绝不完整的显式冻结映射，不能把当前 CLI/worker 测试通过解释为该入口已端到端修复。

## 条件性健壮性问题与完整性边界

- `jobs.py` 对存在但无法验证的 frozen settings 静默回退到当前配置。正常同版本提交不触发，升级后快照不兼容或 payload 损坏时可能违反冻结契约；宜明确报错，只有完全缺失快照的旧任务才进入兼容处理。
- 结果 JSON 丢失或损坏时，存储读取异常会形成裸 500，缺少结构化的“结果不可读”诊断。本机 92 条结果记录对应文件均存在，本次未发现真实丢失。
- 索引重建检查文件存在性和大小，不等于逐文件 SHA-256 验证。同长度损坏可能在重建后仍显示 ready；但项目已明确将完整哈希、行数及时间范围检查交给 `verify-package --full`。本次将其作为完整性边界记录，未直接认定为必须在每次快速重建时全量扫描的缺陷。

## 验证记录

- 前端单元测试：178 passed。
- TypeScript 类型检查：通过。
- Playwright：默认 9 worker 首轮 3 项超时；限制为 1 worker、60 秒测试上限后，10/10 通过（17.1 秒）。未把首轮超时报为确定功能缺陷。
- Vite 生产构建：通过；单独构建，没有重新生成或覆盖用户已有 API schema 改动。
- 配置、性能、偏好与既有审查回归：55 passed。
- 同步范围、恢复、发布围栏、进度回归：最终补丁 62 passed，含 17 项新同步范围测试。
- Reviewer 独立配置测试：12 passed。常见警告为既有 Starlette/httpx 弃用提示。
- Reviewer 独立同步范围测试：最终 17 passed；CLI/worker 实现验收通过，HTTP 恢复入口的不完整旧映射边界另记于 S10，尚未修复。
- Reviewer 已复核剩余发现：确认研究异常误判、预算漏计、恢复重复入队/共享检查点、行情混读、预览取消/缓存和两项前端流程问题；独立复现业务完整性门禁与重复补算恢复。将 S06 收盘后补款降为业务口径待明确；哈希重建项未列为已确认缺陷。其他故障注入来自 Executor，Reviewer 未逐项重跑完整实验。
- 主智能体已独立重新执行 S03、S04、S06 三个完整复现，断言全部符合上述现象。可复制脚本保存在本机 `.local/system-audit-20261028/repro_01_failed_scores_complete.py`、`repro_02_budget_interruption_resume.py`、`repro_03_contribution_after_last_bar.py`，均只使用内存测试夹具，无真实数据写入。
- 主智能体也已在显式临时数据目录和临时包目录独立复现 S05：响应标识为 v1、最后时间为 1 月 5 日，却包含 v2 的 1 月 6 日行数据。
- 修复前启动的后端全量 pytest 已完成：1058 passed、1 failed、1 warning，耗时 18 分 13 秒。唯一失败是 `test_share_workflow.py::test_market_dry_run_reports_universe_without_creating_data_job` 的 ImportError：进程先导入修改前的 jobs 模块，稍后才导入已修改、引用新 normalize_sync_scope 的 cli 模块，形成运行期间跨源码版本的混合导入。随后在新进程重跑该失败用例、新同步范围用例及配置用例，30 passed。结合最终补丁同步相关 62 passed，可确认该导入错误未在新进程复现；本次仍没有宣称“最终源码重新跑完全部 1059 项且全绿”。
- 本机健康接口 200；没有 running/queued 任务或资源租约；92 条结果文件均存在。没有宣称已验证全部行情文件内容和所有历史结果计算正确性。

## 当前变更范围

实现仅修改 `configuration.py`、`cli.py`、`jobs.py`，测试修改 `test_configuration.py` 并新增 `test_sync_scope_regressions.py`。用户原有 `frontend/src/generated/api-schema.d.ts` 改动保留。其余发现尚未实施修复；运行中的后端也没有重启。
