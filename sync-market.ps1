param(
    [ValidateSet('history', 'daily', 'repair')][string]$Mode = 'daily',
    [ValidateSet('1min', '1day')][string[]]$Frequency = @('1day', '1min'),
    [string]$StartDate = '',
    [string]$EndDate = '',
    [switch]$RetryFailed,
    [switch]$Fresh,
    [switch]$Initialize,
    [switch]$InstallSchedule,
    [switch]$DryRun,
    [ValidateRange(1, 4)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $marketMode = if ($Mode -eq 'daily') { 'incremental' } elseif ($Mode -eq 'repair') { 'repair' } else { 'history' }
    if ($Frequency -contains '1min' -and $marketMode -eq 'history') { throw '1 分钟历史行情只从本地数据包读取；请先登记数据包或导入文件，近期缺口使用 -Mode repair。' }
    if ($Frequency -contains '1min' -and $Initialize) { throw '1 分钟行情不支持全市场在线初始化。' }
    if ($marketMode -ne 'incremental' -and -not $StartDate) { throw '历史导入或修补必须显式提供 -StartDate。' }
    $syncArgs = @('run', '--frozen', 'stockregression', 'data', 'sync', '--scope', 'market', '--mode', $marketMode)
    foreach ($item in $Frequency) { $syncArgs += @('--frequency', $item) }
    if ($StartDate) { $syncArgs += @('--start', $StartDate) }
    if ($EndDate) { $syncArgs += @('--end', $EndDate) }
    $syncArgs += @('--workers', $Workers)
    if (-not $Fresh) { $syncArgs += '--resume' }
    if ($RetryFailed) { $syncArgs += '--retry-failed' }
    if ($Initialize) { $syncArgs += '--initialize' }
    if ($DryRun) { Write-Output 'dry-run：只检查日线和 1 分钟基础行情；不会下载其他周期。'; return }
    & uv @syncArgs
    $syncExitCode = $LASTEXITCODE
    if ($InstallSchedule -and -not $DryRun) {
        & uv run --frozen stockregression schedule install
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    exit $syncExitCode
} finally {
    Pop-Location
}
