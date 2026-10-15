param(
    [ValidateSet('history', 'daily', 'repair')][string]$Mode = 'history',
    [switch]$RetryFailed,
    [switch]$Fresh,
    [switch]$InstallSchedule,
    [switch]$DryRun,
    [ValidateRange(1, 4)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $syncArgs = @('run', '--frozen', 'stockregression', 'sync', '--scope', 'market', '--mode', $Mode, '--workers', $Workers)
    if (-not $Fresh) { $syncArgs += '--resume' }
    if ($RetryFailed) { $syncArgs += '--retry-failed' }
    if ($DryRun) { $syncArgs += '--dry-run' }
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
