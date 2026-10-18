param(
    [Parameter(Mandatory = $true)][int]$ParentPid,
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$logDirectory = Join-Path $root '.local'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$log = Join-Path $logDirectory 'restart.log'
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Process -Id $ParentPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
}
if (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue) {
    "$(Get-Date -Format o) 原服务未在 2 分钟内退出，取消重启" | Out-File -LiteralPath $log -Append -Encoding utf8
    exit 1
}
Start-Sleep -Seconds 1
Push-Location -LiteralPath $root
try {
    "$(Get-Date -Format o) 正在重新启动项目" | Out-File -LiteralPath $log -Append -Encoding utf8
    $env:STOCKREGRESSION_NO_BROWSER = '1'
    $env:STOCKREGRESSION_SERVE_PORT = [string]$Port
    & (Join-Path $root 'start.ps1') *>> $log
} catch {
    "$(Get-Date -Format o) 重启失败：$($_.Exception.Message)" | Out-File -LiteralPath $log -Append -Encoding utf8
    exit 1
} finally {
    Pop-Location
}
