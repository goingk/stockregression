$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '未安装 uv。请按 README 安装 uv 后重新启动。'
}
& uv sync --frozen --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python 依赖安装失败，请查看以上输出。' }
$frontendRoot = Join-Path $PSScriptRoot 'frontend'
$builtIndex = Join-Path $frontendRoot 'dist\index.html'
$needsBuild = -not (Test-Path -LiteralPath $builtIndex)
if (-not $needsBuild) {
    $builtTime = (Get-Item -LiteralPath $builtIndex).LastWriteTimeUtc
    $sourceFiles = @(Get-ChildItem -LiteralPath (Join-Path $frontendRoot 'src') -Recurse -File) + @(Get-Item -LiteralPath (Join-Path $frontendRoot 'package.json'), (Join-Path $frontendRoot 'package-lock.json'), (Join-Path $frontendRoot 'index.html'), (Join-Path $frontendRoot 'vite.config.ts'))
    $sourceFiles += @(Get-ChildItem -LiteralPath $frontendRoot -Filter 'tsconfig*.json' -File)
    $needsBuild = @($sourceFiles | Where-Object { $_.LastWriteTimeUtc -gt $builtTime }).Count -gt 0
}
if ($needsBuild) {
    Push-Location -LiteralPath (Join-Path $PSScriptRoot 'frontend')
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw '前端依赖安装失败。' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw '前端构建失败。' }
    } finally { Pop-Location }
}
$servePort = if ($env:STOCKREGRESSION_SERVE_PORT) { [int]$env:STOCKREGRESSION_SERVE_PORT } else { 8000 }
$serveArgs = @('run', '--frozen', 'stockregression', 'serve', '--port', $servePort)
if ($env:STOCKREGRESSION_NO_BROWSER -ne '1') { $serveArgs += '--open-browser' }
& uv @serveArgs
