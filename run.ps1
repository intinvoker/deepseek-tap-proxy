# DeepSeek Tap Proxy — 一键启动 (PowerShell)
#   .\run.ps1                         默认 msg0 锁
#   .\run.ps1 -LockMode prefix        前缀锁
#   .\run.ps1 -LockMode none          禁用锁
#   .\run.ps1 -DisableLock            等价
param(
    [int]$Port = 11808,
    [string]$Target = "https://api.deepseek.com",
    [ValidateSet("msg0", "prefix", "none")]
    [string]$LockMode = "prefix",
    [switch]$DisableLock
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── 本地虚拟环境 ──
$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts" "python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  [init] 创建虚拟环境 .venv ..." -ForegroundColor DarkYellow
    $systemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $systemPython) { $systemPython = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
    if (-not $systemPython) { Write-Error "Python not found"; exit 1 }
    & $systemPython -m venv $VenvDir
    if (-not (Test-Path $VenvPython)) { Write-Error "虚拟环境创建失败"; exit 1 }
    Write-Host "  [init] 安装依赖 ..." -ForegroundColor DarkYellow
    & $VenvPython -m pip install -e $PSScriptRoot --quiet
    if ($LASTEXITCODE -ne 0) { Write-Error "依赖安装失败"; exit 1 }
    Write-Host "  [init] 完成" -ForegroundColor Green
}

$python = $VenvPython

$lock = if ($DisableLock) { "none" } else { $LockMode }

Write-Host ""
Write-Host "  ╔════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   DeepSeek Tap Proxy          ║" -ForegroundColor Cyan
Write-Host "  ╚════════════════════════════════╝" -ForegroundColor Cyan
Write-Host "  Port:    $Port" -ForegroundColor Gray
Write-Host "  Target:  $Target" -ForegroundColor Gray
Write-Host "  Lock:    $lock  (msg0 / prefix / none)" -ForegroundColor Gray
Write-Host "  Data:    $PSScriptRoot\data\" -ForegroundColor Gray
Write-Host ""

& $python -m deepseek_tap_proxy --port $Port --target $Target --lock-mode $lock
