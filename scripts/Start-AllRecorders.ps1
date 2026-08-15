param(
    [string]$OutDir = ".\data",
    [string]$LogLevel = "INFO",
    [ValidateSet("all", "core", "external", "oracle", "research", "binance", "polymarket", "polymarket-rtds", "bybit", "hyperliquid", "deribit", "coinbase", "chainlink", "fng")]
    [string]$Profile = "all"
)

# Notes:
# - CHAINLINK_BACKEND may be set to auto, datastreams_api, or datastreams_ws before launch.
# - CHAINLINK_BACKEND=auto only starts direct Data Streams WS when API key, secret, and BTC/USD feed ID are configured.
# - If direct Data Streams is unavailable, the public live fallback is the separate Polymarket RTDS Chainlink BTC/USD service, not onchain/page equivalence.
# - ENABLE_CHAINLINK_PUBLIC_DELAYED=0 disables the delayed public Chainlink BTC/USD comparison recorder in shared profiles.
# - ENABLE_CHAINLINK_LIVE=0 disables the shared Chainlink live-reference service in shared profiles.
# - ENABLE_POLYMARKET_RTDS=0 disables the public Polymarket RTDS Chainlink BTC/USD stream in shared profiles.
# - ENABLE_POLYMARKET_STRIKE=0 disables the lean Polymarket strike recorder in shared profiles.
# - Profile selection only chooses which services start; per-service transport details still come from env/config.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$activateCandidates = @(
    ".venv\Scripts\Activate.ps1",
    "venv\Scripts\Activate.ps1"
)

$activated = $false
foreach ($candidate in $activateCandidates) {
    if (Test-Path $candidate) {
        . $candidate
        $activated = $true
        break
    }
}

$command = switch ($Profile) {
    "all" { "run-all" }
    "core" { "run-core-sources" }
    "external" { "run-external-venues" }
    "oracle" { "run-oracle-strike" }
    "research" { "run-research-stack" }
    "binance" { "run-binance" }
    "polymarket" { "run-polymarket" }
    "polymarket-rtds" { "run-polymarket-rtds" }
    "bybit" { "run-bybit" }
    "hyperliquid" { "run-hyperliquid" }
    "deribit" { "run-deribit" }
    "coinbase" { "run-coinbase" }
    "chainlink" { "run-chainlink" }
    "fng" { "run-fng" }
}

# --- Plan A: collapse the recorder->file->bot latency ---------------------
# The live bot tails these files; with the old 2s flush the model decided on
# ~2.5s-stale BTC (caused ~18% spurious entries). Flush ~5x/sec instead so the
# bot sees fresh data. Same bytes/data, just written sooner -> no dataset or
# feature-parity change. Env-overridable: set the var before launch to change.
foreach ($v in @(
    'BINANCE_FLUSH_INTERVAL_SECONDS',
    'HYPERLIQUID_FLUSH_INTERVAL_SECONDS',
    'POLYMARKET_FLUSH_INTERVAL_SECONDS',
    'CHAINLINK_FLUSH_INTERVAL_SECONDS',
    'CHAINLINK_ONCHAIN_FLUSH_INTERVAL_SECONDS',
    'CHAINLINK_PUBLIC_DELAYED_FLUSH_INTERVAL_SECONDS'
)) {
    if (-not (Test-Path "env:$v")) { Set-Item "env:$v" "0.05" }
}

Write-Host "Starting recorder stack in one console window"
Write-Host "  Flush interval:     0.05s (Plan A low-latency; override via *_FLUSH_INTERVAL_SECONDS)"
Write-Host "  Profile:            $Profile"
Write-Host "  Command:            py -m market_recorders $command"
Write-Host "  Output base path:   $OutDir"
Write-Host "  Log level:          $LogLevel"
Write-Host "  Enable flags:       ENV-driven overrides still apply in run-all mode"
if (-not $activated) {
    Write-Host "  Virtualenv:         not auto-activated (edit Start-AllRecorders.ps1 if your env lives elsewhere)"
}

py -m market_recorders $command --out $OutDir --log-level $LogLevel
