# Launch the poly_l2_only_bot (validated final strategy, $1.50 GTC entries).
#
# DRY-RUN by default. Pass --live-orders for REAL money.
# Green text prints when it places an order.
# Listens ONLY to Polymarket L2 data (no chainlink / binance / other sources).
#
# signature_type=2 is baked in: that is the proxy holding this wallet's USDC
# (verified $12.47 balance + allowance under sig_type 2; types 0/1/3 are empty).
#
# Usage:
#   .\scripts\start_poly_l2_bot.ps1                 # dry-run
#   .\scripts\start_poly_l2_bot.ps1 --live-orders   # live, real $1.50 orders

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
$env:PYTHONIOENCODING = "utf-8"
$env:POLYMARKET_SIGNATURE_TYPE = "2"   # the funded proxy for this wallet

if ($args -contains "--live-orders") {
    $env:ENABLE_REAL_ORDERS = "1"
    Write-Host "[start_poly_l2_bot] LIVE ORDERS enabled (real money, ~`$1.50/order, sig_type=2)" -ForegroundColor Red
} else {
    Remove-Item Env:\ENABLE_REAL_ORDERS -ErrorAction SilentlyContinue
    Write-Host "[start_poly_l2_bot] DRY-RUN (no real orders). Pass --live-orders to trade." -ForegroundColor Cyan
}

py -3 -u -m poly_l2_only_bot @args
