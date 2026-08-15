# ============================================================================
# COMMAND 2 of 2 - start the live bot (real orders).
#
# The bot is now SELF-CONTAINED: it computes its own bias in-process from the
# raw recorder files every ~90 s. No need to run compute_bias.ps1 first, no
# need to run canonical_daemon.py separately. Only TWO PowerShell windows are
# required end-to-end:
#   (1) .\Start-AllRecorders.ps1   -- the recorder stack
#   (2) .\scripts\start_live_bot.ps1 -MinBet  -- this script
# That's it.
#
# -MinBet : validation mode. Forces every order to target the ~$1 Polymarket
#           minimum (MIN=MAX notional = 1.0). Use this for live truth-check;
#           drop the flag for full ($1-$2) sizing.
# ============================================================================
param([switch]$MinBet)

$ErrorActionPreference = "Stop"
Set-Location ""

if ($MinBet) {
    $env:LIVE_BOT_MIN_NOTIONAL = "1.0"
    $env:LIVE_BOT_MAX_NOTIONAL = "1.0"
    Write-Host '[start_live_bot] *** MIN-BET TEST MODE: every order targets the ~$1 minimum ***'
} else {
    Write-Host "[start_live_bot] Full sizing (`$1-`$2). Pass -MinBet for the ~`$1 validation run."
}

$env:ENABLE_REAL_ORDERS = "1"
Write-Host "[start_live_bot] Starting live bot. Bias is computed in-process every ~90s; no compute_bias step needed."
Write-Host "[start_live_bot] (override refresh cadence via LIVE_BOT_BIAS_REFRESH_S, window via LIVE_BOT_BIAS_WINDOW_HOURS)"
py -3 -m live_bot.main --live
