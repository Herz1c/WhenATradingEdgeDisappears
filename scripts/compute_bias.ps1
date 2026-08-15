# ============================================================================
# COMMAND 1 of 2 - compute today's canonical bias.
#
# Builds ONLY the live_reference_events_v1 canonical (the Chainlink-residual
# bias the bot reads). It does NOT build the heavy dense_close / tabular
# datasets. Features are computed exactly like the training/testing datasets
# (same code path, same inputs; no RTDS, no on-chain).
#
# Exit 0 => fresh active bias is ready; safe to run COMMAND 2 (start bot).
# Exit 1 => bias not ready; do NOT start the bot.
# ============================================================================
$ErrorActionPreference = "Stop"
Set-Location ""

$today = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
Write-Host "[compute_bias] Building canonical bias for $today (UTC) ..."

py -3 -u -m chainlink_recorder.cli build-live-reference-events --date-from $today --date-to $today --max-workers 8
if ($LASTEXITCODE -ne 0) {
    Write-Host "[compute_bias] build FAILED (exit $LASTEXITCODE)"
    exit 1
}

Write-Host "[compute_bias] Verifying the bias is fresh and active ..."
py -3 scripts\check_bias_ready.py
$ready = $LASTEXITCODE
if ($ready -ne 0) {
    Write-Host "[compute_bias] RESULT: NOT READY - do not start the bot yet."
    exit 1
}
Write-Host "[compute_bias] RESULT: READY - you can run scripts\start_live_bot.ps1"
exit 0
