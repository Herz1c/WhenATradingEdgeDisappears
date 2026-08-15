# ============================================================================
# Wrapper for the scheduled run: compute the bias, then (only if it's fresh
# and active) start the live bot. Single command for a Windows Scheduled Task.
# ============================================================================
$ErrorActionPreference = "Continue"
Set-Location ""

Write-Host "[run_scheduled] $(Get-Date -Format o) - computing bias ..."
& "$PSScriptRoot\compute_bias.ps1"
$ready = $LASTEXITCODE

if ($ready -ne 0) {
    Write-Host "[run_scheduled] bias NOT ready (exit $ready) - NOT starting the bot."
    exit 1
}

Write-Host "[run_scheduled] bias ready - starting live bot ..."
& "$PSScriptRoot\start_live_bot.ps1"
