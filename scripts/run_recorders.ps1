# Session-proof launcher for the full recorder stack.
# Registered as scheduled task BTCResearch_Recorders.
$ErrorActionPreference = 'Continue'
Set-Location ''
& powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-AllRecorders.ps1 *>> logs\recorders_runner.log
