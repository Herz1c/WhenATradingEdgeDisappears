# Session-proof launcher for the v1 lock shadow bot (tcn_double_strategy_v1).
# Registered as scheduled task BTCResearch_ShadowV1 so it survives CC session
# exits and reboots. Direct capture ON (this instance is the feed archiver).
$ErrorActionPreference = 'Continue'
Set-Location ''

$env:TCN_DIRECT_CAPTURE = '1'
$env:TCN_DIRECT_CAPTURE_ROOT = 'data\tcn_direct_capture_v6\raw'
$env:TCN_MAX_PM_SOURCE_AGE_S = '1.0'
$env:TCN_EV_MARGIN = '0.001'
$env:FV_PM_BOOK_SOURCE = 'direct'
$env:FV_FEATURE_SOURCE = 'direct'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

& python -m live_bot.tcn_shadow_bot `
    --decisions-dir logs\tcn_shadow_bot_direct_capture_v6 --log-level INFO `
    *>> logs\tcn_shadow_bot_direct_capture_v6\runner.log
