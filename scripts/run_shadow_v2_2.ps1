# Session-proof launcher for the v2.2 portfolio shadow bot (tcn_v2_2 lock:
# calibrated ensemble + base-5/boost-7.5 sizing, DD budget -15).
# Registered as scheduled task BTCResearch_ShadowV2_2. Capture OFF (v1 archives feeds).
$ErrorActionPreference = 'Continue'
Set-Location ''

$env:TCN_ARTIFACTS_DIRS = 'artifacts\tcn_v2_c64_b7_ttc15_150_seed11;artifacts\tcn_v2_c64_b7_ttc15_150_seed7;artifacts\tcn_v2_c64_b7_ttc15_150_seed23;artifacts\tcn_v2_c64_b7_ttc15_150_seed42;artifacts\tcn_v2_c64_b7_ttc15_150_seed101'
$env:TCN_DATASET_DIR = 'data\datasets\btc_5m_episodes_v2_200ms'
$env:TCN_STRATEGY_LOCK = 'artifacts\strategy_locks\tcn_v2_2_lock.json'
$env:TCN_DIRECT_CAPTURE = '0'
$env:TCN_MAX_PM_SOURCE_AGE_S = '1.0'
$env:TCN_EV_MARGIN = '0.001'
$env:FV_PM_BOOK_SOURCE = 'direct'
$env:FV_FEATURE_SOURCE = 'direct'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

& python -m live_bot.tcn_shadow_bot `
    --decisions-dir logs\tcn_v2_2_shadow --log-level INFO `
    *>> logs\tcn_v2_2_shadow\runner.log
