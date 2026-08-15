param(
    [string]$OutDir = ".\data",
    [string]$LogLevel = "INFO"
)

if ($env:CHAINLINK_BACKEND) {
    Write-Host "CHAINLINK_BACKEND:    $($env:CHAINLINK_BACKEND)"
}
else {
    Write-Host "CHAINLINK_BACKEND:    auto (direct Data Streams WS when configured)"
}
Write-Host "PUBLIC_DELAYED:       use ENABLE_CHAINLINK_PUBLIC_DELAYED=0 to disable the delayed comparison layer"

& "$PSScriptRoot\Start-AllRecorders.ps1" -Profile chainlink -OutDir $OutDir -LogLevel $LogLevel
