param(
    [string]$OutDir = ".\data",
    [string]$LogLevel = "INFO"
)

& "$PSScriptRoot\Start-AllRecorders.ps1" -Profile binance -OutDir $OutDir -LogLevel $LogLevel
