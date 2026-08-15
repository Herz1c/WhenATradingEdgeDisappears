param(
    [string]$OutDir = ".\data",
    [string]$LogLevel = "INFO"
)

& "$PSScriptRoot\Start-AllRecorders.ps1" -Profile polymarket -OutDir $OutDir -LogLevel $LogLevel
