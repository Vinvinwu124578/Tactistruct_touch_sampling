param(
    [switch]$ShowPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$cleanPath = (($env:PATH -split ";") | Where-Object {
    $_ -and ($_ -notlike "*WindowsApps*")
}) -join ";"
Remove-Item Env:PATH -ErrorAction SilentlyContinue
Remove-Item Env:Path -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("Path", $cleanPath, "Process")
$env:Path = $cleanPath

if ($ShowPath) {
    Write-Host "WindowsApps in PATH before Python: $($env:PATH -like '*WindowsApps*')"
    Write-Host $env:PATH
}

$python = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe"
$process = Start-Process -FilePath $python -ArgumentList $PythonArgs -NoNewWindow -Wait -PassThru
exit $process.ExitCode
