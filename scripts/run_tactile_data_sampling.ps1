param(
    [string]$TactileLabRoot = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master",
    [string]$PythonExe = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe",
    [string]$NameOutput = "",
    [string]$Task = "Obj-Push-Tactile-v0",
    [ValidateRange(1, 1024)]
    [int]$NumEnvs = 1,
    [ValidateRange(1, 1000000000)]
    [int]$Samples = 256,
    [ValidateRange(0, 1000000000)]
    [int]$WarmupSteps = 30,
    [ValidateRange(1, 1000000000)]
    [int]$StepsPerSample = 4,
    [ValidateSet("zero", "random", "sweep")]
    [string]$ActionMode = "random",
    [double]$ActionScale = 0.35,
    [int]$Seed = 1,
    [string]$OutputRoot = "",
    [int]$TactileImgSize = 32,
    [ValidateSet("depth", "depth_original", "rgb")]
    [string]$TactileImageType = "depth",
    [string]$SensorType = "right_angle_tactip",
    [double]$CameraUpdatePeriod = 0.0,
    [switch]$Gui,
    [switch]$SavePng,
    [switch]$RenderTactile,
    [switch]$ResetOutput
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$collectorScript = Join-Path $scriptRoot "collect_tactile_samples.py"

if (-not (Test-Path -LiteralPath $TactileLabRoot)) {
    throw "Tactile_Lab root not found: $TactileLabRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $collectorScript)) {
    throw "Collector script not found: $collectorScript"
}

if ([string]::IsNullOrWhiteSpace($NameOutput)) {
    $NameOutput = "tactile_samples_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $TactileLabRoot "outputs\tactile_samples"
}

$datasetDir = Join-Path $OutputRoot $NameOutput

Write-Host ""
Write-Host "==== Tactile_Lab TacTip data sampling ===="
Write-Host "[INFO] dataset=$NameOutput"
Write-Host "[INFO] task=$Task | envs=$NumEnvs | samples=$Samples | action=$ActionMode"
Write-Host "[INFO] tactile=$TactileImageType ${TactileImgSize}x$TactileImgSize | sensor=$SensorType"
Write-Host "[INFO] output=$datasetDir"

$cleanPath = (($env:PATH -split ";") | Where-Object {
    $_ -and ($_ -notlike "*WindowsApps*")
}) -join ";"
Remove-Item Env:PATH -ErrorAction SilentlyContinue
Remove-Item Env:Path -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("Path", $cleanPath, "Process")
$env:Path = $cleanPath

$collectArgs = @(
    $collectorScript,
    "--task", $Task,
    "--num_envs", "$NumEnvs",
    "--samples", "$Samples",
    "--warmup_steps", "$WarmupSteps",
    "--steps_per_sample", "$StepsPerSample",
    "--action_mode", $ActionMode,
    "--action_scale", "$ActionScale",
    "--seed", "$Seed",
    "--output_root", $OutputRoot,
    "--name_output", $NameOutput,
    "--tactile_img_size", "$TactileImgSize",
    "--tactile_image_type", $TactileImageType,
    "--sensor_type", $SensorType,
    "--camera_update_period", "$CameraUpdatePeriod",
    "--enable_cameras"
)

if (-not $Gui) {
    $collectArgs += "--headless"
}
if ($SavePng) {
    $collectArgs += "--save_png"
}
if ($RenderTactile) {
    $collectArgs += "--render_tactile"
}
if ($ResetOutput) {
    $collectArgs += "--reset_output"
}

Push-Location $TactileLabRoot
try {
    $process = Start-Process -FilePath $PythonExe -ArgumentList $collectArgs -WorkingDirectory $TactileLabRoot -NoNewWindow -Wait -PassThru
}
finally {
    Pop-Location
}

if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}

Write-Host ""
Write-Host "[DONE] Dataset: $(Join-Path $datasetDir 'dataset.npz')"
Write-Host "[DONE] Manifest: $(Join-Path $datasetDir 'manifest.csv')"
