param(
    [string]$TactileLabRoot = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master",
    [string]$PythonExe = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe",
    [string]$TouchSDFRoot = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master",
    [string]$DatasetName = "touch_charts_gt_isaac_shapenet_smoke",
    [string]$DatasetPath = "",
    [int]$SampleIdx = 0,
    [switch]$PerObject,
    [string]$ObjectIndex = "",
    [ValidateSet("original_touch", "dashboard", "sampling_process", "both", "all")]
    [string]$HtmlStyle = "original_touch",
    [int]$MaxTactileImages = 12,
    [string]$OutputHtml = "",
    [int]$MeshPoints = 120000,
    [switch]$UseCdn
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$visualizerScript = Join-Path $scriptRoot "visualize_touch_charts_html.py"

if (-not (Test-Path -LiteralPath $TactileLabRoot)) {
    throw "Tactile_Lab root not found: $TactileLabRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $visualizerScript)) {
    throw "Visualizer script not found: $visualizerScript"
}

$cleanPath = (($env:PATH -split ";") | Where-Object {
    $_ -and ($_ -notlike "*WindowsApps*")
}) -join ";"
Remove-Item Env:PATH -ErrorAction SilentlyContinue
Remove-Item Env:Path -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("Path", $cleanPath, "Process")
$env:Path = $cleanPath

$vizArgs = @(
    $visualizerScript,
    "--sample_idx", "$SampleIdx",
    "--mesh_points", "$MeshPoints",
    "--html_style", $HtmlStyle,
    "--max_tactile_images", "$MaxTactileImages"
)

if (-not [string]::IsNullOrWhiteSpace($DatasetPath)) {
    $vizArgs += @("--dataset_path", $DatasetPath)
}
else {
    $vizArgs += @("--touchsdf_root", $TouchSDFRoot, "--dataset_name", $DatasetName)
}

if (-not [string]::IsNullOrWhiteSpace($OutputHtml)) {
    $vizArgs += @("--output_html", $OutputHtml)
}
if ($PerObject) {
    $vizArgs += "--per_object"
}
if (-not [string]::IsNullOrWhiteSpace($ObjectIndex)) {
    $vizArgs += @("--object_index", $ObjectIndex)
}
if ($UseCdn) {
    $vizArgs += @("--include_plotlyjs", "cdn")
}

Push-Location $TactileLabRoot
try {
    & $PythonExe @vizArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    exit $exitCode
}
