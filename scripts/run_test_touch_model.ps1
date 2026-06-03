param(
    [string]$TactileLabRoot = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master",
    [string]$PythonExe = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe",
    [string[]]$DatasetPaths = @(
        "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master\results\touch_charts_gt_isaac_live_sampling_viz_20touch_floor.pkl"
    ),
    [string]$Checkpoint = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master\results\touch_model_patch_only\touch_model_best.pt",
    [string]$OutputDir = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master\results\touch_model_patch_only_eval",
    [ValidateRange(1, 4096)]
    [int]$BatchSize = 32,
    [ValidateRange(0, 64)]
    [int]$NumWorkers = 0,
    [ValidateRange(0, 1000000)]
    [int]$MaxSamples = 0,
    [ValidateRange(0, 1000)]
    [int]$NumHtmlSamples = 4,
    [string]$Device = "",
    [switch]$NoNormalizeImages
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $TactileLabRoot "scripts\test_touch_model.py"
if (-not (Test-Path $ScriptPath)) {
    throw "Cannot find evaluation script: $ScriptPath"
}
if (-not (Test-Path $Checkpoint)) {
    throw "Cannot find checkpoint: $Checkpoint"
}

$cmdArgs = @(
    $ScriptPath,
    "--dataset"
) + $DatasetPaths + @(
    "--checkpoint", $Checkpoint,
    "--output_dir", $OutputDir,
    "--batch_size", $BatchSize,
    "--num_workers", $NumWorkers,
    "--max_samples", $MaxSamples,
    "--num_html_samples", $NumHtmlSamples
)

if ($Device -ne "") {
    $cmdArgs += @("--device", $Device)
}
if ($NoNormalizeImages) {
    $cmdArgs += "--no_normalize_images"
}

Write-Host "==== Test Touch Model: tactile image -> local patch ===="
Write-Host "[INFO] Python=$PythonExe"
Write-Host "[INFO] checkpoint=$Checkpoint"
Write-Host "[INFO] output=$OutputDir"
Write-Host "[INFO] dataset=$($DatasetPaths -join ', ')"

& $PythonExe $cmdArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
