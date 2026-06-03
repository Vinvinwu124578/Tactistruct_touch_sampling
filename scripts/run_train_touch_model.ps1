param(
    [string]$TactileLabRoot = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master",
    [string]$PythonExe = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe",
    [string[]]$DatasetPaths = @(
        "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master\results\touch_charts_gt_isaac_live_sampling_viz_20touch_floor.pkl"
    ),
    [string]$OutputDir = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master\results\touch_model_patch_only",
    [ValidateRange(1, 1000000)]
    [int]$Epochs = 100,
    [ValidateRange(1, 4096)]
    [int]$BatchSize = 32,
    [double]$Lr = 0.0001,
    [double]$WeightDecay = 0.0001,
    [ValidateRange(0, 64)]
    [int]$NumWorkers = 0,
    [ValidateRange(0, 1000000)]
    [int]$NumPoints = 0,
    [ValidateRange(0.0, 0.9)]
    [double]$ValFraction = 0.10,
    [int]$Seed = 41,
    [string]$Device = "",
    [ValidateRange(1, 4096)]
    [int]$ImageLatentDim = 256,
    [ValidateRange(1, 32)]
    [int]$ImagePoolSize = 4,
    [ValidateRange(1, 4096)]
    [int]$HiddenDim = 512,
    [double]$CenterWeight = 0.25,
    [double]$SpreadWeight = 0.50,
    [double]$AxisWeight = 0.50,
    [double]$RadialWeight = 0.25,
    [double]$OrderedWeight = 0.0,
    [switch]$NoNormalizeImages,
    [string]$Resume = "",
    [ValidateRange(0, 1000000)]
    [int]$SaveEvery = 10
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $TactileLabRoot "scripts\train_touch_model.py"
if (-not (Test-Path $ScriptPath)) {
    throw "Cannot find training script: $ScriptPath"
}

$cmdArgs = @(
    $ScriptPath,
    "--dataset"
) + $DatasetPaths + @(
    "--output_dir", $OutputDir,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--lr", $Lr,
    "--weight_decay", $WeightDecay,
    "--num_workers", $NumWorkers,
    "--num_points", $NumPoints,
    "--val_fraction", $ValFraction,
    "--seed", $Seed,
    "--image_latent_dim", $ImageLatentDim,
    "--image_pool_size", $ImagePoolSize,
    "--hidden_dim", $HiddenDim,
    "--center_weight", $CenterWeight,
    "--spread_weight", $SpreadWeight,
    "--axis_weight", $AxisWeight,
    "--radial_weight", $RadialWeight,
    "--ordered_weight", $OrderedWeight,
    "--save_every", $SaveEvery
)

if ($Device -ne "") {
    $cmdArgs += @("--device", $Device)
}
if ($NoNormalizeImages) {
    $cmdArgs += "--no_normalize_images"
}
if ($Resume -ne "") {
    $cmdArgs += @("--resume", $Resume)
}

Write-Host "==== Train Touch Model: tactile image -> local patch ===="
Write-Host "[INFO] Python=$PythonExe"
Write-Host "[INFO] output=$OutputDir"
Write-Host "[INFO] dataset=$($DatasetPaths -join ', ')"

& $PythonExe $cmdArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
