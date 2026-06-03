param(
    [int]$PollSeconds = 300,
    [int]$Epochs = 200,
    [int]$BatchSize = 32,
    [int]$EvalSamples = 100,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"

$Root = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master"
$Python = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe"
$DatasetName = "touch_charts_gt_isaac_tactile_gym_clone_touch_model_large1000_fast_stable"
$DatasetRoot = Join-Path $Root "outputs\touch_model_clone_data"
$DatasetPkl = Join-Path $DatasetRoot ($DatasetName + ".pkl")
$DatasetDir = Join-Path $DatasetRoot $DatasetName
$TrainOut = Join-Path $Root "outputs\touch_model_patch_only_formal_large1000_train200"
$EvalOut = Join-Path $Root "outputs\touch_model_patch_only_formal_large1000_eval100"
$LogDir = Join-Path $Root "outputs\touch_model_clone_logs"
$Log = Join-Path $LogDir "touch_model_formal_large1000_after_collect.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Content -LiteralPath $Log -Value ("[START] {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

function Write-FormalLog {
    param([string]$Message)
    Add-Content -LiteralPath $Log -Value $Message
}

Write-FormalLog ("[INFO] waiting_for_dataset={0}" -f $DatasetPkl)
while (-not (Test-Path -LiteralPath $DatasetPkl)) {
    $sampleCount = 0
    if (Test-Path -LiteralPath $DatasetDir) {
        $sampleCount = (Get-ChildItem -LiteralPath $DatasetDir -Filter "*.npy" -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    Write-FormalLog ("[WAIT] {0} pkl_not_ready partial_npy={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $sampleCount)
    Start-Sleep -Seconds $PollSeconds
}

Write-FormalLog ("[READY] {0} dataset_pkl_found" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

$trainArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "scripts\run_train_touch_model.ps1"),
    "-TactileLabRoot", $Root,
    "-PythonExe", $Python,
    "-DatasetPaths", $DatasetPkl,
    "-OutputDir", $TrainOut,
    "-Epochs", $Epochs,
    "-BatchSize", $BatchSize,
    "-Lr", "0.0001",
    "-WeightDecay", "0.0001",
    "-NumWorkers", "0",
    "-NumPoints", "500",
    "-ValFraction", "0.10",
    "-Seed", "61",
    "-ImageLatentDim", "256",
    "-ImagePoolSize", "4",
    "-HiddenDim", "512",
    "-CenterWeight", "0.25",
    "-SpreadWeight", "0.50",
    "-AxisWeight", "0.50",
    "-RadialWeight", "0.25",
    "-SaveEvery", "10"
)
if ($Device -ne "") {
    $trainArgs += @("-Device", $Device)
}

Write-FormalLog ("[TRAIN] {0} output={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $TrainOut)
& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" @trainArgs 1>> $Log 2>> $Log
if ($LASTEXITCODE -ne 0) {
    Write-FormalLog ("[FAIL] training_exit={0}" -f $LASTEXITCODE)
    exit $LASTEXITCODE
}

$checkpoint = Join-Path $TrainOut "touch_model_best.pt"
$evalArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "scripts\run_test_touch_model.ps1"),
    "-TactileLabRoot", $Root,
    "-PythonExe", $Python,
    "-DatasetPaths", $DatasetPkl,
    "-Checkpoint", $checkpoint,
    "-OutputDir", $EvalOut,
    "-BatchSize", $BatchSize,
    "-NumWorkers", "0",
    "-MaxSamples", $EvalSamples,
    "-NumHtmlSamples", "12"
)
if ($Device -ne "") {
    $evalArgs += @("-Device", $Device)
}

Write-FormalLog ("[EVAL] {0} output={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $EvalOut)
& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" @evalArgs 1>> $Log 2>> $Log
if ($LASTEXITCODE -ne 0) {
    Write-FormalLog ("[FAIL] eval_exit={0}" -f $LASTEXITCODE)
    exit $LASTEXITCODE
}

Write-FormalLog ("[DONE] {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-FormalLog ("[DONE] checkpoint={0}" -f $checkpoint)
Write-FormalLog ("[DONE] html={0}" -f (Join-Path $EvalOut "prediction_examples.html"))
