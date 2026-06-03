param(
    [string]$TactileLabRoot = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master",
    [string]$PythonExe = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe",
    [string]$ShapeNetRoot = "C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore",
    [string]$TouchSDFRoot = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master",
    [string]$NameOutput = "",
    [string]$CategoryIds = "",
    [string]$ObjectIds = "",
    [ValidateRange(0, 1000000000)]
    [int]$StartObjectIndex = 0,
    [ValidateRange(0, 1000000000)]
    [int]$MaxObjects = 0,
    [ValidateRange(1, 1000000000)]
    [int]$ChartsPerObject = 8,
    [ValidateRange(1, 1000000000)]
    [int]$TotalTargetCharts = 320,
    [ValidateRange(1, 1000000000)]
    [int]$PointsPerTouch = 200,
    [ValidateRange(4, 4096)]
    [int]$TactileImgSize = 256,
    [ValidateSet("tactip_deformation", "legacy_depth_proxy")]
    [string]$TactileMode = "tactip_deformation",
    [string]$SensorType = "tactip",
    [double]$Scale = 0.30,
    [double]$ObjectWorldX = 0.50,
    [double]$ObjectWorldY = 0.0,
    [double]$ObjectWorldZ = 0.0,
    [double]$ObjectBottomZ = 0.020,
    [double]$PatchRadius = 0.045,
    [int]$DenseSurfaceSamples = 50000,
    [ValidateSet("stratified_hemisphere", "top_farthest")]
    [string]$ContactSamplingStrategy = "stratified_hemisphere",
    [int]$ContactAzimuthBins = 8,
    [int]$ContactElevationBins = 4,
    [double]$ContactMinElevationDeg = 5.0,
    [double]$ContactMaxElevationDeg = 88.0,
    [int]$ContactOversampleFactor = 48,
    [double]$IndentationDepth = 0.008,
    [double]$SurfaceClearance = 0.0040,
    [double]$RobotApproachOffset = 0.030,
    [int]$RobotApproachSteps = 3,
    [double]$RobotInwardStepSize = 0.002,
    [double]$RobotJointStepSize = 0.120,
    [double]$RobotWrist3StepSize = 0.060,
    [double]$RobotWrist3TargetStepSize = 0.350,
    [double]$RobotTouchTriggerRatio = 0.70,
    [double]$RobotJointLimitMargin = 0.05,
    [double]$RobotObjectClearance = 0.012,
    [double]$RobotBodyClearance = 0.060,
    [double]$RobotTransitHeight = 0.32,
    [double]$RobotLateralClearance = 0.18,
    [double]$RobotEllipsoidClearance = 0.18,
    [switch]$RobotPlannerAllowDirect,
    [switch]$DisableRobotObjectCollisionGuard,
    [ValidateSet("touchsdf", "tactile_lab")]
    [string]$RobotHomePreset = "touchsdf",
    [double]$RobotHomeBaseJoint = [double]::NaN,
    [double]$MaxDeformationDepth = 0.010,
    [double]$DeformationScale = 100.0,
    [double]$CameraDistance = 0.035,
    [double]$DepthSpan = 0.035,
    [int]$Seed = 41,
    [switch]$Gui,
    [switch]$VisualizeSampling,
    [switch]$NoUr5Tactip,
    [double]$VisualizeSamplingPause = 0.0,
    [double]$VisualizeStartupPause = 0.5,
    [double]$VisualizeMarkerScale = 0.010,
    [int]$VisualizePrepassContacts = 0,
    [double]$VisualizePrepassPause = 0.0,
    [ValidateRange(1, 64)]
    [int]$RenderSteps = 1,
    [ValidateRange(1, 120)]
    [int]$SimRenderInterval = 1,
    [ValidateRange(0.1, 100.0)]
    [double]$SimSpeedFactor = 1.0,
    [switch]$VisualizeEachCandidate,
    [switch]$VisualizeFollowContact,
    [switch]$DisableFloor,
    [double]$FloorZ = 0.0,
    [double]$FloorSize = 0.90,
    [double]$FloorOpacity = 0.42,
    [string]$FloorColor = "0.42,0.46,0.52",
    [switch]$VisualizePointMarkers,
    [switch]$VisualizeDirectionLines,
    [double]$StandbyHoverGap = 0.030,
    [double]$RobotInitialHoverHeight = 0.20,
    [switch]$SavePng,
    [switch]$SavePerSampleNpy,
    [switch]$ResetOutput,
    [switch]$VisualizeHtml,
    [switch]$NoOriginalTouchHtml,
    [ValidateSet("original_touch", "dashboard", "sampling_process", "both", "all")]
    [string]$HtmlStyle = "original_touch",
    [int]$VisualizeSampleIdx = 0,
    [int]$VisualizeMeshPoints = 120000,
    [switch]$VisualizerUseCdn
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$collectorScript = Join-Path $scriptRoot "collect_shapenet_touch_charts_isaac.py"
$visualizerScript = Join-Path $scriptRoot "visualize_touch_charts_html.py"

if (-not (Test-Path -LiteralPath $TactileLabRoot)) {
    throw "Tactile_Lab root not found: $TactileLabRoot"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $ShapeNetRoot)) {
    throw "ShapeNet root not found: $ShapeNetRoot"
}
if (-not (Test-Path -LiteralPath $TouchSDFRoot)) {
    throw "TouchSDF root not found: $TouchSDFRoot"
}
if (-not (Test-Path -LiteralPath $collectorScript)) {
    throw "Collector script not found: $collectorScript"
}
$AutoOriginalTouchHtml = (-not $NoOriginalTouchHtml) -and ($VisualizeSampling -or $SavePng)
if (($VisualizeHtml -or $AutoOriginalTouchHtml) -and (-not (Test-Path -LiteralPath $visualizerScript))) {
    throw "Visualizer script not found: $visualizerScript"
}

if ([string]::IsNullOrWhiteSpace($NameOutput)) {
    $NameOutput = "isaac_shapenet_tactip_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
if (-not $NameOutput.StartsWith("touch_charts_gt_")) {
    $DatasetName = "touch_charts_gt_$NameOutput"
}
else {
    $DatasetName = $NameOutput
}

Write-Host ""
Write-Host "==== Isaac ShapeNet TacTip touch charts ===="
Write-Host "[INFO] dataset=$DatasetName"
Write-Host "[INFO] ShapeNet=$ShapeNetRoot"
Write-Host "[INFO] target=$TotalTargetCharts | charts_per_object=$ChartsPerObject | points_per_touch=$PointsPerTouch"
Write-Host "[INFO] scale=$Scale | patch_radius=$PatchRadius | img=${TactileImgSize}x$TactileImgSize"
Write-Host "[INFO] contact_sampling=touchsdf_robot_touch_spherical | candidate_strategy=$ContactSamplingStrategy | oversample=$ContactOversampleFactor"
Write-Host "[INFO] object_world_pos=($ObjectWorldX, $ObjectWorldY, $ObjectWorldZ) | object_bottom_z=$ObjectBottomZ"
Write-Host "[INFO] tactile_mode=$TactileMode | sensor=$SensorType | indentation=$IndentationDepth | clearance=$SurfaceClearance | max_deform=$MaxDeformationDepth"
Write-Host "[INFO] robot_dynamics: approach_offset=$RobotApproachOffset | approach_steps=$RobotApproachSteps | inward_step<=$RobotInwardStepSize | joint_step=$RobotJointStepSize | wrist3_step<=$RobotWrist3StepSize | wrist3_target<=$RobotWrist3TargetStepSize | touch_trigger_ratio=$RobotTouchTriggerRatio | joint_limit_margin=$RobotJointLimitMargin | object_clearance=$RobotObjectClearance | body_clearance=$RobotBodyClearance | transit_height=$RobotTransitHeight | lateral_clearance=$RobotLateralClearance | ellipsoid_clearance=$RobotEllipsoidClearance | planner_direct=True | object_guard=disabled_for_touchsdf_motion | home_preset=$RobotHomePreset | home_base=$RobotHomeBaseJoint"
Write-Host "[INFO] live_sampling_viz=$VisualizeSampling | ur5_tactip=$(-not $NoUr5Tactip) | startup_pause=$VisualizeStartupPause | sample_pause=$VisualizeSamplingPause | prepass=$VisualizePrepassContacts | render_steps=$RenderSteps | sim_render_interval=$SimRenderInterval | sim_speed=$SimSpeedFactor"
Write-Host "[INFO] touchsdf_motion=rest -> object_center+[0,0,0.2] -> sphere -> inward | point_markers=$VisualizePointMarkers | direction_lines=$VisualizeDirectionLines"
Write-Host "[INFO] output=$(Join-Path $TouchSDFRoot "results\$DatasetName.pkl")"

$cleanPath = (($env:PATH -split ";") | Where-Object {
    $_ -and ($_ -notlike "*WindowsApps*")
}) -join ";"
Remove-Item Env:PATH -ErrorAction SilentlyContinue
Remove-Item Env:Path -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("Path", $cleanPath, "Process")
$env:Path = $cleanPath

$warpCachePath = Join-Path $TactileLabRoot ("outputs\warp_cache\run_{0}" -f $PID)
New-Item -ItemType Directory -Force -Path $warpCachePath | Out-Null
$env:WARP_CACHE_PATH = $warpCachePath
Write-Host "[INFO] WARP_CACHE_PATH=$env:WARP_CACHE_PATH"

$collectArgs = @(
    $collectorScript,
    "--shapenet_root", $ShapeNetRoot,
    "--touchsdf_root", $TouchSDFRoot,
    "--name_output", $NameOutput,
    "--start_object_index", "$StartObjectIndex",
    "--max_objects", "$MaxObjects",
    "--charts_per_object", "$ChartsPerObject",
    "--total_target_charts", "$TotalTargetCharts",
    "--points_per_touch", "$PointsPerTouch",
    "--tactile_img_size", "$TactileImgSize",
    "--tactile_mode", $TactileMode,
    "--sensor_type", $SensorType,
    "--scale", "$Scale",
    "--object_world_x", "$ObjectWorldX",
    "--object_world_y", "$ObjectWorldY",
    "--object_world_z", "$ObjectWorldZ",
    "--object_bottom_z", "$ObjectBottomZ",
    "--patch_radius", "$PatchRadius",
    "--dense_surface_samples", "$DenseSurfaceSamples",
    "--contact_sampling_strategy", $ContactSamplingStrategy,
    "--contact_azimuth_bins", "$ContactAzimuthBins",
    "--contact_elevation_bins", "$ContactElevationBins",
    "--contact_min_elevation_deg", "$ContactMinElevationDeg",
    "--contact_max_elevation_deg", "$ContactMaxElevationDeg",
    "--contact_oversample_factor", "$ContactOversampleFactor",
    "--indentation_depth", "$IndentationDepth",
    "--surface_clearance", "$SurfaceClearance",
    "--robot_approach_offset", "$RobotApproachOffset",
    "--robot_approach_steps", "$RobotApproachSteps",
    "--robot_inward_step_size", "$RobotInwardStepSize",
    "--robot_joint_step_size", "$RobotJointStepSize",
    "--robot_wrist3_step_size", "$RobotWrist3StepSize",
    "--robot_wrist3_target_step_size", "$RobotWrist3TargetStepSize",
    "--robot_touch_trigger_ratio", "$RobotTouchTriggerRatio",
    "--robot_joint_limit_margin", "$RobotJointLimitMargin",
    "--robot_object_clearance", "$RobotObjectClearance",
    "--robot_body_clearance", "$RobotBodyClearance",
    "--robot_transit_height", "$RobotTransitHeight",
    "--robot_lateral_clearance", "$RobotLateralClearance",
    "--robot_ellipsoid_clearance", "$RobotEllipsoidClearance",
    "--robot_home_preset", $RobotHomePreset,
    "--robot_home_base_joint", "$RobotHomeBaseJoint",
    "--standby_hover_gap", "$StandbyHoverGap",
    "--robot_initial_hover_height", "$RobotInitialHoverHeight",
    "--max_deformation_depth", "$MaxDeformationDepth",
    "--deformation_scale", "$DeformationScale",
    "--camera_distance", "$CameraDistance",
    "--depth_span", "$DepthSpan",
    "--render_steps", "$RenderSteps",
    "--sim_render_interval", "$SimRenderInterval",
    "--sim_speed_factor", "$SimSpeedFactor",
    "--floor_z", "$FloorZ",
    "--floor_size", "$FloorSize",
    "--floor_opacity", "$FloorOpacity",
    "--floor_color", $FloorColor,
    "--seed", "$Seed",
    "--enable_cameras"
)

if (-not [string]::IsNullOrWhiteSpace($CategoryIds)) {
    $collectArgs += @("--category_ids", $CategoryIds)
}
if (-not [string]::IsNullOrWhiteSpace($ObjectIds)) {
    $collectArgs += @("--object_ids", $ObjectIds)
}
if (-not $NoUr5Tactip) {
    $collectArgs += "--spawn_ur5_tactip"
}
if ($DisableRobotObjectCollisionGuard) {
    $collectArgs += "--disable_robot_object_collision_guard"
}
$collectArgs += "--no-robot_planner_skip_direct"
if ($VisualizeSampling) {
    $collectArgs += @(
        "--visualize_sampling",
        "--visualize_sampling_pause", "$VisualizeSamplingPause",
        "--visualize_startup_pause", "$VisualizeStartupPause",
        "--visualize_marker_scale", "$VisualizeMarkerScale",
        "--visualize_prepass_contacts", "$VisualizePrepassContacts",
        "--visualize_prepass_pause", "$VisualizePrepassPause"
    )
    if ($VisualizeEachCandidate) {
        $collectArgs += "--visualize_each_candidate"
    }
if ($VisualizeFollowContact) {
    $collectArgs += "--visualize_follow_contact"
}
if ($DisableFloor) {
    $collectArgs += "--disable_floor"
}
if ($VisualizePointMarkers) {
    $collectArgs += "--visualize_point_markers"
}
    if ($VisualizeDirectionLines) {
        $collectArgs += "--visualize_direction_lines"
    }
}
if (-not $Gui -and -not $VisualizeSampling) {
    $collectArgs += "--headless"
}
if ($SavePng) {
    $collectArgs += "--save_png"
}
if ($SavePerSampleNpy) {
    $collectArgs += "--save_per_sample_npy"
}
if ($ResetOutput) {
    $collectArgs += "--reset_output"
}

Push-Location $TactileLabRoot
try {
    & $PythonExe @collectArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    exit $exitCode
}

$datasetPath = Join-Path $TouchSDFRoot "results\$DatasetName.pkl"
$datasetNpyPath = Join-Path $TouchSDFRoot "results\$DatasetName.npy"
$manifestPath = Join-Path $TouchSDFRoot "results\$DatasetName\manifest.csv"
if (-not (Test-Path -LiteralPath $datasetPath)) {
    Write-Host ""
    Write-Host "[FAIL] Collector exited without creating dataset: $datasetPath" -ForegroundColor Red
    Write-Host "[FAIL] This means no valid [TOUCH] samples were collected; check the preceding [MISS]/[SAFE] logs."
    exit 1
}

$ShouldGenerateHtml = $VisualizeHtml -or $AutoOriginalTouchHtml
if ($ShouldGenerateHtml) {
    $HtmlStyleToUse = if ($VisualizeHtml) { $HtmlStyle } else { "original_touch" }
    Write-Host "[INFO] generating $HtmlStyleToUse HTML visualization and original_touch intermediate folders..."
    $visualizerArgs = @(
        $visualizerScript,
        "--touchsdf_root", $TouchSDFRoot,
        "--dataset_name", $DatasetName,
        "--sample_idx", "$VisualizeSampleIdx",
        "--mesh_points", "$VisualizeMeshPoints",
        "--html_style", $HtmlStyleToUse,
        "--per_object",
        "--save_original_touch_intermediates"
    )
    if ($VisualizerUseCdn) {
        $visualizerArgs += @("--include_plotlyjs", "cdn")
    }
    Push-Location $TactileLabRoot
    try {
        & $PythonExe @visualizerArgs
        $vizExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($vizExitCode -ne 0) {
        if ($VisualizeHtml) {
            exit $vizExitCode
        }
        Write-Warning "HTML visualization failed with exit code $vizExitCode, but dataset files were created."
    }
}

Write-Host ""
Write-Host "[DONE] Dataset: $datasetPath"
if (Test-Path -LiteralPath $datasetNpyPath) {
    Write-Host "[DONE] NPY: $datasetNpyPath"
}
if (Test-Path -LiteralPath $manifestPath) {
    Write-Host "[DONE] Manifest: $manifestPath"
}
Write-Host "[DONE] Train with: python model\train_touch.py --dataset_name $DatasetName"
