param(
    [switch]$ProbeOnly
)

$ErrorActionPreference = "Stop"

$Root = "C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master"
$Python = "C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe"
$Script = Join-Path $Root "scripts\collect_shapenet_touch_charts_isaac_tactile_gym_clone.py"
$ShapeNet = "C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore"
$TouchSdf = "C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master"
$Name = "isaac_tactile_gym_clone_touch_model_large1000_fast_stable_aligned_gt"
$OutRoot = Join-Path $Root "outputs\touch_model_clone_data"
$LogDir = Join-Path $Root "outputs\touch_model_clone_logs"
$WarpCache = Join-Path $Root ("outputs\warp_cache\touch_model_clone_aligned_gt_{0}" -f (Get-Random))
$StdoutLog = Join-Path $LogDir "collect_clone_large1000_aligned_gt_ps_stdout.log"
$StderrLog = Join-Path $LogDir "collect_clone_large1000_aligned_gt_ps_stderr.log"

New-Item -ItemType Directory -Force -Path $OutRoot, $LogDir, $WarpCache | Out-Null
Set-Content -LiteralPath $StdoutLog -Value ""
Set-Content -LiteralPath $StderrLog -Value ""
$env:WARP_CACHE_PATH = $WarpCache
Set-Location $Root

function Write-Log {
    param([string]$Message)
    Add-Content -LiteralPath $StdoutLog -Value $Message
}

Write-Log ("[START] {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Log ("[INFO] dataset=touch_charts_gt_{0}" -f $Name)
Write-Log ("[INFO] output_root={0}" -f $OutRoot)
Write-Log ("[INFO] WARP_CACHE_PATH={0}" -f $env:WARP_CACHE_PATH)
Write-Log "[INFO] gt_origin=actual_sensor_center_ray"

if ($ProbeOnly) {
    Write-Log "[PROBE] aligned_gt launcher script reached logging successfully."
    exit 0
}

$ArgsList = @(
    "--shapenet_root", $ShapeNet,
    "--touchsdf_root", $TouchSdf,
    "--output_root", $OutRoot,
    "--name_output", $Name,
    "--category_ids", "02942699",
    "--start_object_index", "0",
    "--max_objects", "50",
    "--charts_per_object", "20",
    "--total_target_charts", "1000",
    "--points_per_touch", "500",
    "--tactile_img_size", "256",
    "--scale", "0.30",
    "--object_world_x", "0.50",
    "--object_world_y", "0.0",
    "--object_world_z", "0.0",
    "--object_bottom_z", "0.02",
    "--patch_radius", "0.03",
    "--dense_surface_samples", "30000",
    "--contact_sampling_strategy", "surface_normal_coverage",
    "--contact_azimuth_bins", "8",
    "--contact_elevation_bins", "4",
    "--contact_min_elevation_deg", "-5",
    "--contact_max_elevation_deg", "90",
    "--contact_oversample_factor", "16",
    "--reachable_candidate_filter",
    "--reachable_candidate_keep", "120",
    "--reachable_candidate_diversify",
    "--reachable_candidate_diversity_score_weight", "0.35",
    "--reachable_candidate_spatial_bins", "3",
    "--reachable_candidate_first_pass_max_penalty", "0.55",
    "--reachable_candidate_max_failures", "10",
    "--min_contact_separation", "0.025",
    "--gt_patch_source", "tactip_raycast",
    "--gt_uniform_resample",
    "--gt_uniform_resample_space", "xy",
    "--gt_uniform_dedup_tol", "0.000001",
    "--gt_uniform_raycast_min_unique_ratio", "0.65",
    "--no-gt_allow_mesh_fallback",
    "--tactile_mode", "tactip_deformation",
    "--sensor_type", "tactip",
    "--max_deformation_depth", "0.010",
    "--deformation_scale", "100.0",
    "--indentation_depth", "0.008",
    "--surface_clearance", "0.004",
    "--robot_approach_offset", "0.03",
    "--robot_approach_steps", "1",
    "--robot_inward_step_size", "0.004",
    "--robot_joint_step_size", "0.50",
    "--robot_wrist3_step_size", "0.40",
    "--robot_wrist3_target_step_size", "1.40",
    "--tactile_gym_camera_check_stride", "1",
    "--tactile_gym_refine_steps", "6",
    "--tactile_gym_max_mean_gray", "90",
    "--tactile_gym_max_pixel_deformation", "0.008",
    "--tactile_gym_max_overdeep_fraction", "0.0",
    "--tactile_gym_inward_target", "surface_stop",
    "--robot_initial_hover_height", "0.20",
    "--render_steps", "1",
    "--sim_render_interval", "180",
    "--sim_speed_factor", "20",
    "--seed", "71",
    "--reset_output",
    "--spawn_ur5_tactip",
    "--headless",
    "--enable_cameras",
    "--livestream", "0",
    "--save_per_sample_npy",
    "--no-save_touch_chart_html",
    "--no-save_preview_png"
)

$ErrorActionPreference = "Continue"
function Quote-CmdArg {
    param([string]$Value)
    '"' + ($Value -replace '"', '\"') + '"'
}

$CommandParts = @($Python, $Script) + $ArgsList
$CommandLine = (($CommandParts | ForEach-Object { Quote-CmdArg $_ }) -join " ")
$CommandLine = $CommandLine + " 1>> " + (Quote-CmdArg $StdoutLog) + " 2>> " + (Quote-CmdArg $StderrLog)
& cmd.exe /d /c $CommandLine
$ExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Write-Log ("[END] {0} exit={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $ExitCode)
exit $ExitCode
