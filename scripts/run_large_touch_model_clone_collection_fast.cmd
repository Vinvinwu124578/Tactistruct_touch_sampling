@echo off
setlocal

set "ROOT=C:\Users\wudaw\Downloads\Tactile_Lab-master\Tactile_Lab-master"
set "PYTHON=C:\Users\wudaw\anaconda3\envs\env_tactile_lab\python.exe"
set "SHAPENET=C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore"
set "TOUCHSDF=C:\Users\wudaw\Downloads\TouchSDF-master (3)\TouchSDF-master"
set "NAME=isaac_tactile_gym_clone_touch_model_large1000_fast"
set "OUTROOT=%ROOT%\outputs\touch_model_clone_data"
set "LOGDIR=%ROOT%\outputs\touch_model_clone_logs"
set "STDOUT=%LOGDIR%\collect_clone_large1000_fast_stdout.log"
set "STDERR=%LOGDIR%\collect_clone_large1000_fast_stderr.log"
set "WARP_CACHE_PATH=%ROOT%\outputs\warp_cache\touch_model_clone_large1000_fast_%RANDOM%%RANDOM%"

if not exist "%OUTROOT%" mkdir "%OUTROOT%"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%WARP_CACHE_PATH%" mkdir "%WARP_CACHE_PATH%"

cd /d "%ROOT%"

echo [START] %DATE% %TIME% > "%STDOUT%"
echo [INFO] dataset=touch_charts_gt_%NAME% >> "%STDOUT%"
echo [INFO] output_root=%OUTROOT% >> "%STDOUT%"
echo [INFO] WARP_CACHE_PATH=%WARP_CACHE_PATH% >> "%STDOUT%"

"%PYTHON%" ".\scripts\collect_shapenet_touch_charts_isaac_tactile_gym_clone.py" ^
  --shapenet_root "%SHAPENET%" ^
  --touchsdf_root "%TOUCHSDF%" ^
  --output_root "%OUTROOT%" ^
  --name_output "%NAME%" ^
  --category_ids 02942699 ^
  --start_object_index 0 ^
  --max_objects 50 ^
  --charts_per_object 20 ^
  --total_target_charts 1000 ^
  --points_per_touch 500 ^
  --tactile_img_size 256 ^
  --scale 0.30 ^
  --object_world_x 0.50 ^
  --object_world_y 0.0 ^
  --object_world_z 0.0 ^
  --object_bottom_z 0.02 ^
  --patch_radius 0.03 ^
  --dense_surface_samples 12000 ^
  --contact_sampling_strategy stratified_hemisphere ^
  --contact_azimuth_bins 8 ^
  --contact_elevation_bins 4 ^
  --contact_min_elevation_deg 5 ^
  --contact_max_elevation_deg 88 ^
  --contact_oversample_factor 16 ^
  --gt_patch_source tactip_raycast ^
  --tactile_mode tactip_deformation ^
  --sensor_type tactip ^
  --max_deformation_depth 0.010 ^
  --deformation_scale 100.0 ^
  --indentation_depth 0.008 ^
  --surface_clearance 0.004 ^
  --robot_approach_offset 0.03 ^
  --robot_approach_steps 1 ^
  --robot_inward_step_size 0.010 ^
  --robot_joint_step_size 0.40 ^
  --robot_wrist3_step_size 0.30 ^
  --robot_wrist3_target_step_size 1.20 ^
  --tactile_gym_camera_check_stride 6 ^
  --tactile_gym_refine_steps 2 ^
  --robot_initial_hover_height 0.20 ^
  --render_steps 1 ^
  --sim_render_interval 60 ^
  --seed 51 ^
  --reset_output ^
  --spawn_ur5_tactip ^
  --headless ^
  --enable_cameras ^
  --livestream 0 ^
  --save_per_sample_npy ^
  --no-save_touch_chart_html ^
  --no-save_preview_png ^
  >> "%STDOUT%" 2> "%STDERR%"

set "EXITCODE=%ERRORLEVEL%"
echo [END] %DATE% %TIME% exit=%EXITCODE% >> "%STDOUT%"
exit /b %EXITCODE%
