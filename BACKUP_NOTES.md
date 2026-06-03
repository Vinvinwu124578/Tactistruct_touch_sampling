# Tactile Lab Code Backup

This backup contains the current runnable code and scripts from `Tactile_Lab-master`.

Included:

- `scripts/`
- `source/`
- `third_party/`
- project config files and README

Excluded:

- `outputs/` generated datasets, logs, visualizations, and model artifacts
- external ShapeNet, IsaacLab, TouchSDF, and robot asset folders

Current collection launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_large_touch_model_clone_collection_aligned_gt.ps1
```

Important current GT behavior:

- GT patches use TacTip raycast hits only.
- Dense mesh-surface fallback is disabled by default.
- If TacTip raycast GT fails for a touch, that touch is skipped rather than filled with unrelated object surface points.
