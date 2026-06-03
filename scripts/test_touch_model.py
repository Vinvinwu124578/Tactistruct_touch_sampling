from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from touch_model import TouchChartDataset, TouchChartModel, TouchModelConfig, touch_model_loss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained patch-only touch_model checkpoint.")
    parser.add_argument("--dataset", nargs="+", required=True, help="Touchchart .pkl/.npy files, dirs, or globs.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="touch_model_best.pt or touch_model_last.pt")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "touch_model_patch_only_eval",
        help="Directory for metrics and prediction dumps.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the whole dataset.")
    parser.add_argument("--num_html_samples", type=int, default=4, help="Number of prediction examples in HTML.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device, for example cuda or cpu.",
    )
    parser.add_argument("--no_normalize_images", action="store_true")
    return parser.parse_args()


def _model_config_from_checkpoint(checkpoint: dict[str, Any]) -> TouchModelConfig:
    config = dict(checkpoint.get("model_config", {}))
    if "image_pool_size" not in config:
        weight = checkpoint.get("model_state", {}).get("image_encoder.proj.1.weight")
        if weight is not None and len(weight.shape) == 2:
            input_dim = int(weight.shape[1])
            if input_dim == 128:
                config["image_pool_size"] = 1
            elif input_dim % 128 == 0:
                config["image_pool_size"] = int(round((input_dim // 128) ** 0.5))
    return TouchModelConfig(**config)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}


def _evaluate(
    model: TouchChartModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["image"])
            _, metrics = touch_model_loss(
                outputs,
                batch,
                local_weight=loss_weights.get("local", 1.0),
                center_weight=loss_weights.get("center", 0.25),
                spread_weight=loss_weights.get("spread", 0.50),
                axis_weight=loss_weights.get("axis", 0.50),
                radial_weight=loss_weights.get("radial", 0.25),
                ordered_weight=loss_weights.get("ordered", 0.0),
            )
            batch_size = int(batch["image"].shape[0])
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def _tensor_to_list(x: torch.Tensor) -> list[list[float]]:
    return np.round(x.detach().cpu().numpy(), 6).tolist()


def _write_html(path: Path, samples: list[dict[str, Any]]) -> None:
    # Uses Plotly from CDN to keep this file lightweight.
    payload = json.dumps(samples)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>touch_model patch prediction</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    #plot {{ width: 100%; height: 760px; }}
    select {{ font-size: 14px; padding: 4px 8px; }}
  </style>
</head>
<body>
  <h3>touch_model prediction vs ground truth (local normalized patch)</h3>
  <label>sample <select id="sample"></select></label>
  <div id="plot"></div>
  <script>
    const samples = {payload};
    const select = document.getElementById("sample");
    samples.forEach((sample, i) => {{
      const option = document.createElement("option");
      option.value = i;
      option.textContent = sample.name;
      select.appendChild(option);
    }});
    function xyz(points, axis) {{
      const idx = axis === "x" ? 0 : axis === "y" ? 1 : 2;
      return points.map(p => p[idx]);
    }}
    function render(i) {{
      const s = samples[i];
      const traces = [
        {{
          type: "scatter3d",
          mode: "markers",
          name: "gt local patch",
          x: xyz(s.gt, "x"),
          y: xyz(s.gt, "y"),
          z: xyz(s.gt, "z"),
          marker: {{ size: 3, color: "#1f77b4", opacity: 0.85 }}
        }},
        {{
          type: "scatter3d",
          mode: "markers",
          name: "pred local patch",
          x: xyz(s.pred, "x"),
          y: xyz(s.pred, "y"),
          z: xyz(s.pred, "z"),
          marker: {{ size: 3, color: "#d62728", opacity: 0.85 }}
        }}
      ];
      Plotly.newPlot("plot", traces, {{
        scene: {{ aspectmode: "data", xaxis: {{ title: "x" }}, yaxis: {{ title: "y" }}, zaxis: {{ title: "z" }} }},
        margin: {{ l: 0, r: 0, t: 24, b: 0 }},
        legend: {{ x: 0.02, y: 0.98 }}
      }});
    }}
    select.addEventListener("change", e => render(Number(e.target.value)));
    render(0);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _save_examples(
    model: TouchChartModel,
    dataset: TouchChartDataset,
    output_dir: Path,
    device: torch.device,
    count: int,
) -> None:
    count = max(0, min(count, len(dataset)))
    if count == 0:
        return
    pred_local: list[np.ndarray] = []
    gt_local: list[np.ndarray] = []
    raw_points: list[np.ndarray] = []
    patch_radius: list[float] = []
    html_samples: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for idx in range(count):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            pred = model(image)["local_points"][0].cpu()
            gt = sample["local_points"].cpu()
            pred_local.append(pred.numpy())
            gt_local.append(gt.numpy())
            raw_points.append(sample["raw_points"].numpy())
            patch_radius.append(float(sample["patch_radius"].item()))
            html_samples.append(
                {
                    "name": f"{idx:04d}",
                    "pred": _tensor_to_list(pred),
                    "gt": _tensor_to_list(gt),
                }
            )

    np.savez_compressed(
        output_dir / "prediction_examples.npz",
        pred_local=np.asarray(pred_local, dtype=np.float32),
        gt_local=np.asarray(gt_local, dtype=np.float32),
        raw_points=np.asarray(raw_points, dtype=np.float32),
        patch_radius=np.asarray(patch_radius, dtype=np.float32),
    )
    _write_html(output_dir / "prediction_examples.html", html_samples)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = _model_config_from_checkpoint(checkpoint)
    loss_weights = checkpoint.get("train_config", {}).get(
        "loss_weights",
        {"local": 1.0, "center": 0.25, "spread": 0.50, "axis": 0.50, "radial": 0.25, "ordered": 0.0},
    )
    dataset = TouchChartDataset(
        args.dataset,
        num_points=model_config.num_points,
        normalize_images=not args.no_normalize_images,
    )
    eval_dataset = dataset
    if args.max_samples > 0 and args.max_samples < len(dataset):
        eval_dataset = Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TouchChartModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = _evaluate(model, loader, device, loss_weights)
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_summary": dataset.summary(),
        "evaluated_samples": len(eval_dataset),
        "metrics": metrics,
        "model_config": model_config.to_dict(),
        "loss_weights": loss_weights,
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    _save_examples(model, dataset, args.output_dir, device, args.num_html_samples)
    print(f"[EVAL] samples={len(eval_dataset)} loss={metrics['loss']:.6f} local={metrics['local_chamfer']:.6f}", flush=True)
    print(f"[DONE] metrics={args.output_dir / 'metrics.json'}", flush=True)
    if args.num_html_samples > 0:
        print(f"[DONE] html={args.output_dir / 'prediction_examples.html'}", flush=True)


if __name__ == "__main__":
    main()
