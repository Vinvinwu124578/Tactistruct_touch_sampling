from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from touch_model import TouchChartDataset, TouchChartModel
from test_touch_model import _model_config_from_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a card gallery for touch_model predictions.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_count", type=int, default=377)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _select_sample_files(dataset_dir: Path, sample_start: int, sample_count: int) -> list[tuple[int, Path]]:
    sample_end = sample_start + sample_count
    files: list[tuple[int, Path]] = []
    for path in dataset_dir.glob("*.npy"):
        sample_id = _numeric_stem(path)
        if sample_id is not None and sample_start <= sample_id < sample_end:
            files.append((sample_id, path))
    files.sort(key=lambda item: item[0])
    expected = list(range(sample_start, sample_end))
    actual = [sample_id for sample_id, _ in files]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise RuntimeError(f"Expected contiguous sample ids {sample_start}..{sample_end - 1}; missing={missing[:20]}")
    return files


def _load_obj_index(path: Path) -> str:
    sample = np.load(path, allow_pickle=True)
    if isinstance(sample, np.ndarray) and sample.shape == ():
        sample = sample.item()
    if not isinstance(sample, dict) or "obj_index" not in sample:
        return ""
    return str(np.asarray(sample["obj_index"], dtype=object).reshape(-1)[0])


def _chamfer_np(pred: np.ndarray, gt: np.ndarray) -> float:
    diff = pred[:, None, :] - gt[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(dist.min(axis=1).mean() + dist.min(axis=0).mean())


def _metrics_for(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    center_err = float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0)))
    radial_gt = float(np.percentile(np.linalg.norm(gt[:, :2], axis=1), 95))
    radial_pred = float(np.percentile(np.linalg.norm(pred[:, :2], axis=1), 95))
    z_gt = float(np.percentile(np.abs(gt[:, 2]), 95))
    z_pred = float(np.percentile(np.abs(pred[:, 2]), 95))
    return {
        "chamfer_m": _chamfer_np(pred, gt),
        "center_err_m": center_err,
        "radial95_gt_m": radial_gt,
        "radial95_pred_m": radial_pred,
        "radial_ratio": float(radial_pred / (radial_gt + 1.0e-9)),
        "zabs95_gt_m": z_gt,
        "zabs95_pred_m": z_pred,
        "z_ratio": float(z_pred / (z_gt + 1.0e-9)),
        "spread_gt_m": [float(x) for x in np.ptp(gt, axis=0)],
        "spread_pred_m": [float(x) for x in np.ptp(pred, axis=0)],
    }


def _save_tactile(img: np.ndarray, path: Path) -> None:
    arr = np.asarray(img).reshape(img.shape[-2], img.shape[-1])
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    positive = arr[arr > 1.0e-8]
    vmax = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    preview = np.clip(arr / max(vmax, 1.0e-8), 0.0, 1.0)
    preview = np.power(preview, 0.65)
    plt.figure(figsize=(2.15, 2.15), dpi=110)
    plt.imshow(preview, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _limits(points: np.ndarray, axes: list[int], min_span: float) -> tuple[tuple[float, float], tuple[float, float]]:
    vals = points[:, axes]
    lo = vals.min(axis=0)
    hi = vals.max(axis=0)
    center = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo) * 1.25, min_span)
    return (float(center[0] - half[0]), float(center[0] + half[0])), (
        float(center[1] - half[1]),
        float(center[1] + half[1]),
    )


def _save_overlay(gt: np.ndarray, pred: np.ndarray, path: Path, mode: str) -> None:
    all_points = np.vstack([gt, pred])
    if mode == "xy":
        ax0, ax1 = 0, 1
        xlabel, ylabel = "x", "y"
        xlim, ylim = _limits(all_points, [0, 1], 0.018)
    else:
        ax0, ax1 = 0, 2
        xlabel, ylabel = "x", "z"
        xlim, ylim = _limits(all_points, [0, 2], 0.018)
    plt.figure(figsize=(2.65, 2.35), dpi=120)
    plt.scatter(gt[:, ax0], gt[:, ax1], s=6, c="#2563eb", alpha=0.62, linewidths=0, label="GT")
    plt.scatter(pred[:, ax0], pred[:, ax1], s=6, c="#dc2626", alpha=0.62, linewidths=0, label="Pred")
    plt.axhline(0, color="#d1d5db", linewidth=0.7)
    plt.axvline(0, color="#d1d5db", linewidth=0.7)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.tick_params(labelsize=7)
    plt.grid(True, color="#eef2f7", linewidth=0.6)
    plt.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.2)
    plt.tight_layout(pad=0.35)
    plt.savefig(path)
    plt.close()


def _classify(metrics: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if (
        metrics["chamfer_m"] <= 0.0030
        and metrics["center_err_m"] <= 0.0060
        and 0.70 <= metrics["radial_ratio"] <= 1.30
        and 0.70 <= metrics["z_ratio"] <= 1.30
    ):
        tags.append("good")
    if metrics["chamfer_m"] > 0.0040:
        tags.append("high_chamfer")
    if metrics["center_err_m"] > 0.0080:
        tags.append("center_offset")
    if metrics["radial_ratio"] < 0.70:
        tags.append("xy_shrunk")
    if metrics["radial_ratio"] > 1.30:
        tags.append("xy_large")
    if metrics["z_ratio"] < 0.70:
        tags.append("z_shallow")
    if metrics["z_ratio"] > 1.30:
        tags.append("z_tall")
    if not tags:
        tags.append("middle")
    return tags


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _write_html(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    buttons = [
        ("all", "All"),
        ("good", "Good"),
        ("xy_shrunk", "XY Shrunk"),
        ("z_shallow", "Z Shallow"),
        ("center_offset", "Center Offset"),
        ("high_chamfer", "High Chamfer"),
        ("middle", "Middle"),
    ]
    button_html = "\n".join(f'<button data-filter="{key}">{label}</button>' for key, label in buttons)
    cards: list[str] = []
    for record in records:
        tags = " ".join(record["tags"])
        badge = ", ".join(record["tags"])
        cards.append(
            f"""
    <article class="card" data-tags="{html.escape(tags)}">
      <div class="card-head">
        <div><b>sample {record['sample_id']:06d}</b><span>{html.escape(record['obj_index'])}</span></div>
        <em>{html.escape(badge)}</em>
      </div>
      <div class="media-grid">
        <figure><img src="{record['tactile']}" loading="lazy"><figcaption>tactile image</figcaption></figure>
        <figure><img src="{record['xy']}" loading="lazy"><figcaption>overlay XY</figcaption></figure>
        <figure><img src="{record['xz']}" loading="lazy"><figcaption>overlay XZ</figcaption></figure>
      </div>
      <div class="metrics">
        <span>Chamfer <b>{record['chamfer_m'] * 1000:.2f} mm</b></span>
        <span>Center <b>{record['center_err_m'] * 1000:.2f} mm</b></span>
        <span>XY ratio <b>{record['radial_ratio']:.2f}</b></span>
        <span>Z ratio <b>{record['z_ratio']:.2f}</b></span>
        <span>GT r95 <b>{record['radial95_gt_m'] * 1000:.1f} mm</b></span>
        <span>Pred r95 <b>{record['radial95_pred_m'] * 1000:.1f} mm</b></span>
      </div>
    </article>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>touch_model prediction all samples</title>
  <style>
    :root {{ --bg:#f6f7fb; --ink:#172033; --muted:#667085; --line:#d9dee8; --card:#ffffff; --blue:#2563eb; --red:#dc2626; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family: Arial, Helvetica, sans-serif; }}
    header {{ position: sticky; top:0; z-index:10; background:rgba(246,247,251,.96); border-bottom:1px solid var(--line); padding:14px 18px 12px; }}
    h1 {{ margin:0 0 5px; font-size:20px; font-weight:700; }}
    .sub {{ color:var(--muted); font-size:13px; line-height:1.45; }}
    .legend {{ margin-top:8px; display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; }}
    .blue {{ background:var(--blue); }} .red {{ background:var(--red); }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; align-items:center; }}
    button {{ border:1px solid var(--line); background:white; color:var(--ink); border-radius:6px; padding:6px 10px; cursor:pointer; font-size:12px; }}
    button.active {{ background:#172033; color:white; border-color:#172033; }}
    #count {{ color:var(--muted); font-size:12px; margin-left:4px; }}
    main {{ padding:16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap:14px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; box-shadow:0 1px 2px rgba(16,24,40,.04); }}
    .card-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:10px; }}
    .card-head b {{ font-size:14px; }}
    .card-head span {{ display:block; color:var(--muted); font-size:11px; margin-top:3px; word-break:break-all; }}
    .card-head em {{ font-style:normal; color:#475467; background:#eef2f7; border-radius:999px; padding:3px 8px; font-size:11px; white-space:nowrap; }}
    .media-grid {{ display:grid; grid-template-columns: 0.8fr 1fr 1fr; gap:8px; align-items:stretch; }}
    figure {{ margin:0; border:1px solid #edf0f5; border-radius:6px; overflow:hidden; background:#fbfcfe; }}
    img {{ display:block; width:100%; height:180px; object-fit:contain; background:white; }}
    figcaption {{ font-size:11px; color:var(--muted); padding:5px 7px; border-top:1px solid #edf0f5; }}
    .metrics {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:6px; margin-top:9px; }}
    .metrics span {{ background:#f8fafc; border:1px solid #edf0f5; border-radius:6px; padding:6px 7px; color:var(--muted); font-size:11px; }}
    .metrics b {{ color:var(--ink); font-size:12px; }}
    .hidden {{ display:none; }}
  </style>
</head>
<body>
  <header>
    <h1>touch_model prediction vs ground truth - all {summary['count']} samples</h1>
    <div class="sub">Same card-style view as the tactile quality gallery. This page uses samples <b>{summary['sample_start']}.npy through {summary['sample_end']}.npy</b> and checkpoint <code>{html.escape(summary['checkpoint_name'])}</code>. Units are local patch meters for this dataset.</div>
    <div class="legend"><span><i class="dot blue"></i>GT</span><span><i class="dot red"></i>Prediction</span><span>Good = Chamfer &lt;= 3 mm, Center &lt;= 6 mm, XY/Z ratios in [0.7, 1.3]</span></div>
    <div class="toolbar">{button_html}<span id="count"></span></div>
  </header>
  <main><section class="grid" id="grid">{''.join(cards)}</section></main>
  <script>
    const buttons = [...document.querySelectorAll('button[data-filter]')];
    const cards = [...document.querySelectorAll('.card')];
    const count = document.getElementById('count');
    function applyFilter(name) {{
      let shown = 0;
      cards.forEach(card => {{
        const tags = card.dataset.tags.split(/\\s+/);
        const visible = name === 'all' || tags.includes(name);
        card.classList.toggle('hidden', !visible);
        if (visible) shown++;
      }});
      buttons.forEach(b => b.classList.toggle('active', b.dataset.filter === name));
      count.textContent = `${{shown}} / ${{cards.length}}`;
    }}
    buttons.forEach(b => b.addEventListener('click', () => applyFilter(b.dataset.filter)));
    applyFilter('all');
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = args.output_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    files = _select_sample_files(args.dataset_dir, args.sample_start, args.sample_count)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = _model_config_from_checkpoint(checkpoint)
    dataset = TouchChartDataset([path for _, path in files], num_points=config.num_points, normalize_images=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TouchChartModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_batches: list[np.ndarray] = []
    gt_batches: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device))["local_points"].detach().cpu().numpy()
            pred_batches.append(pred)
            gt_batches.append(batch["local_points"].detach().cpu().numpy())
            image_batches.append(batch["image"].detach().cpu().numpy())

    pred_all = np.concatenate(pred_batches, axis=0).astype(np.float64)
    gt_all = np.concatenate(gt_batches, axis=0).astype(np.float64)
    image_all = np.concatenate(image_batches, axis=0).astype(np.float64)

    records: list[dict[str, Any]] = []
    for (sample_id, path), pred, gt, image in zip(files, pred_all, gt_all, image_all):
        metrics = _metrics_for(pred, gt)
        tags = _classify(metrics)
        prefix = f"{sample_id:06d}"
        tactile_png = thumb_dir / f"{prefix}_tactile.png"
        xy_png = thumb_dir / f"{prefix}_overlay_xy.png"
        xz_png = thumb_dir / f"{prefix}_overlay_xz.png"
        _save_tactile(image, tactile_png)
        _save_overlay(gt, pred, xy_png, "xy")
        _save_overlay(gt, pred, xz_png, "xz")
        record = {
            "sample_id": int(sample_id),
            "file": path.name,
            "obj_index": _load_obj_index(path),
            "tags": tags,
            "tactile": str(tactile_png.relative_to(args.output_dir)).replace("\\", "/"),
            "xy": str(xy_png.relative_to(args.output_dir)).replace("\\", "/"),
            "xz": str(xz_png.relative_to(args.output_dir)).replace("\\", "/"),
        }
        record.update(metrics)
        records.append(record)

    summary: dict[str, Any] = {
        "count": len(records),
        "sample_start": args.sample_start,
        "sample_end": args.sample_start + args.sample_count - 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.parent.name + "/" + args.checkpoint.name,
        "dataset_dir": str(args.dataset_dir),
        "tag_counts": {},
    }
    for key in ["chamfer_m", "center_err_m", "radial_ratio", "z_ratio"]:
        summary[key] = _percentiles([float(record[key]) for record in records])
    for record in records:
        for tag in record["tags"]:
            summary["tag_counts"][tag] = summary["tag_counts"].get(tag, 0) + 1

    summary_path = args.output_dir / "prediction_all_summary.json"
    html_path = args.output_dir / "prediction_all_gallery.html"
    summary_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    _write_html(html_path, records, summary)

    print(
        json.dumps(
            {
                "html": str(html_path),
                "summary": str(summary_path),
                "count": len(records),
                "tag_counts": summary["tag_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
