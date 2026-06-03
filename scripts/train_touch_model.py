from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from touch_model import TouchChartDataset, TouchChartModel, TouchModelConfig, touch_model_loss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image-only touch model on TouchChart tactile images and local patch point clouds."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more touchchart .pkl/.npy files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "touch_model_patch_only",
        help="Directory for checkpoints and training config.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--num_points",
        type=int,
        default=0,
        help="Predicted patch point count. 0 means infer from the first dataset.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, for example cuda or cpu.",
    )
    parser.add_argument("--image_latent_dim", type=int, default=256)
    parser.add_argument(
        "--image_pool_size",
        type=int,
        default=4,
        help="Adaptive pooled feature grid size before the image latent projection. 4 keeps coarse spatial layout.",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--local_weight", type=float, default=1.0)
    parser.add_argument("--center_weight", type=float, default=0.25)
    parser.add_argument("--spread_weight", type=float, default=0.50)
    parser.add_argument("--axis_weight", type=float, default=0.50)
    parser.add_argument("--radial_weight", type=float, default=0.25)
    parser.add_argument(
        "--ordered_weight",
        type=float,
        default=0.0,
        help="Optional point-wise Huber loss. Keep 0 for randomly sampled unordered patches.",
    )
    parser.add_argument("--no_normalize_images", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--save_every", type=int, default=10, help="Save epoch checkpoint every N epochs. 0 disables.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}


def _split_dataset(dataset: Dataset[Any], val_fraction: float, seed: int) -> tuple[Dataset[Any], Dataset[Any] | None]:
    if len(dataset) < 2 or val_fraction <= 0.0:
        return dataset, None
    val_count = int(round(len(dataset) * val_fraction))
    val_count = max(1, min(val_count, len(dataset) - 1))
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    return train_dataset, val_dataset


def _run_epoch(
    model: TouchChartModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    local_weight: float,
    center_weight: float,
    spread_weight: float,
    axis_weight: float,
    radial_weight: float,
    ordered_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["image"])
            loss, metrics = touch_model_loss(
                outputs,
                batch,
                local_weight=local_weight,
                center_weight=center_weight,
                spread_weight=spread_weight,
                axis_weight=axis_weight,
                radial_weight=radial_weight,
                ordered_weight=ordered_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = int(batch["image"].shape[0])
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def _save_checkpoint(
    path: Path,
    model: TouchChartModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": config,
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TouchChartDataset(
        args.dataset,
        num_points=args.num_points or None,
        normalize_images=not args.no_normalize_images,
    )
    train_dataset, val_dataset = _split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    model_config = TouchModelConfig(
        image_latent_dim=args.image_latent_dim,
        image_pool_size=args.image_pool_size,
        fusion_hidden_dim=args.hidden_dim,
        num_points=dataset.num_points,
    )
    device = torch.device(args.device)
    model = TouchChartModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_metric = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    train_config = {
        "dataset": args.dataset,
        "dataset_summary": dataset.summary(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": str(device),
        "model_config": model_config.to_dict(),
        "target": "local_points = raw_patch_points / patch_radius",
        "uses_scale_metadata": False,
        "loss_weights": {
            "local": args.local_weight,
            "center": args.center_weight,
            "spread": args.spread_weight,
            "axis": args.axis_weight,
            "radial": args.radial_weight,
            "ordered": args.ordered_weight,
        },
    }
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    print("[INFO] touch_model target: tactile image -> local normalized patch", flush=True)
    print(f"[INFO] dataset: {json.dumps(dataset.summary(), indent=2)}", flush=True)
    print(f"[INFO] output_dir: {args.output_dir}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.local_weight,
            args.center_weight,
            args.spread_weight,
            args.axis_weight,
            args.radial_weight,
            args.ordered_weight,
        )
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                device,
                None,
                args.local_weight,
                args.center_weight,
                args.spread_weight,
                args.axis_weight,
                args.radial_weight,
                args.ordered_weight,
            )
            monitor = val_metrics["loss"]
        else:
            val_metrics = {}
            monitor = train_metrics["loss"]

        parts = [
            f"[EPOCH {epoch:04d}/{args.epochs:04d}]",
            f"train_loss={train_metrics['loss']:.6f}",
            f"train_local={train_metrics['local_chamfer']:.6f}",
            f"train_spread={train_metrics['spread']:.6f}",
        ]
        if val_metrics:
            parts.extend(
                [
                    f"val_loss={val_metrics['loss']:.6f}",
                    f"val_local={val_metrics['local_chamfer']:.6f}",
                    f"val_spread={val_metrics['spread']:.6f}",
                ]
            )
        print(" ".join(parts), flush=True)

        _save_checkpoint(args.output_dir / "touch_model_last.pt", model, optimizer, epoch, best_metric, train_config)
        if monitor < best_metric:
            best_metric = monitor
            _save_checkpoint(args.output_dir / "touch_model_best.pt", model, optimizer, epoch, best_metric, train_config)
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                args.output_dir / f"touch_model_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_metric,
                train_config,
            )

    print(f"[DONE] best_loss={best_metric:.6f}", flush=True)
    print(f"[DONE] best_checkpoint={args.output_dir / 'touch_model_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
