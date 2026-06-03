from __future__ import annotations

import glob
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


_EPS = 1.0e-8


@dataclass
class TouchModelConfig:
    image_channels: int = 1
    image_latent_dim: int = 256
    image_pool_size: int = 4
    fusion_hidden_dim: int = 512
    num_points: int = 500

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TouchImageEncoder(nn.Module):
    """CNN branch from tactile image to z_img in R^256."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 256, pool_size: int = 4) -> None:
        super().__init__()
        pool_size = max(int(pool_size), 1)
        self.features = nn.Sequential(
            ConvNormAct(in_channels, 32, stride=2),
            ResBlock(32),
            ConvNormAct(32, 64, stride=2),
            ResBlock(64),
            ConvNormAct(64, 128, stride=2),
            ResBlock(128),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pool_size * pool_size, latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(image))


class TouchChartModel(nn.Module):
    """Image-only touch model that predicts a local normalized tactile patch."""

    def __init__(self, config: TouchModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = TouchModelConfig(**kwargs)
        self.config = config
        self.image_encoder = TouchImageEncoder(
            config.image_channels,
            config.image_latent_dim,
            config.image_pool_size,
        )
        out_dim = config.num_points * 3
        self.decoder = nn.Sequential(
            nn.Linear(config.image_latent_dim, config.fusion_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.fusion_hidden_dim, out_dim),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        z_img = self.image_encoder(image)
        points = self.decoder(z_img).view(image.shape[0], self.config.num_points, 3)
        return {
            "local_points": points,
        }


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance for B x P x 3 point clouds."""

    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2).values.mean(dim=1)
    target_to_pred = dist.min(dim=1).values.mean(dim=1)
    return (pred_to_target + target_to_pred).mean()


def center_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred.mean(dim=1), target.mean(dim=1))


def spread_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_std = pred.std(dim=1, unbiased=False)
    target_std = target.std(dim=1, unbiased=False)
    return torch.nn.functional.smooth_l1_loss(pred_std, target_std)


def axis_distribution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_sorted = pred.sort(dim=1).values
    target_sorted = target.sort(dim=1).values
    return torch.nn.functional.smooth_l1_loss(pred_sorted, target_sorted)


def radial_distribution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_radius = pred_centered.norm(dim=-1).sort(dim=1).values
    target_radius = target_centered.norm(dim=-1).sort(dim=1).values
    return torch.nn.functional.smooth_l1_loss(pred_radius, target_radius)


def ordered_point_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred, target)


def touch_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    local_weight: float = 1.0,
    center_weight: float = 0.25,
    spread_weight: float = 0.50,
    axis_weight: float = 0.50,
    radial_weight: float = 0.25,
    ordered_weight: float = 0.0,
    object_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = outputs["local_points"]
    target = batch["local_points"]
    local = chamfer_loss(pred, target)
    center = center_loss(pred, target)
    spread = spread_loss(pred, target)
    axis = axis_distribution_loss(pred, target)
    radial = radial_distribution_loss(pred, target)
    ordered = ordered_point_loss(pred, target)
    loss = (
        local_weight * local
        + center_weight * center
        + spread_weight * spread
        + axis_weight * axis
        + radial_weight * radial
        + ordered_weight * ordered
    )
    metrics: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "local_chamfer": local.detach(),
        "center": center.detach(),
        "spread": spread.detach(),
        "axis_dist": axis.detach(),
        "radial_dist": radial.detach(),
    }
    if ordered_weight > 0.0:
        metrics["ordered"] = ordered.detach()
    if object_weight > 0.0 and "object_points" in outputs and "object_points" in batch:
        obj = chamfer_loss(outputs["object_points"], batch["object_points"])
        loss = loss + object_weight * obj
        metrics["object_chamfer"] = obj.detach()
    metrics["loss"] = loss.detach()
    return loss, metrics


def expand_touchchart_paths(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        text = str(item)
        matches = [Path(p) for p in glob.glob(text)] if any(ch in text for ch in "*?[") else [Path(text)]
        for path in matches:
            if path.is_dir():
                sidecar = path.parent / f"{path.name}.pkl"
                local_pkl = path / f"{path.name}.pkl"
                any_pkl = sorted(path.glob("*.pkl"))
                any_npy = sorted(path.glob("*.npy"))
                if sidecar.exists():
                    paths.append(sidecar)
                elif local_pkl.exists():
                    paths.append(local_pkl)
                elif any_pkl:
                    paths.extend(any_pkl)
                elif any_npy:
                    paths.extend(any_npy)
                else:
                    raise FileNotFoundError(f"No .pkl or per-sample .npy touchchart file found for directory: {path}")
            else:
                paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
    else:
        raise ValueError(f"Unsupported touchchart file type: {path.suffix}")
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict-like touchchart file, got {type(data).__name__}: {path}")
    return data


def _metadata_path_for(data_path: Path) -> Path | None:
    candidates = [
        data_path.parent / data_path.stem / "metadata.json",
        data_path.with_suffix("") / "metadata.json",
        data_path.parent / "metadata.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_metadata(data_path: Path) -> dict[str, Any]:
    path = _metadata_path_for(data_path)
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_first(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required key. Tried: {', '.join(keys)}")


def _optional_first(data: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_per_sample_float(
    data: dict[str, Any],
    metadata: dict[str, Any],
    keys: Iterable[str],
    metadata_keys: Iterable[str],
    n: int,
    default: float,
) -> np.ndarray:
    value = _optional_first(data, keys)
    if value is None:
        for key in metadata_keys:
            if key in metadata:
                value = metadata[key]
                break
    if value is None:
        value = default
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((n,), float(arr[0]), dtype=np.float32)
    if arr.size != n:
        raise ValueError(f"Expected scalar or {n} values for {list(keys)}, got {arr.size}")
    return arr.astype(np.float32)


def _prepare_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[:, None, :, :]
    elif images.ndim == 4 and images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
        images = np.transpose(images, (0, 3, 1, 2))
    if images.ndim != 4:
        raise ValueError(f"Expected tactile images with shape N,H,W or N,C,H,W, got {images.shape}")
    if images.shape[1] == 3:
        images = images.mean(axis=1, keepdims=True)
    if images.shape[1] != 1:
        raise ValueError(f"TouchChartModel expects one tactile image channel, got {images.shape[1]}")
    return np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_images(images: np.ndarray) -> np.ndarray:
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    std = images.std(axis=(1, 2, 3), keepdims=True)
    return (images - mean) / np.maximum(std, 1.0e-6)


def _fit_point_count(points: np.ndarray, target_count: int) -> np.ndarray:
    count = points.shape[0]
    if count == target_count:
        return points
    if count > target_count:
        idx = np.linspace(0, count - 1, target_count, dtype=np.int64)
        return points[idx]
    repeats = int(np.ceil(target_count / max(count, 1)))
    tiled = np.tile(points, (repeats, 1))
    return tiled[:target_count]


class TouchChartDataset(Dataset):
    """Loads touchchart pkl/npy files for image-to-local-patch training."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        num_points: int | None = None,
        normalize_images: bool = True,
    ) -> None:
        self.paths = expand_touchchart_paths(paths)
        if not self.paths:
            raise ValueError("No touchchart paths were provided.")

        image_chunks: list[np.ndarray] = []
        point_chunks: list[np.ndarray] = []
        radius_chunks: list[np.ndarray] = []
        source_chunks: list[np.ndarray] = []

        inferred_points: int | None = None
        for path in self.paths:
            data = _load_mapping(path)
            metadata = _load_metadata(path)
            images = _prepare_images(np.asarray(_get_first(data, ("tactile_imgs", "tactile_images", "images"))))
            points = np.asarray(_get_first(data, ("pointclouds", "points", "patches")), dtype=np.float32)
            points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"Expected pointclouds with shape N,P,3, got {points.shape} in {path}")
            if images.shape[0] != points.shape[0]:
                raise ValueError(f"Image/sample count mismatch in {path}: {images.shape[0]} vs {points.shape[0]}")
            n = images.shape[0]
            inferred_points = points.shape[1] if inferred_points is None else inferred_points

            patch_radius = _as_per_sample_float(
                data,
                metadata,
                keys=("patch_radius", "patch_radii"),
                metadata_keys=("patch_radius",),
                n=n,
                default=1.0,
            )

            image_chunks.append(images)
            point_chunks.append(points)
            radius_chunks.append(patch_radius.astype(np.float32))
            source_chunks.append(np.asarray([str(path)] * n, dtype=object))

        self.images_raw = np.concatenate(image_chunks, axis=0).astype(np.float32)
        self.images = _normalize_images(self.images_raw) if normalize_images else self.images_raw
        self.points = np.concatenate(point_chunks, axis=0).astype(np.float32)
        self.patch_radius = np.concatenate(radius_chunks, axis=0).astype(np.float32)
        self.source_path = np.concatenate(source_chunks, axis=0)
        self.num_points = int(num_points or inferred_points or self.points.shape[1])

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        points = _fit_point_count(self.points[index], self.num_points).astype(np.float32)
        radius = float(max(self.patch_radius[index], _EPS))
        return {
            "image": torch.from_numpy(self.images[index].astype(np.float32)),
            "local_points": torch.from_numpy(points / radius),
            "raw_points": torch.from_numpy(points),
            "patch_radius": torch.tensor(radius, dtype=torch.float32),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "paths": [str(path) for path in self.paths],
            "samples": len(self),
            "num_points": self.num_points,
            "image_shape": list(self.images.shape[1:]),
            "patch_radius_min": float(self.patch_radius.min()),
            "patch_radius_max": float(self.patch_radius.max()),
        }
