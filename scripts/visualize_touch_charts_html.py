"""Create Plotly HTML visualizations for TouchSDF touch chart datasets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh


parser = argparse.ArgumentParser(description="Visualize one TouchSDF touch chart sample as an interactive HTML file.")
parser.add_argument("--dataset_path", type=str, default="", help="Path to touch_charts_gt_*.pkl or .npy.")
parser.add_argument("--touchsdf_root", type=str, default="", help="TouchSDF root, used with --dataset_name.")
parser.add_argument("--dataset_name", type=str, default="", help="Dataset name, e.g. touch_charts_gt_isaac_shapenet_smoke.")
parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize.")
parser.add_argument("--per_object", action="store_true", default=False, help="Write one HTML file per object.")
parser.add_argument("--object_index", type=str, default="", help="Only visualize this obj_index when --per_object is set.")
parser.add_argument(
    "--html_style",
    choices=["original_touch", "dashboard", "sampling_process", "both", "all"],
    default="original_touch",
    help=(
        "original_touch matches Tactistruct's two-layer 3D touch point cloud; "
        "dashboard keeps the older multi-panel view; sampling_process animates contacts in sampling order."
    ),
)
parser.add_argument("--max_tactile_images", type=int, default=12, help="Max tactile thumbnails per object page.")
parser.add_argument("--output_html", type=str, default="", help="HTML output path. Defaults beside the dataset manifest.")
parser.add_argument("--mesh_points", type=int, default=5000, help="Number of source mesh surface points to draw.")
parser.add_argument("--include_plotlyjs", choices=["embed", "cdn"], default="embed", help="Embed Plotly JS or use CDN.")
parser.add_argument(
    "--save_original_touch_intermediates",
    action="store_true",
    default=False,
    help="Write per-sample original_touch folders containing original_touch.html and original_touch.npz.",
)
args = parser.parse_args()


ORIGINAL_TOUCH_COLORSCALE = [
    [0.0, "rgb(110,0,0)"],
    [0.15, "rgb(210,25,20)"],
    [0.30, "rgb(255,118,32)"],
    [0.50, "rgb(255,238,82)"],
    [0.70, "rgb(35,215,255)"],
    [0.85, "rgb(25,100,255)"],
    [1.0, "rgb(35,0,170)"],
]
ORIGINAL_MESH_MARKER = {"size": 1.15, "color": "rgba(70,95,255,0.26)", "opacity": 0.26}
ORIGINAL_MESH_OUTLINE_MAX_EDGES = 6000
ORIGINAL_MESH_OUTLINE_LINE = {"color": "rgba(20,24,35,0.62)", "width": 1.4}
ORIGINAL_TOUCH_MARKER_SIZE = 4.2
ORIGINAL_TOUCH_MARKER_LINE = {"color": "rgba(30,30,30,0.35)", "width": 0.35}
ORIGINAL_NORMAL_LINE = {"color": "rgba(10,10,10,0.92)", "width": 5}
ORIGINAL_NORMAL_CONE_COLOR = "rgba(10,10,10,0.92)"


def _categorical_touch_color(relative_id: int, total: int) -> str:
    """Return a high-contrast color for separating individual tactile patches."""
    if total <= 1:
        hue = 215.0
    else:
        hue = (relative_id * 137.508) % 360.0
    return f"hsl({hue:.1f}, 82%, 44%)"


def _resolve_dataset_path() -> Path:
    if args.dataset_path:
        path = Path(args.dataset_path).expanduser().resolve()
    else:
        if not args.touchsdf_root or not args.dataset_name:
            raise ValueError("Provide either --dataset_path or both --touchsdf_root and --dataset_name.")
        name = args.dataset_name
        if not name.startswith("touch_charts_gt_"):
            name = "touch_charts_gt_" + name
        path = Path(args.touchsdf_root).expanduser().resolve() / "results" / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _load_dataset(path: Path) -> dict:
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    return np.load(path, allow_pickle=True).item()


def _dataset_folder(path: Path) -> Path:
    return path.parent / path.stem


def _load_metadata(path: Path) -> dict:
    metadata_path = _dataset_folder(path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _value(data: dict, key: str, idx: int, default=None):
    if key not in data:
        return default
    value = data[key][idx]
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return value[0]
    return value


def _load_scaled_mesh(mesh_path: str, scale: float, metadata: dict | None = None) -> trimesh.Trimesh | None:
    if not mesh_path:
        return None
    path = Path(str(mesh_path))
    if not path.exists():
        return None
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = (vertices - center) * scale
    if metadata:
        object_world_pos = np.asarray(metadata.get("object_world_pos", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        vertices = vertices + object_world_pos
        object_bottom_z = metadata.get("object_bottom_z")
        if object_bottom_z is not None:
            target_bottom_z = float(object_world_pos[2]) + float(object_bottom_z)
            vertices[:, 2] += target_bottom_z - float(vertices[:, 2].min())
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def _axis_range(points: np.ndarray, pad: float = 0.003) -> list[float]:
    lo = float(np.min(points))
    hi = float(np.max(points))
    if abs(hi - lo) < 1.0e-9:
        return [lo - pad, hi + pad]
    extra = max((hi - lo) * 0.08, pad)
    return [lo - extra, hi + extra]


def _make_output_path(dataset_path: Path, sample_idx: int) -> Path:
    if args.output_html:
        return Path(args.output_html).expanduser().resolve()
    out_dir = _dataset_folder(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"sample_{sample_idx:06d}_touch_chart.html"


def _safe_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe.strip("_") or "object"


def _make_object_output_path(dataset_path: Path, obj_index: str, group_count: int) -> Path:
    if args.output_html:
        requested = Path(args.output_html).expanduser().resolve()
        if group_count == 1 and requested.suffix:
            requested.parent.mkdir(parents=True, exist_ok=True)
            return requested
        out_dir = requested
    else:
        out_dir = _dataset_folder(dataset_path) / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_safe_name(obj_index)}_touch_charts.html"


def _make_original_touch_output_path(dataset_path: Path, obj_index: str, group_count: int) -> Path:
    if args.output_html and args.html_style == "original_touch":
        requested = Path(args.output_html).expanduser().resolve()
        if group_count == 1 and requested.suffix:
            requested.parent.mkdir(parents=True, exist_ok=True)
            return requested
        out_dir = requested
    else:
        out_dir = _dataset_folder(dataset_path) / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_safe_name(obj_index)}_original_touch.html"


def _make_sampling_process_output_path(dataset_path: Path, obj_index: str, group_count: int) -> Path:
    if args.output_html and args.html_style == "sampling_process":
        requested = Path(args.output_html).expanduser().resolve()
        if group_count == 1 and requested.suffix:
            requested.parent.mkdir(parents=True, exist_ok=True)
            return requested
        out_dir = requested
    else:
        out_dir = _dataset_folder(dataset_path) / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_safe_name(obj_index)}_sampling_process.html"


def _make_original_touch_sample_dir(dataset_path: Path, idx: int, obj_index: str) -> Path:
    out_dir = _dataset_folder(dataset_path) / "original_touch" / f"{idx:06d}_{_safe_name(obj_index)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _object_groups(data: dict, n_samples: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx in range(n_samples):
        obj_index = str(_value(data, "obj_index", idx, f"sample_{idx:06d}"))
        if args.object_index and obj_index != args.object_index:
            continue
        groups.setdefault(obj_index, []).append(idx)
    return groups


def _tactile_montage(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((1, 1), dtype=np.float32)
    count = min(len(images), max(args.max_tactile_images, 1))
    images = [np.asarray(img).squeeze().astype(np.float32) for img in images[:count]]
    height, width = images[0].shape
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    pad = max(width // 32, 2)
    montage = np.zeros((rows * height + (rows - 1) * pad, cols * width + (cols - 1) * pad), dtype=np.float32)
    for image_idx, image in enumerate(images):
        row = image_idx // cols
        col = image_idx % cols
        y0 = row * (height + pad)
        x0 = col * (width + pad)
        montage[y0 : y0 + height, x0 : x0 + width] = image
    return montage


def _mesh_display_points(mesh: trimesh.Trimesh | None) -> np.ndarray | None:
    if mesh is None:
        return None
    target_count = max(int(args.mesh_points), 1)
    if len(mesh.vertices) >= target_count:
        chosen = np.random.choice(len(mesh.vertices), size=target_count, replace=False)
        return np.asarray(mesh.vertices[chosen], dtype=np.float32)
    points, _ = trimesh.sample.sample_surface(mesh, target_count)
    return np.asarray(points, dtype=np.float32)


def _mesh_outline_lines(mesh: trimesh.Trimesh | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.edges_unique) == 0:
        return None
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) > ORIGINAL_MESH_OUTLINE_MAX_EDGES:
        chosen = np.random.choice(len(edges), size=ORIGINAL_MESH_OUTLINE_MAX_EDGES, replace=False)
        edges = edges[chosen]
    segments = np.asarray(mesh.vertices[edges], dtype=np.float32)
    line_count = len(segments)
    xs = np.empty(line_count * 3, dtype=np.float32)
    ys = np.empty(line_count * 3, dtype=np.float32)
    zs = np.empty(line_count * 3, dtype=np.float32)
    xs[0::3], xs[1::3], xs[2::3] = segments[:, 0, 0], segments[:, 1, 0], np.nan
    ys[0::3], ys[1::3], ys[2::3] = segments[:, 0, 1], segments[:, 1, 1], np.nan
    zs[0::3], zs[1::3], zs[2::3] = segments[:, 0, 2], segments[:, 1, 2], np.nan
    return xs, ys, zs


def _world_touch_points(data: dict, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    world_parts = []
    color_parts = []
    for idx in indices:
        patch_local = np.asarray(data["pointclouds"][idx], dtype=np.float32)
        rot_m = np.asarray(_value(data, "rot_M_wrld_list", idx, np.eye(3)), dtype=np.float32)
        contact = np.asarray(_value(data, "pos_wrld_list", idx, np.zeros(3)), dtype=np.float32).reshape(3)
        world_parts.append(patch_local @ rot_m.T + contact)
        color_parts.append(patch_local[:, 2])
    return np.concatenate(world_parts, axis=0), np.concatenate(color_parts, axis=0)


def _touch_patch_world(data: dict, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch_local = np.asarray(data["pointclouds"][idx], dtype=np.float32)
    rot_m = np.asarray(_value(data, "rot_M_wrld_list", idx, np.eye(3)), dtype=np.float32)
    contact = np.asarray(_value(data, "pos_wrld_list", idx, np.zeros(3)), dtype=np.float32).reshape(3)
    patch_world = patch_local @ rot_m.T + contact
    return patch_world, patch_local[:, 2], contact, rot_m


def _press_axis(contact: np.ndarray, rot_m: np.ndarray, axis_len: float) -> np.ndarray:
    press_dir = np.asarray(rot_m[:, 2], dtype=np.float32)
    press_dir /= np.linalg.norm(press_dir) + 1.0e-12
    return np.stack([contact - press_dir * axis_len, contact, contact + press_dir * axis_len * 0.35], axis=0)


def _color_range(values: np.ndarray) -> tuple[float, float]:
    cmin = float(np.nanmin(values))
    cmax = float(np.nanmax(values))
    if abs(cmax - cmin) < 1.0e-9:
        pad = max(abs(cmin) * 0.05, 1.0e-6)
        return cmin - pad, cmax + pad
    return cmin, cmax


def _save_original_touch_sample(dataset_path: Path, data: dict, metadata: dict, idx: int) -> Path:
    obj_index = str(_value(data, "obj_index", idx, f"sample_{idx:06d}"))
    output_dir = _make_original_touch_sample_dir(dataset_path, idx, obj_index)
    scale = float(metadata.get("scale", 1.0))
    mesh_path = str(_value(data, "mesh_path", idx, ""))
    mesh = _load_scaled_mesh(mesh_path, scale, metadata)
    original_vertices = (
        np.asarray(mesh.vertices, dtype=np.float32) if mesh is not None else np.zeros((0, 3), dtype=np.float32)
    )
    original_faces = (
        np.asarray(mesh.faces, dtype=np.int32) if mesh is not None else np.zeros((0, 3), dtype=np.int32)
    )
    patch_world, signed_distance, contact, rot_m = _touch_patch_world(data, idx)
    tactile = np.asarray(data["tactile_imgs"][idx]).squeeze().astype(np.float32)
    tactile_reference = np.asarray(_value(data, "tactile_reference_depth", idx, np.zeros_like(tactile)), dtype=np.float32)
    tactile_current = np.asarray(_value(data, "tactile_current_depth", idx, np.zeros_like(tactile)), dtype=np.float32)
    patch_local = np.asarray(data["pointclouds"][idx], dtype=np.float32)

    np.savez_compressed(
        output_dir / "original_touch.npz",
        original_vertices=original_vertices,
        original_faces=original_faces,
        pointcloud=patch_world.astype(np.float32),
        pointcloud_local=patch_local.astype(np.float32),
        signed_distance=np.asarray(signed_distance, dtype=np.float32),
        contact=contact.astype(np.float32),
        rot_M_wrld=rot_m.astype(np.float32),
        tactile_img=tactile.astype(np.float32),
        tactile_reference_depth=tactile_reference.astype(np.float32),
        tactile_current_depth=tactile_current.astype(np.float32),
    )
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_id": int(idx),
                "obj_index": obj_index,
                "mesh_path": mesh_path,
                "source_dataset": str(dataset_path),
                "npz_path": str(output_dir / "original_touch.npz"),
                "html_path": str(output_dir / "original_touch.html"),
            },
            f,
            indent=2,
        )

    cmin, cmax = _color_range(np.asarray(signed_distance, dtype=np.float32))
    fig = go.Figure()
    if original_vertices.size:
        fig.add_trace(
            go.Scatter3d(
                x=original_vertices[:, 0],
                y=original_vertices[:, 1],
                z=original_vertices[:, 2],
                mode="markers",
                name="original mesh",
                marker=ORIGINAL_MESH_MARKER,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=patch_world[:, 0],
            y=patch_world[:, 1],
            z=patch_world[:, 2],
            mode="markers",
            name="original touch",
            marker={
                "size": ORIGINAL_TOUCH_MARKER_SIZE,
                "color": signed_distance,
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": ORIGINAL_TOUCH_COLORSCALE,
                "showscale": True,
                "colorbar": {"title": "local z"},
                "line": ORIGINAL_TOUCH_MARKER_LINE,
            },
        )
    )
    if original_vertices.size:
        all_points = np.vstack([original_vertices, patch_world])
    else:
        all_points = patch_world
    fig.update_layout(
        title={"text": f"{dataset_path.stem} | sample {idx:06d} | original_touch", "x": 0.5},
        template="plotly_white",
        height=760,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        scene={
            "xaxis": {"title": "x", "range": _axis_range(all_points[:, 0])},
            "yaxis": {"title": "y", "range": _axis_range(all_points[:, 1])},
            "zaxis": {"title": "z", "range": _axis_range(all_points[:, 2])},
            "aspectmode": "data",
        },
    )
    include_plotlyjs = True if args.include_plotlyjs == "embed" else "cdn"
    fig.write_html(str(output_dir / "original_touch.html"), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_dir


def _original_touch_object_html(
    dataset_path: Path, data: dict, metadata: dict, obj_index: str, indices: list[int], group_count: int
) -> Path:
    scale = float(metadata.get("scale", 1.0))
    mesh_path = str(_value(data, "mesh_path", indices[0], ""))
    mesh = _load_scaled_mesh(mesh_path, scale, metadata)
    mesh_points = _mesh_display_points(mesh)
    mesh_outline = _mesh_outline_lines(mesh)
    patches = []
    patch_colors = []
    contacts = []
    normals = []
    for idx in indices:
        patch_world, patch_color, contact, rot_m = _touch_patch_world(data, idx)
        normal = -np.asarray(rot_m[:, 2], dtype=np.float32)
        normal /= np.linalg.norm(normal) + 1.0e-12
        patches.append(patch_world)
        patch_colors.append(patch_color)
        contacts.append(contact)
        normals.append(normal)
    all_points = np.concatenate(patches, axis=0) if patches else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(patch_colors, axis=0) if patch_colors else np.zeros((0,), dtype=np.float32)
    contacts_array = np.asarray(contacts, dtype=np.float32) if contacts else np.zeros((0, 3), dtype=np.float32)
    normals_array = np.asarray(normals, dtype=np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
    cmin, cmax = _color_range(all_colors)

    fig = go.Figure()
    if mesh_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=mesh_points[:, 0],
                y=mesh_points[:, 1],
                z=mesh_points[:, 2],
                mode="markers",
                name="original mesh",
                marker=ORIGINAL_MESH_MARKER,
            )
        )

    if mesh_outline is not None:
        outline_x, outline_y, outline_z = mesh_outline
        fig.add_trace(
            go.Scatter3d(
                x=outline_x,
                y=outline_y,
                z=outline_z,
                mode="lines",
                name="original mesh outline",
                line=ORIGINAL_MESH_OUTLINE_LINE,
                hoverinfo="skip",
            )
        )

    for relative_id, idx in enumerate(indices):
        patch_world = patches[relative_id]
        patch_color = patch_colors[relative_id]
        sample_color = _categorical_touch_color(relative_id, len(indices))
        label = f"sample {idx:06d} / touch {relative_id:02d}"
        fig.add_trace(
            go.Scatter3d(
                x=patch_world[:, 0],
                y=patch_world[:, 1],
                z=patch_world[:, 2],
                mode="markers",
                name=label,
                legendgroup=f"sample_{idx:06d}",
                marker={
                    "size": ORIGINAL_TOUCH_MARKER_SIZE,
                    "color": sample_color,
                    "opacity": 0.94,
                    "line": ORIGINAL_TOUCH_MARKER_LINE,
                },
                customdata=patch_color,
                hovertemplate=(
                    f"{label}<br>"
                    "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>"
                    "local z=%{customdata:.5f}<extra></extra>"
                ),
            )
        )

    if len(contacts_array):
        fig.add_trace(
            go.Scatter3d(
                x=contacts_array[:, 0],
                y=contacts_array[:, 1],
                z=contacts_array[:, 2],
                mode="markers+text",
                name="sample numbers",
                text=[f"{idx:03d}" for idx in indices],
                textposition="top center",
                marker={
                    "size": 8.5,
                    "color": [_categorical_touch_color(i, len(indices)) for i in range(len(indices))],
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "line": {"color": "#111827", "width": 1.2},
                },
                textfont={"size": 20, "color": "#111827"},
                hovertext=[f"sample {idx:06d} / touch {i:02d}" for i, idx in enumerate(indices)],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    if len(contacts_array) and len(normals_array):
        normal_len = max(float(metadata.get("patch_radius", 0.03)) * 1.8, 0.025)
        normal_tips = contacts_array + normals_array * normal_len
        normal_lines = np.empty((len(contacts_array) * 3, 3), dtype=np.float32)
        normal_lines[0::3] = contacts_array
        normal_lines[1::3] = normal_tips
        normal_lines[2::3] = np.nan
        normal_hover = []
        for i, idx in enumerate(indices):
            normal_hover.extend(
                [
                    f"sample {idx:06d} outward normal<br>"
                    f"n=({normals_array[i,0]:+.3f}, {normals_array[i,1]:+.3f}, {normals_array[i,2]:+.3f})",
                    f"sample {idx:06d} outward normal<br>"
                    f"n=({normals_array[i,0]:+.3f}, {normals_array[i,1]:+.3f}, {normals_array[i,2]:+.3f})",
                    "",
                ]
            )
        fig.add_trace(
            go.Scatter3d(
                x=normal_lines[:, 0],
                y=normal_lines[:, 1],
                z=normal_lines[:, 2],
                mode="lines",
                name="surface normals (outward)",
                line=ORIGINAL_NORMAL_LINE,
                hovertext=normal_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Cone(
                x=normal_tips[:, 0],
                y=normal_tips[:, 1],
                z=normal_tips[:, 2],
                u=normals_array[:, 0],
                v=normals_array[:, 1],
                w=normals_array[:, 2],
                name="normal arrowheads",
                anchor="tip",
                sizemode="absolute",
                sizeref=normal_len * 0.28,
                colorscale=[[0.0, ORIGINAL_NORMAL_CONE_COLOR], [1.0, ORIGINAL_NORMAL_CONE_COLOR]],
                showscale=False,
                hovertext=[
                    f"sample {idx:06d} outward normal<br>"
                    f"n=({normals_array[i,0]:+.3f}, {normals_array[i,1]:+.3f}, {normals_array[i,2]:+.3f})"
                    for i, idx in enumerate(indices)
                ],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    if mesh_points is not None and len(all_points):
        focus_parts = [mesh_points, all_points, contacts_array]
    elif len(all_points):
        focus_parts = [all_points, contacts_array]
    else:
        focus_parts = [mesh_points] if mesh_points is not None else []
    if len(contacts_array) and len(normals_array):
        focus_parts.append(contacts_array + normals_array * max(float(metadata.get("patch_radius", 0.03)) * 1.8, 0.025))
    focus_points = np.vstack(focus_parts) if focus_parts else np.zeros((0, 3), dtype=np.float32)

    scene = {
        "xaxis": {"title": "x"},
        "yaxis": {"title": "y"},
        "zaxis": {"title": "z"},
        "aspectmode": "data",
        "dragmode": "orbit",
    }
    if len(focus_points):
        scene["xaxis"]["range"] = _axis_range(focus_points[:, 0])
        scene["yaxis"]["range"] = _axis_range(focus_points[:, 1])
        scene["zaxis"]["range"] = _axis_range(focus_points[:, 2])
    fig.update_layout(
        title={
            "text": f"{dataset_path.stem} | {obj_index} | original_touch ({len(indices)} samples)",
            "x": 0.5,
        },
        template="plotly_white",
        height=760,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        scene=scene,
        dragmode="orbit",
        legend={"itemsizing": "constant"},
    )

    output_html = _make_original_touch_output_path(dataset_path, obj_index, group_count)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else "cdn"
    fig.write_html(
        str(output_html),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    return output_html


def _sampling_process_object_html(
    dataset_path: Path, data: dict, metadata: dict, obj_index: str, indices: list[int], group_count: int
) -> Path:
    scale = float(metadata.get("scale", 1.0))
    mesh_path = str(_value(data, "mesh_path", indices[0], ""))
    mesh_points = _mesh_display_points(_load_scaled_mesh(mesh_path, scale, metadata))

    patches = []
    patch_colors = []
    contacts = []
    axes = []
    tactile_images = []
    for idx in indices:
        patch_world, patch_color, contact, rot_m = _touch_patch_world(data, idx)
        patches.append(patch_world)
        patch_colors.append(patch_color)
        contacts.append(contact)
        tactile_images.append(np.asarray(data["tactile_imgs"][idx]).squeeze().astype(np.float32))

    contacts_array = np.stack(contacts, axis=0)
    all_patches = np.concatenate(patches, axis=0)
    all_colors = np.concatenate(patch_colors, axis=0)
    world_for_span = all_patches if mesh_points is None else np.vstack([mesh_points, all_patches, contacts_array])
    span = float(np.max(world_for_span.max(axis=0) - world_for_span.min(axis=0)))
    axis_len = max(span * 0.12, float(metadata.get("indentation_depth", 0.004)) * 8.0, float(metadata.get("patch_radius", 0.012)) * 1.5)
    for idx in indices:
        _, _, contact, rot_m = _touch_patch_world(data, idx)
        axes.append(_press_axis(contact, rot_m, axis_len))

    cmin, cmax = _color_range(all_colors)
    world_with_axes = np.vstack([world_for_span, *axes])
    world_ranges = [_axis_range(world_with_axes[:, axis]) for axis in range(3)]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.72, 0.28],
        subplot_titles=("Sampling order on ShapeNet object", "Current deformation tactile image"),
        horizontal_spacing=0.035,
    )

    if mesh_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=mesh_points[:, 0],
                y=mesh_points[:, 1],
                z=mesh_points[:, 2],
                mode="markers",
                name="source mesh",
                marker={"size": 1, "color": "rgba(120,130,150,0.28)"},
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="source mesh"), row=1, col=1)

    fig.add_trace(
        go.Scatter3d(
            x=contacts_array[:, 0],
            y=contacts_array[:, 1],
            z=contacts_array[:, 2],
            mode="markers+text",
            name="sampled contacts",
            text=[str(i) for i in range(len(indices))],
            textposition="top center",
            marker={"size": 4.2, "color": np.arange(len(indices)), "colorscale": "Turbo", "opacity": 0.9},
        ),
        row=1,
        col=1,
    )

    first_patch = patches[0]
    first_color = patch_colors[0]
    first_contact = contacts_array[0]
    first_axis = axes[0]
    first_tactile = tactile_images[0]
    fig.add_trace(
        go.Scatter3d(
            x=first_patch[:, 0],
            y=first_patch[:, 1],
            z=first_patch[:, 2],
            mode="markers",
            name="current GT patch",
            marker={
                "size": 3.5,
                "color": first_color,
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": ORIGINAL_TOUCH_COLORSCALE,
                "showscale": True,
                "colorbar": {"title": "local z"},
            },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[first_contact[0]],
            y=[first_contact[1]],
            z=[first_contact[2]],
            mode="markers",
            name="current contact",
            marker={"size": 7, "color": "#ffcc00", "symbol": "diamond"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=first_axis[:, 0],
            y=first_axis[:, 1],
            z=first_axis[:, 2],
            mode="lines+markers",
            name="TacTip press direction",
            marker={"size": [3, 6, 3], "color": ["#222222", "#ffcc00", "#222222"]},
            line={"width": 6, "color": "#222222"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(first_tactile),
            colorscale="Gray",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"title": "deform"},
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    frames = []
    for relative_id, idx in enumerate(indices):
        patch = patches[relative_id]
        patch_color = patch_colors[relative_id]
        contact = contacts_array[relative_id]
        axis = axes[relative_id]
        tactile = tactile_images[relative_id]
        frames.append(
            go.Frame(
                name=str(relative_id),
                traces=[2, 3, 4, 5],
                data=[
                    go.Scatter3d(
                        x=patch[:, 0],
                        y=patch[:, 1],
                        z=patch[:, 2],
                        mode="markers",
                        marker={
                            "size": 3.5,
                            "color": patch_color,
                            "cmin": cmin,
                            "cmax": cmax,
                            "colorscale": ORIGINAL_TOUCH_COLORSCALE,
                            "showscale": True,
                        },
                    ),
                    go.Scatter3d(
                        x=[contact[0]],
                        y=[contact[1]],
                        z=[contact[2]],
                        mode="markers",
                        marker={"size": 7, "color": "#ffcc00", "symbol": "diamond"},
                    ),
                    go.Scatter3d(
                        x=axis[:, 0],
                        y=axis[:, 1],
                        z=axis[:, 2],
                        mode="lines+markers",
                        marker={"size": [3, 6, 3], "color": ["#222222", "#ffcc00", "#222222"]},
                        line={"width": 6, "color": "#222222"},
                    ),
                    go.Heatmap(z=np.flipud(tactile), colorscale="Gray", zmin=0, zmax=1, showscale=True),
                ],
            )
        )
    fig.frames = frames

    steps = []
    for relative_id, idx in enumerate(indices):
        steps.append(
            {
                "label": str(relative_id),
                "method": "animate",
                "args": [
                    [str(relative_id)],
                    {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                ],
            }
        )

    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, scaleanchor="x2", row=1, col=2)
    fig.update_layout(
        title={"text": f"{dataset_path.stem} | {obj_index} | sampling process ({len(indices)} touches)", "x": 0.5},
        template="plotly_white",
        height=820,
        margin={"l": 20, "r": 20, "t": 80, "b": 80},
        legend={"orientation": "h", "y": -0.06},
        scene={
            "xaxis": {"title": "x", "range": world_ranges[0]},
            "yaxis": {"title": "y", "range": world_ranges[1]},
            "zaxis": {"title": "z", "range": world_ranges[2]},
            "aspectmode": "data",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 750, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}},
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[{"active": 0, "steps": steps, "x": 0.18, "y": -0.09, "len": 0.78}],
    )

    output_html = _make_sampling_process_output_path(dataset_path, obj_index, group_count)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else "cdn"
    fig.write_html(str(output_html), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_html


def _object_html(dataset_path: Path, data: dict, metadata: dict, obj_index: str, indices: list[int], group_count: int) -> Path:
    scale = float(metadata.get("scale", 1.0))
    mesh_path = str(_value(data, "mesh_path", indices[0], ""))
    mesh = _load_scaled_mesh(mesh_path, scale, metadata)
    mesh_points = None
    if mesh is not None:
        mesh_points, _ = trimesh.sample.sample_surface(mesh, min(args.mesh_points, max(len(mesh.faces), 1)))

    tactile_images = [np.asarray(data["tactile_imgs"][idx]).squeeze().astype(np.float32) for idx in indices]
    montage = _tactile_montage(tactile_images)

    local_parts = []
    world_parts = []
    color_parts = []
    contacts = []
    for relative_id, idx in enumerate(indices):
        patch_local = np.asarray(data["pointclouds"][idx], dtype=np.float32)
        rot_m = np.asarray(_value(data, "rot_M_wrld_list", idx, np.eye(3)), dtype=np.float32)
        contact = np.asarray(_value(data, "pos_wrld_list", idx, np.zeros(3)), dtype=np.float32).reshape(3)
        local_parts.append(patch_local)
        world_parts.append(patch_local @ rot_m.T + contact)
        color_parts.append(np.full(patch_local.shape[0], relative_id, dtype=np.float32))
        contacts.append(contact)

    patch_local_all = np.concatenate(local_parts, axis=0)
    patch_world_all = np.concatenate(world_parts, axis=0)
    sample_colors = np.concatenate(color_parts, axis=0)
    contacts = np.stack(contacts, axis=0)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}]],
        column_widths=[0.25, 0.35, 0.40],
        subplot_titles=(
            f"Tactile images ({min(len(indices), args.max_tactile_images)}/{len(indices)})",
            "GT patches in TCP frame",
            "ShapeNet mesh and all contact patches",
        ),
        horizontal_spacing=0.04,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(montage),
            colorscale="Gray",
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=patch_local_all[:, 0],
            y=patch_local_all[:, 1],
            z=patch_local_all[:, 2],
            mode="markers",
            name="GT patches local",
            marker={"size": 2.5, "color": sample_colors, "colorscale": "Turbo", "opacity": 0.9},
        ),
        row=1,
        col=2,
    )
    if mesh_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=mesh_points[:, 0],
                y=mesh_points[:, 1],
                z=mesh_points[:, 2],
                mode="markers",
                name="source mesh",
                marker={"size": 1.2, "color": "rgba(120,140,170,0.22)"},
            ),
            row=1,
            col=3,
        )
    fig.add_trace(
        go.Scatter3d(
            x=patch_world_all[:, 0],
            y=patch_world_all[:, 1],
            z=patch_world_all[:, 2],
            mode="markers",
            name="GT patches world",
            marker={"size": 2.4, "color": sample_colors, "colorscale": "Turbo", "opacity": 0.92},
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter3d(
            x=contacts[:, 0],
            y=contacts[:, 1],
            z=contacts[:, 2],
            mode="markers",
            name="contact points",
            marker={"size": 5, "color": "#0b5fff", "symbol": "diamond"},
        ),
        row=1,
        col=3,
    )

    local_ranges = [_axis_range(patch_local_all[:, axis]) for axis in range(3)]
    world_points = patch_world_all if mesh_points is None else np.vstack([mesh_points, patch_world_all])
    world_ranges = [_axis_range(world_points[:, axis]) for axis in range(3)]
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", row=1, col=1)
    fig.update_layout(
        title={"text": f"{dataset_path.stem} | {obj_index} | {len(indices)} touch charts", "x": 0.5},
        template="plotly_white",
        height=760,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        legend={"orientation": "h", "y": -0.02},
        scene={
            "xaxis": {"title": "x", "range": local_ranges[0]},
            "yaxis": {"title": "y", "range": local_ranges[1]},
            "zaxis": {"title": "z", "range": local_ranges[2]},
            "aspectmode": "data",
        },
        scene2={
            "xaxis": {"title": "x", "range": world_ranges[0]},
            "yaxis": {"title": "y", "range": world_ranges[1]},
            "zaxis": {"title": "z", "range": world_ranges[2]},
            "aspectmode": "data",
        },
    )

    output_html = _make_object_output_path(dataset_path, obj_index, group_count)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else "cdn"
    fig.write_html(str(output_html), include_plotlyjs=include_plotlyjs, full_html=True)
    return output_html


def main():
    dataset_path = _resolve_dataset_path()
    data = _load_dataset(dataset_path)
    metadata = _load_metadata(dataset_path)
    n_samples = int(data["tactile_imgs"].shape[0])

    if args.per_object:
        groups = _object_groups(data, n_samples)
        if not groups:
            raise RuntimeError("No object groups matched the requested filters.")
        outputs = []
        for obj_index, indices in groups.items():
            if args.html_style in ("original_touch", "both", "all"):
                outputs.append(_original_touch_object_html(dataset_path, data, metadata, obj_index, indices, len(groups)))
            if args.html_style in ("sampling_process", "all"):
                outputs.append(_sampling_process_object_html(dataset_path, data, metadata, obj_index, indices, len(groups)))
            if args.html_style in ("dashboard", "both", "all"):
                outputs.append(_object_html(dataset_path, data, metadata, obj_index, indices, len(groups)))
            if args.save_original_touch_intermediates:
                for idx in indices:
                    outputs.append(_save_original_touch_sample(dataset_path, data, metadata, idx))
        print(f"[DONE] Object HTML files: {len(outputs)}")
        for output in outputs:
            print(f"[DONE] {output}")
        return

    if args.sample_idx < 0 or args.sample_idx >= n_samples:
        raise IndexError(f"--sample_idx must be in [0, {n_samples - 1}], got {args.sample_idx}.")

    idx = args.sample_idx
    tactile = np.asarray(data["tactile_imgs"][idx]).squeeze().astype(np.float32)
    patch_local = np.asarray(data["pointclouds"][idx], dtype=np.float32)
    rot_m = np.asarray(_value(data, "rot_M_wrld_list", idx, np.eye(3)), dtype=np.float32)
    contact = np.asarray(_value(data, "pos_wrld_list", idx, np.zeros(3)), dtype=np.float32).reshape(3)
    obj_index = str(_value(data, "obj_index", idx, "unknown"))
    mesh_path = str(_value(data, "mesh_path", idx, ""))
    scale = float(metadata.get("scale", 1.0))

    patch_world = patch_local @ rot_m.T + contact
    mesh = _load_scaled_mesh(mesh_path, scale, metadata)
    mesh_points = None
    if mesh is not None:
        mesh_points, _ = trimesh.sample.sample_surface(mesh, min(args.mesh_points, max(len(mesh.faces), 1)))

    if args.html_style == "original_touch":
        obj_index = str(_value(data, "obj_index", idx, f"sample_{idx:06d}"))
        output = _original_touch_object_html(dataset_path, data, metadata, obj_index, [idx], 1)
        print(f"[DONE] HTML visualization: {output}")
        return
    if args.html_style == "sampling_process":
        obj_index = str(_value(data, "obj_index", idx, f"sample_{idx:06d}"))
        output = _sampling_process_object_html(dataset_path, data, metadata, obj_index, [idx], 1)
        print(f"[DONE] HTML visualization: {output}")
        return

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}]],
        column_widths=[0.25, 0.35, 0.40],
        subplot_titles=(
            "Tactile image",
            "GT patch in TCP frame",
            "ShapeNet mesh and contact patch",
        ),
        horizontal_spacing=0.04,
    )

    fig.add_trace(
        go.Heatmap(
            z=np.flipud(tactile),
            colorscale="Gray",
            showscale=True,
            colorbar={"title": "depth", "len": 0.55, "x": 0.255},
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=patch_local[:, 0],
            y=patch_local[:, 1],
            z=patch_local[:, 2],
            mode="markers",
            name="GT patch local",
            marker={"size": 3, "color": patch_local[:, 2], "colorscale": "Viridis", "opacity": 0.95},
        ),
        row=1,
        col=2,
    )

    if mesh_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=mesh_points[:, 0],
                y=mesh_points[:, 1],
                z=mesh_points[:, 2],
                mode="markers",
                name="source mesh",
                marker={"size": 1.2, "color": "rgba(120,140,170,0.24)"},
            ),
            row=1,
            col=3,
        )

    fig.add_trace(
        go.Scatter3d(
            x=patch_world[:, 0],
            y=patch_world[:, 1],
            z=patch_world[:, 2],
            mode="markers",
            name="GT patch world",
            marker={"size": 3, "color": "#d94f37", "opacity": 0.95},
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[contact[0]],
            y=[contact[1]],
            z=[contact[2]],
            mode="markers",
            name="contact point",
            marker={"size": 6, "color": "#0b5fff", "symbol": "diamond"},
        ),
        row=1,
        col=3,
    )

    local_ranges = [_axis_range(patch_local[:, axis]) for axis in range(3)]
    world_points = patch_world if mesh_points is None else np.vstack([mesh_points, patch_world])
    world_ranges = [_axis_range(world_points[:, axis]) for axis in range(3)]

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", row=1, col=1)
    fig.update_layout(
        title={
            "text": f"{dataset_path.stem} | sample {idx} | {obj_index}",
            "x": 0.5,
        },
        template="plotly_white",
        height=760,
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        legend={"orientation": "h", "y": -0.02},
        scene={
            "xaxis": {"title": "x", "range": local_ranges[0]},
            "yaxis": {"title": "y", "range": local_ranges[1]},
            "zaxis": {"title": "z", "range": local_ranges[2]},
            "aspectmode": "data",
        },
        scene2={
            "xaxis": {"title": "x", "range": world_ranges[0]},
            "yaxis": {"title": "y", "range": world_ranges[1]},
            "zaxis": {"title": "z", "range": world_ranges[2]},
            "aspectmode": "data",
        },
    )

    output_html = _make_output_path(dataset_path, idx)
    include_plotlyjs = True if args.include_plotlyjs == "embed" else "cdn"
    fig.write_html(str(output_html), include_plotlyjs=include_plotlyjs, full_html=True)
    print(f"[DONE] HTML visualization: {output_html}")


if __name__ == "__main__":
    main()
