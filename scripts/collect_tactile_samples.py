# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect tactile samples from a Tactile_Lab Isaac Lab environment.

The script enables the Obj-Push tactile camera path at runtime, steps the
environment with simple scripted actions, and saves tactile images plus useful
state labels for quick downstream experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect tactile samples from Tactile_Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Obj-Push-Tactile-v0", help="Gym task name.")
parser.add_argument("--samples", type=int, default=256, help="Total samples to save across all environments.")
parser.add_argument("--warmup_steps", type=int, default=30, help="Steps to run before saving samples.")
parser.add_argument("--steps_per_sample", type=int, default=4, help="Environment steps between saved samples.")
parser.add_argument("--action_mode", choices=["zero", "random", "sweep"], default="random", help="Action policy.")
parser.add_argument("--action_scale", type=float, default=0.35, help="Magnitude for random/sweep actions.")
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument("--output_root", type=str, default="outputs/tactile_samples", help="Root output directory.")
parser.add_argument("--name_output", type=str, default="", help="Dataset folder name. Timestamped if empty.")
parser.add_argument("--reset_output", action="store_true", default=False, help="Delete existing output folder first.")
parser.add_argument("--save_png", action="store_true", default=False, help="Also save each tactile image as PNG.")
parser.add_argument("--tactile_img_size", type=int, default=32, help="Tactile camera image size.")
parser.add_argument(
    "--tactile_image_type",
    choices=["depth", "depth_original", "rgb"],
    default="depth",
    help="Image returned by TactileSensor.",
)
parser.add_argument("--sensor_type", type=str, default="right_angle_tactip", help="Reference image folder name.")
parser.add_argument("--camera_update_period", type=float, default=0.0, help="Tactile camera update period.")
parser.add_argument("--render_tactile", action="store_true", default=False, help="Show OpenCV tactile debug window.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.samples < 1:
    parser.error("--samples must be at least 1.")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps cannot be negative.")
if args_cli.steps_per_sample < 1:
    parser.error("--steps_per_sample must be at least 1.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Tactile_Lab.tasks  # noqa: F401
from Tactile_Lab.utility.tactile_sensor import TactileSensorCfg


def _make_output_dir() -> Path:
    name = args_cli.name_output.strip()
    if not name:
        name = "tactile_samples_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_root).expanduser().resolve() / name
    if output_dir.exists() and args_cli.reset_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.save_png:
        (output_dir / "images").mkdir(exist_ok=True)
    return output_dir


def _configure_tactile_env(env_cfg):
    env_cfg.obs_type = "tactile"
    env_cfg.tactile_img_size = args_cli.tactile_img_size
    env_cfg.tactile_sensor_cfg = TactileSensorCfg(
        tactile_img_size=args_cli.tactile_img_size,
        tactile_image_type=args_cli.tactile_image_type,
        if_render_tactile=args_cli.render_tactile,
        sensor_type=args_cli.sensor_type,
        tactile_depth_enhance_scale=1 / 0.0097,
        offset_pos=(0.0, -0.065, 0.0),
        offset_rot=(-0.707, 0.707, 0.0, 0.0),
        update_period=args_cli.camera_update_period,
    )
    channels = 3 if args_cli.tactile_image_type == "rgb" else 1
    env_cfg.observation_space = {"image": [channels, args_cli.tactile_img_size, args_cli.tactile_img_size], "feature": 6}
    return env_cfg


def _make_actions(env, step_idx: int) -> torch.Tensor:
    shape = env.action_space.shape
    device = env.unwrapped.device
    if args_cli.action_mode == "zero":
        return torch.zeros(shape, device=device)
    if args_cli.action_mode == "random":
        return (2.0 * torch.rand(shape, device=device) - 1.0) * args_cli.action_scale

    action = torch.zeros(shape, device=device)
    phase = step_idx * 0.05
    action[..., 0] = torch.sin(torch.tensor(phase, device=device)) * args_cli.action_scale
    if action.shape[-1] > 1:
        action[..., 1] = torch.cos(torch.tensor(phase * 0.7, device=device)) * args_cli.action_scale
    return action


def _tensor_or_zeros(value, shape):
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return value.detach().cpu().numpy()


def _save_png(image_chw: np.ndarray, path: Path):
    import cv2

    image = np.moveaxis(image_chw, 0, -1)
    if image.shape[-1] == 1:
        gray = image[..., 0]
        if gray.max(initial=0.0) <= 1.0:
            gray = gray * 255.0
        cv2.imwrite(str(path), np.clip(gray, 0, 255).astype(np.uint8))
    else:
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def _policy_obs(obs):
    return obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    output_dir = _make_output_dir()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg = _configure_tactile_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] observation_space={env.observation_space}")
    print(f"[INFO] action_space={env.action_space}")
    print(f"[INFO] output_dir={output_dir}")

    rows = []
    images = []
    features = []
    actions_out = []
    rewards_out = []
    terminated_out = []
    truncated_out = []
    ee_pose_w = []
    object_pose_w = []
    goal_pos_w = []
    goal_quat_w = []
    contact_force_w = []

    obs, _ = env.reset()
    latest_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for step_idx in range(args_cli.warmup_steps):
        latest_action = _make_actions(env, step_idx)
        obs, _, _, _, _ = env.step(latest_action)

    sample_idx = 0
    step_idx = args_cli.warmup_steps
    while simulation_app.is_running() and sample_idx < args_cli.samples:
        for _ in range(args_cli.steps_per_sample):
            latest_action = _make_actions(env, step_idx)
            obs, reward, terminated, truncated, _ = env.step(latest_action)
            step_idx += 1

        obs_dict = _policy_obs(obs)
        image_batch = obs_dict["image"].detach().cpu().numpy().astype(np.float32)
        feature_batch = obs_dict["feature"].detach().cpu().numpy().astype(np.float32)
        action_batch = latest_action.detach().cpu().numpy().astype(np.float32)
        reward_batch = reward.detach().cpu().numpy().astype(np.float32)
        term_batch = terminated.detach().cpu().numpy().astype(bool)
        trunc_batch = truncated.detach().cpu().numpy().astype(bool)

        unwrapped = env.unwrapped
        ee_batch = _tensor_or_zeros(getattr(unwrapped, "ee_pose_world", None), (args_cli.num_envs, 7))
        object_pose_batch = np.concatenate(
            [
                unwrapped._object.data.root_pos_w.detach().cpu().numpy(),
                unwrapped._object.data.root_quat_w.detach().cpu().numpy(),
            ],
            axis=-1,
        ).astype(np.float32)
        goal_pos_batch = unwrapped.current_goal_pose_worldframe_all_envs[:, 0:3].detach().cpu().numpy().astype(np.float32)
        goal_quat_batch = unwrapped.current_goal_quat_worldframe_all_envs.detach().cpu().numpy().astype(np.float32)
        contact_batch = _tensor_or_zeros(getattr(unwrapped, "tip_contact_force", None), (args_cli.num_envs, 1, 3))

        for env_id in range(args_cli.num_envs):
            if sample_idx >= args_cli.samples:
                break
            image_file = ""
            if args_cli.save_png:
                image_file = f"images/sample_{sample_idx:06d}_env_{env_id}.png"
                _save_png(image_batch[env_id], output_dir / image_file)

            rows.append(
                {
                    "sample_id": sample_idx,
                    "env_id": env_id,
                    "sim_step": step_idx,
                    "reward": float(reward_batch[env_id]),
                    "terminated": bool(term_batch[env_id]),
                    "truncated": bool(trunc_batch[env_id]),
                    "image_file": image_file,
                }
            )
            images.append(image_batch[env_id])
            features.append(feature_batch[env_id])
            actions_out.append(action_batch[env_id])
            rewards_out.append(reward_batch[env_id])
            terminated_out.append(term_batch[env_id])
            truncated_out.append(trunc_batch[env_id])
            ee_pose_w.append(ee_batch[env_id])
            object_pose_w.append(object_pose_batch[env_id])
            goal_pos_w.append(goal_pos_batch[env_id])
            goal_quat_w.append(goal_quat_batch[env_id])
            contact_force_w.append(contact_batch[env_id])
            sample_idx += 1

        if sample_idx % max(args_cli.num_envs * 10, 1) == 0 or sample_idx >= args_cli.samples:
            print(f"[INFO] collected {sample_idx}/{args_cli.samples}")

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        tactile_images=np.stack(images, axis=0),
        features=np.stack(features, axis=0),
        actions=np.stack(actions_out, axis=0),
        rewards=np.asarray(rewards_out, dtype=np.float32),
        terminated=np.asarray(terminated_out, dtype=bool),
        truncated=np.asarray(truncated_out, dtype=bool),
        ee_pose_w=np.stack(ee_pose_w, axis=0).astype(np.float32),
        object_pose_w=np.stack(object_pose_w, axis=0).astype(np.float32),
        goal_pos_w=np.stack(goal_pos_w, axis=0).astype(np.float32),
        goal_quat_w=np.stack(goal_quat_w, axis=0).astype(np.float32),
        contact_force_w=np.stack(contact_force_w, axis=0).astype(np.float32),
    )

    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "samples": sample_idx,
        "tactile_image_type": args_cli.tactile_image_type,
        "tactile_img_size": args_cli.tactile_img_size,
        "sensor_type": args_cli.sensor_type,
        "action_mode": args_cli.action_mode,
        "action_scale": args_cli.action_scale,
        "warmup_steps": args_cli.warmup_steps,
        "steps_per_sample": args_cli.steps_per_sample,
        "dataset_path": str(dataset_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    print(f"[DONE] Dataset: {dataset_path}")
    print(f"[DONE] Manifest: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
