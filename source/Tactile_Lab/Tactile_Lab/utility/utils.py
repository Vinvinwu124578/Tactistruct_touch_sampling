import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul, sample_uniform, quat_from_angle_axis, quat_slerp, quat_unique
import os
import argparse
import shutil
import json


def positional_encoding(x: torch.Tensor, num_freqs: int = 4):
    """
    Apply sinusoidal positional encoding to input tensor.
    Args:
        x (torch.Tensor): Input tensor of shape (N, D)
        num_freqs (int): Number of frequency bands
    Returns:
        torch.Tensor: Encoded tensor of shape (N, D * 2 * num_freqs)
    """
    N, D = x.shape
    freq_bands = 2 ** torch.arange(num_freqs, dtype=torch.float32, device=x.device) * torch.pi
    x = x.unsqueeze(-1)  # (N, D, 1)
    freqs = freq_bands.view(1, 1, -1)  # (1, 1, F)

    x_freq = x * freqs  # (N, D, F)
    sin_x = torch.sin(x_freq)
    cos_x = torch.cos(x_freq)

    return torch.cat([sin_x, cos_x], dim=-1).view(N, D * 2 * num_freqs)  # (N, D * 2F)


def rpy_from_quat(quat: torch.Tensor):
    """
    Convert quaternion to roll, pitch, yaw. Batch version.
    """
    return torch.stack(euler_xyz_from_quat(quat), dim=-1)


def quat_from_rpy(rpy: torch.Tensor):
    """
    Convert roll, pitch, yaw to quaternion. Batch version.
    """
    return quat_from_euler_xyz(*rpy.unbind(-1))


def rpy_quat_convert(rpy):
    """
    For single RPY: [1,3] or [3,], using for cfg robot or camera rpy to quat conversion.
    Convert between RPY (Rad) and quaternion.
    """
    if rpy is not None:
        rpy = torch.tensor(rpy, dtype=torch.float32) 
        rpy = rpy.unsqueeze(0)                  # (1,3)
        quat = quat_from_euler_xyz(*rpy.unbind(-1))

        return quat.squeeze(0).tolist()


def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Batch wrapper for quat_slerp.

    Args:
        q1: (N,4) start quaternions (w,x,y,z)
        q2: (N,4) target quaternions
        tau: interpolation coefficient [0,1]

    Returns:
        (N,4) interpolated quaternions
    """
    assert q1.shape == q2.shape
    assert q1.shape[-1] == 4

    N = q1.shape[0]
    out = torch.empty_like(q1)

    for i in range(N):
        out[i] = quat_slerp(q1[i], q2[i], tau)

    return out


def quat_to_rotvec(q: torch.Tensor):
    q = quat_unique(q)                # ensure shortest representation
    w, xyz = q[:, 0], q[:, 1:4]
    angle = 2.0 * torch.atan2(
        torch.norm(xyz, dim=-1),
        torch.clamp(w, min=1e-6),
    )
    axis = xyz / (torch.norm(xyz, dim=-1, keepdim=True) + 1e-6)
    return axis * angle.unsqueeze(-1)  # (N, 3)


def make_dir(dir, check=True):
    if check:
        check_dir(dir)
    os.makedirs(dir, exist_ok=True)


def check_dir(dir):
    if os.path.isdir(dir):
        str_input = input(f"\n{dir} \nSave Directory already exists, would you like to continue (y,n)? ")
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)


def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_json_obj(obj, name):
    with open(name + ".json", "w") as fp:
        json.dump(obj, fp)


def load_json_obj(name):
    with open(name + ".json", "r") as fp:
        return json.load(fp)
