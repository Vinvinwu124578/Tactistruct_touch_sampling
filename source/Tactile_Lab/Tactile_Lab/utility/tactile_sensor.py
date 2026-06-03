# utils/tactile_sensor.py

from __future__ import annotations
import os
import numpy as np
import torch
import cv2
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

@configclass
class TactileSensorCfg:
    # ------------------------------------------------------------------
    # General tactile settings
    # ------------------------------------------------------------------
    sensor_type: str = "tactip"
    tactile_image_type: str = "depth"   # depth | depth_original | rgb
    tactile_img_size: int = 32

    if_render_tactile: bool = False
    if_save_tactile_reference_image: bool = False

    # ------------------------------------------------------------------
    # Depth / deformation processing
    # ------------------------------------------------------------------
    tactile_depth_enhance_scale: float = 1 / 0.0197
    if_depth: bool = False   # derived

    # ------------------------------------------------------------------
    # Camera intrinsics (ALL VARIABLE)
    # ------------------------------------------------------------------
    near_plane: float = 0.001
    far_plane: float = 1.0
    focal_length: float = 26.5           # mm
    focus_distance: float = 400.0        # mm
    horizontal_aperture: float = 20.955  # mm

    # ------------------------------------------------------------------
    # Camera extrinsics / wiring (ALL VARIABLE)
    # ------------------------------------------------------------------
    prim_path: str = "/World/envs/env_.*/Robot/tcp_link/tactile_cam"
    update_period: float = 0.1
    update_latest_camera_pose: bool = True

    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.065)
    offset_rot: tuple[float, float, float, float] = (0, 1, 0, 0)
    offset_convention: str = "ros"

    # ------------------------------------------------------------------
    # Advanced: override camera data types (optional)
    # ------------------------------------------------------------------
    camera_data_types: list[str] | None = None

    # ------------------------------------------------------------------
    # Built artifact (derived)
    # ------------------------------------------------------------------
    tactile_camera: TiledCameraCfg | None = None

    idx: int | None = None  # for multi-sensor setups (e.g. bipush), can be used to differentiate sensors in rendering
    # ------------------------------------------------------------------
    # Post-init logic (THIS is the key)
    # ------------------------------------------------------------------
    def __post_init__(self):
        # -------- depth mode --------
        self.if_depth = "depth" in self.tactile_image_type

        # -------- camera outputs --------
        if self.camera_data_types is not None:
            data_types = self.camera_data_types
        else:
            if self.tactile_image_type == "rgb":
                data_types = ["rgb"]
            else:
                data_types = ["distance_to_image_plane"]

        # -------- build camera cfg --------
        self.tactile_camera = TiledCameraCfg(
            prim_path=self.prim_path,
            update_period=self.update_period,
            height=self.tactile_img_size,
            width=self.tactile_img_size,
            update_latest_camera_pose=self.update_latest_camera_pose,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.focal_length,
                focus_distance=self.focus_distance,
                horizontal_aperture=self.horizontal_aperture,
                clipping_range=(self.near_plane, self.far_plane),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=self.offset_pos,
                rot=self.offset_rot,
                convention=self.offset_convention,
            ),
        )


class TactileSensor:
    """
    Tactile sensor wrapper that preserves the original env-level API.
    Designed to be a drop-in replacement for tactile-related methods
    in EdgeFollowEnv / ObjPushEnv.
    """

    def __init__(
        self,
        tactile_camera,
        cfg,
        device,
        num_envs: int,
        file_anchor: str,
    ):
        """
        Args:
            tactile_camera: IsaacLab TiledCamera
            cfg: env cfg (expects sensor_type, tactile_img_size, tactile_image_type)
            device: torch device
            num_envs: number of environments
            file_anchor: __file__ from the env (used to resolve paths)
        """
        self._tactile_camera = tactile_camera
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._file_anchor = file_anchor

        self._if_loaded_tactile_reference_image = False
        self.no_deformation_image: torch.Tensor | None = None
        self.idx = self.cfg.idx  # for multi-sensor setups (e.g. bipush), can be used to differentiate sensors in rendering

    # ------------------------------------------------------------------
    # Reference image handling
    # ------------------------------------------------------------------

    def _get_tactile_reference_img_path(self):
        saved_file_dir = os.path.join(
            os.path.dirname(os.path.abspath(self._file_anchor)),
            "../../../tactile_lab_assets/reference_images",
            self.cfg.sensor_type,
            f"{self.cfg.tactile_img_size}x{self.cfg.tactile_img_size}",
        )
        return saved_file_dir

    def _load_tactile_reference_images(self):
        saved_file_dir = self._get_tactile_reference_img_path()
        nodef_gray_savefile = os.path.join(saved_file_dir, "nodef_dep.npy")

        if not os.path.exists(nodef_gray_savefile):
            raise FileNotFoundError(
                f"Tactile reference image not found: {nodef_gray_savefile}"
            )

        self.no_deformation_image = torch.from_numpy(
            np.load(nodef_gray_savefile)
        ).float().to(self.device)

        self._if_loaded_tactile_reference_image = True

    def _save_tactile_reference_images(self):
        if self.num_envs != 1:
            raise ValueError(
                "Reference images can only be saved when num_envs == 1."
            )

        no_deformation_image = (
            self._tactile_camera.data.output["distance_to_image_plane"][0]
            .detach()
            .cpu()
            .numpy()
        )

        saved_file_dir = self._get_tactile_reference_img_path()
        os.makedirs(saved_file_dir, exist_ok=True)

        save_path = os.path.join(saved_file_dir, "nodef_dep.npy")
        np.save(save_path, no_deformation_image)

        self._plot_tactile_depth_image(no_deformation_image, pause_time=0.01)

        print(f"[TactileSensor] Reference image saved to {save_path}")
        print("Safely exit the script and comment out _save_tactile_reference_images().")
        exit()

    # ------------------------------------------------------------------
    # Main tactile image access 
    # ------------------------------------------------------------------

    def _get_tactile_images_tensors(self):  # EXACT semantics preserved
        """
        Returns tactile image tensor (uint8, [0,255]).
        """
        img = None
        # --------------------------------------------------
        # depth_original
        # --------------------------------------------------
        if self.cfg.tactile_image_type == 'depth_original':
            cam_img = self._tactile_camera.data.output["distance_to_image_plane"]

            # avoid invalid depth
            if cam_img.max() > 20:
                cam_img = torch.clamp(cam_img, 0, 1.0)

            cam_min, cam_max = cam_img.min(), cam_img.max()
            img = ((cam_img - cam_min) / (cam_max - cam_min) * 255).to(torch.uint8)

        else:
            # --------------------------------------------------
            # depth deformation 
            # --------------------------------------------------
            if self.cfg.tactile_image_type == 'depth':
                cam_img_original = self._tactile_camera.data.output["distance_to_image_plane"]

                cam_img = torch.clamp(
                    torch.abs(cam_img_original - self.no_deformation_image)
                    * self.cfg.tactile_depth_enhance_scale,
                    0,
                    1.0,
                )
                img = (cam_img * 255).to(torch.uint8)

            # --------------------------------------------------
            # rgb
            # --------------------------------------------------
            if self.cfg.tactile_image_type == 'rgb':
                cam_img = self._tactile_camera.data.output["rgb"]
                img = cam_img.clone()  # avoid in-place modification for rendering
                cam_img = cam_img/255.0

        # --------------------------------------------------
        # safety check
        # --------------------------------------------------
        if img is None:
            raise ValueError(f"Unknown image_type: {self.cfg.tactile_image_type}")

        # --------------------------------------------------
        # debug rendering (unchanged)
        # --------------------------------------------------
        if self.cfg.if_render_tactile:
            self._render_closed = False
            if not self._render_closed:
                suffix = f"_sensor{self.idx}" if self.idx is not None else ""
                if self.cfg.tactile_image_type == 'rgb':
                    self._render_tactile_img(
                        cam_img[0],
                        title=f"tactile_window_rgb{suffix}",
                    )
                if self.cfg.if_depth:
                    self._render_tactile_img(
                        img[0],
                        title=f"tactile_window_depth{suffix}",
                    )
        return cam_img

    # ------------------------------------------------------------------
    # Debug utilities
    # ------------------------------------------------------------------

    def _plot_tactile_depth_image(self, tactile_depth_image_tensor, pause_time=1):
        import matplotlib.pyplot as plt

        if isinstance(tactile_depth_image_tensor, torch.Tensor):
            img = tactile_depth_image_tensor.detach().cpu().numpy()
        else:
            img = tactile_depth_image_tensor

        plt.imshow(img, cmap="gray")
        plt.title("Tactile Depth Image")
        plt.pause(pause_time)
        plt.close()

    def _render_tactile_img(self, img, title="tactile_window"):
        """
        Render tactile image using OpenCV (debug only).
        Preserves original semantics exactly.
        """

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Convert RGB -> BGR for OpenCV
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize to fixed resolution for visualization
        img = cv2.resize(
            img,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,  # preserve pixels
        )

        if not self._render_closed:
            cv2.imshow(title, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to close
                cv2.destroyWindow(title)
                self._render_closed = True
