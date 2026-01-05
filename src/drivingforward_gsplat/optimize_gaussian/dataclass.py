from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class OptimizeGaussianConfig:
    gaussian_ply_path: Optional[str] = None
    output_dir: str = "output/gaussians"
    output_ply_name: str = "optimized.ply"
    debug_dir_name: str = "debug"
    device: str = "cuda"
    lr: float = 5e-3
    raw_steps: int = 1000
    phase2_steps: int = 1000
    phase3_steps: int = 1000
    phase1_cam_count: int = 2
    phase2_cam_count: int = 4
    jitter_views_per_cam: int = 4
    fixer_ratio: float = 0.33
    danger_percentile: float = 0.25
    lambda_fix_low: float = 1.0
    lambda_fix_lpips: float = 0.1
    blur_sigma: float = 1.5
    gamma: float = 5.0
    lambda_sigma: float = 0.1
    sigma_min: float = 0.03
    merge_every: int = 200
    merge_voxel_size: float = 0.05
    merge_small_scale: float = 0.02
    merge_thin_opacity: float = 0.05
    background_freeze_steps: Optional[int] = 500
    background_remove_step: Optional[int] = None
    log_every: int = 50
    sky_erode_kernel: int = 3
    sky_erode_iter: int = 1
    sam3_prompt: str = "sky"
    sam3_invert: bool = True
    sam3_model_id: str = "facebook/sam3"
    sam3_device: str = "cuda"
    sam3_dtype: str = "auto"
    sam3_mask_threshold: float = 0.5
    sam3_resize_longest_side: Optional[int] = None
    use_lpips: bool = True
    lpips_net: str = "vgg"
    background_color: List[float] = None
    random_seed: int = 0
    phase_settings: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizeGaussianConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            gaussian_ply_path=data.get("gaussian_ply_path"),
            output_dir=data.get("output_dir", cls.output_dir),
            output_ply_name=data.get("output_ply_name", cls.output_ply_name),
            debug_dir_name=data.get("debug_dir_name", cls.debug_dir_name),
            device=data.get("device", cls.device),
            lr=float(data.get("lr", cls.lr)),
            raw_steps=int(data.get("raw_steps", cls.raw_steps)),
            phase2_steps=int(data.get("phase2_steps", cls.phase2_steps)),
            phase3_steps=int(data.get("phase3_steps", cls.phase3_steps)),
            phase1_cam_count=int(data.get("phase1_cam_count", cls.phase1_cam_count)),
            phase2_cam_count=int(data.get("phase2_cam_count", cls.phase2_cam_count)),
            jitter_views_per_cam=int(
                data.get("jitter_views_per_cam", cls.jitter_views_per_cam)
            ),
            fixer_ratio=float(data.get("fixer_ratio", cls.fixer_ratio)),
            danger_percentile=float(
                data.get("danger_percentile", cls.danger_percentile)
            ),
            lambda_fix_low=float(data.get("lambda_fix_low", cls.lambda_fix_low)),
            lambda_fix_lpips=float(data.get("lambda_fix_lpips", cls.lambda_fix_lpips)),
            blur_sigma=float(data.get("blur_sigma", cls.blur_sigma)),
            gamma=float(data.get("gamma", cls.gamma)),
            lambda_sigma=float(data.get("lambda_sigma", cls.lambda_sigma)),
            sigma_min=float(data.get("sigma_min", cls.sigma_min)),
            merge_every=int(data.get("merge_every", cls.merge_every)),
            merge_voxel_size=float(data.get("merge_voxel_size", cls.merge_voxel_size)),
            merge_small_scale=float(
                data.get("merge_small_scale", cls.merge_small_scale)
            ),
            merge_thin_opacity=float(
                data.get("merge_thin_opacity", cls.merge_thin_opacity)
            ),
            background_freeze_steps=data.get(
                "background_freeze_steps", cls.background_freeze_steps
            ),
            background_remove_step=data.get(
                "background_remove_step", cls.background_remove_step
            ),
            log_every=int(data.get("log_every", cls.log_every)),
            sky_erode_kernel=int(data.get("sky_erode_kernel", cls.sky_erode_kernel)),
            sky_erode_iter=int(data.get("sky_erode_iter", cls.sky_erode_iter)),
            sam3_prompt=data.get("sam3_prompt", cls.sam3_prompt),
            sam3_invert=bool(data.get("sam3_invert", cls.sam3_invert)),
            sam3_model_id=data.get("sam3_model_id", cls.sam3_model_id),
            sam3_device=data.get("sam3_device", cls.sam3_device),
            sam3_dtype=data.get("sam3_dtype", cls.sam3_dtype),
            sam3_mask_threshold=float(
                data.get("sam3_mask_threshold", cls.sam3_mask_threshold)
            ),
            sam3_resize_longest_side=data.get(
                "sam3_resize_longest_side", cls.sam3_resize_longest_side
            ),
            use_lpips=bool(data.get("use_lpips", cls.use_lpips)),
            lpips_net=data.get("lpips_net", cls.lpips_net),
            background_color=data.get("background_color", cls.background_color),
            random_seed=int(data.get("random_seed", cls.random_seed)),
            phase_settings=data.get("phase_settings", cls.phase_settings),
        )
