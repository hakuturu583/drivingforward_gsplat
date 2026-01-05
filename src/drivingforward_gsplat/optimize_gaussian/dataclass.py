from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class LossConfig:
    weight: float


@dataclass
class PhotometricLossConfig(LossConfig):
    pass


@dataclass
class FixerLossConfig(LossConfig):
    low_freq_weight: float = 1.0
    lpips_weight: float = 0.1
    danger_percentile: Optional[float] = None
    blur_sigma: Optional[float] = None
    gamma: Optional[float] = None


@dataclass
class MinScaleLossConfig(LossConfig):
    sigma_min: Optional[float] = None


@dataclass
class SkyMaskConfig:
    erode_kernel: int = 3
    erode_iter: int = 1
    sam3_prompt: str = "sky"
    sam3_invert: bool = True
    sam3_model_id: str = "facebook/sam3"
    sam3_device: str = "cuda"
    sam3_dtype: str = "auto"
    sam3_mask_threshold: float = 0.5
    sam3_resize_longest_side: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "SkyMaskConfig":
        return cls(
            erode_kernel=int(data.get("erode_kernel", cls.erode_kernel)),
            erode_iter=int(data.get("erode_iter", cls.erode_iter)),
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
        )


@dataclass
class PhaseConfig:
    photometric_loss: Optional[PhotometricLossConfig] = None
    fixer_loss: Optional[FixerLossConfig] = None
    minscale_loss: Optional[MinScaleLossConfig] = None
    steps: Optional[int] = None
    cam_count: Optional[int] = None
    jitter_views_per_cam: Optional[int] = None
    jitter_cm: Optional[float] = None
    sigma_min: Optional[float] = None
    lambda_sigma: Optional[float] = None
    danger_percentile: Optional[float] = None
    blur_sigma: Optional[float] = None
    gamma: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "PhaseConfig":
        photometric = data.get("photometric_loss")
        fixer = data.get("fixer_loss")
        minscale = data.get("minscale_loss")
        return cls(
            photometric_loss=PhotometricLossConfig(photometric["weight"])
            if isinstance(photometric, dict) and "weight" in photometric
            else None,
            fixer_loss=FixerLossConfig(
                weight=fixer["weight"],
                low_freq_weight=fixer.get("low_freq_weight", 1.0),
                lpips_weight=fixer.get("lpips_weight", 0.1),
                danger_percentile=fixer.get("danger_percentile"),
                blur_sigma=fixer.get("blur_sigma"),
                gamma=fixer.get("gamma"),
            )
            if isinstance(fixer, dict) and "weight" in fixer
            else None,
            minscale_loss=MinScaleLossConfig(
                weight=minscale["weight"],
                sigma_min=minscale.get("sigma_min"),
            )
            if isinstance(minscale, dict) and "weight" in minscale
            else None,
            steps=data.get("steps"),
            cam_count=data.get("cam_count"),
            jitter_views_per_cam=data.get("jitter_views_per_cam"),
            jitter_cm=data.get("jitter_cm"),
            sigma_min=data.get("sigma_min"),
            lambda_sigma=data.get("lambda_sigma"),
            danger_percentile=data.get("danger_percentile"),
            blur_sigma=data.get("blur_sigma"),
            gamma=data.get("gamma"),
        )


@dataclass
class OptimizeGaussianConfig:
    gaussian_ply_path: Optional[str] = None
    output_dir: str = "output/gaussians"
    output_ply_name: str = "optimized.ply"
    debug_dir_name: str = "debug"
    device: str = "cuda"
    lr: float = 5e-3
    fixer_ratio: float = 0.33
    danger_percentile: float = 0.25
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
    sky_mask: SkyMaskConfig = SkyMaskConfig()
    use_lpips: bool = True
    lpips_net: str = "vgg"
    background_color: List[float] = None
    random_seed: int = 0
    phase_settings: Optional[Dict[str, PhaseConfig]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizeGaussianConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        raw_phase_settings = data.get("phase_settings", {})
        raw_sky_mask = data.get("sky_mask", {})
        phase_settings = None
        if isinstance(raw_phase_settings, dict) and raw_phase_settings:
            phase_settings = {
                key: PhaseConfig.from_dict(value)
                for key, value in raw_phase_settings.items()
                if isinstance(value, dict)
            }
        sky_mask = (
            SkyMaskConfig.from_dict(raw_sky_mask) if raw_sky_mask else cls.sky_mask
        )
        return cls(
            gaussian_ply_path=data.get("gaussian_ply_path"),
            output_dir=data.get("output_dir", cls.output_dir),
            output_ply_name=data.get("output_ply_name", cls.output_ply_name),
            debug_dir_name=data.get("debug_dir_name", cls.debug_dir_name),
            device=data.get("device", cls.device),
            lr=float(data.get("lr", cls.lr)),
            fixer_ratio=float(data.get("fixer_ratio", cls.fixer_ratio)),
            danger_percentile=float(
                data.get("danger_percentile", cls.danger_percentile)
            ),
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
            sky_mask=sky_mask,
            use_lpips=bool(data.get("use_lpips", cls.use_lpips)),
            lpips_net=data.get("lpips_net", cls.lpips_net),
            background_color=data.get("background_color", cls.background_color),
            random_seed=int(data.get("random_seed", cls.random_seed)),
            phase_settings=phase_settings,
        )
