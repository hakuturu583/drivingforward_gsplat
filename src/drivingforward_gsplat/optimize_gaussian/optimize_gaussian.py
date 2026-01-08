from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from gsplat import exporter
from PIL import Image
from plyfile import PlyData

from drivingforward_gsplat.predict_gaussian import CAM_ORDER
from drivingforward_gsplat.optimize_gaussian.dataclass import OptimizeGaussianConfig
from drivingforward_gsplat.optimize_gaussian.train import erode_mask, optimize_gaussians
from drivingforward_gsplat.utils.gaussian_ply import save_gaussians_tensors_as_inria_ply


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.array(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def _sam3_mask_from_render(
    image: torch.Tensor, cfg: OptimizeGaussianConfig
) -> torch.Tensor:
    from drivingforward_gsplat.mask.sam3_mask import Sam3MaskConfig, cutout_with_sam3

    sam3_cfg = Sam3MaskConfig(
        model_id=cfg.sky_mask.sam3_model_id,
        device=cfg.sky_mask.sam3_device,
        dtype=cfg.sky_mask.sam3_dtype,
        mask_threshold=cfg.sky_mask.sam3_mask_threshold,
        resize_longest_side=cfg.sky_mask.sam3_resize_longest_side,
    )
    mask = cutout_with_sam3(image, cfg.sky_mask.sam3_prompt, sam3_cfg)
    if cfg.sky_mask.sam3_invert:
        mask = ~mask
    return mask


def _load_gaussians_from_ply(path: str) -> Dict[str, torch.Tensor]:
    ply = PlyData.read(path)
    data = ply["vertex"].data
    means = np.stack([data["x"], data["y"], data["z"]], axis=1)
    rotations = np.stack(
        [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1
    )
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
    opacities = np.stack([data["opacity"]], axis=1)
    sh_fields = [name for name in data.dtype.names if name.startswith("sh_")]
    sh_fields = sorted(sh_fields, key=lambda x: int(x.split("_")[1]))
    if sh_fields:
        sh_values = np.stack([data[name] for name in sh_fields], axis=1)
        if sh_values.shape[1] % 3 != 0:
            raise ValueError("SH coefficient count is not divisible by 3.")
        sh_dim = sh_values.shape[1] // 3
        sh_values = sh_values.reshape(sh_values.shape[0], sh_dim, 3)
    else:
        sh_values = np.zeros((means.shape[0], 1, 3), dtype=np.float32)
    return {
        "means": torch.from_numpy(means).float(),
        "rotations": torch.from_numpy(rotations).float(),
        "scales": torch.from_numpy(scales).float(),
        "opacities": torch.from_numpy(opacities).float(),
        "shs": torch.from_numpy(sh_values).float(),
    }


def _save_gaussians_to_ply(path: str, gaussians: Dict[str, torch.Tensor]) -> None:
    means = gaussians["means"].detach().cpu()
    rotations = gaussians["rotations"].detach().cpu()
    scales = gaussians["scales"].detach().cpu()
    opacities = gaussians["opacities"].detach().cpu().squeeze(-1)
    shs = gaussians["shs"].detach().cpu()
    sh0 = shs[:, :1, :]
    shN = shs[:, 1:, :]
    if shN.numel() == 0:
        shN = shs.new_zeros((shs.shape[0], 0, 3))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    exporter.export_splats(
        means=means,
        scales=scales,
        quats=rotations,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=path,
    )


def _save_gaussians_to_inria_ply(path: str, gaussians: Dict[str, torch.Tensor]) -> None:
    save_gaussians_tensors_as_inria_ply(
        gaussians["means"],
        gaussians["rotations"],
        gaussians["scales"],
        gaussians["opacities"],
        gaussians["shs"],
        path,
    )


def _prepare_view_entries(
    sample: Dict,
    output_token: str,
    cfg: OptimizeGaussianConfig,
    image_root: str,
) -> tuple[List[Dict], List[Dict]]:
    raw_images = sample[("color", 0, 0)]
    k_all = sample[("K", 0)]
    extrinsics = sample["extrinsics"]

    if isinstance(raw_images, np.ndarray):
        raw_images = torch.from_numpy(raw_images)
    if isinstance(k_all, np.ndarray):
        k_all = torch.from_numpy(k_all)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics)

    def _squeeze_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        if tensor.dim() >= 1 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    raw_images = _squeeze_batch(raw_images)
    k_all = _squeeze_batch(k_all)
    extrinsics = _squeeze_batch(extrinsics)

    extrinsics_inv = torch.inverse(extrinsics)

    raw_views: List[Dict] = []
    fixer_views: List[Dict] = []

    bg_color = cfg.background_color or [1.0, 1.0, 1.0]
    output_dir = Path(image_root) / output_token
    for cam_idx, cam_name in enumerate(CAM_ORDER):
        raw = raw_images[cam_idx].float()
        k = k_all[cam_idx]
        if k.shape[-1] == 4:
            k = k[:3, :3]
        viewmat = extrinsics_inv[cam_idx]
        world_view = viewmat.transpose(0, 1)

        input_path = output_dir / f"{cam_name}_render_raw.png"
        fixer_path = output_dir / f"{cam_name}_render.png"
        input_render = _load_image(input_path) if input_path.exists() else None
        fixer_rgb = _load_image(fixer_path) if fixer_path.exists() else None
        if input_render is None:
            raise FileNotFoundError(
                f"Rendered image not found for SAM3 mask: {input_path}"
            )
        if input_render.shape[-2:] != raw.shape[-2:]:
            input_render = torch.nn.functional.interpolate(
                input_render.unsqueeze(0),
                size=raw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if fixer_rgb is not None and fixer_rgb.shape[-2:] != raw.shape[-2:]:
            fixer_rgb = torch.nn.functional.interpolate(
                fixer_rgb.unsqueeze(0),
                size=raw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        mask_source = input_render
        mask = _sam3_mask_from_render(mask_source, cfg)
        mask = mask.to(dtype=torch.float32).unsqueeze(0)
        mask = erode_mask(mask, cfg.sky_mask.erode_kernel, cfg.sky_mask.erode_iter)

        raw_views.append(
            {
                "type": "raw",
                "cam_idx": cam_idx,
                "cam_name": cam_name,
                "rgb": raw,
                "mask": mask,
                "K": k,
                "viewmat": viewmat,
                "world_view_transform": world_view,
                "height": raw.shape[-2],
                "width": raw.shape[-1],
                "bg_color": bg_color,
            }
        )

        if input_render is not None and fixer_rgb is not None:
            fixer_views.append(
                {
                    "type": "fixer",
                    "cam_idx": cam_idx,
                    "cam_name": cam_name,
                    "fixer_rgb": fixer_rgb.float(),
                    "raw_render_rgb": input_render.float(),
                    "mask": mask,
                    "K": k,
                    "viewmat": viewmat,
                    "world_view_transform": world_view,
                    "height": raw.shape[-2],
                    "width": raw.shape[-1],
                    "bg_color": bg_color,
                }
            )
    return raw_views, fixer_views


def optimize_from_prediction(
    optimize_cfg_path: str,
    sample: Dict,
    output_token: str,
    output_dir: str,
    image_root: str = "output/images",
) -> str:
    cfg = OptimizeGaussianConfig.from_yaml(optimize_cfg_path)
    if cfg.gaussian_ply_path is None:
        cfg.gaussian_ply_path = os.path.join(output_dir, "output.ply")
    cfg.output_dir = output_dir

    gaussians = _load_gaussians_from_ply(cfg.gaussian_ply_path)
    raw_views, fixer_views = _prepare_view_entries(
        sample, output_token, cfg, image_root
    )
    optimized = optimize_gaussians(cfg, gaussians, raw_views, fixer_views)

    output_path = os.path.join(cfg.output_dir, cfg.output_ply_name)
    _save_gaussians_to_ply(output_path, optimized)
    optimized_inria = os.path.join(cfg.output_dir, "optimized_inria.ply")
    _save_gaussians_to_inria_ply(optimized_inria, optimized)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize gsplat gaussians.")
    parser.add_argument(
        "--optimize-config",
        default="configs/optimize_gaussian.yaml",
        help="Optimize gaussian config yaml file path.",
    )
    parser.add_argument(
        "--gaussian-ply",
        required=True,
        help="Path to initial gaussian PLY.",
    )
    parser.add_argument(
        "--image-root",
        default="output/images",
        help="Root directory containing Fixer inputs/outputs.",
    )
    parser.add_argument(
        "--output-token", required=True, help="Prediction output token."
    )
    parser.add_argument("--output-dir", required=True, help="Output gaussian dir.")
    args = parser.parse_args()

    raise RuntimeError(
        "Standalone optimize_gaussian requires prediction sample context. "
        "Run via drivingforward_gsplat.predict_gaussian with --optimize-gaussian-config."
    )


if __name__ == "__main__":
    main()
