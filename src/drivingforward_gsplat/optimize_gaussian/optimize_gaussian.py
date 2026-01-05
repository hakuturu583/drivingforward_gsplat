from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from drivingforward_gsplat.predict_gaussian import CAM_ORDER
from drivingforward_gsplat.optimize_gaussian.train import (
    OptimizeGaussianConfig,
    erode_mask,
    optimize_gaussians,
)


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        array = np.array(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


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
    means = gaussians["means"].cpu().numpy().astype(np.float32)
    rotations = gaussians["rotations"].cpu().numpy().astype(np.float32)
    scales = gaussians["scales"].cpu().numpy().astype(np.float32)
    opacities = gaussians["opacities"].cpu().numpy().astype(np.float32)
    shs = gaussians["shs"].cpu().numpy().astype(np.float32)
    shs = shs.reshape(shs.shape[0], -1)

    num = means.shape[0]
    sh_dim = shs.shape[1]
    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("opacity", "f4"),
    ] + [(f"sh_{idx}", "f4") for idx in range(sh_dim)]

    data = np.zeros(num, dtype=dtype_fields)
    data["x"] = means[:, 0]
    data["y"] = means[:, 1]
    data["z"] = means[:, 2]
    data["rot_0"] = rotations[:, 0]
    data["rot_1"] = rotations[:, 1]
    data["rot_2"] = rotations[:, 2]
    data["rot_3"] = rotations[:, 3]
    data["scale_0"] = scales[:, 0]
    data["scale_1"] = scales[:, 1]
    data["scale_2"] = scales[:, 2]
    data["opacity"] = opacities[:, 0]
    for idx in range(sh_dim):
        data[f"sh_{idx}"] = shs[:, idx]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def _prepare_view_entries(
    sample: Dict,
    output_token: str,
    cfg: OptimizeGaussianConfig,
    image_root: str,
) -> tuple[List[Dict], List[Dict]]:
    raw_images = sample[("color", 0, 0)]
    masks = sample.get("mask")
    k_all = sample[("K", 0)]
    extrinsics = sample["extrinsics"]

    if isinstance(raw_images, np.ndarray):
        raw_images = torch.from_numpy(raw_images)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    if isinstance(k_all, np.ndarray):
        k_all = torch.from_numpy(k_all)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics)

    extrinsics_inv = torch.inverse(extrinsics)

    raw_views: List[Dict] = []
    fixer_views: List[Dict] = []

    bg_color = cfg.background_color or [1.0, 1.0, 1.0]
    output_dir = Path(image_root) / output_token
    for cam_idx, cam_name in enumerate(CAM_ORDER):
        raw = raw_images[cam_idx].float()
        mask = masks[cam_idx].float() if masks is not None else torch.ones_like(raw[:1])
        mask = mask[:1]
        mask = erode_mask(mask, cfg.sky_erode_kernel, cfg.sky_erode_iter)
        k = k_all[cam_idx]
        if k.shape[-1] == 4:
            k = k[:3, :3]
        viewmat = extrinsics_inv[cam_idx]
        world_view = viewmat.transpose(0, 1)

        raw_views.append(
            {
                "type": "raw",
                "cam_idx": cam_idx,
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

        input_path = output_dir / f"{cam_name}_render_raw.png"
        fixer_path = output_dir / f"{cam_name}_render.png"
        if input_path.exists() and fixer_path.exists():
            input_render = _load_image(input_path)
            fixer_rgb = _load_image(fixer_path)
            if input_render.shape[-2:] != raw.shape[-2:]:
                input_render = torch.nn.functional.interpolate(
                    input_render.unsqueeze(0),
                    size=raw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if fixer_rgb.shape[-2:] != raw.shape[-2:]:
                fixer_rgb = torch.nn.functional.interpolate(
                    fixer_rgb.unsqueeze(0),
                    size=raw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            fixer_views.append(
                {
                    "type": "fixer",
                    "cam_idx": cam_idx,
                    "fixer_rgb": fixer_rgb.float(),
                    "input_render": input_render.float(),
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
