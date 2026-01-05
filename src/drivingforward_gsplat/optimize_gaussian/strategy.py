from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class MergeConfig:
    every: int = 200
    voxel_size: float = 0.05
    voxel_size_distance_scale: float = 0.0
    small_scale: float = 0.02
    color_bin: float = 0.1
    prune_thin_opacity: Optional[float] = None


def _weighted_merge(
    values: torch.Tensor, inverse: torch.Tensor, weights: torch.Tensor, num: int
) -> torch.Tensor:
    flattened = values.reshape(values.shape[0], -1)
    out = torch.zeros(
        (num, flattened.shape[1]), device=values.device, dtype=values.dtype
    )
    out.index_add_(0, inverse, flattened * weights[:, None])
    denom = torch.zeros(num, device=values.device, dtype=values.dtype)
    denom.index_add_(0, inverse, weights)
    denom = denom.clamp_min(1e-6).unsqueeze(1)
    out = out / denom
    return out.view(num, *values.shape[1:])


def merge_gaussians(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
    bg_mask: torch.Tensor,
    cfg: MergeConfig,
    camera_center: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if cfg.voxel_size <= 0:
        return means, rotations, scales, opacities, shs, bg_mask

    fg_mask = ~bg_mask
    small = scales.min(dim=1).values < cfg.small_scale
    merge_mask = fg_mask & small
    prune_mask = torch.zeros_like(merge_mask)
    if cfg.prune_thin_opacity is not None and cfg.prune_thin_opacity > 0:
        prune_mask = fg_mask & (opacities.squeeze(-1) < cfg.prune_thin_opacity)
        merge_mask = merge_mask & ~prune_mask
    if not torch.any(merge_mask) and not torch.any(prune_mask):
        return means, rotations, scales, opacities, shs, bg_mask

    merge_idx = torch.where(merge_mask)[0]
    keep_idx = torch.where(~merge_mask & ~prune_mask)[0]

    if merge_idx.numel() > 0:
        merge_means = means[merge_idx]
        if (
            camera_center is not None
            and cfg.voxel_size_distance_scale is not None
            and cfg.voxel_size_distance_scale > 0
        ):
            distances = torch.linalg.norm(
                merge_means - camera_center.to(device=means.device, dtype=means.dtype),
                dim=1,
            )
            voxel_sizes = cfg.voxel_size * (
                1.0 + distances * cfg.voxel_size_distance_scale
            )
            voxel_sizes = voxel_sizes.clamp_min(1e-6).unsqueeze(1)
            voxels = torch.floor(merge_means / voxel_sizes).to(torch.int32)
        else:
            voxels = torch.floor(merge_means / cfg.voxel_size).to(torch.int32)
        if cfg.color_bin > 0:
            base_color = shs[merge_idx][:, 0, :].clamp(0.0, 1.0)
            color_bins = torch.floor(base_color / cfg.color_bin).to(torch.int32)
            voxels = torch.cat([voxels, color_bins], dim=1)
        unique, inverse = torch.unique(voxels, dim=0, return_inverse=True)
        num = unique.shape[0]

        weights = opacities[merge_idx].squeeze(-1).clamp_min(1e-6)
        merged_means = _weighted_merge(merge_means, inverse, weights, num)
        merged_scales = _weighted_merge(scales[merge_idx], inverse, weights, num)
        merged_opacities = _weighted_merge(opacities[merge_idx], inverse, weights, num)
        merged_shs = _weighted_merge(shs[merge_idx], inverse, weights, num)

        merged_rots = _weighted_merge(rotations[merge_idx], inverse, weights, num)
        merged_rots = torch.nn.functional.normalize(merged_rots, dim=1, eps=1e-6)

        new_means = torch.cat([means[keep_idx], merged_means], dim=0)
        new_rot = torch.cat([rotations[keep_idx], merged_rots], dim=0)
        new_scales = torch.cat([scales[keep_idx], merged_scales], dim=0)
        new_opacity = torch.cat([opacities[keep_idx], merged_opacities], dim=0)
        new_shs = torch.cat([shs[keep_idx], merged_shs], dim=0)

        merged_bg = torch.zeros(num, dtype=torch.bool, device=bg_mask.device)
        new_bg_mask = torch.cat([bg_mask[keep_idx], merged_bg], dim=0)
        return new_means, new_rot, new_scales, new_opacity, new_shs, new_bg_mask

    return (
        means[keep_idx],
        rotations[keep_idx],
        scales[keep_idx],
        opacities[keep_idx],
        shs[keep_idx],
        bg_mask[keep_idx],
    )
