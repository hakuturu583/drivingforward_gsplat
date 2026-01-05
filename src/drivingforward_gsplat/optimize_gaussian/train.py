from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from gsplat.optimizers import SelectiveAdam

from drivingforward_gsplat.models.gaussian import rendering as gs_render
from drivingforward_gsplat.optimize_gaussian.losses import (
    FixerLoss,
    FixerLossConfig,
    charbonnier,
    masked_mean,
)
from drivingforward_gsplat.optimize_gaussian.strategy import (
    MergeConfig,
    merge_gaussians,
)


@dataclass
class OptimizeGaussianConfig:
    gaussian_ply_path: Optional[str] = None
    output_dir: str = "output/gaussians"
    output_ply_name: str = "optimized.ply"
    device: str = "cuda"
    lr: float = 5e-3
    raw_steps: int = 1000
    phase2_steps: int = 1000
    phase3_steps: int = 1000
    phase1_cam_count: int = 2
    phase2_cam_count: int = 4
    fixer_ratio: float = 0.33
    danger_percentile: float = 0.25
    lambda_fix: float = 0.02
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
    background_freeze_steps: int = 500
    background_remove_step: Optional[int] = None
    sky_erode_kernel: int = 3
    sky_erode_iter: int = 1
    use_lpips: bool = True
    lpips_net: str = "vgg"
    background_color: List[float] = None
    random_seed: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "OptimizeGaussianConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            gaussian_ply_path=data.get("gaussian_ply_path"),
            output_dir=data.get("output_dir", cls.output_dir),
            output_ply_name=data.get("output_ply_name", cls.output_ply_name),
            device=data.get("device", cls.device),
            lr=float(data.get("lr", cls.lr)),
            raw_steps=int(data.get("raw_steps", cls.raw_steps)),
            phase2_steps=int(data.get("phase2_steps", cls.phase2_steps)),
            phase3_steps=int(data.get("phase3_steps", cls.phase3_steps)),
            phase1_cam_count=int(data.get("phase1_cam_count", cls.phase1_cam_count)),
            phase2_cam_count=int(data.get("phase2_cam_count", cls.phase2_cam_count)),
            fixer_ratio=float(data.get("fixer_ratio", cls.fixer_ratio)),
            danger_percentile=float(
                data.get("danger_percentile", cls.danger_percentile)
            ),
            lambda_fix=float(data.get("lambda_fix", cls.lambda_fix)),
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
            background_freeze_steps=int(
                data.get("background_freeze_steps", cls.background_freeze_steps)
            ),
            background_remove_step=data.get(
                "background_remove_step", cls.background_remove_step
            ),
            sky_erode_kernel=int(data.get("sky_erode_kernel", cls.sky_erode_kernel)),
            sky_erode_iter=int(data.get("sky_erode_iter", cls.sky_erode_iter)),
            use_lpips=bool(data.get("use_lpips", cls.use_lpips)),
            lpips_net=data.get("lpips_net", cls.lpips_net),
            background_color=data.get("background_color", cls.background_color),
            random_seed=int(data.get("random_seed", cls.random_seed)),
        )


def erode_mask(mask: torch.Tensor, kernel: int, iters: int) -> torch.Tensor:
    if kernel <= 1 or iters <= 0:
        return mask
    mask = mask.float()
    pad = kernel // 2
    inv = 1.0 - mask
    for _ in range(iters):
        inv = F.max_pool2d(inv, kernel, stride=1, padding=pad)
    return (1.0 - inv).clamp(0.0, 1.0)


def compute_background_mask(
    means: torch.Tensor,
    view: Dict,
) -> torch.Tensor:
    k = view["K"]
    viewmat = view["viewmat"]
    mask = view["mask"]
    h, w = mask.shape[-2:]

    points = torch.cat(
        [
            means,
            torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype),
        ],
        dim=1,
    )
    cam = (viewmat @ points.T).T
    z = cam[:, 2].clamp_min(1e-6)
    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    u = (cam[:, 0] / z) * fx + cx
    v = (cam[:, 1] / z) * fy + cy
    u = u.round().long()
    v = v.round().long()
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    bg = torch.zeros(means.shape[0], device=means.device, dtype=torch.bool)
    if torch.any(in_bounds):
        mask_vals = mask[0, v[in_bounds], u[in_bounds]]
        bg[in_bounds] = mask_vals < 0.5
    return bg


def _prepare_optimizers(
    params: Dict[str, torch.nn.Parameter], lr: float
) -> Dict[str, torch.optim.Optimizer]:
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for name, param in params.items():
        opt = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        for group in opt.param_groups:
            group["lr"] = lr
        optimizers[name] = opt
    return optimizers


def _normalize_rotations(rotations: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(rotations, dim=1, eps=1e-6)


def _clamp_scales(
    scales: torch.Tensor, sigma_min: float, fg_mask: torch.Tensor
) -> None:
    scales.clamp_min_(1e-6)
    if sigma_min > 0 and torch.any(fg_mask):
        scales[fg_mask] = scales[fg_mask].clamp_min(sigma_min)


def _clamp_opacities(opacities: torch.Tensor) -> None:
    opacities.clamp_(1e-4, 1.0 - 1e-4)


def _select_views(
    views: List[Dict],
    cam_indices: List[int],
    kind: Optional[str] = None,
) -> List[Dict]:
    filtered = [v for v in views if v["cam_idx"] in cam_indices]
    if kind is None:
        return filtered
    return [v for v in filtered if v["type"] == kind]


def optimize_gaussians(
    cfg: OptimizeGaussianConfig,
    gaussians: Dict[str, torch.Tensor],
    raw_views: List[Dict],
    fixer_views: List[Dict],
) -> Dict[str, torch.Tensor]:
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    device = torch.device(cfg.device)

    def _move_view(view: Dict) -> Dict:
        moved = dict(view)
        for key in (
            "rgb",
            "mask",
            "K",
            "viewmat",
            "world_view_transform",
            "fixer_rgb",
            "input_render",
        ):
            if key in moved and torch.is_tensor(moved[key]):
                moved[key] = moved[key].to(device=device, dtype=torch.float32)
        return moved

    raw_views = [_move_view(view) for view in raw_views]
    fixer_views = [_move_view(view) for view in fixer_views]

    means = torch.nn.Parameter(gaussians["means"].to(device))
    rotations = torch.nn.Parameter(gaussians["rotations"].to(device))
    scales = torch.nn.Parameter(gaussians["scales"].to(device))
    opacities = torch.nn.Parameter(gaussians["opacities"].to(device))
    shs = torch.nn.Parameter(gaussians["shs"].to(device))
    params = {
        "means": means,
        "quats": rotations,
        "scales": scales,
        "opacities": opacities,
        "shs": shs,
    }

    bg_mask = compute_background_mask(means, raw_views[0])
    fg_mask = ~bg_mask

    _clamp_scales(scales.data, cfg.sigma_min, fg_mask)
    merge_cfg = MergeConfig(
        every=cfg.merge_every,
        voxel_size=cfg.merge_voxel_size,
        small_scale=cfg.merge_small_scale,
        thin_opacity=cfg.merge_thin_opacity,
    )
    (
        means.data,
        rotations.data,
        scales.data,
        opacities.data,
        shs.data,
        bg_mask,
    ) = merge_gaussians(
        means.data,
        rotations.data,
        scales.data,
        opacities.data,
        shs.data,
        bg_mask,
        merge_cfg,
    )
    fg_mask = ~bg_mask
    rotations.data = _normalize_rotations(rotations.data)
    _clamp_opacities(opacities.data)

    optimizers = _prepare_optimizers(params, cfg.lr)
    fixer_loss = FixerLoss(
        FixerLossConfig(
            danger_percentile=cfg.danger_percentile,
            blur_sigma=cfg.blur_sigma,
            gamma=cfg.gamma,
            lambda_fix_low=cfg.lambda_fix_low,
            lambda_fix_lpips=cfg.lambda_fix_lpips,
            use_lpips=cfg.use_lpips,
            lpips_net=cfg.lpips_net,
        )
    )

    all_cams = sorted({v["cam_idx"] for v in raw_views})
    phase1_cams = all_cams[: max(1, min(cfg.phase1_cam_count, len(all_cams)))]
    phase2_cams = all_cams[: max(1, min(cfg.phase2_cam_count, len(all_cams)))]

    phases = [
        ("raw", cfg.raw_steps, phase1_cams),
        ("mix", cfg.phase2_steps, phase2_cams),
        ("mix", cfg.phase3_steps, all_cams),
    ]

    global_step = 0
    for phase_name, steps, cam_indices in phases:
        if steps <= 0:
            continue
        raw_subset = _select_views(raw_views, cam_indices, kind="raw")
        fixer_subset = _select_views(fixer_views, cam_indices, kind="fixer")
        for _ in range(steps):
            global_step += 1
            if phase_name == "raw" or not fixer_subset:
                view = random.choice(raw_subset)
            else:
                if random.random() < cfg.fixer_ratio and fixer_subset:
                    view = random.choice(fixer_subset)
                else:
                    view = random.choice(raw_subset)

            rendered = gs_render.render(
                novel_FovX=0.0,
                novel_FovY=0.0,
                novel_height=view["height"],
                novel_width=view["width"],
                novel_world_view_transform=view["world_view_transform"],
                novel_full_proj_transform=view["world_view_transform"],
                novel_camera_center=None,
                novel_K=view["K"],
                pts_xyz=means,
                pts_rgb=None,
                rotations=rotations,
                scales=scales,
                opacity=opacities,
                shs=shs,
                bg_color=view["bg_color"],
                with_postprocess=False,
            )

            mask = view["mask"]
            if view["type"] == "raw":
                diff = rendered - view["rgb"]
                loss_raw = charbonnier(diff)
                loss_raw = torch.mean(loss_raw, dim=0, keepdim=True)
                loss = masked_mean(loss_raw, mask)
            else:
                loss = cfg.lambda_fix * fixer_loss(
                    rendered,
                    view["fixer_rgb"],
                    view["input_render"],
                    mask,
                )

            if cfg.lambda_sigma > 0:
                sigma_loss = F.relu(cfg.sigma_min - scales).pow(2.0)
                if torch.any(fg_mask):
                    sigma_loss = sigma_loss[fg_mask].mean()
                    loss = loss + cfg.lambda_sigma * sigma_loss

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            if (
                cfg.background_freeze_steps > 0
                and global_step <= cfg.background_freeze_steps
            ):
                visibility = fg_mask
            else:
                visibility = torch.ones_like(fg_mask, dtype=torch.bool)
            for opt in optimizers.values():
                opt.step(visibility=visibility)

            rotations.data = _normalize_rotations(rotations.data)
            _clamp_scales(scales.data, cfg.sigma_min, fg_mask)
            _clamp_opacities(opacities.data)

            if cfg.background_remove_step and global_step == cfg.background_remove_step:
                keep = ~bg_mask
                means = torch.nn.Parameter(means.data[keep])
                rotations = torch.nn.Parameter(rotations.data[keep])
                scales = torch.nn.Parameter(scales.data[keep])
                opacities = torch.nn.Parameter(opacities.data[keep])
                shs = torch.nn.Parameter(shs.data[keep])
                bg_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
                fg_mask = ~bg_mask
                params = {
                    "means": means,
                    "quats": rotations,
                    "scales": scales,
                    "opacities": opacities,
                    "shs": shs,
                }
                optimizers = _prepare_optimizers(params, cfg.lr)

            if cfg.merge_every > 0 and (global_step % cfg.merge_every == 0):
                (
                    new_means,
                    new_rots,
                    new_scales,
                    new_opacities,
                    new_shs,
                    bg_mask,
                ) = merge_gaussians(
                    means.data,
                    rotations.data,
                    scales.data,
                    opacities.data,
                    shs.data,
                    bg_mask,
                    merge_cfg,
                )
                means = torch.nn.Parameter(new_means)
                rotations = torch.nn.Parameter(new_rots)
                scales = torch.nn.Parameter(new_scales)
                opacities = torch.nn.Parameter(new_opacities)
                shs = torch.nn.Parameter(new_shs)
                fg_mask = ~bg_mask
                rotations.data = _normalize_rotations(rotations.data)
                params = {
                    "means": means,
                    "quats": rotations,
                    "scales": scales,
                    "opacities": opacities,
                    "shs": shs,
                }
                optimizers = _prepare_optimizers(params, cfg.lr)

    return {
        "means": means.data.detach().cpu(),
        "rotations": rotations.data.detach().cpu(),
        "scales": scales.data.detach().cpu(),
        "opacities": opacities.data.detach().cpu(),
        "shs": shs.data.detach().cpu(),
    }
