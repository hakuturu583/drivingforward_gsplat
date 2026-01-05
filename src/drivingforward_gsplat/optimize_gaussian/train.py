from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import os
import torch
import torch.nn.functional as F
import yaml
from gsplat.optimizers import SelectiveAdam
from tqdm import tqdm

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
from drivingforward_gsplat.utils.gaussian_ply import save_gaussians_tensors_as_inria_ply
from drivingforward_gsplat.utils.misc import to_pil_rgb


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
    phase1_jitter_cm: float = 1.0
    phase2_jitter_cm: float = 3.0
    phase3_jitter_cm: float = 5.0
    jitter_views_per_cam: int = 4
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
            phase1_jitter_cm=float(data.get("phase1_jitter_cm", cls.phase1_jitter_cm)),
            phase2_jitter_cm=float(data.get("phase2_jitter_cm", cls.phase2_jitter_cm)),
            phase3_jitter_cm=float(data.get("phase3_jitter_cm", cls.phase3_jitter_cm)),
            jitter_views_per_cam=int(
                data.get("jitter_views_per_cam", cls.jitter_views_per_cam)
            ),
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


def _make_jittered_views(
    views: List[Dict],
    jitter_cm: float,
    per_view: int,
    rng: random.Random,
) -> List[Dict]:
    if jitter_cm <= 0 or per_view <= 0:
        return []
    jitter_m = jitter_cm / 100.0
    jittered: List[Dict] = []
    for view in views:
        viewmat = view["viewmat"]
        for _ in range(per_view):
            dx = rng.uniform(-jitter_m, jitter_m)
            dz = rng.uniform(-jitter_m, jitter_m)
            offset = torch.tensor(
                [dx, 0.0, dz],
                device=viewmat.device,
                dtype=viewmat.dtype,
            )
            new_viewmat = viewmat.clone()
            new_viewmat[:3, 3] = new_viewmat[:3, 3] + offset
            new_world_view = new_viewmat.transpose(0, 1)
            jittered_view = dict(view)
            jittered_view["viewmat"] = new_viewmat
            jittered_view["world_view_transform"] = new_world_view
            jittered.append(jittered_view)
    return jittered


def _save_debug_snapshot(
    cfg: OptimizeGaussianConfig,
    phase_name: str,
    phase_step: int,
    view: Dict,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
) -> None:
    debug_root = os.path.join(cfg.output_dir, cfg.debug_dir_name)
    phase_dir = os.path.join(debug_root, f"phase_{phase_name}_step_{phase_step:04d}")
    os.makedirs(phase_dir, exist_ok=True)

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
    image_path = os.path.join(phase_dir, "novel_view.png")
    to_pil_rgb(rendered).save(image_path)

    inria_path = os.path.join(phase_dir, "gaussians_inria.ply")
    save_gaussians_tensors_as_inria_ply(
        means.detach().cpu(),
        rotations.detach().cpu(),
        scales.detach().cpu(),
        opacities.detach().cpu(),
        shs.detach().cpu(),
        inria_path,
    )


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
        ("raw", cfg.raw_steps, phase1_cams, cfg.phase1_jitter_cm),
        ("mix", cfg.phase2_steps, phase2_cams, cfg.phase2_jitter_cm),
        ("mix", cfg.phase3_steps, all_cams, cfg.phase3_jitter_cm),
    ]

    global_step = 0
    rng = random.Random(cfg.random_seed)
    for phase_name, steps, cam_indices, jitter_cm in phases:
        if steps <= 0:
            continue
        print(
            f"[optimize] phase={phase_name} steps={steps} cams={cam_indices} "
            f"jitter_cm={jitter_cm} per_cam={cfg.jitter_views_per_cam}"
        )
        raw_subset = _select_views(raw_views, cam_indices, kind="raw")
        fixer_subset = _select_views(fixer_views, cam_indices, kind="fixer")
        raw_subset = raw_subset + _make_jittered_views(
            raw_subset, jitter_cm, cfg.jitter_views_per_cam, rng
        )
        progress = tqdm(range(steps), desc=f"optimize/{phase_name}", leave=False)
        last_view = None
        for _ in progress:
            global_step += 1
            if phase_name == "raw" or not fixer_subset:
                view = random.choice(raw_subset)
            else:
                if random.random() < cfg.fixer_ratio and fixer_subset:
                    view = random.choice(fixer_subset)
                else:
                    view = random.choice(raw_subset)
            last_view = view

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
            loss_raw_val = torch.tensor(0.0, device=device)
            loss_fix_val = torch.tensor(0.0, device=device)
            loss_sigma_val = torch.tensor(0.0, device=device)
            if view["type"] == "raw":
                diff = rendered - view["rgb"]
                loss_raw = charbonnier(diff)
                loss_raw = torch.mean(loss_raw, dim=0, keepdim=True)
                loss_raw_val = masked_mean(loss_raw, mask)
                loss = loss_raw_val
            else:
                loss_fix_val = cfg.lambda_fix * fixer_loss(
                    rendered,
                    view["fixer_rgb"],
                    view["input_render"],
                    mask,
                )
                loss = loss_fix_val

            if cfg.lambda_sigma > 0:
                sigma_loss = F.relu(cfg.sigma_min - scales).pow(2.0)
                if torch.any(fg_mask):
                    sigma_loss = sigma_loss[fg_mask].mean()
                    loss_sigma_val = cfg.lambda_sigma * sigma_loss
                    loss = loss + loss_sigma_val

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.background_freeze_steps is None:
                visibility = fg_mask
            elif (
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

            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                message = (
                    f"[optimize] step={global_step} phase={phase_name} "
                    f"type={view['type']} loss={loss.item():.6f} "
                    f"raw={loss_raw_val.item():.6f} fix={loss_fix_val.item():.6f} "
                    f"sigma={loss_sigma_val.item():.6f} "
                    f"gaussians={means.shape[0]}"
                )
                print(message)
                progress.set_postfix_str(f"loss={loss.item():.4f} g={means.shape[0]}")

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
                print(
                    f"[optimize] removed background at step={global_step} "
                    f"gaussians={means.shape[0]}"
                )

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
                print(
                    f"[optimize] merged at step={global_step} gaussians={means.shape[0]}"
                )

        if last_view is not None:
            _save_debug_snapshot(
                cfg,
                phase_name,
                global_step,
                last_view,
                means,
                rotations,
                scales,
                opacities,
                shs,
            )
            print(
                f"[optimize] saved debug snapshot for phase={phase_name} "
                f"step={global_step}"
            )

    return {
        "means": means.data.detach().cpu(),
        "rotations": rotations.data.detach().cpu(),
        "scales": scales.data.detach().cpu(),
        "opacities": opacities.data.detach().cpu(),
        "shs": shs.data.detach().cpu(),
    }
