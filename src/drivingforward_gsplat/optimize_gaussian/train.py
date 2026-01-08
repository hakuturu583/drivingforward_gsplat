from __future__ import annotations

import random
from typing import Dict, List, Optional

import os
import torch
import torch.nn.functional as F
from gsplat.optimizers import SelectiveAdam
from tqdm import tqdm

from drivingforward_gsplat.models.gaussian import rendering as gs_render
from drivingforward_gsplat.optimize_gaussian.dataclass import (
    OptimizeGaussianConfig,
    PhaseConfig,
)
from drivingforward_gsplat.optimize_gaussian.losses import (
    FixerLoss,
    FixerLossParams,
    charbonnier,
    masked_mean,
)
from drivingforward_gsplat.optimize_gaussian.strategy import (
    MergeConfig,
    merge_gaussians,
)
from drivingforward_gsplat.utils.gaussian_ply import save_gaussians_tensors_as_inria_ply
from drivingforward_gsplat.utils.misc import to_pil_rgb


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


def _camera_center_from_view(view: Dict) -> torch.Tensor:
    world_view = view["world_view_transform"]
    return world_view.inverse()[3, :3]


def _select_views(
    views: List[Dict],
    cam_indices: List[int],
    kind: Optional[str] = None,
) -> List[Dict]:
    filtered = [v for v in views if v["cam_idx"] in cam_indices]
    if kind is None:
        return filtered
    return [v for v in filtered if v["type"] == kind]


def _apply_optional(value: Optional[float], fallback: float) -> float:
    return fallback if value is None else float(value)


def _resolve_phase_loss(
    cfg: OptimizeGaussianConfig,
    phase_id: str,
) -> Dict[str, float]:
    base = {
        "photometric_loss_weight": 1.0,
        "fixer_loss_weight": 0.02,
        "fixer_low_freq_weight": 1.0,
        "fixer_lpips_weight": 0.1,
        "fixer_use_lpips": True,
        "fixer_lpips_net": "vgg",
        "lambda_sigma_min": 0.1,
        "lambda_sigma_max": 0.1,
        "min_scale": 0.03,
        "max_scale": None,
        "opacity_sparsity_weight": 0.0,
        "scale_ratio_weight": 0.0,
        "scale_ratio_max": 2.0,
        "scale_ratio_gamma": 2.0,
        "danger_percentile": 0.25,
        "blur_sigma": 1.5,
        "gamma": 5.0,
        "jitter_cm": 0.0,
    }
    if cfg.phase_settings and phase_id in cfg.phase_settings:
        phase_cfg: PhaseConfig = cfg.phase_settings[phase_id]
        if phase_cfg.photometric_loss is not None:
            base["photometric_loss_weight"] = float(phase_cfg.photometric_loss.weight)
        if phase_cfg.fixer_loss is not None:
            base["fixer_loss_weight"] = float(phase_cfg.fixer_loss.weight)
            base["fixer_low_freq_weight"] = float(phase_cfg.fixer_loss.low_freq_weight)
            base["fixer_lpips_weight"] = float(phase_cfg.fixer_loss.lpips_weight)
            base["fixer_use_lpips"] = (
                phase_cfg.fixer_loss.use_lpips
                if phase_cfg.fixer_loss.use_lpips is not None
                else base["fixer_use_lpips"]
            )
            base["fixer_lpips_net"] = (
                phase_cfg.fixer_loss.lpips_net
                if phase_cfg.fixer_loss.lpips_net is not None
                else base["fixer_lpips_net"]
            )
            base["danger_percentile"] = _apply_optional(
                phase_cfg.fixer_loss.danger_percentile, base["danger_percentile"]
            )
            base["blur_sigma"] = _apply_optional(
                phase_cfg.fixer_loss.blur_sigma, base["blur_sigma"]
            )
            base["gamma"] = _apply_optional(phase_cfg.fixer_loss.gamma, base["gamma"])
        if phase_cfg.minscale_loss is not None:
            base["lambda_sigma_min"] = float(phase_cfg.minscale_loss.weight)
            base["lambda_sigma_max"] = float(phase_cfg.minscale_loss.weight)
            base["min_scale"] = _apply_optional(
                phase_cfg.minscale_loss.min_scale, base["min_scale"]
            )
            base["max_scale"] = _apply_optional(
                phase_cfg.minscale_loss.max_scale, base["max_scale"]
            )
            base["lambda_sigma_min"] = _apply_optional(
                phase_cfg.minscale_loss.min_weight, base["lambda_sigma_min"]
            )
            base["lambda_sigma_max"] = _apply_optional(
                phase_cfg.minscale_loss.max_weight, base["lambda_sigma_max"]
            )
        if phase_cfg.opacity_sparsity_loss is not None:
            base["opacity_sparsity_weight"] = float(
                phase_cfg.opacity_sparsity_loss.weight
            )
        if phase_cfg.scale_ratio_loss is not None:
            base["scale_ratio_weight"] = float(phase_cfg.scale_ratio_loss.weight)
            base["scale_ratio_max"] = _apply_optional(
                phase_cfg.scale_ratio_loss.max_ratio, base["scale_ratio_max"]
            )
            base["scale_ratio_gamma"] = _apply_optional(
                phase_cfg.scale_ratio_loss.gamma, base["scale_ratio_gamma"]
            )
        base["jitter_cm"] = _apply_optional(phase_cfg.jitter_cm, base["jitter_cm"])
    return base


def _resolve_phase_runtime(
    cfg: OptimizeGaussianConfig,
    phase_id: str,
    default_steps: int,
    default_cam_count: int,
    default_jitter_views_per_cam: int,
) -> Dict[str, int]:
    steps = default_steps
    cam_count = default_cam_count
    jitter_views_per_cam = default_jitter_views_per_cam
    fixer_view_use_ratio = 0.33
    if cfg.phase_settings and phase_id in cfg.phase_settings:
        phase_cfg = cfg.phase_settings[phase_id]
        if phase_cfg.steps is not None:
            steps = int(phase_cfg.steps)
        if phase_cfg.cam_count is not None:
            cam_count = int(phase_cfg.cam_count)
        if phase_cfg.jitter_views_per_cam is not None:
            jitter_views_per_cam = int(phase_cfg.jitter_views_per_cam)
        if phase_cfg.fixer_view_use_ratio is not None:
            fixer_view_use_ratio = float(phase_cfg.fixer_view_use_ratio)
    return {
        "steps": steps,
        "cam_count": cam_count,
        "jitter_views_per_cam": jitter_views_per_cam,
        "fixer_view_use_ratio": fixer_view_use_ratio,
    }


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
            jittered_view["is_jittered"] = True
            if "raw_render_rgb" in view:
                jittered_view["raw_render_rgb"] = view["raw_render_rgb"]
            if "fixer_rgb" in view:
                jittered_view["fixer_rgb"] = view["fixer_rgb"]
            jittered.append(jittered_view)
    return jittered


def _save_debug_image(
    cfg: OptimizeGaussianConfig,
    base_name: str,
    view: Dict,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
) -> None:
    debug_root = os.path.join(cfg.output_dir, cfg.debug_dir_name)
    os.makedirs(debug_root, exist_ok=True)

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
    image_path = os.path.join(debug_root, f"{base_name}.png")
    to_pil_rgb(rendered).save(image_path)


def _save_debug_ply(
    cfg: OptimizeGaussianConfig,
    base_name: str,
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
) -> None:
    debug_root = os.path.join(cfg.output_dir, cfg.debug_dir_name)
    os.makedirs(debug_root, exist_ok=True)
    inria_path = os.path.join(debug_root, f"{base_name}.ply")
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
            "raw_render_rgb",
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

    initial_phase_loss = _resolve_phase_loss(cfg, "phase1")
    initial_sigma_min = float(initial_phase_loss["min_scale"])
    _clamp_scales(scales.data, initial_sigma_min, fg_mask)
    default_jitter_views_per_cam = 4
    merge_cfg = MergeConfig(
        every=cfg.merge.every,
        voxel_size=cfg.merge.voxel_size,
        voxel_size_distance_scale=cfg.merge.voxel_size_distance_scale,
        small_scale=cfg.merge.small_scale,
        prune_scale=cfg.merge.prune_scale,
        prune_thin_opacity=cfg.merge.prune_thin_opacity,
        color_bin=cfg.merge.color_bin,
    )

    optimizers = _prepare_optimizers(params, cfg.lr)
    fixer_loss = FixerLoss(
        FixerLossParams(
            danger_percentile=float(initial_phase_loss["danger_percentile"]),
            blur_sigma=float(initial_phase_loss["blur_sigma"]),
            gamma=float(initial_phase_loss["gamma"]),
            low_freq_weight=float(initial_phase_loss["fixer_low_freq_weight"]),
            lpips_weight=float(initial_phase_loss["fixer_lpips_weight"]),
            use_lpips=bool(initial_phase_loss["fixer_use_lpips"]),
            lpips_net=str(initial_phase_loss["fixer_lpips_net"]),
        )
    )

    if raw_views:
        views_by_cam: Dict[int, Dict] = {}
        for view in raw_views:
            cam_idx = int(view["cam_idx"])
            if cam_idx not in views_by_cam:
                views_by_cam[cam_idx] = view
        for cam_idx, view in views_by_cam.items():
            _save_debug_image(
                cfg,
                f"step_0000_cam{cam_idx}",
                view,
                means,
                rotations,
                scales,
                opacities,
                shs,
            )
        _save_debug_ply(cfg, "step_0000", means, rotations, scales, opacities, shs)
        print("[optimize] saved debug snapshot for phase=init step=0")

    all_cams = sorted({v["cam_idx"] for v in raw_views})
    phase1_runtime = _resolve_phase_runtime(
        cfg,
        "phase1",
        default_steps=1000,
        default_cam_count=2,
        default_jitter_views_per_cam=default_jitter_views_per_cam,
    )
    phase2_runtime = _resolve_phase_runtime(
        cfg,
        "phase2",
        default_steps=1000,
        default_cam_count=4,
        default_jitter_views_per_cam=default_jitter_views_per_cam,
    )
    phase3_runtime = _resolve_phase_runtime(
        cfg,
        "phase3",
        default_steps=1000,
        default_cam_count=len(all_cams),
        default_jitter_views_per_cam=default_jitter_views_per_cam,
    )

    phase1_cams = all_cams[: max(1, min(phase1_runtime["cam_count"], len(all_cams)))]
    phase2_cams = all_cams[: max(1, min(phase2_runtime["cam_count"], len(all_cams)))]
    phase3_cams = all_cams[: max(1, min(phase3_runtime["cam_count"], len(all_cams)))]

    phases = [
        ("phase1", "raw", phase1_runtime, phase1_cams),
        ("phase2", "mix", phase2_runtime, phase2_cams),
        ("phase3", "mix", phase3_runtime, phase3_cams),
    ]

    global_step = 0
    rng = random.Random(cfg.random_seed)
    for phase_id, phase_mode, runtime, cam_indices in phases:
        steps = int(runtime["steps"])
        fixer_view_use_ratio = float(runtime["fixer_view_use_ratio"])
        if steps <= 0:
            continue
        phase_loss_cfg = _resolve_phase_loss(cfg, phase_id)
        phase_merge_enabled = phase_id == "phase1"
        if cfg.phase_settings and phase_id in cfg.phase_settings:
            phase_cfg = cfg.phase_settings[phase_id]
            if phase_cfg.merge_enabled is not None:
                phase_merge_enabled = bool(phase_cfg.merge_enabled)
        phase_sigma_min = float(phase_loss_cfg["min_scale"])
        phase_sigma_max = phase_loss_cfg.get("max_scale")
        lambda_raw = float(phase_loss_cfg["photometric_loss_weight"])
        lambda_fix = float(phase_loss_cfg["fixer_loss_weight"])
        lambda_sigma_min = float(phase_loss_cfg["lambda_sigma_min"])
        lambda_sigma_max = float(phase_loss_cfg["lambda_sigma_max"])
        opacity_sparsity_weight = float(phase_loss_cfg["opacity_sparsity_weight"])
        scale_ratio_weight = float(phase_loss_cfg["scale_ratio_weight"])
        scale_ratio_max = float(phase_loss_cfg["scale_ratio_max"])
        scale_ratio_gamma = float(phase_loss_cfg["scale_ratio_gamma"])
        jitter_cm = float(phase_loss_cfg.get("jitter_cm", 0.0))
        jitter_views_per_cam = int(runtime["jitter_views_per_cam"])
        fixer_loss.set_config(
            FixerLossParams(
                danger_percentile=float(phase_loss_cfg["danger_percentile"]),
                blur_sigma=float(phase_loss_cfg["blur_sigma"]),
                gamma=float(phase_loss_cfg["gamma"]),
                low_freq_weight=float(phase_loss_cfg["fixer_low_freq_weight"]),
                lpips_weight=float(phase_loss_cfg["fixer_lpips_weight"]),
                use_lpips=bool(phase_loss_cfg["fixer_use_lpips"]),
                lpips_net=str(phase_loss_cfg["fixer_lpips_net"]),
            )
        )
        print(
            f"[optimize] phase={phase_id} steps={steps} cams={cam_indices} "
            f"jitter_cm={jitter_cm} per_cam={jitter_views_per_cam}"
        )
        raw_subset = _select_views(raw_views, cam_indices, kind="raw")
        fixer_subset = _select_views(fixer_views, cam_indices, kind="fixer")
        raw_subset = raw_subset + _make_jittered_views(
            raw_subset, jitter_cm, jitter_views_per_cam, rng
        )
        progress = tqdm(range(steps), desc=f"optimize/{phase_id}", leave=False)
        last_view = None
        for _ in progress:
            global_step += 1
            if phase_mode == "raw" or not fixer_subset:
                view = random.choice(raw_subset)
            else:
                if random.random() < fixer_view_use_ratio and fixer_subset:
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
                if view.get("is_jittered"):
                    raw_render = rendered
                    fixer_rgb = gs_render.postprocess_by_fixer(raw_render)
                    loss_fix_val = lambda_fix * fixer_loss(
                        rendered,
                        fixer_rgb,
                        raw_render,
                        mask,
                    )
                    loss = loss_fix_val
                else:
                    diff = rendered - view["rgb"]
                    loss_raw = charbonnier(diff)
                    loss_raw = torch.mean(loss_raw, dim=0, keepdim=True)
                    loss_raw_val = masked_mean(loss_raw, mask) * lambda_raw
                    loss = loss_raw_val
            else:
                raw_render = view.get("raw_render_rgb")
                fixer_rgb = view.get("fixer_rgb")
                if fixer_rgb is not None and raw_render is not None:
                    loss_fix_val = lambda_fix * fixer_loss(
                        rendered,
                        fixer_rgb,
                        raw_render,
                        mask,
                    )
                loss = loss_fix_val

            if lambda_sigma_min > 0 or lambda_sigma_max > 0:
                min_loss = None
                max_loss = None
                if lambda_sigma_min > 0:
                    min_loss = F.relu(phase_sigma_min - scales).pow(2.0)
                if lambda_sigma_max > 0 and phase_sigma_max is not None:
                    max_loss = F.relu(scales - phase_sigma_max).pow(2.0)
                if torch.any(fg_mask):
                    loss_sigma_val = torch.tensor(0.0, device=device)
                    if min_loss is not None:
                        loss_sigma_val = (
                            loss_sigma_val + lambda_sigma_min * min_loss[fg_mask].mean()
                        )
                    if max_loss is not None:
                        loss_sigma_val = (
                            loss_sigma_val + lambda_sigma_max * max_loss[fg_mask].mean()
                        )
                    loss = loss + loss_sigma_val

            if opacity_sparsity_weight > 0:
                sparsity_loss = opacities.mean()
                loss = loss + opacity_sparsity_weight * sparsity_loss

            if scale_ratio_weight > 0:
                max_scale = scales.max(dim=1).values
                min_scale = scales.min(dim=1).values.clamp_min(1e-6)
                ratio = max_scale / min_scale
                excess = F.relu(ratio - scale_ratio_max)
                ratio_loss = torch.exp(scale_ratio_gamma * excess).clamp_max(1e6) - 1.0
                if torch.any(fg_mask):
                    ratio_loss = ratio_loss[fg_mask].mean()
                    loss = loss + scale_ratio_weight * ratio_loss

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
            _clamp_scales(scales.data, phase_sigma_min, fg_mask)
            _clamp_opacities(opacities.data)

            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                message = (
                    f"[optimize] step={global_step} phase={phase_id} "
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

            if (
                phase_merge_enabled
                and cfg.merge.every > 0
                and (global_step % cfg.merge.every == 0)
            ):
                camera_center = None
                if (
                    merge_cfg.voxel_size_distance_scale is not None
                    and merge_cfg.voxel_size_distance_scale > 0
                ):
                    camera_center = _camera_center_from_view(view)
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
                    camera_center=camera_center,
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

        if raw_subset:
            views_by_cam: Dict[int, Dict] = {}
            for view in raw_subset:
                cam_idx = int(view["cam_idx"])
                if cam_idx not in views_by_cam:
                    views_by_cam[cam_idx] = view
            for cam_idx, view in views_by_cam.items():
                _save_debug_image(
                    cfg,
                    f"step_{global_step:04d}_cam{cam_idx}",
                    view,
                    means,
                    rotations,
                    scales,
                    opacities,
                    shs,
                )
        if last_view is not None:
            _save_debug_ply(
                cfg,
                f"step_{global_step:04d}",
                means,
                rotations,
                scales,
                opacities,
                shs,
            )
        if raw_subset or last_view is not None:
            print(
                f"[optimize] saved debug snapshot for phase={phase_id} "
                f"step={global_step}"
            )

    return {
        "means": means.data.detach().cpu(),
        "rotations": rotations.data.detach().cpu(),
        "scales": scales.data.detach().cpu(),
        "opacities": opacities.data.detach().cpu(),
        "shs": shs.data.detach().cpu(),
    }
