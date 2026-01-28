from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from drivingforward_gsplat.clip_guidance.loss import (
    ClipLoss,
    ClipLossConfig,
    EdgeLoss,
    EdgeLossConfig,
    MusiqLoss,
    MusiqLossConfig,
    ShRegConfig,
    ShRegLoss,
)
from drivingforward_gsplat.clip_guidance.loss.edge_loss import _soft_canny
from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
from drivingforward_gsplat.i2i.instruct_pix2pix import instruct_pix2pix_i2i
from drivingforward_gsplat.models.gaussian import rendering as gs_render
from drivingforward_gsplat.predict_gaussian import (
    CAM_ORDER,
    GtPoseDrivingForwardModel,
    _add_batch_dim_to_inputs,
    _generate_context_gaussians_mf,
    _move_inputs_to_device,
)
from drivingforward_gsplat.utils.gaussian_ply import (
    _gather_gaussians,
    save_gaussians_tensors_as_inria_ply,
)
from drivingforward_gsplat.utils.misc import get_config, to_pil_rgb


@dataclass
class PredictConfig:
    model_config: str = "configs/nuscenes/main.yaml"
    split: str = "eval_MF"
    index: int = 0
    torchscript_dir: str = "torchscript"
    novel_view_mode: str = "MF"

    @classmethod
    def from_dict(cls, data: Dict) -> "PredictConfig":
        return cls(
            model_config=data.get("model_config", cls.model_config),
            split=data.get("split", cls.split),
            index=int(data.get("index", cls.index)),
            torchscript_dir=data.get("torchscript_dir", cls.torchscript_dir),
            novel_view_mode=data.get("novel_view_mode", cls.novel_view_mode),
        )


@dataclass
class TrainingConfig:
    device: str = "cuda"
    steps: int = 500
    lr: float = 1e-2
    sh_lr: Optional[float] = None
    opacity_lr: Optional[float] = None
    optimize_opacity: bool = False
    seed: int = 0
    log_every: int = 10
    save_every: int = 100

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingConfig":
        return cls(
            device=data.get("device", cls.device),
            steps=int(data.get("steps", cls.steps)),
            lr=float(data.get("lr", cls.lr)),
            sh_lr=(
                float(data["sh_lr"])
                if isinstance(data.get("sh_lr"), (int, float))
                else None
            ),
            opacity_lr=(
                float(data["opacity_lr"])
                if isinstance(data.get("opacity_lr"), (int, float))
                else None
            ),
            optimize_opacity=bool(data.get("optimize_opacity", cls.optimize_opacity)),
            seed=int(data.get("seed", cls.seed)),
            log_every=int(data.get("log_every", cls.log_every)),
            save_every=int(data.get("save_every", cls.save_every)),
        )


@dataclass
class PoseJitterConfig:
    translation_m: float = 1.0
    yaw_deg: float = 360.0
    pitch_deg: float = 30.0
    batch_size: int = 4
    base_cameras: List[str] = field(default_factory=lambda: ["CAM_FRONT"])
    background_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @classmethod
    def from_dict(cls, data: Dict) -> "PoseJitterConfig":
        defaults = cls()
        return cls(
            translation_m=float(data.get("translation_m", defaults.translation_m)),
            yaw_deg=float(data.get("yaw_deg", defaults.yaw_deg)),
            pitch_deg=float(data.get("pitch_deg", defaults.pitch_deg)),
            batch_size=int(data.get("batch_size", defaults.batch_size)),
            base_cameras=list(data.get("base_cameras", defaults.base_cameras)),
            background_color=list(
                data.get("background_color", defaults.background_color)
            ),
        )


@dataclass
class OutputConfig:
    dir: str = "output/clip_guidance"
    ply_name: str = "optimized_inria.ply"
    save_initial: bool = True
    save_renders_every: int = 100

    @classmethod
    def from_dict(cls, data: Dict) -> "OutputConfig":
        return cls(
            dir=data.get("dir", cls.dir),
            ply_name=data.get("ply_name", cls.ply_name),
            save_initial=bool(data.get("save_initial", cls.save_initial)),
            save_renders_every=int(
                data.get("save_renders_every", cls.save_renders_every)
            ),
        )


@dataclass
class Step1Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    i2i: ClipLossConfig = field(default_factory=ClipLossConfig)
    edge_select: EdgeLossConfig = field(default_factory=EdgeLossConfig)
    background_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @classmethod
    def from_dict(cls, data: Dict) -> "Step1Config":
        return cls(
            training=TrainingConfig.from_dict(data.get("training", {})),
            i2i=ClipLossConfig.from_dict(data.get("i2i", {})),
            edge_select=EdgeLossConfig.from_dict(data.get("edge_select", {})),
            background_color=list(
                data.get("background_color", cls().background_color)
            ),
        )


@dataclass
class Step2Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pose_jitter: PoseJitterConfig = field(default_factory=PoseJitterConfig)
    clip_loss: ClipLossConfig = field(default_factory=ClipLossConfig)
    edge_loss: EdgeLossConfig = field(default_factory=EdgeLossConfig)
    sh_reg: ShRegConfig = field(default_factory=ShRegConfig)
    musiq_loss: MusiqLossConfig = field(default_factory=MusiqLossConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "Step2Config":
        return cls(
            training=TrainingConfig.from_dict(data.get("training", {})),
            pose_jitter=PoseJitterConfig.from_dict(data.get("pose_jitter", {})),
            clip_loss=ClipLossConfig.from_dict(data.get("clip_loss", {})),
            edge_loss=EdgeLossConfig.from_dict(data.get("edge_loss", {})),
            sh_reg=ShRegConfig.from_dict(data.get("sh_reg", {})),
            musiq_loss=MusiqLossConfig.from_dict(data.get("musiq_loss", {})),
        )


@dataclass
class ClipGuidanceConfig:
    predict: PredictConfig = field(default_factory=PredictConfig)
    step1: Step1Config = field(default_factory=Step1Config)
    step2: Step2Config = field(default_factory=Step2Config)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ClipGuidanceConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            predict=PredictConfig.from_dict(data.get("predict", {})),
            step1=Step1Config.from_dict(data.get("step1", {})),
            step2=Step2Config.from_dict(data.get("step2", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
        )


def _resolve_cam_indices(cameras: Sequence[str]) -> List[int]:
    indices: List[int] = []
    for cam in cameras:
        if cam not in CAM_ORDER:
            raise ValueError(f"Unknown camera name: {cam}")
        indices.append(CAM_ORDER.index(cam))
    return indices


def _ensure_tensor(value: torch.Tensor | List | Tuple) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _build_base_views(
    sample: Dict,
    cam_indices: Sequence[int],
    background_color: Sequence[float],
) -> List[Dict]:
    raw_images = sample[("color", 0, 0)]
    k_all = sample[("K", 0)]
    extrinsics = sample["extrinsics"]

    raw_images = _ensure_tensor(raw_images)
    k_all = _ensure_tensor(k_all)
    extrinsics = _ensure_tensor(extrinsics)

    if raw_images.dim() >= 5 and raw_images.shape[0] == 1:
        raw_images = raw_images.squeeze(0)
    if k_all.dim() >= 4 and k_all.shape[0] == 1:
        k_all = k_all.squeeze(0)
    if extrinsics.dim() >= 4 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics.squeeze(0)

    extrinsics_inv = torch.inverse(extrinsics)
    views: List[Dict] = []
    for cam_idx in cam_indices:
        k = k_all[cam_idx]
        if k.shape[-1] == 4:
            k = k[:3, :3]
        viewmat = extrinsics_inv[cam_idx]
        world_view = viewmat.transpose(0, 1)
        height = int(raw_images[cam_idx].shape[-2])
        width = int(raw_images[cam_idx].shape[-1])
        views.append(
            {
                "cam_idx": cam_idx,
                "K": k,
                "viewmat": viewmat,
                "world_view_transform": world_view,
                "height": height,
                "width": width,
                "bg_color": list(background_color),
            }
        )
    return views


def _rotation_matrix_yaw_pitch(
    yaw_rad: float, pitch_rad: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)

    # Camera-local yaw (about +Y) then pitch (about +X).
    rot_yaw = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device,
        dtype=dtype,
    )
    rot_pitch = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        device=device,
        dtype=dtype,
    )
    return rot_yaw @ rot_pitch


def _sample_pose(
    view: Dict,
    cfg: PoseJitterConfig,
    rng: random.Random,
    device: torch.device,
) -> Dict:
    viewmat = view["viewmat"].to(device=device)
    dtype = viewmat.dtype
    yaw = math.radians(rng.uniform(0.0, cfg.yaw_deg))
    pitch = math.radians(rng.uniform(-cfg.pitch_deg, cfg.pitch_deg))
    rot_delta = _rotation_matrix_yaw_pitch(yaw, pitch, device, dtype)

    dx = rng.uniform(-cfg.translation_m, cfg.translation_m)
    dz = rng.uniform(-cfg.translation_m, cfg.translation_m)
    translation = torch.tensor([dx, 0.0, dz], device=device, dtype=dtype)

    new_viewmat = viewmat.clone()
    # Apply rotation in camera-local coordinates; adjust translation consistently.
    new_viewmat[:3, :3] = rot_delta @ viewmat[:3, :3]
    new_viewmat[:3, 3] = rot_delta @ (viewmat[:3, 3] - translation)
    world_view = new_viewmat.transpose(0, 1)

    return {
        "cam_idx": view["cam_idx"],
        "K": view["K"].to(device=device, dtype=dtype),
        "viewmat": new_viewmat,
        "world_view_transform": world_view,
        "height": view["height"],
        "width": view["width"],
        "bg_color": view["bg_color"],
    }


def _render_views(
    views: List[Dict],
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    shs: torch.Tensor,
) -> torch.Tensor:
    rendered: List[torch.Tensor] = []
    for view in views:
        image = gs_render.render(
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
        if image.dim() == 3:
            rendered.append(image)
        else:
            rendered.append(image.squeeze(0))
    return torch.stack(rendered, dim=0)


def _normalize_reference_images(
    reference_images: torch.Tensor | Sequence,
) -> List:
    if torch.is_tensor(reference_images):
        images = reference_images
        if images.dim() == 5 and images.shape[0] == 1:
            images = images.squeeze(0)
        if images.dim() == 4:
            return [images[idx] for idx in range(images.shape[0])]
        if images.dim() == 3:
            return [images]
        raise ValueError(
            "reference_images tensor must be 3D or 4D (optionally with batch)."
        )
    return list(reference_images)


def _images_to_tensor(
    images: Sequence,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensors = []
    for image in images:
        pil = to_pil_rgb(image)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] != 3:
            arr = arr[:, :, :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
    return batch


def _save_rendered_views(
    output_dir: str,
    views: Sequence[Dict],
    rendered: torch.Tensor,
    step_label: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for view, image in zip(views, rendered):
        cam_name = CAM_ORDER[view["cam_idx"]]
        render_path = os.path.join(output_dir, f"render_{step_label}_{cam_name}.png")
        to_pil_rgb(image.detach()).save(render_path)


def _generate_i2i_images(
    reference_images: torch.Tensor | Sequence,
    camera_names: Sequence[str],
    prompt: str,
    device: torch.device,
    seed: Optional[int | list[int]],
    output_dir: Optional[str],
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
) -> List[List[Image.Image]]:
    images = _normalize_reference_images(reference_images)
    if len(camera_names) < len(images):
        raise ValueError("camera_names length must be >= number of reference images.")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if isinstance(seed, list) and not seed:
        seed = None
    i2i_images: List[List[Image.Image]] = []
    for cam_idx, image in enumerate(images):
        pil_image = to_pil_rgb(image)
        edited = instruct_pix2pix_i2i(
            image=pil_image,
            prompt=prompt,
            negative_prompt=None,
            device=str(device),
            torch_dtype=torch_dtype,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        )
        edited_images = edited if isinstance(edited, list) else [edited]
        if output_dir:
            cam_name = camera_names[cam_idx]
            cam_dir = os.path.join(output_dir, "i2i", cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            if seed is None:
                seeds = [None] * len(edited_images)
            else:
                seeds = seed if isinstance(seed, list) else [seed] * len(edited_images)
            for idx, (img, seed_value) in enumerate(zip(edited_images, seeds)):
                if seed_value is None:
                    name = f"seed_none_{idx:02d}.png"
                else:
                    name = f"seed_{seed_value}.png"
                img.save(os.path.join(cam_dir, name))
        i2i_images.append(edited_images)
    return i2i_images


def _ensure_shs_layout(shs: torch.Tensor) -> torch.Tensor:
    if shs.dim() != 3:
        return shs
    if shs.shape[-1] == 3:
        return shs
    if shs.shape[1] == 3:
        return shs.transpose(1, 2).contiguous()
    return shs


def _predict_initial_gaussians(
    cfg: ClipGuidanceConfig,
    device: torch.device,
) -> Tuple[
    Dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    base_cfg = get_config(
        cfg.predict.model_config,
        mode="eval",
        novel_view_mode=cfg.predict.novel_view_mode,
    )
    augmentation = {
        "image_shape": (
            int(base_cfg["training"]["height"]),
            int(base_cfg["training"]["width"]),
        ),
        "jittering": (0.0, 0.0, 0.0, 0.0),
        "crop_train_borders": (),
        "crop_eval_borders": (),
    }
    dataset = EnvNuScenesDataset(
        cfg.predict.split,
        cameras=CAM_ORDER,
        back_context=base_cfg["data"]["back_context"],
        forward_context=base_cfg["data"]["forward_context"],
        data_transform=get_transforms("train", **augmentation),
        depth_type=None,
        with_pose=True,
        with_ego_pose=True,
        with_mask=True,
    )
    sample = dataset[cfg.predict.index]
    inputs = _add_batch_dim_to_inputs(sample)
    inputs = _move_inputs_to_device(inputs, device)
    model = GtPoseDrivingForwardModel(base_cfg, cfg.predict.torchscript_dir, device)
    model.set_eval()
    with torch.no_grad():
        outputs = model.estimate(inputs)
        if getattr(model, "gaussian", False):
            model.gs_net = model.models["gs_net"]
            for cam in range(model.num_cams):
                if model.novel_view_mode == "MF":
                    _generate_context_gaussians_mf(model, inputs, outputs, cam)
                else:
                    model.get_gaussian_data(inputs, outputs, cam)
    gathered = _gather_gaussians(outputs, model.num_cams, model.novel_view_mode, 0)
    if gathered is None:
        raise RuntimeError("Failed to gather initial gaussians from prediction.")
    means, rotations, scales, opacities, shs = gathered
    shs = _ensure_shs_layout(shs)
    return sample, (means, rotations, scales, opacities, shs)


class ClipGuidanceTrainer:
    def __init__(self, cfg: ClipGuidanceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.step2.training.device)

    def train(self) -> None:
        step1_cfg = self.cfg.step1
        step2_cfg = self.cfg.step2

        torch.manual_seed(step1_cfg.training.seed)
        sample, gathered = _predict_initial_gaussians(self.cfg, self.device)
        means, rotations, scales, opacities, shs_init = gathered
        means = means.to(self.device)
        rotations = rotations.to(self.device)
        scales = scales.to(self.device)
        opacities = opacities.to(self.device)
        shs_init = shs_init.to(self.device)

        shs = torch.nn.Parameter(shs_init.clone())

        step1_dir = os.path.join(self.cfg.output.dir, "step1")
        step2_dir = os.path.join(self.cfg.output.dir, "step2")
        os.makedirs(self.cfg.output.dir, exist_ok=True)
        os.makedirs(step1_dir, exist_ok=True)
        os.makedirs(step2_dir, exist_ok=True)

        prompt = ", ".join(step1_cfg.i2i.prompts)
        i2i_images_by_cam = _generate_i2i_images(
            reference_images=sample[("color", 0, 0)],
            camera_names=CAM_ORDER,
            prompt=prompt,
            device=self.device,
            seed=step1_cfg.i2i.seed,
            output_dir=self.cfg.output.dir,
            num_inference_steps=step1_cfg.i2i.i2i_num_inference_steps,
            guidance_scale=step1_cfg.i2i.i2i_guidance_scale,
            image_guidance_scale=step1_cfg.i2i.i2i_image_guidance_scale,
        )

        step1_cam_indices = _resolve_cam_indices(CAM_ORDER)
        step1_views = _build_base_views(
            sample, step1_cam_indices, step1_cfg.background_color
        )

        edge_loss_fn_step1 = EdgeLoss(
            sigma=step1_cfg.edge_select.sigma,
            low_threshold=step1_cfg.edge_select.low_threshold,
            high_threshold=step1_cfg.edge_select.high_threshold,
            soft_k=step1_cfg.edge_select.soft_k,
            render_low_threshold=step1_cfg.edge_select.render_low_threshold,
            render_high_threshold=step1_cfg.edge_select.render_high_threshold,
            ref_low_threshold=step1_cfg.edge_select.ref_low_threshold,
            ref_high_threshold=step1_cfg.edge_select.ref_high_threshold,
        )
        (
            render_low,
            render_high,
            ref_low,
            ref_high,
        ) = edge_loss_fn_step1._resolve_thresholds()
        torch_dtype = means.dtype
        raw_images = _normalize_reference_images(sample[("color", 0, 0)])
        selected_i2i_images = []
        for cam_idx in step1_cam_indices:
            reference = _images_to_tensor(
                [raw_images[cam_idx]], self.device, torch_dtype
            )
            ref_edges = _soft_canny(
                reference,
                step1_cfg.edge_select.sigma,
                ref_low,
                ref_high,
                step1_cfg.edge_select.soft_k,
            )
            best_score = None
            best_image = None
            for i2i_image in i2i_images_by_cam[cam_idx]:
                candidate = _images_to_tensor(
                    [i2i_image], self.device, torch_dtype
                )
                cand_edges = _soft_canny(
                    candidate,
                    step1_cfg.edge_select.sigma,
                    ref_low,
                    ref_high,
                    step1_cfg.edge_select.soft_k,
                )
                score = F.l1_loss(cand_edges, ref_edges).item()
                if best_score is None or score < best_score:
                    best_score = score
                    best_image = i2i_image
            if best_image is None:
                raise RuntimeError("Failed to select i2i image for step1.")
            selected_i2i_images.append(best_image)
        target_rgb_tensor = _images_to_tensor(
            selected_i2i_images, self.device, torch_dtype
        )

        sh_lr_step1 = (
            step1_cfg.training.sh_lr
            if step1_cfg.training.sh_lr is not None
            else step1_cfg.training.lr
        )
        optimizer_step1 = torch.optim.Adam([{"params": [shs], "lr": sh_lr_step1}])

        if self.cfg.output.save_initial:
            init_path = os.path.join(step1_dir, "initial_inria.ply")
            save_gaussians_tensors_as_inria_ply(
                means.detach(),
                rotations.detach(),
                scales.detach(),
                opacities.detach(),
                shs_init.detach(),
                init_path,
            )
            initial_views = _build_base_views(
                sample,
                _resolve_cam_indices(CAM_ORDER),
                step1_cfg.background_color,
            )
            initial_renders = _render_views(
                initial_views, means, rotations, scales, opacities, shs_init
            )
            _save_rendered_views(step1_dir, initial_views, initial_renders, "initial")

        for step in range(1, step1_cfg.training.steps + 1):
            optimizer_step1.zero_grad(set_to_none=True)
            rendered = _render_views(
                step1_views, means, rotations, scales, opacities, shs
            )
            rgb_loss = F.l1_loss(rendered, target_rgb_tensor)
            rgb_loss.backward()
            optimizer_step1.step()

            if step % step1_cfg.training.log_every == 0 or step == 1:
                print(
                    f"[clip-guidance step1] step={step:05d} "
                    f"rgb_l1={rgb_loss.item():.4f}"
                )

            if (
                step1_cfg.training.save_every > 0
                and step % step1_cfg.training.save_every == 0
            ):
                out_path = os.path.join(step1_dir, self.cfg.output.ply_name)
                save_gaussians_tensors_as_inria_ply(
                    means.detach(),
                    rotations.detach(),
                    scales.detach(),
                    opacities.detach(),
                    shs.detach(),
                    out_path,
                )

            if (
                self.cfg.output.save_renders_every > 0
                and step % self.cfg.output.save_renders_every == 0
            ):
                _save_rendered_views(step1_dir, step1_views, rendered, f"{step:05d}")

        final_path_step1 = os.path.join(step1_dir, self.cfg.output.ply_name)
        save_gaussians_tensors_as_inria_ply(
            means.detach(),
            rotations.detach(),
            scales.detach(),
            opacities.detach(),
            shs.detach(),
            final_path_step1,
        )
        final_renders_step1 = _render_views(
            step1_views, means, rotations, scales, opacities, shs
        )
        _save_rendered_views(step1_dir, step1_views, final_renders_step1, "final")

        shs_step1 = shs.detach().clone()

        torch.manual_seed(step2_cfg.training.seed)
        rng = random.Random(step2_cfg.training.seed)

        params = []
        sh_lr = (
            step2_cfg.training.sh_lr
            if step2_cfg.training.sh_lr is not None
            else step2_cfg.training.lr
        )
        params.append({"params": [shs], "lr": sh_lr})

        opacity_param = None
        if step2_cfg.training.optimize_opacity:
            opacity_param = torch.nn.Parameter(opacities.clone())
            opacities = opacity_param
            opacity_lr = (
                step2_cfg.training.opacity_lr
                if step2_cfg.training.opacity_lr is not None
                else step2_cfg.training.lr
            )
            params.append({"params": [opacity_param], "lr": opacity_lr})

        optimizer = torch.optim.Adam(params)

        cam_indices = _resolve_cam_indices(step2_cfg.pose_jitter.base_cameras)
        base_views = _build_base_views(
            sample, cam_indices, step2_cfg.pose_jitter.background_color
        )

        clip_loss_fn = ClipLoss(
            step2_cfg.clip_loss.model_id,
            step2_cfg.clip_loss.prompts,
            step2_cfg.clip_loss.image_size,
            self.device,
            sample[("color", 0, 0)],
            CAM_ORDER,
            step2_cfg.clip_loss.seed,
            self.cfg.output.dir,
            step2_cfg.clip_loss.i2i_num_inference_steps,
            step2_cfg.clip_loss.i2i_guidance_scale,
            step2_cfg.clip_loss.i2i_image_guidance_scale,
            i2i_images=i2i_images_by_cam,
            save_i2i=False,
        )
        edge_loss_fn = EdgeLoss(
            sigma=step2_cfg.edge_loss.sigma,
            low_threshold=step2_cfg.edge_loss.low_threshold,
            high_threshold=step2_cfg.edge_loss.high_threshold,
            soft_k=step2_cfg.edge_loss.soft_k,
            render_low_threshold=step2_cfg.edge_loss.render_low_threshold,
            render_high_threshold=step2_cfg.edge_loss.render_high_threshold,
            ref_low_threshold=step2_cfg.edge_loss.ref_low_threshold,
            ref_high_threshold=step2_cfg.edge_loss.ref_high_threshold,
        )
        musiq_loss_fn = None
        if step2_cfg.musiq_loss.weight > 0:
            musiq_loss_fn = MusiqLoss(
                step2_cfg.musiq_loss.model_id,
                step2_cfg.musiq_loss.image_size,
                self.device,
            )
        sh_reg_fn = ShRegLoss(
            step2_cfg.sh_reg.degree_weights,
            step2_cfg.sh_reg.norm,
            self.device,
        )

        if self.cfg.output.save_initial:
            init_path = os.path.join(step2_dir, "initial_inria.ply")
            save_gaussians_tensors_as_inria_ply(
                means.detach(),
                rotations.detach(),
                scales.detach(),
                opacities.detach(),
                shs.detach(),
                init_path,
            )
            initial_views = _build_base_views(
                sample,
                _resolve_cam_indices(CAM_ORDER),
                step2_cfg.pose_jitter.background_color,
            )
            initial_renders = _render_views(
                initial_views, means, rotations, scales, opacities, shs
            )
            _save_rendered_views(step2_dir, initial_views, initial_renders, "initial")

        for step in range(1, step2_cfg.training.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            views = [
                _sample_pose(
                    base_views[rng.randrange(len(base_views))],
                    step2_cfg.pose_jitter,
                    rng,
                    self.device,
                )
                for _ in range(step2_cfg.pose_jitter.batch_size)
            ]

            rendered = _render_views(views, means, rotations, scales, opacities, shs)
            with torch.no_grad():
                rendered_init = _render_views(
                    views, means, rotations, scales, opacities, shs_init
                )

            cam_indices = [view["cam_idx"] for view in views]
            clip_loss = clip_loss_fn.compute(rendered, cam_indices)

            musiq_loss = torch.tensor(0.0, device=self.device)
            if musiq_loss_fn is not None:
                musiq_loss = musiq_loss_fn.compute(rendered)

            edge_loss = torch.tensor(0.0, device=self.device)
            if step2_cfg.edge_loss.weight > 0:
                edge_loss = edge_loss_fn.compute(rendered, rendered_init)

            reg_loss = sh_reg_fn.compute(shs, shs_step1)

            total = (
                step2_cfg.clip_loss.weight * clip_loss
                + step2_cfg.musiq_loss.weight * musiq_loss
                + step2_cfg.edge_loss.weight * edge_loss
                + step2_cfg.sh_reg.weight * reg_loss
            )
            total.backward()
            optimizer.step()
            if opacity_param is not None:
                with torch.no_grad():
                    opacities.clamp_(1e-4, 1.0 - 1e-4)

            if step % step2_cfg.training.log_every == 0 or step == 1:
                clip_w = step2_cfg.clip_loss.weight * clip_loss
                musiq_w = step2_cfg.musiq_loss.weight * musiq_loss
                edge_w = step2_cfg.edge_loss.weight * edge_loss
                sh_w = step2_cfg.sh_reg.weight * reg_loss
                print(
                    f"[clip-guidance step2] step={step:05d} "
                    f"total={total.item():.4f} "
                    f"clip={clip_loss.item():.4f}*{step2_cfg.clip_loss.weight:.4f}"
                    f"={clip_w.item():.4f} "
                    f"musiq_loss={musiq_loss.item():.4f}*"
                    f"{step2_cfg.musiq_loss.weight:.4f}={musiq_w.item():.4f} "
                    f"edge={edge_loss.item():.4f}*{step2_cfg.edge_loss.weight:.4f}"
                    f"={edge_w.item():.4f} "
                    f"sh_reg={reg_loss.item():.4f}*{step2_cfg.sh_reg.weight:.4f}"
                    f"={sh_w.item():.4f}"
                )

            if (
                step2_cfg.training.save_every > 0
                and step % step2_cfg.training.save_every == 0
            ):
                out_path = os.path.join(step2_dir, self.cfg.output.ply_name)
                save_gaussians_tensors_as_inria_ply(
                    means.detach(),
                    rotations.detach(),
                    scales.detach(),
                    opacities.detach(),
                    shs.detach(),
                    out_path,
                )

            if (
                self.cfg.output.save_renders_every > 0
                and step % self.cfg.output.save_renders_every == 0
            ):
                _save_rendered_views(step2_dir, views, rendered, f"{step:05d}")

        final_path = os.path.join(step2_dir, self.cfg.output.ply_name)
        save_gaussians_tensors_as_inria_ply(
            means.detach(),
            rotations.detach(),
            scales.detach(),
            opacities.detach(),
            shs.detach(),
            final_path,
        )

        final_views = _build_base_views(
            sample,
            _resolve_cam_indices(CAM_ORDER),
            step2_cfg.pose_jitter.background_color,
        )
        final_renders = _render_views(
            final_views, means, rotations, scales, opacities, shs
        )
        _save_rendered_views(step2_dir, final_views, final_renders, "final")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-guided SH optimization trainer.")
    parser.add_argument(
        "--config",
        default="configs/clip_guidance.yaml",
        help="Clip guidance YAML config path.",
    )
    args = parser.parse_args()
    cfg = ClipGuidanceConfig.from_yaml(args.config)
    trainer = ClipGuidanceTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
