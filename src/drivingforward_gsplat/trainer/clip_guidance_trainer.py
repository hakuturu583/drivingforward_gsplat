from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
import pyiqa
from transformers import CLIPModel, CLIPTokenizer

from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
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

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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
    seed: int = 0
    log_every: int = 10
    save_every: int = 100

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingConfig":
        return cls(
            device=data.get("device", cls.device),
            steps=int(data.get("steps", cls.steps)),
            lr=float(data.get("lr", cls.lr)),
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
class ClipLossConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    prompts: List[str] = field(default_factory=list)
    weight: float = 1.0
    image_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "ClipLossConfig":
        defaults = cls()
        prompts = data.get("prompts", defaults.prompts)
        if isinstance(prompts, str):
            prompts = [prompts]
        return cls(
            model_id=data.get("model_id", defaults.model_id),
            prompts=list(prompts),
            weight=float(data.get("weight", defaults.weight)),
            image_size=data.get("image_size", defaults.image_size),
        )


@dataclass
class EdgeLossConfig:
    weight: float = 0.2
    sigma: float = 1.0
    low_threshold: float = 0.1
    high_threshold: float = 0.2
    soft_k: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict) -> "EdgeLossConfig":
        return cls(
            weight=float(data.get("weight", cls.weight)),
            sigma=float(data.get("sigma", cls.sigma)),
            low_threshold=float(data.get("low_threshold", cls.low_threshold)),
            high_threshold=float(data.get("high_threshold", cls.high_threshold)),
            soft_k=float(data.get("soft_k", cls.soft_k)),
        )


@dataclass
class ShRegConfig:
    weight: float = 0.01
    degree_weights: List[float] = field(default_factory=list)
    norm: str = "l2"

    @classmethod
    def from_dict(cls, data: Dict) -> "ShRegConfig":
        defaults = cls()
        return cls(
            weight=float(data.get("weight", defaults.weight)),
            degree_weights=list(data.get("degree_weights", defaults.degree_weights)),
            norm=str(data.get("norm", defaults.norm)),
        )


@dataclass
class MusiqLossConfig:
    model_id: str = "musiq"
    weight: float = 0.0
    image_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "MusiqLossConfig":
        defaults = cls()
        return cls(
            model_id=data.get("model_id", defaults.model_id),
            weight=float(data.get("weight", defaults.weight)),
            image_size=data.get("image_size", defaults.image_size),
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
class ClipGuidanceConfig:
    predict: PredictConfig = field(default_factory=PredictConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pose_jitter: PoseJitterConfig = field(default_factory=PoseJitterConfig)
    clip_loss: ClipLossConfig = field(default_factory=ClipLossConfig)
    edge_loss: EdgeLossConfig = field(default_factory=EdgeLossConfig)
    sh_reg: ShRegConfig = field(default_factory=ShRegConfig)
    musiq_loss: MusiqLossConfig = field(default_factory=MusiqLossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ClipGuidanceConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            predict=PredictConfig.from_dict(data.get("predict", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            pose_jitter=PoseJitterConfig.from_dict(data.get("pose_jitter", {})),
            clip_loss=ClipLossConfig.from_dict(data.get("clip_loss", {})),
            edge_loss=EdgeLossConfig.from_dict(data.get("edge_loss", {})),
            sh_reg=ShRegConfig.from_dict(data.get("sh_reg", {})),
            musiq_loss=MusiqLossConfig.from_dict(data.get("musiq_loss", {})),
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


def _gaussian_kernel(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = radius * 2 + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def _soft_canny(
    images: torch.Tensor,
    sigma: float,
    low: float,
    high: float,
    soft_k: float,
) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(0)
    gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    device = gray.device
    dtype = gray.dtype
    if sigma > 0:
        kernel = _gaussian_kernel(sigma, device, dtype)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        padding = kernel.shape[-1] // 2
        gray = F.conv2d(gray, kernel, padding=padding)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    low_mask = torch.sigmoid((mag - low) * soft_k)
    high_mask = torch.sigmoid((mag - high) * soft_k)
    high_dilate = F.max_pool2d(high_mask, kernel_size=3, stride=1, padding=1)
    edges = high_mask + (1.0 - high_mask) * low_mask * high_dilate
    return edges


def _preprocess_clip_images(
    images: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(0)
    images = images.clamp(0.0, 1.0)
    images = F.interpolate(
        images, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(_CLIP_MEAN, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(_CLIP_STD, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    return (images - mean) / std


def _preprocess_musiq_images(
    images: torch.Tensor,
    image_size: Optional[int],
) -> torch.Tensor:
    if images.dim() == 3:
        images = images.unsqueeze(0)
    images = images.clamp(0.0, 1.0)
    if image_size is not None:
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return images


def _make_sh_degree_weights(d_sh: int, weights: List[float]) -> torch.Tensor:
    degree = int(math.sqrt(d_sh) - 1)
    if (degree + 1) ** 2 != d_sh:
        degree = max(degree, 0)
    if not weights:
        weights = [1.0] * (degree + 1)
    if len(weights) < degree + 1:
        weights = weights + [weights[-1]] * (degree + 1 - len(weights))
    if len(weights) > degree + 1:
        weights = weights[: degree + 1]

    weight_map = torch.ones(d_sh, dtype=torch.float32)
    for l in range(degree + 1):
        start = l * l
        end = (l + 1) * (l + 1)
        weight_map[start:end] = float(weights[l])
    return weight_map


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
        self.device = torch.device(cfg.training.device)

    def train(self) -> None:
        rng = random.Random(self.cfg.training.seed)
        torch.manual_seed(self.cfg.training.seed)

        sample, gathered = _predict_initial_gaussians(self.cfg, self.device)
        means, rotations, scales, opacities, shs_init = gathered
        means = means.to(self.device)
        rotations = rotations.to(self.device)
        scales = scales.to(self.device)
        opacities = opacities.to(self.device)
        shs_init = shs_init.to(self.device)

        shs = torch.nn.Parameter(shs_init.clone())
        optimizer = torch.optim.Adam([shs], lr=self.cfg.training.lr)

        cam_indices = _resolve_cam_indices(self.cfg.pose_jitter.base_cameras)
        base_views = _build_base_views(
            sample, cam_indices, self.cfg.pose_jitter.background_color
        )

        clip_model = CLIPModel.from_pretrained(self.cfg.clip_loss.model_id).to(
            self.device
        )
        clip_model.eval()
        tokenizer = CLIPTokenizer.from_pretrained(self.cfg.clip_loss.model_id)
        if not self.cfg.clip_loss.prompts:
            raise ValueError("clip_loss.prompts must contain at least one prompt.")
        text_inputs = tokenizer(
            self.cfg.clip_loss.prompts,
            padding=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

        image_size = (
            self.cfg.clip_loss.image_size
            if self.cfg.clip_loss.image_size is not None
            else int(clip_model.config.vision_config.image_size)
        )

        musiq_metric = None
        if self.cfg.musiq_loss.weight > 0:
            musiq_metric = pyiqa.create_metric(
                self.cfg.musiq_loss.model_id, device=self.device
            )
            musiq_metric.eval()

        sh_weight_map = _make_sh_degree_weights(
            shs.shape[1], self.cfg.sh_reg.degree_weights
        ).to(self.device, dtype=shs.dtype)

        os.makedirs(self.cfg.output.dir, exist_ok=True)
        if self.cfg.output.save_initial:
            init_path = os.path.join(self.cfg.output.dir, "initial_inria.ply")
            save_gaussians_tensors_as_inria_ply(
                means.detach(),
                rotations.detach(),
                scales.detach(),
                opacities.detach(),
                shs_init.detach(),
                init_path,
            )

        for step in range(1, self.cfg.training.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            views = [
                _sample_pose(
                    base_views[rng.randrange(len(base_views))],
                    self.cfg.pose_jitter,
                    rng,
                    self.device,
                )
                for _ in range(self.cfg.pose_jitter.batch_size)
            ]

            rendered = _render_views(views, means, rotations, scales, opacities, shs)
            with torch.no_grad():
                rendered_init = _render_views(
                    views, means, rotations, scales, opacities, shs_init
                )

            clip_inputs = _preprocess_clip_images(rendered, image_size)
            image_features = clip_model.get_image_features(pixel_values=clip_inputs)
            image_features = F.normalize(image_features, dim=-1)
            sims = image_features @ text_features.T
            clip_loss = 1.0 - sims.mean()

            musiq_loss = torch.tensor(0.0, device=self.device)
            musiq_score = None
            if musiq_metric is not None:
                musiq_inputs = _preprocess_musiq_images(
                    rendered, self.cfg.musiq_loss.image_size
                )
                musiq_score = musiq_metric(musiq_inputs).mean()
                musiq_loss = -musiq_score

            edge_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.edge_loss.weight > 0:
                edges_cur = _soft_canny(
                    rendered,
                    self.cfg.edge_loss.sigma,
                    self.cfg.edge_loss.low_threshold,
                    self.cfg.edge_loss.high_threshold,
                    self.cfg.edge_loss.soft_k,
                )
                with torch.no_grad():
                    edges_init = _soft_canny(
                        rendered_init,
                        self.cfg.edge_loss.sigma,
                        self.cfg.edge_loss.low_threshold,
                        self.cfg.edge_loss.high_threshold,
                        self.cfg.edge_loss.soft_k,
                    )
                edge_loss = F.l1_loss(edges_cur, edges_init)

            delta = shs - shs_init
            if self.cfg.sh_reg.norm.lower() == "l1":
                reg_loss = (delta.abs() * sh_weight_map[None, :, None]).mean()
            else:
                reg_loss = (delta.pow(2) * sh_weight_map[None, :, None]).mean()

            total = (
                self.cfg.clip_loss.weight * clip_loss
                + self.cfg.musiq_loss.weight * musiq_loss
                + self.cfg.edge_loss.weight * edge_loss
                + self.cfg.sh_reg.weight * reg_loss
            )
            total.backward()
            optimizer.step()

            if step % self.cfg.training.log_every == 0 or step == 1:
                musiq_str = (
                    f" musiq={musiq_score.item():.4f}"
                    if musiq_score is not None
                    else ""
                )
                print(
                    f"[clip-guidance] step={step:05d} "
                    f"total={total.item():.4f} "
                    f"clip={clip_loss.item():.4f} "
                    f"musiq_loss={musiq_loss.item():.4f} "
                    f"edge={edge_loss.item():.4f} "
                    f"sh_reg={reg_loss.item():.4f}"
                    f"{musiq_str}"
                )

            if (
                self.cfg.training.save_every > 0
                and step % self.cfg.training.save_every == 0
            ):
                out_path = os.path.join(self.cfg.output.dir, self.cfg.output.ply_name)
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
                render_path = os.path.join(
                    self.cfg.output.dir, f"render_{step:05d}.png"
                )
                to_pil_rgb(rendered[0].detach()).save(render_path)
                init_render_path = os.path.join(
                    self.cfg.output.dir, f"render_init_{step:05d}.png"
                )
                to_pil_rgb(rendered_init[0].detach()).save(init_render_path)

        final_path = os.path.join(self.cfg.output.dir, self.cfg.output.ply_name)
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
            self.cfg.pose_jitter.background_color,
        )
        final_renders = _render_views(
            final_views, means, rotations, scales, opacities, shs
        )
        for view, image in zip(final_views, final_renders):
            cam_name = CAM_ORDER[view["cam_idx"]]
            render_path = os.path.join(
                self.cfg.output.dir, f"render_final_{cam_name}.png"
            )
            to_pil_rgb(image.detach()).save(render_path)


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
