from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel

from drivingforward_gsplat.i2i.instruct_pix2pix import instruct_pix2pix_i2i
from drivingforward_gsplat.utils.misc import to_pil_rgb

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class ClipLossConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    prompts: List[str] = field(default_factory=list)
    weight: float = 1.0
    image_size: Optional[int] = None
    seed: Optional[int | list[int]] = None
    i2i_num_inference_steps: int = 20
    i2i_guidance_scale: float = 7.5
    i2i_image_guidance_scale: float = 1.5

    @classmethod
    def from_dict(cls, data: dict) -> "ClipLossConfig":
        defaults = cls()
        prompts = data.get("prompts", defaults.prompts)
        if isinstance(prompts, str):
            prompts = [prompts]
        seed = data.get("seed", defaults.seed)
        if isinstance(seed, list):
            seed = [int(item) for item in seed]
        elif isinstance(seed, (int, float)):
            seed = int(seed)
        return cls(
            model_id=data.get("model_id", defaults.model_id),
            prompts=list(prompts),
            weight=float(data.get("weight", defaults.weight)),
            image_size=data.get("image_size", defaults.image_size),
            seed=seed,
            i2i_num_inference_steps=int(
                data.get("i2i_num_inference_steps", defaults.i2i_num_inference_steps)
            ),
            i2i_guidance_scale=float(
                data.get("i2i_guidance_scale", defaults.i2i_guidance_scale)
            ),
            i2i_image_guidance_scale=float(
                data.get("i2i_image_guidance_scale", defaults.i2i_image_guidance_scale)
            ),
        )


def _preprocess_clip_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
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


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] != 3:
        arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


class ClipLoss:
    def __init__(
        self,
        model_id: str,
        prompts: List[str],
        image_size: Optional[int],
        device: torch.device,
        reference_images: Sequence[torch.Tensor | Image.Image | np.ndarray],
        camera_names: Sequence[str],
        seed: Optional[int | list[int]] = None,
        output_dir: Optional[str] = None,
        i2i_num_inference_steps: int = 20,
        i2i_guidance_scale: float = 7.5,
        i2i_image_guidance_scale: float = 1.5,
        i2i_images: Optional[Sequence[Sequence[Image.Image]]] = None,
        save_i2i: bool = True,
    ) -> None:
        if not prompts and i2i_images is None:
            raise ValueError(
                "clip_loss.prompts must contain at least one prompt when i2i_images "
                "is not provided."
            )
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.image_size = (
            image_size
            if image_size is not None
            else int(self.model.config.vision_config.image_size)
        )
        prompt = ", ".join(prompts) if prompts else ""
        reference_images, camera_names = self._normalize_reference_inputs(
            reference_images, camera_names
        )
        self._target_features = self._build_target_features(
            reference_images,
            camera_names,
            prompt,
            seed,
            output_dir,
            i2i_num_inference_steps,
            i2i_guidance_scale,
            i2i_image_guidance_scale,
            i2i_images,
            save_i2i,
        )

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        clip_inputs = _preprocess_clip_images(images, self.image_size)
        image_features = self.model.get_image_features(pixel_values=clip_inputs)
        return F.normalize(image_features, dim=-1)

    def _build_target_features(
        self,
        reference_images: Sequence[torch.Tensor | Image.Image | np.ndarray],
        camera_names: Sequence[str],
        prompt: str,
        seed: Optional[int | list[int]],
        output_dir: Optional[str],
        i2i_num_inference_steps: int,
        i2i_guidance_scale: float,
        i2i_image_guidance_scale: float,
        i2i_images: Optional[Sequence[Sequence[Image.Image]]],
        save_i2i: bool,
    ) -> dict[int, torch.Tensor]:
        target_features: dict[int, torch.Tensor] = {}
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if isinstance(seed, list) and not seed:
            seed = None
        if i2i_images is None and not prompt:
            raise ValueError(
                "clip_loss.prompts must contain at least one prompt when i2i_images "
                "is not provided."
            )
        for cam_idx, image in enumerate(reference_images):
            if i2i_images is not None:
                edited_images = list(i2i_images[cam_idx])
            else:
                pil_image = to_pil_rgb(image)
                edited = instruct_pix2pix_i2i(
                    image=pil_image,
                    prompt=prompt,
                    negative_prompt=None,
                    device=str(self.device),
                    torch_dtype=torch_dtype,
                    seed=seed,
                    num_inference_steps=i2i_num_inference_steps,
                    guidance_scale=i2i_guidance_scale,
                    image_guidance_scale=i2i_image_guidance_scale,
                )
                edited_images = edited if isinstance(edited, list) else [edited]
            if output_dir and save_i2i:
                cam_name = camera_names[cam_idx]
                cam_dir = os.path.join(output_dir, "i2i", cam_name)
                os.makedirs(cam_dir, exist_ok=True)
                if seed is None:
                    seeds = [None] * len(edited_images)
                else:
                    seeds = (
                        seed if isinstance(seed, list) else [seed] * len(edited_images)
                    )
                for idx, (img, seed_value) in enumerate(zip(edited_images, seeds)):
                    if seed_value is None:
                        name = f"seed_none_{idx:02d}.png"
                    else:
                        name = f"seed_{seed_value}.png"
                    img.save(os.path.join(cam_dir, name))
            tensors = [
                _pil_to_tensor(img).to(device=self.device, dtype=torch_dtype)
                for img in edited_images
            ]
            batch = torch.stack(tensors, dim=0)
            with torch.no_grad():
                features = self._encode_images(batch)
                target_features[cam_idx] = features.mean(dim=0)
        return target_features

    @staticmethod
    def _normalize_reference_inputs(
        reference_images: Sequence[torch.Tensor | Image.Image | np.ndarray]
        | torch.Tensor,
        camera_names: Sequence[str],
    ) -> tuple[List[torch.Tensor | Image.Image | np.ndarray], List[str]]:
        if torch.is_tensor(reference_images):
            images = reference_images
            if images.dim() == 5 and images.shape[0] == 1:
                images = images.squeeze(0)
            if images.dim() == 4:
                image_list = [images[idx] for idx in range(images.shape[0])]
            elif images.dim() == 3:
                image_list = [images]
            else:
                raise ValueError(
                    "reference_images tensor must be 3D or 4D (optionally with batch)."
                )
        else:
            image_list = list(reference_images)
        if len(camera_names) < len(image_list):
            raise ValueError(
                "camera_names length must be >= number of reference_images."
            )
        camera_list = list(camera_names[: len(image_list)])
        return image_list, camera_list

    def compute(self, images: torch.Tensor, cam_indices: Sequence[int]) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if len(cam_indices) != images.shape[0]:
            raise ValueError("cam_indices length must match images batch size.")
        features = self._encode_images(images)
        target = []
        for cam_idx in cam_indices:
            if cam_idx not in self._target_features:
                raise ValueError(f"Missing target CLIP features for cam_idx={cam_idx}.")
            target.append(self._target_features[cam_idx])
        target_features = torch.stack(target, dim=0)
        sims = (features * target_features).sum(dim=-1)
        return 1.0 - sims.mean()
