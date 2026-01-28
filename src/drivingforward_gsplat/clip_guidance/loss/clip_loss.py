from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class ClipLossConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    prompts: List[str] = field(default_factory=list)
    weight: float = 1.0
    image_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ClipLossConfig":
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


class ClipLoss:
    def __init__(
        self,
        model_id: str,
        prompts: List[str],
        image_size: Optional[int],
        device: torch.device,
    ) -> None:
        if not prompts:
            raise ValueError("clip_loss.prompts must contain at least one prompt.")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            self.text_features = F.normalize(text_features, dim=-1)
        self.image_size = (
            image_size
            if image_size is not None
            else int(self.model.config.vision_config.image_size)
        )

    def compute(self, images: torch.Tensor) -> torch.Tensor:
        clip_inputs = _preprocess_clip_images(images, self.image_size)
        image_features = self.model.get_image_features(pixel_values=clip_inputs)
        image_features = F.normalize(image_features, dim=-1)
        sims = image_features @ self.text_features.T
        return 1.0 - sims.mean()
