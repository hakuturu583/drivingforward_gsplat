from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import pyiqa


@dataclass
class MusiqLossConfig:
    model_id: str = "musiq"
    weight: float = 0.0
    image_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "MusiqLossConfig":
        defaults = cls()
        return cls(
            model_id=data.get("model_id", defaults.model_id),
            weight=float(data.get("weight", defaults.weight)),
            image_size=data.get("image_size", defaults.image_size),
        )


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


class MusiqLoss:
    def __init__(
        self, model_id: str, image_size: Optional[int], device: torch.device
    ) -> None:
        self.device = device
        self.metric = pyiqa.create_metric(model_id, device=device)
        self.metric.eval()
        self.image_size = image_size

    def compute(self, images: torch.Tensor) -> torch.Tensor:
        musiq_inputs = _preprocess_musiq_images(images, self.image_size)
        musiq_score = self.metric(musiq_inputs).mean()
        return -musiq_score
