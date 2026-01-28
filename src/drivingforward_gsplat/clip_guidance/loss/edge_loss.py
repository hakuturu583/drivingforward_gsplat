from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class EdgeLossConfig:
    weight: float = 0.2
    sigma: float = 1.0
    low_threshold: float = 0.1
    high_threshold: float = 0.2
    render_low_threshold: Optional[float] = None
    render_high_threshold: Optional[float] = None
    ref_low_threshold: Optional[float] = None
    ref_high_threshold: Optional[float] = None
    soft_k: float = 10.0

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeLossConfig":
        low = float(data.get("low_threshold", cls.low_threshold))
        high = float(data.get("high_threshold", cls.high_threshold))
        render_low = data.get("render_low_threshold")
        render_high = data.get("render_high_threshold")
        ref_low = data.get("ref_low_threshold")
        ref_high = data.get("ref_high_threshold")
        return cls(
            weight=float(data.get("weight", cls.weight)),
            sigma=float(data.get("sigma", cls.sigma)),
            low_threshold=low,
            high_threshold=high,
            render_low_threshold=(
                float(render_low) if isinstance(render_low, (int, float)) else None
            ),
            render_high_threshold=(
                float(render_high) if isinstance(render_high, (int, float)) else None
            ),
            ref_low_threshold=(
                float(ref_low) if isinstance(ref_low, (int, float)) else None
            ),
            ref_high_threshold=(
                float(ref_high) if isinstance(ref_high, (int, float)) else None
            ),
            soft_k=float(data.get("soft_k", cls.soft_k)),
        )


def _gaussian_kernel(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    radius = max(1, int((3.0 * sigma) + 0.999))
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


class EdgeLoss:
    def __init__(
        self,
        sigma: float,
        low_threshold: float,
        high_threshold: float,
        soft_k: float,
        render_low_threshold: Optional[float] = None,
        render_high_threshold: Optional[float] = None,
        ref_low_threshold: Optional[float] = None,
        ref_high_threshold: Optional[float] = None,
    ) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.soft_k = soft_k
        self.render_low_threshold = render_low_threshold
        self.render_high_threshold = render_high_threshold
        self.ref_low_threshold = ref_low_threshold
        self.ref_high_threshold = ref_high_threshold

    def _resolve_thresholds(self) -> tuple[float, float, float, float]:
        render_low = (
            self.render_low_threshold
            if self.render_low_threshold is not None
            else self.low_threshold
        )
        render_high = (
            self.render_high_threshold
            if self.render_high_threshold is not None
            else self.high_threshold
        )
        ref_low = (
            self.ref_low_threshold
            if self.ref_low_threshold is not None
            else self.low_threshold
        )
        ref_high = (
            self.ref_high_threshold
            if self.ref_high_threshold is not None
            else self.high_threshold
        )
        return render_low, render_high, ref_low, ref_high

    def compute(self, rendered: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        render_low, render_high, ref_low, ref_high = self._resolve_thresholds()
        edges_cur = _soft_canny(
            rendered, self.sigma, render_low, render_high, self.soft_k
        )
        with torch.no_grad():
            edges_ref = _soft_canny(
                reference, self.sigma, ref_low, ref_high, self.soft_k
            )
        return F.l1_loss(edges_cur, edges_ref)
