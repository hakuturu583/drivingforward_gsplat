from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value.mean()
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(0)
    mask = mask.to(dtype=value.dtype, device=value.device)
    numerator = (value * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return numerator / denom


def _gaussian_kernel(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive for gaussian kernel.")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = radius * 2 + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def blur_image(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    if image.dim() == 3:
        image = image.unsqueeze(0)
    b, c, h, w = image.shape
    kernel = _gaussian_kernel(sigma, image.device, image.dtype)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = kernel.shape[-1] // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=c)
    return blurred if blurred.shape[0] > 1 else blurred.squeeze(0)


@dataclass
class FixerLossConfig:
    danger_percentile: float = 0.25
    blur_sigma: float = 1.5
    gamma: float = 5.0
    lambda_fix_low: float = 1.0
    lambda_fix_lpips: float = 0.1
    use_lpips: bool = True
    lpips_net: str = "vgg"


class FixerLoss:
    def __init__(self, cfg: FixerLossConfig) -> None:
        self.cfg = cfg
        self._lpips = None
        self._lpips_warned = False
        if cfg.use_lpips:
            try:
                import lpips  # type: ignore

                self._lpips = lpips.LPIPS(net=cfg.lpips_net)
            except Exception:
                self._lpips = None

    def _warn_lpips(self) -> None:
        if self._lpips_warned or not self.cfg.use_lpips:
            return
        self._lpips_warned = True
        print("LPIPS unavailable; skipping LPIPS fixer loss.")

    def _danger_mask(
        self,
        fixer_rgb: torch.Tensor,
        input_render: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        diff = torch.mean(torch.abs(fixer_rgb - input_render), dim=0, keepdim=True)
        if mask is not None:
            valid = mask > 0.5
            values = diff[valid]
        else:
            values = diff.view(-1)
        if values.numel() == 0:
            return torch.ones_like(diff)
        percentile = max(0.0, min(1.0, 1.0 - self.cfg.danger_percentile))
        threshold = torch.quantile(values, percentile)
        safe = diff < threshold
        return safe.to(diff.dtype)

    def __call__(
        self,
        rendered: torch.Tensor,
        fixer_rgb: torch.Tensor,
        input_render: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = mask if mask is not None else None
        safe_mask = self._danger_mask(fixer_rgb, input_render, mask)
        if mask is not None:
            combined_mask = mask * safe_mask
        else:
            combined_mask = safe_mask

        diff = torch.mean(torch.abs(fixer_rgb - input_render), dim=0, keepdim=True)
        weights = torch.exp(-self.cfg.gamma * diff)
        if combined_mask is not None:
            weights = weights * combined_mask

        rendered_blur = blur_image(rendered, self.cfg.blur_sigma)
        fixer_blur = blur_image(fixer_rgb, self.cfg.blur_sigma)
        low_loss = charbonnier(rendered_blur - fixer_blur)
        low_loss = masked_mean(low_loss, weights)

        lpips_loss = torch.tensor(0.0, device=rendered.device, dtype=rendered.dtype)
        if self.cfg.use_lpips:
            if self._lpips is None:
                self._warn_lpips()
            else:
                self._lpips = self._lpips.to(rendered.device)
                lpips_mask = combined_mask
                if lpips_mask is not None:
                    rendered_lp = rendered * lpips_mask
                    fixer_lp = fixer_rgb * lpips_mask
                else:
                    rendered_lp = rendered
                    fixer_lp = fixer_rgb
                lpips_val = self._lpips(rendered_lp.unsqueeze(0), fixer_lp.unsqueeze(0))
                lpips_loss = lpips_val.mean()

        return (
            self.cfg.lambda_fix_low * low_loss + self.cfg.lambda_fix_lpips * lpips_loss
        )
