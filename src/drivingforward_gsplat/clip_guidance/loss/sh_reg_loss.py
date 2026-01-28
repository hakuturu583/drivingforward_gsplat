from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ShRegConfig:
    weight: float = 0.01
    degree_weights: List[float] = field(default_factory=list)
    norm: str = "l2"

    @classmethod
    def from_dict(cls, data: dict) -> "ShRegConfig":
        defaults = cls()
        return cls(
            weight=float(data.get("weight", defaults.weight)),
            degree_weights=list(data.get("degree_weights", defaults.degree_weights)),
            norm=str(data.get("norm", defaults.norm)),
        )


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


class ShRegLoss:
    def __init__(self, degree_weights: List[float], norm: str, device: torch.device):
        self.norm = norm
        self.degree_weights = degree_weights
        self.device = device
        self.weight_map: torch.Tensor | None = None

    def _ensure_weight_map(self, shs: torch.Tensor) -> torch.Tensor:
        if self.weight_map is None or self.weight_map.numel() != shs.shape[1]:
            self.weight_map = _make_sh_degree_weights(
                shs.shape[1], self.degree_weights
            ).to(self.device, dtype=shs.dtype)
        return self.weight_map

    def compute(self, shs: torch.Tensor, shs_init: torch.Tensor) -> torch.Tensor:
        weight_map = self._ensure_weight_map(shs)
        delta = shs - shs_init
        if self.norm.lower() == "l1":
            return (delta.abs() * weight_map[None, :, None]).mean()
        return (delta.pow(2) * weight_map[None, :, None]).mean()
