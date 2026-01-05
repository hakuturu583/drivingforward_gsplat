from __future__ import annotations

from typing import Iterable

import torch


def zero_grad_for_indices(
    params: Iterable[torch.nn.Parameter], mask: torch.Tensor
) -> None:
    if mask is None:
        return
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    for param in params:
        if param.grad is None:
            continue
        if param.grad.shape[0] != mask.shape[0]:
            continue
        param.grad[mask] = 0
