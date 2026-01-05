from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from drivingforward_gsplat.utils.misc import ImageLike, to_pil_rgb


@dataclass
class Sam3MaskConfig:
    model_id: str = "facebook/sam3-huge"
    device: str = "cuda"
    dtype: str = "auto"
    mask_threshold: float = 0.5
    resize_longest_side: Optional[int] = None


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@lru_cache(maxsize=2)
def _load_sam3_components(
    model_id: str, device_str: str, dtype_str: str
) -> Tuple[object, torch.nn.Module]:
    from transformers import AutoModelForMaskGeneration, AutoProcessor

    device = _resolve_device(device_str)
    dtype = _resolve_dtype(dtype_str, device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model


def cutout_with_sam3(
    image: ImageLike,
    prompt: str,
    cfg: Sam3MaskConfig,
) -> torch.Tensor:
    """Runs SAM3 with a text prompt and returns a 0/1 mask tensor."""
    pil_image = to_pil_rgb(image)
    if cfg.resize_longest_side:
        w, h = pil_image.size
        scale = cfg.resize_longest_side / max(w, h)
        if scale < 1.0:
            new_size = (int(round(w * scale)), int(round(h * scale)))
            pil_image = pil_image.resize(new_size, resample=Image.BICUBIC)

    processor, model = _load_sam3_components(cfg.model_id, cfg.device, cfg.dtype)
    device = _resolve_device(cfg.device)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    mask_logits = (
        getattr(outputs, "pred_masks", None)
        or getattr(outputs, "masks", None)
        or getattr(outputs, "mask_logits", None)
    )
    if mask_logits is None:
        raise RuntimeError("SAM3 output does not contain masks.")

    if mask_logits.ndim == 4:
        masks = mask_logits[0]
    else:
        masks = mask_logits

    iou_scores = getattr(outputs, "iou_scores", None)
    if iou_scores is not None:
        best_idx = int(torch.argmax(iou_scores[0]).item())
        mask = masks[best_idx]
    else:
        mask = masks[0]

    if mask.max() > 1.0 or mask.min() < 0.0:
        mask = torch.sigmoid(mask)
    mask = (mask >= cfg.mask_threshold).float()

    mask_np = (mask.detach().cpu().numpy() * 255.0).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")
    if mask_pil.size != pil_image.size:
        mask_pil = mask_pil.resize(pil_image.size, resample=Image.NEAREST)
        mask_np = np.array(mask_pil)

    mask_tensor = torch.from_numpy(mask_np > 0).to(dtype=torch.bool)
    return mask_tensor
