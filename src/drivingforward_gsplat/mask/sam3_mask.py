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
    from transformers import Sam3Model, Sam3Processor

    device = _resolve_device(device_str)
    dtype = _resolve_dtype(dtype_str, device)
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, torch_dtype=dtype)
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
    images = [pil_image]
    if prompt:
        try:
            inputs = processor(images=images, text=[prompt], return_tensors="pt")
        except TypeError as exc:
            raise TypeError(
                "SAM3 processor does not accept text prompts. "
                "Set sam3_prompt to empty or switch to a text-capable SAM3 processor."
            ) from exc
    else:
        inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=cfg.mask_threshold,
        mask_threshold=cfg.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )
    if not results:
        raise RuntimeError("SAM3 did not return any segmentation result.")
    result = results[0]
    masks = result.get("masks")
    if masks is None or masks.numel() == 0:
        mask = torch.zeros(pil_image.size[1], pil_image.size[0], dtype=torch.bool)
    else:
        scores = result.get("scores")
        if scores is not None and scores.numel() > 0:
            best_idx = int(torch.argmax(scores).item())
        else:
            best_idx = 0
        mask = masks[best_idx] > 0.5

    mask_np = (mask.detach().cpu().numpy() * 255.0).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")
    if mask_pil.size != pil_image.size:
        mask_pil = mask_pil.resize(pil_image.size, resample=Image.NEAREST)
        mask_np = np.array(mask_pil)

    mask_tensor = torch.from_numpy(mask_np > 0).to(dtype=torch.bool)
    return mask_tensor
