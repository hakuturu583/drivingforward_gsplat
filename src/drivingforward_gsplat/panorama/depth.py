import math
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]


def to_pil_depth(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("L")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.numpy()
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype(np.uint8), mode="L")
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            return Image.fromarray(image.astype(np.uint8)).convert("L")
    raise TypeError(f"Unsupported depth type: {type(image)}")


def normalize_depths(depths: Sequence[ImageLike]) -> List[Image.Image]:
    arrays = []
    for depth in depths:
        if isinstance(depth, Image.Image):
            arrays.append(np.array(depth.convert("L"), dtype=np.float32))
        elif isinstance(depth, torch.Tensor):
            arr = depth.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            arrays.append(arr.numpy().astype(np.float32))
        elif isinstance(depth, np.ndarray):
            arrays.append(depth.astype(np.float32))
        else:
            raise TypeError(f"Unsupported depth type: {type(depth)}")

    stacked = np.stack(arrays, axis=0)
    depth_min = float(np.nanmin(stacked))
    depth_max = float(np.nanmax(stacked))
    if math.isfinite(depth_min) and math.isfinite(depth_max) and depth_max > depth_min:
        norm = (stacked - depth_min) / (depth_max - depth_min)
    else:
        norm = np.zeros_like(stacked, dtype=np.float32)
    norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(frame, mode="L") for frame in norm]
