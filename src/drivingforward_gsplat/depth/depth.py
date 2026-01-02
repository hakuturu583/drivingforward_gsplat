import math
import os
import tempfile
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

from drivingforward_gsplat.utils.misc import to_pil_rgb

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


def resize_depths_to_match(
    images: Sequence[Image.Image], depths: Sequence[Image.Image]
) -> List[Image.Image]:
    if not images:
        return list(depths)
    target_size = images[0].size
    return [
        depth
        if depth.size == target_size
        else depth.resize(target_size, Image.BILINEAR)
        for depth in depths
    ]


def dense_depth_from_anything(
    images: Sequence[ImageLike],
    device: torch.device,
    model_id: str,
    intrinsics: np.ndarray | Sequence[np.ndarray] | None = None,
) -> List[np.ndarray]:
    if intrinsics is not None:
        if isinstance(intrinsics, np.ndarray):
            if intrinsics.ndim == 2:
                if len(images) != 1:
                    raise ValueError(
                        "Expected 1 image to match a single intrinsics matrix."
                    )
            elif intrinsics.ndim == 3:
                if intrinsics.shape[0] != len(images):
                    raise ValueError(
                        "Intrinsics batch size must match number of images."
                    )
            else:
                raise ValueError(
                    "Intrinsics must be a 2D matrix or a batch of 2D matrices."
                )
        else:
            if len(intrinsics) != len(images):
                raise ValueError("Intrinsics must match number of images.")
    model = DepthAnything3.from_pretrained(model_id).to(device=device)
    tmp_dir = tempfile.mkdtemp(prefix="da3_depth_")
    image_paths = []
    for idx, image in enumerate(images):
        pil = to_pil_rgb(image)
        path = os.path.join(tmp_dir, f"{idx:02d}.png")
        pil.save(path)
        image_paths.append(path)
    intrinsics_arr = None
    if intrinsics is not None:
        if isinstance(intrinsics, np.ndarray):
            intrinsics_arr = intrinsics
        else:
            intrinsics_arr = np.stack(list(intrinsics), axis=0)
    prediction = model.inference(image_paths, intrinsics=intrinsics_arr)
    return [depth for depth in prediction.depth]
