from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from skimage import feature

from drivingforward_gsplat.depth.depth import dense_depth_from_anything, to_pil_depth
from drivingforward_gsplat.panorama.panorama import _concat_strip
from drivingforward_gsplat.utils.misc import ImageLike, to_pil_rgb


@dataclass
class ControlNetConfig:
    id: str
    scale: float = 1.0


def control_image_from_canny(image_strip: Image.Image) -> Image.Image:
    gray = np.array(image_strip.convert("L"), dtype=np.float32) / 255.0
    edges = feature.canny(gray, sigma=2.0)
    edge_img = edges.astype(np.uint8) * 255
    return Image.fromarray(edge_img, mode="L").convert("RGB")


def build_control_images(
    images: Sequence[ImageLike],
    controlnet_ids: Sequence[str],
    depth_device: torch.device,
    depth_model_id: str,
    height: Optional[int],
):
    pil_images = [to_pil_rgb(img) for img in images]
    target_width = sum(img.width for img in pil_images)
    target_height = pil_images[0].height
    image_strip = _concat_strip(pil_images, height)
    target_size = (target_width, target_height)
    controls: List[Image.Image] = []
    depths = None
    for controlnet_id in controlnet_ids:
        if "canny" in controlnet_id.lower():
            controls.append(control_image_from_canny(image_strip))
        elif "depth" in controlnet_id.lower():
            if depths is None:
                depths = dense_depth_from_anything(images, depth_device, depth_model_id)
            control_strip = _concat_strip(
                [to_pil_depth(d) for d in depths],
                height or image_strip.height,
            ).convert("RGB")
            controls.append(control_strip)
        else:
            raise ValueError(f"Unsupported controlnet id: {controlnet_id}")
    return image_strip, controls, target_size
