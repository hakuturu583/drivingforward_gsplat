from typing import Optional, Sequence

from PIL import Image

from drivingforward_gsplat.depth.depth import normalize_depths, to_pil_depth
from drivingforward_gsplat.utils.misc import ImageLike, to_pil_rgb


def _concat_strip(images: Sequence[Image.Image], height: Optional[int]) -> Image.Image:
    if not images:
        raise ValueError("No images provided for strip panorama.")
    if height is None:
        height = images[0].height
    resized = []
    for img in images:
        if img.height != height:
            width = int(round(img.width * (height / img.height)))
            resized.append(img.resize((width, height), resample=Image.BICUBIC))
        else:
            resized.append(img)
    total_width = sum(img.width for img in resized)
    out = Image.new(resized[0].mode, (total_width, height))
    x = 0
    for img in resized:
        out.paste(img, (x, 0))
        x += img.width
    return out


def build_strip_panorama(
    images: Sequence[ImageLike],
    height: Optional[int] = None,
    blend_width: int = 0,
    return_target_size: bool = False,
) -> Image.Image:
    if len(images) != 6:
        raise ValueError(f"Expected 6 images, got {len(images)}")
    pil_images = [to_pil_rgb(img) for img in images]
    target_width = sum(img.width for img in pil_images)
    target_height = pil_images[0].height
    strip = _concat_strip(pil_images, height)
    if return_target_size:
        return strip, (target_width, target_height)
    return strip


def build_depth_panorama(
    depths: Sequence[ImageLike],
    height: Optional[int] = None,
    blend_width: int = 0,
) -> Image.Image:
    if len(depths) != 6:
        raise ValueError(f"Expected 6 depth maps, got {len(depths)}")
    depth_frames = normalize_depths(depths)
    pil_depths = [to_pil_depth(d) for d in depth_frames]
    depth_strip = _concat_strip(pil_depths, height)
    return depth_strip.convert("RGB")
