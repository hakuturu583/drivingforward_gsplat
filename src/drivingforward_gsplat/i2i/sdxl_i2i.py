import argparse
import math
import os
import tempfile
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from skimage import feature

from depth_anything_3.api import DepthAnything3
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from huggingface_hub import snapshot_download

from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
from drivingforward_gsplat.utils import misc as utils
from drivingforward_gsplat.i2i.sdxl_i2i_config import SdxlI2IConfig


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]


def _tensor_to_numpy_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        return Image.fromarray(_tensor_to_numpy_uint8(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_pil_depth(image: ImageLike) -> Image.Image:
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


def _normalize_depths(depths: Sequence[ImageLike]) -> List[Image.Image]:
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


def _blend_strip_segments(
    image_strip: Image.Image, widths: Sequence[int], blend_width: int
) -> Image.Image:
    if blend_width <= 0:
        return image_strip
    strip = np.array(image_strip, dtype=np.float32)
    height, width, channels = strip.shape
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)
    for idx in range(1, len(offsets) - 1):
        left = offsets[idx - 1]
        mid = offsets[idx]
        right = offsets[idx + 1]
        bw = min(blend_width, mid - left, right - mid)
        if bw <= 0:
            continue
        l_slice = strip[:, mid - bw : mid, :]
        r_slice = strip[:, mid : mid + bw, :]
        alpha = (1.0 - np.cos(np.linspace(0, math.pi, bw))) * 0.5
        alpha = alpha.reshape(1, bw, 1)
        blended = l_slice * (1.0 - alpha) + r_slice * alpha
        strip[:, mid - bw : mid, :] = blended
        strip[:, mid : mid + bw, :] = blended
    strip = np.clip(strip, 0, 255).astype(np.uint8)
    return Image.fromarray(strip, mode="RGB")


class SdxlStripPanoramaI2I:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_ids: Sequence[str] = ("diffusers/controlnet-depth-sdxl-1.0",),
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = True,
        enable_sequential_offload: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch_dtype

        controlnet = [
            ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
            for controlnet_id in controlnet_ids
        ]
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch_dtype
        )

        if device == "cuda" and enable_cpu_offload:
            if enable_sequential_offload:
                self.pipe.enable_sequential_cpu_offload()
            else:
                self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
        else:
            self.pipe.to(device)

        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except (AttributeError, ImportError):
                pass

    def build_strip_panorama(
        self,
        images: Sequence[ImageLike],
        height: Optional[int] = None,
        blend_width: int = 0,
        return_target_size: bool = False,
    ) -> Image.Image:
        if len(images) != 6:
            raise ValueError(f"Expected 6 images, got {len(images)}")
        pil_images = [_to_pil_rgb(img) for img in images]
        target_width = sum(img.width for img in pil_images)
        target_height = pil_images[0].height
        strip = _concat_strip(pil_images, height)
        if return_target_size:
            return strip, (target_width, target_height)
        return strip

    def build_depth_panorama(
        self,
        depths: Sequence[ImageLike],
        height: Optional[int] = None,
        blend_width: int = 0,
    ) -> Image.Image:
        if len(depths) != 6:
            raise ValueError(f"Expected 6 depth maps, got {len(depths)}")
        depth_frames = _normalize_depths(depths)
        pil_depths = [_to_pil_depth(d) for d in depth_frames]
        depth_strip = _concat_strip(pil_depths, height)
        return depth_strip.convert("RGB")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        images: Sequence[ImageLike],
        depths: Sequence[ImageLike],
        height: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        control_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        image_strip, target_size = self.build_strip_panorama(
            images, height=height, return_target_size=True
        )
        if control_image is None:
            control_strip = self.build_depth_panorama(
                depths, height=height or image_strip.height
            )
        else:
            control_strip = control_image
        if isinstance(control_strip, list):
            control_for_size = control_strip[0]
        else:
            control_for_size = control_strip

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_strip,
            control_image=control_strip,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=control_for_size.height,
            width=control_for_size.width,
        )
        output = result.images[0]
        if height is not None and output.size != target_size:
            output = output.resize(target_size, resample=Image.BICUBIC)
        return output


def _build_env_dataset(cfg, mode: str):
    augmentation = {
        "image_shape": (int(cfg["training"]["height"]), int(cfg["training"]["width"])),
        "jittering": (0.0, 0.0, 0.0, 0.0),
        "crop_train_borders": (),
        "crop_eval_borders": (),
    }
    if mode == "train":
        dataset_args = {
            "cameras": cfg["data"]["cameras"],
            "back_context": cfg["data"]["back_context"],
            "forward_context": cfg["data"]["forward_context"],
            "data_transform": get_transforms("train", **augmentation),
            "depth_type": cfg["data"]["depth_type"]
            if "gt_depth" in cfg["data"]["train_requirements"]
            else None,
            "with_pose": "gt_pose" in cfg["data"]["train_requirements"],
            "with_ego_pose": "gt_ego_pose" in cfg["data"]["train_requirements"],
            "with_mask": "mask" in cfg["data"]["train_requirements"],
        }
    elif mode in ("val", "eval"):
        dataset_args = {
            "cameras": cfg["data"]["cameras"],
            "back_context": cfg["data"]["back_context"],
            "forward_context": cfg["data"]["forward_context"],
            "data_transform": get_transforms("train", **augmentation),
            "depth_type": cfg["data"]["depth_type"]
            if "gt_depth" in cfg["data"]["val_requirements"]
            else None,
            "with_pose": "gt_pose" in cfg["data"]["val_requirements"],
            "with_ego_pose": "gt_ego_pose" in cfg["data"]["val_requirements"],
            "with_mask": "mask" in cfg["data"]["val_requirements"],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if cfg["data"]["dataset"] != "nuscenes":
        raise ValueError(f"Unknown dataset: {cfg['data']['dataset']}")

    if mode == "train":
        split = "train"
    else:
        if cfg["model"]["novel_view_mode"] == "MF":
            split = "eval_MF"
        elif cfg["model"]["novel_view_mode"] == "SF":
            split = "eval_SF"
        else:
            raise ValueError(
                f"Unknown novel view mode: {cfg['model']['novel_view_mode']}"
            )

    return EnvNuScenesDataset(split, **dataset_args)


def _dense_depth_from_anything(
    images: Sequence[ImageLike],
    device: torch.device,
    model_id: str,
) -> List[np.ndarray]:
    model = DepthAnything3.from_pretrained(model_id).to(device=device)
    tmp_dir = tempfile.mkdtemp(prefix="da3_depth_")
    image_paths = []
    for idx, image in enumerate(images):
        pil = _to_pil_rgb(image)
        path = os.path.join(tmp_dir, f"{idx:02d}.png")
        pil.save(path)
        image_paths.append(path)
    prediction = model.inference(image_paths)
    return [depth for depth in prediction.depth]


def _control_image_from_canny(image_strip: Image.Image) -> Image.Image:
    gray = np.array(image_strip.convert("L"), dtype=np.float32) / 255.0
    edges = feature.canny(gray, sigma=2.0)
    edge_img = edges.astype(np.uint8) * 255
    return Image.fromarray(edge_img, mode="L").convert("RGB")


def _build_control_images(
    images: Sequence[ImageLike],
    controlnet_ids: Sequence[str],
    depth_device: torch.device,
    depth_model_id: str,
    height: Optional[int],
):
    pil_images = [_to_pil_rgb(img) for img in images]
    target_width = sum(img.width for img in pil_images)
    target_height = pil_images[0].height
    image_strip = _concat_strip(pil_images, height)
    target_size = (target_width, target_height)
    controls: List[Image.Image] = []
    depths = None
    for controlnet_id in controlnet_ids:
        if "canny" in controlnet_id.lower():
            controls.append(_control_image_from_canny(image_strip))
        elif "depth" in controlnet_id.lower():
            if depths is None:
                depths = _dense_depth_from_anything(
                    images, depth_device, depth_model_id
                )
            control_strip = _concat_strip(
                [_to_pil_depth(d) for d in depths],
                height or image_strip.height,
            ).convert("RGB")
            controls.append(control_strip)
        else:
            raise ValueError(f"Unsupported controlnet id: {controlnet_id}")
    return image_strip, controls, target_size


def _parse_args():
    parser = argparse.ArgumentParser(description="SDXL strip panorama i2i")
    parser.add_argument("--config", required=True, help="Path to SDXL i2i yaml config.")
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = os.getcwd()
    cfg_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(repo_root, args.config)
    )
    i2i_cfg = SdxlI2IConfig.from_yaml(cfg_path)
    config_file = (
        i2i_cfg.config_file
        if os.path.isabs(i2i_cfg.config_file)
        else os.path.join(repo_root, i2i_cfg.config_file)
    )

    cfg = utils.get_config(
        config_file,
        mode="eval",
        weight_path=repo_root,
        novel_view_mode=i2i_cfg.novel_view_mode,
    )
    dataset = _build_env_dataset(cfg, "eval")

    if len(cfg["data"]["cameras"]) != 6:
        raise ValueError(
            f"Expected 6 cameras for strip panorama, got {len(cfg['data']['cameras'])}"
        )

    sample = dataset[i2i_cfg.sample_index]
    images_tensor = sample[("color", 0, 0)]
    cam_order = [
        "CAM_FRONT_RIGHT",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]
    cam_to_index = {name: idx for idx, name in enumerate(cfg["data"]["cameras"])}
    images = [images_tensor[cam_to_index[name]] for name in cam_order]
    depth_device = torch.device(i2i_cfg.depth_device)
    controlnet_ids = [item.id for item in i2i_cfg.control_nets]
    control_scales = [item.scale for item in i2i_cfg.control_nets]
    image_strip, control_strips, target_size = _build_control_images(
        images,
        controlnet_ids,
        depth_device,
        i2i_cfg.depth_model_id,
        i2i_cfg.height,
    )

    i2i = SdxlStripPanoramaI2I(
        model_id=i2i_cfg.model_id,
        controlnet_ids=controlnet_ids,
        enable_cpu_offload=i2i_cfg.cpu_offload,
        enable_sequential_offload=i2i_cfg.sequential_offload,
        enable_xformers=i2i_cfg.xformers,
    )
    os.makedirs(i2i_cfg.output_dir, exist_ok=True)
    image_strip.save(os.path.join(i2i_cfg.output_dir, "input_image.png"))
    for idx, control in enumerate(control_strips):
        control.save(os.path.join(i2i_cfg.output_dir, f"control_map_{idx}.png"))

    result = i2i.generate(
        prompt=i2i_cfg.prompt,
        negative_prompt=i2i_cfg.negative_prompt,
        images=images,
        depths=[],
        height=i2i_cfg.height,
        strength=i2i_cfg.strength,
        num_inference_steps=i2i_cfg.steps,
        guidance_scale=i2i_cfg.guidance_scale,
        controlnet_conditioning_scale=control_scales,
        seed=i2i_cfg.seed,
        control_image=control_strips,
    )
    blended = _blend_strip_segments(
        result,
        widths=[img.width for img in [_to_pil_rgb(img) for img in images]],
        blend_width=i2i_cfg.blend_width,
    )
    blended.save(os.path.join(i2i_cfg.output_dir, "output_image.png"))


if __name__ == "__main__":
    main()
