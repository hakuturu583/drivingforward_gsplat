import argparse
import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
from drivingforward_gsplat.utils import misc as utils


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


class SdxlStripPanoramaI2I:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_id: str = "diffusers/controlnet-depth-sdxl-1.0",
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

        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch_dtype
        )
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
        self, depths: Sequence[ImageLike], height: Optional[int] = None
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
        images: Sequence[ImageLike],
        depths: Sequence[ImageLike],
        height: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        controlnet_conditioning_scale: float = 1.0,
        tile_size: Optional[int] = None,
        tile_overlap: int = 64,
        seed: Optional[int] = None,
    ) -> Image.Image:
        image_strip, target_size = self.build_strip_panorama(
            images, height=height, return_target_size=True
        )
        depth_strip = self.build_depth_panorama(
            depths, height=height or image_strip.height
        )

        if tile_size:
            result = self._generate_tiled(
                prompt=prompt,
                image_strip=image_strip,
                depth_strip=depth_strip,
                strength=strength,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
            )
            if height is not None and result.size != target_size:
                result = result.resize(target_size, resample=Image.BICUBIC)
            return result

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image_strip,
            control_image=depth_strip,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=depth_strip.height,
            width=depth_strip.width,
        )
        output = result.images[0]
        if height is not None and output.size != target_size:
            output = output.resize(target_size, resample=Image.BICUBIC)
        return output

    def _generate_tiled(
        self,
        prompt: str,
        image_strip: Image.Image,
        depth_strip: Image.Image,
        strength: float,
        tile_size: int,
        tile_overlap: int,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        seed: Optional[int],
    ) -> Image.Image:
        width, height = depth_strip.size
        tile_size = min(tile_size, width, height)
        stride = max(1, tile_size - tile_overlap)

        output_accum = np.zeros((height, width, 3), dtype=np.float32)
        weight_accum = np.zeros((height, width, 1), dtype=np.float32)

        win_x = np.hanning(tile_size) if tile_overlap > 0 else np.ones(tile_size)
        win_y = np.hanning(tile_size) if tile_overlap > 0 else np.ones(tile_size)
        weight = (win_y[:, None] * win_x[None, :]).astype(np.float32)
        weight = np.expand_dims(weight, axis=-1)

        tiles = []
        for y in range(0, height, stride):
            y0 = min(y, height - tile_size)
            for x in range(0, width, stride):
                x0 = min(x, width - tile_size)
                tiles.append((x0, y0))
            if y0 == height - tile_size:
                break

        for idx, (x0, y0) in enumerate(tiles):
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            image_tile = image_strip.crop((x0, y0, x1, y1))
            depth_tile = depth_strip.crop((x0, y0, x1, y1))

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + idx)

            if self.device == "cuda":
                torch.cuda.empty_cache()
            result = self.pipe(
                prompt=prompt,
                image=image_tile,
                control_image=depth_tile,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=tile_size,
                width=tile_size,
            )
            tile_img = result.images[0]
            tile_arr = np.array(tile_img, dtype=np.float32)

            output_accum[y0:y1, x0:x1] += tile_arr * weight
            weight_accum[y0:y1, x0:x1] += weight

        output = output_accum / np.maximum(weight_accum, 1e-6)
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)


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


def _parse_args():
    parser = argparse.ArgumentParser(description="SDXL strip panorama i2i")
    parser.add_argument("--config_file", default="configs/nuscenes/main.yaml")
    parser.add_argument("--novel_view_mode", default="MF", choices=("MF", "SF"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="sdxl_strip.png")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--model_id", default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--controlnet_id", default="diffusers/controlnet-depth-sdxl-1.0"
    )
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--sequential_offload", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = os.getcwd()
    config_file = (
        args.config_file
        if os.path.isabs(args.config_file)
        else os.path.join(repo_root, args.config_file)
    )

    cfg = utils.get_config(
        config_file,
        mode="eval",
        weight_path=repo_root,
        novel_view_mode=args.novel_view_mode,
    )
    dataset = _build_env_dataset(cfg, "eval")

    if len(cfg["data"]["cameras"]) != 6:
        raise ValueError(
            f"Expected 6 cameras for strip panorama, got {len(cfg['data']['cameras'])}"
        )

    sample = dataset[args.sample_index]
    images_tensor = sample[("color", 0, 0)]
    depths_tensor = sample["depth"]
    images = [images_tensor[i] for i in range(images_tensor.shape[0])]
    depths = [depths_tensor[i] for i in range(depths_tensor.shape[0])]

    i2i = SdxlStripPanoramaI2I(
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        enable_cpu_offload=args.cpu_offload,
        enable_sequential_offload=args.sequential_offload,
        enable_xformers=args.xformers,
    )
    result = i2i.generate(
        prompt=args.prompt,
        images=images,
        depths=depths,
        height=args.height,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        seed=args.seed,
    )
    result.save(args.output)


if __name__ == "__main__":
    main()
