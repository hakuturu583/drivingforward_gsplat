import argparse
import math
import os
import tempfile
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from skimage import feature

from depth_anything_3.api import DepthAnything3
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from huggingface_hub import snapshot_download

from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
from drivingforward_gsplat.utils import misc as utils
from drivingforward_gsplat.i2i.prompt_config import PromptConfig
from drivingforward_gsplat.i2i.sdxl_panorama_i2i_config import SdxlPanoramaI2IConfig
from drivingforward_gsplat.depth.depth import to_pil_depth
from drivingforward_gsplat.utils.misc import to_pil_rgb
from drivingforward_gsplat.panorama.panorama import (
    ImageLike,
    _concat_strip,
    build_depth_panorama,
    build_strip_panorama,
)


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


def _split_strip_by_widths(
    image_strip: Image.Image, widths: Sequence[int]
) -> List[Image.Image]:
    segments = []
    x0 = 0
    for width in widths:
        x1 = x0 + width
        segments.append(image_strip.crop((x0, 0, x1, image_strip.height)))
        x0 = x1
    return segments


def _output_path_from_nuscenes_filename(
    output_dir: str, nuscenes_filename: str
) -> str:
    rel_path = (
        os.path.relpath(nuscenes_filename, "samples")
        if nuscenes_filename.startswith("samples/")
        else nuscenes_filename
    )
    return os.path.join(output_dir, "samples", rel_path)


class SdxlPanoramaI2I:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_ids: Sequence[str] = ("diffusers/controlnet-depth-sdxl-1.0",),
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = True,
        enable_sequential_offload: bool = False,
        enable_xformers: bool = False,
        ip_adapter_model_id: Optional[str] = None,
        ip_adapter_subfolder: Optional[str] = None,
        ip_adapter_weight_name: Optional[str] = None,
        ip_adapter_image_encoder_folder: Optional[str] = None,
        ip_adapter_scale: float = 1.0,
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
        if ip_adapter_model_id:
            ip_adapter_kwargs = {}
            if ip_adapter_subfolder:
                ip_adapter_kwargs["subfolder"] = ip_adapter_subfolder
            if ip_adapter_weight_name:
                ip_adapter_kwargs["weight_name"] = ip_adapter_weight_name
            if ip_adapter_image_encoder_folder is not None:
                ip_adapter_kwargs["image_encoder_folder"] = ip_adapter_image_encoder_folder
            self.pipe.load_ip_adapter(ip_adapter_model_id, **ip_adapter_kwargs)
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)

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
        ip_adapter_images: Optional[Sequence[Image.Image]] = None,
    ) -> Image.Image:
        image_strip, target_size = build_strip_panorama(
            images, height=height, return_target_size=True
        )
        if control_image is None:
            control_strip = build_depth_panorama(
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

        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image_strip,
            "control_image": control_strip,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "generator": generator,
            "height": control_for_size.height,
            "width": control_for_size.width,
        }
        if ip_adapter_images:
            pipe_kwargs["ip_adapter_image"] = ip_adapter_images
        result = self.pipe(**pipe_kwargs)
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
        split = "eval_SF"

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
        pil = to_pil_rgb(image)
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
    pil_images = [to_pil_rgb(img) for img in images]
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
    parser = argparse.ArgumentParser(description="SDXL panorama i2i")
    parser.add_argument(
        "--config", required=True, help="Path to SDXL panorama i2i yaml config."
    )
    parser.add_argument(
        "--prompt-config",
        required=True,
        help="Path to SDXL prompt yaml config.",
    )
    return parser.parse_args()


def sdxl_panorama_i2i(
    i2i_cfg: SdxlPanoramaI2IConfig, images: Sequence[ImageLike]
) -> List[Image.Image]:
    repo_root = os.getcwd()
    project_root = utils.find_project_root(repo_root)
    if not i2i_cfg.prompt_config:
        raise ValueError("prompt_config is required in SdxlPanoramaI2IConfig.")
    if len(images) != 6:
        raise ValueError(f"Expected 6 camera images, got {len(images)}.")
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

    prompt_path = (
        i2i_cfg.prompt_config
        if os.path.isabs(i2i_cfg.prompt_config)
        else os.path.join(repo_root, i2i_cfg.prompt_config)
    )
    prompt_cfg = PromptConfig.from_yaml(prompt_path)
    prompt = prompt_cfg.prompt
    negative_prompt = prompt_cfg.negative_prompt
    use_ip_adapter = bool(prompt_cfg.reference_images)
    if use_ip_adapter and not i2i_cfg.ip_adapter_model_id:
        raise ValueError(
            "reference_images provided in prompt config but ip_adapter_model_id is not set."
        )
    reference_images = []
    for ref_path in prompt_cfg.reference_images:
        abs_path = (
            ref_path
            if os.path.isabs(ref_path)
            else os.path.join(project_root, ref_path)
        )
        reference_images.append(Image.open(abs_path).convert("RGB"))

    i2i = SdxlPanoramaI2I(
        model_id=i2i_cfg.model_id,
        controlnet_ids=controlnet_ids,
        enable_cpu_offload=i2i_cfg.cpu_offload,
        enable_sequential_offload=i2i_cfg.sequential_offload,
        enable_xformers=i2i_cfg.xformers,
        ip_adapter_model_id=i2i_cfg.ip_adapter_model_id if use_ip_adapter else None,
        ip_adapter_subfolder=i2i_cfg.ip_adapter_subfolder,
        ip_adapter_weight_name=i2i_cfg.ip_adapter_weight_name,
        ip_adapter_image_encoder_folder=i2i_cfg.ip_adapter_image_encoder_folder,
        ip_adapter_scale=i2i_cfg.ip_adapter_scale,
    )
    result = i2i.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=images,
        depths=[],
        height=i2i_cfg.height,
        strength=i2i_cfg.strength,
        num_inference_steps=i2i_cfg.steps,
        guidance_scale=i2i_cfg.guidance_scale,
        controlnet_conditioning_scale=control_scales,
        seed=i2i_cfg.seed,
        control_image=control_strips,
        ip_adapter_images=reference_images or None,
    )
    blended = _blend_strip_segments(
        result,
        widths=[img.width for img in [to_pil_rgb(img) for img in images]],
        blend_width=i2i_cfg.blend_width,
    )
    blended_segments = _split_strip_by_widths(
        blended, widths=[img.width for img in [to_pil_rgb(img) for img in images]]
    )
    return blended_segments


def main():
    args = _parse_args()
    repo_root = os.getcwd()
    cfg_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(repo_root, args.config)
    )
    i2i_cfg = SdxlPanoramaI2IConfig.from_yaml(cfg_path)
    i2i_cfg.prompt_config = args.prompt_config
    config_file = (
        i2i_cfg.config_file
        if os.path.isabs(i2i_cfg.config_file)
        else os.path.join(repo_root, i2i_cfg.config_file)
    )
    cfg = utils.get_config(
        config_file,
        mode="eval",
        weight_path=repo_root,
        novel_view_mode="SF",
    )
    dataset = _build_env_dataset(cfg, "eval")
    if len(cfg["data"]["cameras"]) != 6:
        raise ValueError(
            f"Expected 6 cameras for strip panorama, got {len(cfg['data']['cameras'])}"
        )
    sample = dataset[i2i_cfg.sample_index]
    sample_token = sample.get("token")
    if sample_token is None:
        raise ValueError("Sample token is missing from dataset output.")
    nusc_dataset = dataset._dataset.dataset
    nusc_sample = nusc_dataset.get("sample", sample_token)
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
    cam_filenames = [
        nusc_dataset.get("sample_data", nusc_sample["data"][name])["filename"]
        for name in cam_order
    ]
    blended_segments = sdxl_panorama_i2i(i2i_cfg, images)
    os.makedirs(i2i_cfg.output_dir, exist_ok=True)
    for cam_filename, segment in zip(cam_filenames, blended_segments):
        out_path = _output_path_from_nuscenes_filename(
            i2i_cfg.output_dir, cam_filename
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        segment.save(out_path)


if __name__ == "__main__":
    main()
