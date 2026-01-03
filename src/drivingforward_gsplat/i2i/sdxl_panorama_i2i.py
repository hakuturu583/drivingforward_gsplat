import argparse
import math
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from huggingface_hub import snapshot_download

from drivingforward_gsplat.dataset import EnvNuScenesDataset, get_transforms
from drivingforward_gsplat.utils import misc as utils
from drivingforward_gsplat.i2i.controlnet import build_control_images
from drivingforward_gsplat.i2i.prompt_config import PromptConfig
from drivingforward_gsplat.i2i.sdxl_panorama_i2i_config import SdxlPanoramaI2IConfig
from drivingforward_gsplat.utils.misc import to_pil_rgb
from drivingforward_gsplat.panorama.panorama import (
    ImageLike,
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


def _output_path_from_nuscenes_filename(output_dir: str, nuscenes_filename: str) -> str:
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
        refiner_model_id: Optional[str] = None,
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
        self._cpu_offload_enabled = device == "cuda" and enable_cpu_offload

        controlnet = [
            ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype)
            for controlnet_id in controlnet_ids
        ]
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch_dtype
        )
        self._configure_pipeline(
            self.pipe,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_sequential_offload=enable_sequential_offload,
            enable_xformers=enable_xformers,
        )
        if ip_adapter_model_id:
            ip_adapter_kwargs = {}
            if ip_adapter_subfolder:
                ip_adapter_kwargs["subfolder"] = ip_adapter_subfolder
            if ip_adapter_weight_name:
                ip_adapter_kwargs["weight_name"] = ip_adapter_weight_name
            if ip_adapter_image_encoder_folder is not None:
                ip_adapter_kwargs[
                    "image_encoder_folder"
                ] = ip_adapter_image_encoder_folder
            self.pipe.load_ip_adapter(ip_adapter_model_id, **ip_adapter_kwargs)
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        self.refiner_pipe = None
        if refiner_model_id:
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_model_id, torch_dtype=torch_dtype
            )
            self._configure_pipeline(
                self.refiner_pipe,
                device=device,
                enable_cpu_offload=enable_cpu_offload,
                enable_sequential_offload=enable_sequential_offload,
                enable_xformers=enable_xformers,
            )

    def _maybe_swap_pipelines(self, active_pipe, inactive_pipe) -> None:
        if self.device != "cuda" or self._cpu_offload_enabled:
            return
        if inactive_pipe is not None:
            inactive_pipe.to("cpu")
        active_pipe.to(self.device)
        torch.cuda.empty_cache()

    @staticmethod
    def _configure_pipeline(
        pipe,
        device: str,
        enable_cpu_offload: bool,
        enable_sequential_offload: bool,
        enable_xformers: bool,
    ) -> None:
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except AttributeError:
            pass
        pipe.unet.eval()
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.eval()
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.eval()
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.eval()
        if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
            pipe.controlnet.eval()
        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            pipe.image_encoder.eval()
        if device == "cuda" and enable_cpu_offload:
            if enable_sequential_offload:
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except (AttributeError, ImportError):
                pass

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
        if self.refiner_pipe is not None:
            self._maybe_swap_pipelines(self.pipe, self.refiner_pipe)
        with torch.inference_mode():
            result = self.pipe(**pipe_kwargs)
        output = result.images[0]
        if height is not None and output.size != target_size:
            output = output.resize(target_size, resample=Image.BICUBIC)
        return output

    def refine_images(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        images: Sequence[Image.Image],
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        if self.refiner_pipe is None:
            return list(images)
        self._maybe_swap_pipelines(self.refiner_pipe, self.pipe)
        refined = []
        for idx, image in enumerate(images):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + idx)
            with torch.inference_mode():
                result = self.refiner_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            refined.append(result.images[0])
        self._maybe_swap_pipelines(self.pipe, self.refiner_pipe)
        return refined


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
    image_strip, control_strips, target_size = build_control_images(
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
        refiner_model_id=i2i_cfg.refiner_model_id,
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
    refined_segments = i2i.refine_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=blended_segments,
        strength=i2i_cfg.refiner_strength,
        num_inference_steps=i2i_cfg.steps,
        guidance_scale=i2i_cfg.guidance_scale,
        seed=i2i_cfg.seed,
    )
    return refined_segments


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
        out_path = _output_path_from_nuscenes_filename(i2i_cfg.output_dir, cam_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        segment.save(out_path)


if __name__ == "__main__":
    main()
