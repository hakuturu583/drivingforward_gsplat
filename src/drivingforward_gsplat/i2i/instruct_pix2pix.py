from __future__ import annotations

from typing import Optional

import torch
from PIL import Image, ImageOps

from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)


def instruct_pix2pix_i2i(
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str] = None,
    model_id: str = "timbrooks/instruct-pix2pix",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    seed: Optional[int | list[int]] = None,
    enable_cpu_offload: bool = True,
    enable_xformers: bool = False,
) -> Image.Image | list[Image.Image]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch_dtype = torch.float32

    image = ImageOps.exif_transpose(image).convert("RGB")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except AttributeError:
        pass

    if device == "cuda" and enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except (AttributeError, ImportError):
            pass

    if isinstance(seed, list):
        images = []
        for current_seed in seed:
            generator = torch.Generator(device=device).manual_seed(current_seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    generator=generator,
                )
            images.append(result.images[0])
        return images

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        )
    return result.images[0]
