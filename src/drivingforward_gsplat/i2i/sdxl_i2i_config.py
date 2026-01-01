from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class SdxlI2IConfig:
    prompt: str
    config_file: str = "configs/nuscenes/main.yaml"
    output_dir: str = "output"
    novel_view_mode: str = "MF"
    sample_index: int = 0
    height: Optional[int] = None
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_id: str = "diffusers/controlnet-depth-sdxl-1.0"
    depth_model_id: str = "depth-anything/DA3METRIC-LARGE"
    depth_device: str = "cuda"
    strength: float = 0.6
    steps: int = 30
    guidance_scale: float = 5.0
    controlnet_scale: float = 1.0
    cpu_offload: bool = False
    sequential_offload: bool = False
    xformers: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "SdxlI2IConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
