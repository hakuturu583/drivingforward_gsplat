from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from drivingforward_gsplat.i2i.controlnet import ControlNetConfig


@dataclass
class SdxlPanoramaI2IConfig:
    config_file: str = "configs/nuscenes/main.yaml"
    output_dir: str = "output"
    sample_index: int = 0
    height: Optional[int] = None
    blend_width: int = 0
    prompt_config: Optional[str] = None
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    refiner_strength: float = 0.2
    ip_adapter_model_id: Optional[str] = None
    ip_adapter_subfolder: Optional[str] = None
    ip_adapter_weight_name: Optional[str] = None
    ip_adapter_image_encoder_folder: Optional[str] = None
    ip_adapter_scale: float = 1.0
    control_nets: List[ControlNetConfig] = field(
        default_factory=lambda: [
            ControlNetConfig(id="diffusers/controlnet-depth-sdxl-1.0")
        ]
    )
    depth_model_id: str = "depth-anything/DA3METRIC-LARGE"
    depth_device: str = "cuda"
    strength: float = 0.6
    steps: int = 30
    guidance_scale: float = 5.0
    cpu_offload: bool = False
    sequential_offload: bool = False
    xformers: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "SdxlPanoramaI2IConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if "control_nets" in data:
            data["control_nets"] = [
                ControlNetConfig(**item) for item in data["control_nets"]
            ]
        return cls(**data)
