from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class PromptConfig:
    prompt: str
    negative_prompt: Optional[str] = None
    reference_images: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "PromptConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError(f"Missing prompt in {path}")
        ref_images = data.get("reference_images") or []
        if isinstance(ref_images, str):
            ref_images = [ref_images]
        return cls(
            prompt=prompt,
            negative_prompt=data.get("negative_prompt"),
            reference_images=ref_images,
        )
