from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import yaml


@dataclass
class PromptConfig:
    prompt: List[str]
    negative_prompt: Optional[List[str]] = None
    reference_images: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "PromptConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        prompt = data.get("prompt") or []
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(prompt, Sequence) or not prompt:
            raise ValueError(f"Missing prompt in {path}")
        prompt = [str(item) for item in prompt]
        ref_images = data.get("reference_images") or []
        if isinstance(ref_images, str):
            ref_images = [ref_images]
        negative_prompt = data.get("negative_prompt")
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        if isinstance(negative_prompt, Sequence):
            negative_prompt = [str(item) for item in negative_prompt]
        return cls(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_images=ref_images,
        )
