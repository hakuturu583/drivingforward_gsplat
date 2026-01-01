from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class PromptConfig:
    prompt: str
    negative_prompt: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "PromptConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError(f"Missing prompt in {path}")
        return cls(prompt=prompt, negative_prompt=data.get("negative_prompt"))
