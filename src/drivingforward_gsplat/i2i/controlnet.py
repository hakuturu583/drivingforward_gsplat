from dataclasses import dataclass


@dataclass
class ControlNetConfig:
    id: str
    scale: float = 1.0
