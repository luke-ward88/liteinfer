"""Global-ish configuration knobs.

Kept intentionally tiny — a couple of dataclasses that group the parameters the
CLI and examples pass around, so we don't thread a dozen keyword arguments
through every call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantConfig:
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    group_size: int | None = None  # only meaningful for INT4 group-wise

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"only 4- and 8-bit quantization supported, got {self.bits}")


@dataclass
class SpecDecodeConfig:
    gamma: int = 4  # draft tokens proposed per step
    temperature: float = 1.0
    top_k: int = 0  # 0 disables top-k
    top_p: float = 1.0  # 1.0 disables nucleus filtering
    max_new_tokens: int = 64
    seed: int = 0
