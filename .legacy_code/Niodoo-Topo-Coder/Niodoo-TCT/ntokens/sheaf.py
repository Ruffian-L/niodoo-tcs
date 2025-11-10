"""Toy sheaf encoder that lifts segmented embeddings into a cochain complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .quantization import QuantizedTensor


@dataclass
class SheafPatch:
    name: str
    support: Tuple[int, int]
    value: torch.Tensor


class SheafEncoder:
    """Segment-based sheaf approximation over a 1D cover."""

    def __init__(self, segments: int = 3, overlap: float = 0.25) -> None:
        if segments < 1:
            raise ValueError("segments must be >= 1")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        self.segments = segments
        self.overlap = overlap

    def build(self, tensor: QuantizedTensor) -> Dict[str, torch.Tensor]:
        signal = tensor.dequantize()
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        patches = self._segment(signal)
        restriction = self._build_restriction_matrix(len(patches))
        return {
            "patch_vectors": torch.stack([patch.value for patch in patches], dim=0),
            "patch_supports": torch.tensor([[p.support[0], p.support[1]] for p in patches]),
            "restriction": restriction,
        }

    def _segment(self, signal: torch.Tensor) -> List[SheafPatch]:
        length = signal.size(1)
        seg_width = max(int(length / (self.segments - (self.segments - 1) * self.overlap)), 1)
        step = int(seg_width * (1 - self.overlap)) or 1
        patches: List[SheafPatch] = []
        start = 0
        idx = 0
        while start < length:
            end = min(start + seg_width, length)
            window = signal[:, start:end]
            value = window.mean(dim=1)
            patches.append(SheafPatch(name=f"patch_{idx}", support=(start, end), value=value))
            idx += 1
            if end == length:
                break
            start += step
        return patches

    def _build_restriction_matrix(self, count: int) -> torch.Tensor:
        if count <= 1:
            return torch.eye(count)
        restriction = torch.eye(count)
        for i in range(count - 1):
            restriction[i, i + 1] = 0.5
            restriction[i + 1, i] = 0.5
        return restriction
