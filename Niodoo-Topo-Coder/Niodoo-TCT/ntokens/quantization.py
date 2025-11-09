"""INT8 vector quantization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data and dequantization metadata."""

    values: torch.Tensor
    scale: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        """Reconstruct the approximate floating point tensor."""
        return self.values.float() * self.scale


class Int8VectorQuantizer:
    """Simple symmetric int8 quantizer with optional per-channel scaling."""

    def __init__(
        self,
        per_channel: bool = True,
        epsilon: float = 1e-8,
        device: Optional[torch.device] = None,
    ) -> None:
        self.per_channel = per_channel
        self.epsilon = epsilon
        self.device = device

    def __call__(self, tensor: torch.Tensor | np.ndarray | Sequence[float]) -> QuantizedTensor:
        return self.quantize(tensor)

    def quantize(
        self, tensor: torch.Tensor | np.ndarray | Sequence[float]
    ) -> QuantizedTensor:
        """Quantize the incoming tensor to int8 with symmetric scaling."""

        data = self._to_tensor(tensor)
        if data.dim() == 1:
            axis = 0
        else:
            axis = 1 if self.per_channel else None

        if axis is None:
            max_abs = torch.max(data.abs())
        else:
            max_abs = torch.amax(data.abs(), dim=axis, keepdim=True)

        scale = torch.clamp(max_abs / 127.0, min=self.epsilon)
        quantized = torch.clamp(torch.round(data / scale), -128, 127).to(torch.int8)
        return QuantizedTensor(values=quantized, scale=scale)

    def dequantize(self, qt: QuantizedTensor) -> torch.Tensor:
        """Convenience wrapper to dequantize."""

        return qt.dequantize()

    def _to_tensor(self, tensor: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            data = tensor
        else:
            data = torch.tensor(tensor)
        if self.device is not None:
            data = data.to(self.device)
        return data.float()
