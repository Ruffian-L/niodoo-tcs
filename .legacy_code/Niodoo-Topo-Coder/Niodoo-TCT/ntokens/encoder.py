"""Top-level orchestration for minimal nToken encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .homology import HomologyResult, PersistentHomologyBackend
from .quantization import Int8VectorQuantizer, QuantizedTensor
from .sheaf import SheafEncoder


@dataclass
class NTokensEncoding:
    quantized: QuantizedTensor
    homology: HomologyResult
    sheaf: Dict[str, torch.Tensor]

    def summary(self) -> Dict[str, Any]:
        return {
            **self.homology.summary(),
            "num_patches": int(self.sheaf.get("patch_vectors", torch.empty(0)).shape[0]) if "patch_vectors" in self.sheaf else 0,
        }


class MinimalNTokens:
    """Composable pipeline producing coarse nToken descriptors."""

    def __init__(
        self,
        quantizer: Int8VectorQuantizer | None = None,
        homology_backend: PersistentHomologyBackend | None = None,
        sheaf_encoder: SheafEncoder | None = None,
    ) -> None:
        self.quantizer = quantizer or Int8VectorQuantizer()
        self.homology_backend = homology_backend or PersistentHomologyBackend()
        self.sheaf_encoder = sheaf_encoder or SheafEncoder()

    def encode(self, embeddings: torch.Tensor) -> NTokensEncoding:
        quantized = self.quantizer.quantize(embeddings)
        homology = self.homology_backend.compute(quantized.dequantize())
        sheaf = self.sheaf_encoder.build(quantized)
        return NTokensEncoding(quantized=quantized, homology=homology, sheaf=sheaf)
