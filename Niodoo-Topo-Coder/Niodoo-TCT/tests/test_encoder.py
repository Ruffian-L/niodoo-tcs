"""Basic smoke tests for the minimal nToken encoder."""

import torch

from ntokens import MinimalNTokens


def test_encoder_runs_on_cpu():
    encoder = MinimalNTokens()
    embeddings = torch.randn(2, 16)
    result = encoder.encode(embeddings)
    assert result.quantized.values.dtype == torch.int8
    assert "patch_vectors" in result.sheaf
    assert result.sheaf["patch_vectors"].ndim == 2
