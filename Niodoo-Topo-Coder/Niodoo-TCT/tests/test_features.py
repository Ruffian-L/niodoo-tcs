"""Unit tests for topology feature extraction utilities."""

import torch

from ntokens import (
    HiddenStateFeatureAdapter,
    MinimalNTokens,
    TopologyFeatureExtractor,
    betti_curve,
)


def _dummy_encoding():
    encoder = MinimalNTokens()
    embeddings = torch.randn(4, 16)
    return encoder.encode(embeddings)


def test_betti_curve_returns_expected_shape():
    encoding = _dummy_encoding()
    curve = betti_curve(encoding.homology.diagrams, n_bins=16)
    assert curve.ndim == 2
    assert curve.shape[1] == 16


def test_feature_extractor_outputs_flat_vector():
    encoding = _dummy_encoding()
    extractor = TopologyFeatureExtractor(betti_bins=8)
    features = extractor.from_encoding(encoding)
    assert features.values.ndim == 1
    assert features.values.numel() == sum(section.numel() for section in features.sections.values())


def test_hidden_state_feature_adapter_matches_expected_dim():
    hidden_states = torch.randn(6, 32, 64)
    adapter = HiddenStateFeatureAdapter(pool_mode="mean")
    features = adapter.features(hidden_states)
    assert features.values.ndim == 1
    assert features.values.numel() > 0



