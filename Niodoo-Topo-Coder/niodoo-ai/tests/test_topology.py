"""Tests covering topology augmentation and dataset preparation."""

import json
from typing import Dict, List

from niodoo_ai import DataConfig, TopologyAugmentor, build_datasets


class DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "right"

    def __call__(self, text, truncation, max_length, padding):
        tokens = text.split()
        ids = list(range(min(len(tokens), max_length)))
        if len(ids) < max_length:
            ids.extend([0] * (max_length - len(ids)))
        attention = [1 if idx < len(tokens) else 0 for idx in range(max_length)]
        return {"input_ids": ids, "attention_mask": attention}


def test_topology_augmentor_from_sequence():
    augmentor = TopologyAugmentor()
    bundle = augmentor.from_sequence([0.1, 0.2, 0.3])
    assert bundle.vector.shape[0] == 3
    assert "mean" in bundle.summary
    assert bundle.text.startswith("len=")


def test_build_datasets_with_features(tmp_path):
    train_file = tmp_path / "train.jsonl"
    records: List[Dict[str, object]] = [
        {
            "instruction": "Count nodes",
            "input": "A tree with 3 leaves",
            "output": "The answer is 5",
            "topology_features": [0.0, 0.1, 0.2],
        }
    ]
    with train_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    data_config = DataConfig(
        train_file=train_file,
        eval_file=None,
        max_seq_length=16,
        prompt_template="[INST] {instruction} {input} {topology} [/INST]",
        feature_prefix="[TOPOLOGY]",
        eval_split_ratio=None,
        num_workers=1,
    )

    tokenizer = DummyTokenizer()
    augmentor = TopologyAugmentor()
    train_dataset, eval_dataset = build_datasets(data_config, tokenizer, augmentor)
    assert len(train_dataset) == 1
    assert eval_dataset is None
    example = train_dataset[0]
    assert "topology_vector" in example
    assert example["input_ids"].shape[0] == 16
    assert example["teacher_key"] is None


def test_build_datasets_with_topology_payload(tmp_path):
    train_file = tmp_path / "train_payload.jsonl"
    records: List[Dict[str, object]] = [
        {
            "instruction": "Detect loop",
            "input": "A torus",
            "output": "Two cycles",
            "topology": {
                "vector": [0.5, 0.25, 0.75],
                "betti_numbers": {"b0": 1, "b1": 2, "b2": 1},
                "persistence_entropy": 1.23,
                "sheaf_energy": 0.42,
                "persistence_diagrams": {
                    "h0": [[0.0, 0.1]],
                    "h1": [[0.2, 0.8], [0.3, 0.7]],
                },
            },
        }
    ]
    with train_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    data_config = DataConfig(
        train_file=train_file,
        eval_file=None,
        max_seq_length=32,
        prompt_template="[INST] {instruction} {input} {topology} [/INST]",
        feature_prefix="[TOPOLOGY]",
        eval_split_ratio=None,
        num_workers=1,
    )

    tokenizer = DummyTokenizer()
    augmentor = TopologyAugmentor()
    train_dataset, _ = build_datasets(data_config, tokenizer, augmentor)
    example = train_dataset[0]
    assert example["topology_vector"].shape[0] == 3
    metadata = example["topology_metadata"]
    assert metadata["betti_numbers"]["b1"] == 2
    assert "persistence_diagrams" in metadata
    assert example["teacher_key"] == records[0]["topology"]["source_path"]

