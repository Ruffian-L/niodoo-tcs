"""Dataset utilities for topology-augmented supervised fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from .topology import TopologyAugmentor


def _load_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return _validate_records(records, path)


def _validate_records(records: List[Dict[str, object]], source: Path) -> List[Dict[str, object]]:
    validated: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {index} in {source} is not a mapping")

        instruction = record.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Record {index} in {source} must include a non-empty 'instruction'")

        output = record.get("output")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"Record {index} in {source} must include a non-empty 'output'")

        input_text = record.get("input", "")
        if not isinstance(input_text, str):
            raise ValueError(f"Record {index} in {source} has non-string 'input'")

        has_topology_payload = any(key in record for key in ("topology", "topology_features", "hidden_state_path"))
        if not has_topology_payload:
            raise ValueError(
                f"Record {index} in {source} must include one of 'topology', 'topology_features', or 'hidden_state_path'"
            )

        topology_features = record.get("topology_features")
        if topology_features is not None:
            if not isinstance(topology_features, Iterable) or isinstance(topology_features, (str, bytes)):
                raise ValueError(
                    f"Record {index} in {source} has invalid 'topology_features'; expected a numeric iterable"
                )
            _ = _validate_feature_sequence(topology_features)

        hidden_state_path = record.get("hidden_state_path")
        if hidden_state_path is not None and not isinstance(hidden_state_path, str):
            raise ValueError(f"Record {index} in {source} has non-string 'hidden_state_path'")

        validated_record = dict(record)
        validated_record["instruction"] = instruction
        validated_record["input"] = input_text
        validated_record["output"] = output
        validated.append(validated_record)

    return validated


def _split_records(records: List[Dict[str, object]], ratio: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cutoff = int(len(records) * (1 - ratio))
    return records[:cutoff], records[cutoff:]


def _format_prompt(template: str, instruction: str, input_text: str, topology_text: str) -> str:
    rendered = template.format(
        instruction=instruction.strip(),
        input=input_text.strip(),
        topology=topology_text.strip(),
    )
    return rendered.strip()


def _validate_feature_sequence(sequence: Iterable[object]) -> List[float]:
    values: List[float] = []
    for item in sequence:
        if not isinstance(item, (int, float)):
            raise ValueError("Topology features must be numeric")
        values.append(float(item))
    if not values:
        raise ValueError("Topology feature sequence must not be empty")
    return values


def _validate_topology_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not payload:
        raise ValueError("Topology payload must not be empty")
    vector = payload.get("vector") or payload.get("values")
    if vector is None:
        raise ValueError("Topology payload requires a 'vector' or 'values' key")
    payload = dict(payload)
    payload["vector"] = _validate_feature_sequence(vector)

    diagrams = payload.get("persistence_diagrams")
    if diagrams is not None:
        if not isinstance(diagrams, Mapping):
            raise ValueError("persistence_diagrams must be a mapping of dimensions to pairs")
        normalised: Dict[str, List[List[float]]] = {}
        for dim, entries in diagrams.items():
            if not isinstance(entries, Iterable):
                raise ValueError(f"Diagram for {dim!r} must be iterable")
            normalised_entries: List[List[float]] = []
            for pair in entries:
                if not isinstance(pair, Iterable):
                    raise ValueError("Diagram entries must be iterable length-2 pairs")
                pair_list = list(pair)
                if len(pair_list) != 2:
                    raise ValueError("Diagram pairs must have length 2")
                normalised_entries.append([float(pair_list[0]), float(pair_list[1])])
            normalised[str(dim)] = normalised_entries
        payload["persistence_diagrams"] = normalised

    betti_numbers = payload.get("betti_numbers")
    if betti_numbers is not None:
        if not isinstance(betti_numbers, Mapping):
            raise ValueError("betti_numbers must be a mapping")
        payload["betti_numbers"] = {str(dim): int(count) for dim, count in betti_numbers.items()}

    if "persistence_entropy" in payload:
        payload["persistence_entropy"] = float(payload["persistence_entropy"])
    if "sheaf_energy" in payload:
        payload["sheaf_energy"] = float(payload["sheaf_energy"])

    return payload


class TopologyDataset:
    """Wraps raw JSONL records into tokenised training examples."""

    def __init__(
        self,
        records: Iterable[Dict[str, object]],
        tokenizer: PreTrainedTokenizerBase,
        config: DataConfig,
        augmentor: TopologyAugmentor,
        feature_prefix: Optional[str] = None,
    ) -> None:
        self._records = list(records)
        self.tokenizer = tokenizer
        self.config = config
        self.augmentor = augmentor
        self.feature_prefix = feature_prefix or config.feature_prefix

    def prepare(self) -> Dataset:
        dataset = Dataset.from_list(self._records)
        columns_to_remove = [col for col in dataset.column_names if col not in {"instruction", "input", "output", "topology_features", "hidden_state_path"}]

        def _map(example: Dict[str, object]) -> Dict[str, object]:
            instruction = str(example.get("instruction", ""))
            input_text = str(example.get("input", ""))
            output = str(example.get("output", ""))

            existing_features = example.get("topology_features")
            hidden_state_path = example.get("hidden_state_path")
            topology_payload_raw = example.get("topology") if isinstance(example, dict) else None
            topology_payload = _validate_topology_payload(topology_payload_raw) if isinstance(topology_payload_raw, Mapping) else None

            feature_path: Optional[Path] = None
            topology_diagrams = None
            diagram_summary = None
            if topology_payload is not None:
                bundle = self.augmentor.ensure_bundle(topology_payload=topology_payload)
                topology_diagrams = topology_payload.get("persistence_diagrams")
                diagram_summary = topology_payload.get("diagram_summary")
            elif isinstance(existing_features, (list, tuple)):
                features = _validate_feature_sequence(existing_features)
                bundle = self.augmentor.ensure_bundle(existing_features=features)
            elif hidden_state_path:
                feature_path = Path(str(hidden_state_path))
                if not feature_path.exists():
                    raise FileNotFoundError(f"Hidden state tensor not found: {feature_path}")
                bundle = self.augmentor.ensure_bundle(feature_path=feature_path)
            else:
                raise ValueError(
                    "Each record must include `topology`, `topology_features`, or `hidden_state_path`"
                )

            topology_text = f"{self.feature_prefix} {bundle.text}" if self.feature_prefix else bundle.text
            prompt = _format_prompt(self.config.prompt_template, instruction, input_text, topology_text)
            combined = f"{prompt}\n{output.strip()}"

            tokens = self.tokenizer(
                combined,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

            topology_vector = bundle.vector.detach().cpu().numpy().astype(np.float32)
            if topology_payload is not None:
                topology_metadata = {
                    key: value
                    for key, value in topology_payload.items()
                    if key not in {"vector", "values"}
                }
            else:
                topology_metadata = {
                    "betti_sum": bundle.summary.get("betti_sum", 0.0),
                    "betti_max": bundle.summary.get("betti_max", 0.0),
                    "entropy": bundle.summary.get("entropy", 0.0),
                    "sheaf_energy": bundle.summary.get("sheaf_energy", 0.0),
                }

            teacher_key = None
            if topology_payload is not None:
                for candidate in ("source_path", "id"):
                    value = topology_payload.get(candidate)
                    if value:
                        teacher_key = str(value)
                        break
            if teacher_key is None and feature_path is not None:
                teacher_key = str(feature_path.resolve())
            if teacher_key is None and example.get("id") is not None:
                teacher_key = str(example["id"])

            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"],
                "topology_vector": topology_vector,
                "prompt": prompt,
                "response": output,
                "topology_metadata": topology_metadata,
                "teacher_key": teacher_key,
                "topology_diagrams": topology_diagrams,
                "diagram_summary": diagram_summary,
            }

        dataset = dataset.map(_map, remove_columns=columns_to_remove, desc="tokenising-topology", num_proc=None)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "topology_vector"])
        return dataset


def build_datasets(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    augmentor: TopologyAugmentor,
) -> Tuple[Dataset, Optional[Dataset]]:
    train_records = _load_records(config.train_file)
    eval_records: List[Dict[str, object]]

    if config.eval_file:
        eval_records = _load_records(config.eval_file)
    elif config.eval_split_ratio:
        train_records, eval_records = _split_records(train_records, config.eval_split_ratio)
    else:
        eval_records = []

    train_dataset = TopologyDataset(train_records, tokenizer, config, augmentor).prepare()
    eval_dataset = None
    if eval_records:
        eval_dataset = TopologyDataset(eval_records, tokenizer, config, augmentor).prepare()

    return train_dataset, eval_dataset

