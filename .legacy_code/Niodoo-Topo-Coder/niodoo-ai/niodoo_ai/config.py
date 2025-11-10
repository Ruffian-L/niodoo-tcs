"""Configuration handling for topology-aware QLoRA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def _dtype_from_string(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    lowered = value.lower()
    if lowered not in mapping:
        raise ValueError(f"Unsupported dtype string: {value}")
    return mapping[lowered]


@dataclass
class ModelConfig:
    base_model: str
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    trust_remote_code: bool = False

    def torch_dtype(self) -> torch.dtype:
        return _dtype_from_string(self.compute_dtype)


@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])


@dataclass
class DataConfig:
    train_file: Path
    eval_file: Optional[Path] = None
    max_seq_length: int = 4096
    prompt_template: str = "[INST] {instruction}\n{input}\n{topology} [/INST]"
    feature_prefix: str = "[TOPOLOGY]"
    eval_split_ratio: Optional[float] = None
    num_workers: int = 2


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 2.0
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0


@dataclass
class RuntimeConfig:
    output_dir: Path
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 200
    seed: int = 42
    bf16: bool = True


@dataclass
class TrainingConfig:
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    topology: Optional["TopologyTrainingConfig"] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainingConfig":
        model = ModelConfig(**raw["model"])
        lora = LoRAConfig(**raw.get("lora", {}))
        data_section = raw["data"].copy()
        data_section["train_file"] = Path(data_section["train_file"]).expanduser()
        if data_section.get("eval_file") is not None:
            data_section["eval_file"] = Path(data_section["eval_file"]).expanduser()
        if data_section.get("eval_split_ratio") is not None:
            ratio = data_section["eval_split_ratio"]
            if not 0.0 < ratio < 1.0:
                raise ValueError("eval_split_ratio must be in (0, 1)")
        data = DataConfig(**data_section)
        optimizer = OptimizerConfig(**raw.get("optimizer", {}))
        runtime_section = raw["runtime"].copy()
        runtime_section["output_dir"] = Path(runtime_section["output_dir"]).expanduser()
        runtime = RuntimeConfig(**runtime_section)
        topology_section = raw.get("topology")
        topology = None
        if topology_section is not None:
            topology = TopologyTrainingConfig.from_dict(topology_section)
        return cls(model=model, lora=lora, data=data, optimizer=optimizer, runtime=runtime, topology=topology)


@dataclass
class TopologyTrainingConfig:
    lambda_weight: float = 0.05
    lambda_teacher: float = 0.0
    lambda_sinkhorn: float = 0.0
    projection: str = "mean"
    teacher_cache: Optional[Path] = None
    teacher_match_field: str = "source_path"
    sinkhorn_p: float = 1.0
    sinkhorn_blur: float = 0.05
    sinkhorn_scaling: float = 0.8
    max_sinkhorn_points: int = 128

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TopologyTrainingConfig":
        section = dict(raw)
        cache = section.get("teacher_cache")
        if cache is not None:
            section["teacher_cache"] = Path(cache).expanduser()
        projection = section.get("projection")
        if projection is not None and projection not in {"mean", "cls"}:
            raise ValueError("topology.projection must be 'mean' or 'cls'")
        max_points = section.get("max_sinkhorn_points")
        if max_points is not None and max_points <= 0:
            raise ValueError("topology.max_sinkhorn_points must be positive")
        for key in ("lambda_weight", "lambda_teacher", "lambda_sinkhorn", "sinkhorn_blur"):
            if key in section and section[key] < 0:
                raise ValueError(f"topology.{key} must be non-negative")
        return cls(**section)


def load_training_config(path: Path | str) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Config file must be a mapping")
    return TrainingConfig.from_dict(raw)



