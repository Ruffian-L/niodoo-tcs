"""Configuration parsing tests."""

from pathlib import Path

import yaml

from niodoo_ai import TrainingConfig, load_training_config


def _write_sample_config(path: Path) -> None:
    sample = {
        "model": {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "load_in_4bit": True,
            "compute_dtype": "bfloat16",
            "quant_type": "nf4",
        },
        "lora": {
            "r": 64,
            "alpha": 16,
            "dropout": 0.05,
        },
        "data": {
            "train_file": str(path.parent / "train.jsonl"),
            "eval_split_ratio": 0.1,
            "max_seq_length": 1024,
        },
        "optimizer": {
            "learning_rate": 2e-4,
            "num_train_epochs": 2.0,
        },
        "runtime": {
            "output_dir": str(path.parent / "outputs"),
            "logging_steps": 5,
        },
        "topology": {
            "lambda_weight": 0.05,
            "lambda_teacher": 0.01,
            "projection": "cls",
            "teacher_cache": str(path.parent / "cache.jsonl"),
            "teacher_match_field": "source_path",
        },
    }
    path.write_text(yaml.safe_dump(sample), encoding="utf-8")


def test_load_training_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_sample_config(config_path)
    config = load_training_config(config_path)
    assert isinstance(config, TrainingConfig)
    assert config.model.base_model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config.data.max_seq_length == 1024
    assert config.runtime.output_dir.exists() is False
    assert config.topology is not None
    assert config.topology.lambda_teacher == 0.01



