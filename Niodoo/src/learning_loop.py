"""System 2 learning loop for adaptive QLoRA fine-tuning.

This module keeps a buffer of low-quality Granite interactions, requests
self-revisions, and triggers a QLoRA fine-tune once enough examples have been
collected. All operations are reproducible via a TOML configuration file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class WeightUpdateCallback(TrainerCallback):
    """PHASE 0: Diagnostic callback to track weight updates during training.
    
    This callback captures model weights before and after training steps
    to verify that Trainer.train() actually updates model weights.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.initial_weights: Optional[Dict[str, torch.Tensor]] = None
        self.weight_updates: List[float] = []
        self.step_count = 0
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture initial weights at training start."""
        if model is None:
            return
        
        self.logger.info("🔍 DIAGNOSTIC: WeightUpdateCallback initialized")
        self.initial_weights = {}
        
        # Capture initial weights from LoRA adapters
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                self.initial_weights[name] = param.data.clone().detach()
                self.logger.debug(f"Captured initial weight: {name}, shape={param.shape}")
        
        if not self.initial_weights:
            self.logger.warning("⚠️  No trainable LoRA parameters found!")
        else:
            self.logger.info(f"Captured {len(self.initial_weights)} initial weight tensors")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Track weight updates after each training step."""
        if model is None or self.initial_weights is None:
            return
        
        self.step_count += 1
        
        # Compute weight differences for LoRA parameters
        total_diff = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower() and name in self.initial_weights:
                initial = self.initial_weights[name]
                current = param.data.clone().detach()
                
                # Compute absolute difference
                diff = torch.abs(current - initial).sum().item()
                total_diff += diff
                param_count += 1
        
        if param_count > 0:
            avg_diff = total_diff / param_count
            self.weight_updates.append(avg_diff)
            
            if self.step_count % 10 == 0:
                self.logger.info(
                    f"Step {self.step_count}: avg_weight_diff={avg_diff:.9f}, "
                    f"total_params_tracked={param_count}"
                )
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Report final weight update statistics."""
        if model is None or not self.weight_updates:
            return
        
        final_diff = self.weight_updates[-1] if self.weight_updates else 0.0
        max_diff = max(self.weight_updates) if self.weight_updates else 0.0
        
        self.logger.info(
            f"🔍 DIAGNOSTIC: Training complete - "
            f"steps={self.step_count}, "
            f"final_weight_diff={final_diff:.9f}, "
            f"max_weight_diff={max_diff:.9f}"
        )
        
        if final_diff < 1e-6:
            self.logger.warning(
                f"⚠️  WARNING: Weight update magnitude very small ({final_diff:.9f}), "
                "weights may not be updating!"
            )
        else:
            self.logger.info("✅ Weight updates confirmed - training is working correctly")


@dataclass
class LearningLoopConfig:
    base_model: str
    adapter_path: pathlib.Path
    buffer_path: pathlib.Path
    log_path: pathlib.Path
    trigger_threshold: int = 1  # Persistent training: train on every sample (curator is the gate)
    quality_threshold: int = 6  # Curator gate: only samples below this score enter training
    generation_endpoint: str = "http://127.0.0.1:8000/v1/completions"
    max_seq_length: int = 2048
    train_epochs: int = 3
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation: int = 16
    warmup_steps: int = 25
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: DEFAULT_TARGET_MODULES.copy()
    )

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "LearningLoopConfig":
        data = tomllib.loads(path.read_text())
        base = data.get("learning_loop", {})
        resolved = {
            "base_model": base.get("base_model", "ibm-granite/granite-3b-code-instruct"),
            "adapter_path": pathlib.Path(base.get("adapter_path", "models/system2_adapters")),
            "buffer_path": pathlib.Path(base.get("buffer_path", "storage/system2_learning_buffer.jsonl")),
            "log_path": pathlib.Path(base.get("log_path", "logs/learning_loop.log")),
            "trigger_threshold": int(base.get("trigger_threshold", 1)),  # Default: persistent training
            "quality_threshold": int(base.get("quality_threshold", 6)),
            "generation_endpoint": base.get("generation_endpoint", "http://127.0.0.1:8000/v1/completions"),
            "max_seq_length": int(base.get("max_seq_length", 2048)),
            "train_epochs": int(base.get("train_epochs", 3)),
            "learning_rate": float(base.get("learning_rate", 1e-4)),
            "train_batch_size": int(base.get("train_batch_size", 1)),
            "gradient_accumulation": int(base.get("gradient_accumulation", 16)),
            "warmup_steps": int(base.get("warmup_steps", 25)),
            "lora_r": int(base.get("lora_r", 16)),
            "lora_alpha": int(base.get("lora_alpha", 16)),
            "lora_dropout": float(base.get("lora_dropout", 0.05)),
            "target_modules": base.get("target_modules", DEFAULT_TARGET_MODULES),
        }
        resolved["target_modules"] = list(resolved["target_modules"])
        return cls(**resolved)


class LearningLoop:
    def __init__(self, config: LearningLoopConfig) -> None:
        self.config = config
        self.config.adapter_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger("learning_loop")
        file_handler = logging.FileHandler(self.config.log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(file_handler)
        self._logger.propagate = False

        self.buffer: List[Dict[str, Any]] = []
        if self.config.buffer_path.exists():
            with self.config.buffer_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self.buffer.append(json.loads(line))
        self._logger.info("loaded %d buffered samples", len(self.buffer))

    # ------------------------------------------------------------------
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        quality = sample.get("quality_score")
        if quality is None:
            self._logger.info("skipping sample – missing curator score")
            return {
                "status": "skipped",
                "reason": "missing-score",
                "buffer_count": len(self.buffer),
            }

        if quality >= self.config.quality_threshold:
            self._logger.info("skipping sample – score %s above threshold", quality)
            return {
                "status": "skipped",
                "reason": "score-ok",
                "buffer_count": len(self.buffer),
            }

        improved = self._generate_revision(sample)
        record = {
            "prompt": sample["prompt"],
            "original_response": sample["response"],
            "feedback": sample.get("feedback", ""),
            "revised_response": improved,
            "quality_score": quality,
            "rouge_l": sample.get("rouge_l"),
            "timestamp": time.time(),
        }
        self.buffer.append(record)
        self._persist_buffer()
        
        # Persistent training mode: train immediately on every sample that passes curator gate
        # The curator (quality_threshold) is the gate, so we train persistently
        triggered = False
        training_summary: Optional[Dict[str, Any]] = None
        
        # Check if we should trigger training (persistent mode: trigger_threshold=1)
        if len(self.buffer) >= self.config.trigger_threshold:
            self._logger.info(
                "persistent training: triggering QLoRA fine-tune (%d buffered, threshold=%d)",
                len(self.buffer),
                self.config.trigger_threshold,
            )
            # Persistent training: submit to async training service if available
            # This allows training to happen in background without blocking the main loop
            training_service_endpoint = os.getenv("TRAINING_SERVICE_ENDPOINT")
            if training_service_endpoint:
                self._logger.info("persistent training: submitting QLoRA job to service (%d sample(s))", len(self.buffer))
                try:
                    import requests
                    job_response = requests.post(
                        f"{training_service_endpoint}/training/jobs/python",
                        json={
                            "buffer_path": str(self.config.buffer_path),
                            "config_path": str(self.config.adapter_path.parent / "learning_loop.toml") if self.config.adapter_path.parent.exists() else "config/learning_loop.toml",
                            "adapter_path": str(self.config.adapter_path),
                        },
                        timeout=10,
                    )
                    job_response.raise_for_status()
                    job_data = job_response.json()
                    training_summary = {
                        "status": "submitted",
                        "job_id": job_data.get("job_id"),
                        "message": "Persistent training job submitted to training service",
                        "buffer_consumed": len(self.buffer),
                        "mode": "persistent",
                    }
                    triggered = True
                    # Clear buffer after successful submission (persistent mode: train immediately, don't accumulate)
                    self.buffer.clear()
                    self._persist_buffer()
                    self._logger.info("persistent training: job submitted, buffer cleared")
                except Exception as e:
                    self._logger.warning("Failed to submit to training service: %s. Falling back to sync training.", e)
                    # Fall back to synchronous training (will block, but ensures training happens)
                    training_summary = self.trigger_qlora_finetune(force=True)
                    triggered = training_summary.get("status") == "trained" if training_summary else False
            else:
                # No training service - use synchronous training (will block main loop)
                self._logger.info("persistent training: triggering QLoRA fine-tune synchronously (%d sample(s))", len(self.buffer))
                training_summary = self.trigger_qlora_finetune(force=True)
                triggered = training_summary.get("status") == "trained" if training_summary else False

        return {
            "status": "queued",
            "buffer_count": len(self.buffer),
            "triggered_training": triggered,
            "training_summary": training_summary,
            "adapter_path": str(self.config.adapter_path) if triggered else None,
            "reason": None,
        }

    def trigger_qlora_finetune(self, *, force: bool = False) -> Dict[str, Any]:
        if not self.buffer:
            self._logger.info("skipping training – buffer empty")
            return {
                "status": "skipped",
                "reason": "empty-buffer",
                "buffer_count": 0,
            }
        if not force and len(self.buffer) < self.config.trigger_threshold:
            self._logger.info(
                "skipping training – buffer %d below threshold %d",
                len(self.buffer),
                self.config.trigger_threshold,
            )
            return {
                "status": "skipped",
                "reason": "threshold-not-met",
                "buffer_count": len(self.buffer),
            }

        start = time.time()
        tokenizer, model = self._prepare_model()
        dataset = self._build_dataset(tokenizer)
        self._logger.info(
            "training with %d examples (epochs=%d batch=%d accum=%d lr=%.2e)",
            len(dataset),
            self.config.train_epochs,
            self.config.train_batch_size,
            self.config.gradient_accumulation,
            self.config.learning_rate,
        )

        precision_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                precision_kwargs["bf16"] = True
            else:
                precision_kwargs["fp16"] = True

        training_args = TrainingArguments(
            output_dir=str(self.config.adapter_path / "tmp"),
            overwrite_output_dir=True,
            num_train_epochs=self.config.train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=1,
            save_strategy="no",
            **precision_kwargs,
        )

        # PHASE 0: Add weight update callback for diagnostics
        weight_callback = WeightUpdateCallback(self._logger)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[weight_callback],
        )
        trainer.train()

        self.config.adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.config.adapter_path)
        tokenizer.save_pretrained(self.config.adapter_path)

        duration = time.time() - start
        summary = {
            "status": "trained",
            "buffer_consumed": len(dataset),
            "duration_sec": duration,
            "adapter_path": str(self.config.adapter_path),
        }
        self._logger.info("training complete in %.1fs", duration)

        self.buffer.clear()
        self._persist_buffer()
        return summary

    # ------------------------------------------------------------------
    def _generate_revision(self, sample: Dict[str, Any]) -> str:
        payload = {
            "model": sample.get("revision_model", self.config.base_model),
            "prompt": self._build_revision_prompt(sample),
            "max_tokens": min(self.config.max_seq_length // 2, 1024),
            "temperature": 0.1,
        }
        response = requests.post(
            self.config.generation_endpoint,
            json=payload,
            timeout=float(os.getenv("NIODOO_REVISION_TIMEOUT", "30")),
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("revision generation returned no choices")
        text = choices[0].get("text", "")
        if not isinstance(text, str):
            raise RuntimeError("revision generation returned non-string text")
        return text.strip()

    def _build_revision_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            "You are NIODOO's adaptive editor. Improve the response using the "
            "curator feedback while preserving factual accuracy."
            "\n<prompt>\n"
            f"{sample['prompt'].strip()}\n"  # noqa: E501
            "</prompt>\n<original>\n"
            f"{sample['response'].strip()}\n"
            "</original>\n<feedback>\n"
            f"{sample.get('feedback', '').strip()}\n"
            "</feedback>\nProvide an improved response only."
        )

    def _prepare_model(self) -> tuple[AutoTokenizer, torch.nn.Module]:
        target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        adapter_config_path = self.config.adapter_path / "adapter_config.json"
        if adapter_config_path.exists():
            self._logger.info("loading existing adapters from %s", self.config.adapter_path)
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.adapter_path,
                device_map="auto",
                dtype=target_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.config.adapter_path, use_fast=True)
            # Enable training mode and gradients for LoRA layers
            model.train()
            # Enable gradient computation for input embeddings (required for LoRA)
            model.enable_input_require_grads()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, model

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            device_map="auto",
            dtype=target_dtype,
        )
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        # Enable training mode
        model.train()
        # Enable gradient computation for input embeddings
        model.enable_input_require_grads()
        return tokenizer, model

    def _build_dataset(self, tokenizer: AutoTokenizer) -> Dataset:
        texts = [
            "### Prompt:\n"
            f"{entry['prompt']}\n\n"
            "### Curator Feedback:\n"
            f"{entry.get('feedback', '')}\n\n"
            "### Target Response:\n"
            f"{entry['revised_response']}"
            for entry in self.buffer
        ]

        dataset = Dataset.from_dict({"text": texts})

        def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            tokens = tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
            )
            tokens["labels"] = [ids[:] for ids in tokens["input_ids"]]
            return tokens

        return dataset.map(_tokenize, batched=True)

    def _persist_buffer(self) -> None:
        with self.config.buffer_path.open("w", encoding="utf-8") as fh:
            for record in self.buffer:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
def _load_config(path_str: str) -> LearningLoopConfig:
    path = pathlib.Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return LearningLoopConfig.from_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="System 2 learning loop controller")
    parser.add_argument(
        "--config",
        default="config/learning_loop.toml",
        help="Path to the learning-loop TOML configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process-sample", help="Queue a sample for training")
    process_parser.add_argument("--sample", required=True, help="Curator sample as JSON string")

    train_parser = subparsers.add_parser("train-now", help="Force training immediately")
    train_parser.add_argument("--force", action="store_true", help="Force even if below threshold")

    subparsers.add_parser("show-buffer", help="Print buffered sample count")

    args = parser.parse_args()
    config = _load_config(args.config)
    loop = LearningLoop(config)

    if args.command == "process-sample":
        sample = json.loads(args.sample)
        result = loop.process_sample(sample)
        print(json.dumps(result, indent=2))
    elif args.command == "train-now":
        result = loop.trigger_qlora_finetune(force=args.force)
        print(json.dumps(result, indent=2))
    elif args.command == "show-buffer":
        print(json.dumps({"buffer_count": len(loop.buffer)}, indent=2))
    else:  # pragma: no cover - argparse guarantees reachable
        raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
