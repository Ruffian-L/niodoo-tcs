"""QLoRA fine-tuning orchestration for topology-aware training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    Trainer,
    TrainingArguments,
)

from geomloss import SamplesLoss

from .config import TopologyTrainingConfig, TrainingConfig
from .data import build_datasets
from .topology import TopologyAugmentor


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_tokenizer(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _bitsandbytes_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    if not config.model.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=config.model.torch_dtype(),
        bnb_4bit_quant_type=config.model.quant_type,
    )


def _load_base_model(config: TrainingConfig, for_training: bool) -> torch.nn.Module:
    quant_config = _bitsandbytes_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        torch_dtype=config.model.torch_dtype(),
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quant_config,
        device_map="auto",
    )
    if for_training and config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def _lora_config(config: TrainingConfig) -> PeftLoraConfig:
    return PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )


def _training_arguments(config: TrainingConfig) -> TrainingArguments:
    runtime = config.runtime
    return TrainingArguments(
        output_dir=str(runtime.output_dir),
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_ratio=config.optimizer.warmup_ratio,
        num_train_epochs=config.optimizer.num_train_epochs,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        max_grad_norm=config.optimizer.max_grad_norm,
        logging_steps=runtime.logging_steps,
        evaluation_strategy=runtime.evaluation_strategy,
        eval_steps=runtime.eval_steps,
        save_steps=runtime.save_steps,
        bf16=runtime.bf16,
        fp16=not runtime.bf16,
        seed=runtime.seed,
        report_to=["none"],
    )


def _topology_data_collator(features):
    metadata = [feature.pop("topology_metadata", {}) for feature in features]
    teacher_keys = [feature.pop("teacher_key", None) for feature in features]
    diagrams = [feature.pop("topology_diagrams", None) for feature in features]
    diagram_summary = [feature.pop("diagram_summary", None) for feature in features]
    batch = default_data_collator(features)
    batch["topology_metadata"] = metadata
    batch["teacher_key"] = teacher_keys
    batch["topology_diagrams"] = diagrams
    batch["diagram_summary"] = diagram_summary
    return batch


class TopologyAwareTrainer(Trainer):
    def __init__(
        self,
        *args,
        topology_config: Optional[TopologyTrainingConfig] = None,
        topology_head: Optional[torch.nn.Module] = None,
        teacher_cache: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topology_config = topology_config
        self.topology_head = topology_head
        self.teacher_cache = teacher_cache or {}
        self.lambda_topology = topology_config.lambda_weight if topology_config else 0.0
        self.lambda_teacher = topology_config.lambda_teacher if topology_config else 0.0
        self.lambda_sinkhorn = topology_config.lambda_sinkhorn if topology_config else 0.0
        self.max_sinkhorn_points = topology_config.max_sinkhorn_points if topology_config else 0

        self._sinkhorn_loss = None
        if self.lambda_sinkhorn > 0 and topology_config is not None:
            self._sinkhorn_loss = SamplesLoss(
                loss="sinkhorn",
                p=topology_config.sinkhorn_p,
                blur=topology_config.sinkhorn_blur,
                scaling=topology_config.sinkhorn_scaling,
                backend="auto",
            )

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.topology_config:
            return hidden_states.mean(dim=1)
        if self.topology_config.projection == "cls":
            return hidden_states[:, 0]
        return hidden_states.mean(dim=1)

    def _teacher_vector_loss(self, predictions: torch.Tensor, teacher_keys: Optional[List[Optional[str]]]) -> torch.Tensor:
        if not teacher_keys:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        matched_pred = []
        matched_target = []
        for idx, key in enumerate(teacher_keys):
            if not key:
                continue
            entry = self.teacher_cache.get(str(key))
            if entry is None:
                # Attempt to resolve basename for filesystem keys
                candidate = Path(str(key)).name
                entry = self.teacher_cache.get(candidate)
            vector: Optional[torch.Tensor]
            if isinstance(entry, dict):
                vector = entry.get("vector")  # type: ignore[index]
            else:
                vector = entry  # Backwards compatibility
            if vector is None:
                continue
            matched_pred.append(predictions[idx])
            matched_target.append(vector.to(predictions.device, dtype=predictions.dtype))
        if not matched_pred:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        pred_batch = torch.stack(matched_pred, dim=0)
        target_batch = torch.stack(matched_target, dim=0)
        return F.mse_loss(pred_batch, target_batch)

    def _prepare_sinkhorn_points(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 2:
            tensor = tensor.view(tensor.size(0), -1)
        if self.max_sinkhorn_points and tensor.size(1) > self.max_sinkhorn_points:
            device = tensor.device
            indices = torch.linspace(0, tensor.size(1) - 1, steps=self.max_sinkhorn_points, device=device)
            indices = indices.round().long()
            tensor = tensor.index_select(1, indices)
        return tensor.unsqueeze(-1)

    def _sinkhorn_distance(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._sinkhorn_loss is None:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        pred_points = self._prepare_sinkhorn_points(predictions)
        target_points = self._prepare_sinkhorn_points(targets)
        return self._sinkhorn_loss(pred_points, target_points)

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        _ = inputs.pop("topology_metadata", None)
        teacher_keys = inputs.pop("teacher_key", None)
        topology_targets = inputs.pop("topology_vector", None)
        _ = inputs.pop("topology_diagrams", None)
        _ = inputs.pop("diagram_summary", None)

        inputs = self._prepare_inputs(inputs)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        lm_loss = outputs.loss

        total_loss = lm_loss
        loss_dict = {"loss_lm": lm_loss.detach()}

        if topology_targets is not None and self.lambda_topology > 0:
            assert self.topology_head is not None, "Topology head must be initialised for topology loss"
            hidden_states = outputs.hidden_states[-1]
            pooled = self._pool_hidden_states(hidden_states)
            predictions = self.topology_head(pooled)
            topology_targets = topology_targets.to(predictions.device, dtype=predictions.dtype)
            topo_loss = F.mse_loss(predictions, topology_targets)
            total_loss = total_loss + self.lambda_topology * topo_loss
            loss_dict["loss_topology"] = topo_loss.detach()

            if self.lambda_teacher > 0:
                teacher_loss = self._teacher_vector_loss(predictions, teacher_keys)
                total_loss = total_loss + self.lambda_teacher * teacher_loss
                loss_dict["loss_teacher"] = teacher_loss.detach()

            if self.lambda_sinkhorn > 0 and self._sinkhorn_loss is not None:
                sinkhorn_loss = self._sinkhorn_distance(predictions, topology_targets)
                total_loss = total_loss + self.lambda_sinkhorn * sinkhorn_loss
                loss_dict["loss_sinkhorn"] = sinkhorn_loss.detach()

        if return_outputs:
            outputs.losses = loss_dict  # type: ignore[attr-defined]
            return total_loss, outputs
        return total_loss

    def compute_alignment_metrics(self, dataloader) -> Dict[str, float]:
        if self.topology_head is None or dataloader is None:
            return {}

        model = self.model
        training_mode = model.training
        model.eval()

        device = next(model.parameters()).device
        dtype = self.topology_head.weight.dtype

        sinkhorn = self._sinkhorn_loss
        if sinkhorn is None and self.topology_config is not None:
            sinkhorn = SamplesLoss(
                loss="sinkhorn",
                p=self.topology_config.sinkhorn_p,
                blur=self.topology_config.sinkhorn_blur,
                scaling=self.topology_config.sinkhorn_scaling,
                backend="auto",
            )

        total = 0
        mse_total = 0.0
        sink_total = 0.0
        betti_sq_errors: List[float] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = dict(batch)
                metadata = batch.pop("topology_metadata", None)
                batch.pop("teacher_key", None)
                batch.pop("topology_diagrams", None)
                batch.pop("diagram_summary", None)

                topology_targets = batch.pop("topology_vector", None)
                if topology_targets is None:
                    continue
                topology_targets = topology_targets.to(device=device, dtype=dtype)

                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                inputs.pop("labels", None)  # labels unused for forward metrics

                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1]
                pooled = self._pool_hidden_states(hidden_states)
                predictions = self.topology_head(pooled)

                batch_size = topology_targets.size(0)
                total += batch_size

                mse_total += F.mse_loss(predictions, topology_targets, reduction="mean").item() * batch_size

                if sinkhorn is not None:
                    pred_points = self._prepare_sinkhorn_points(predictions)
                    target_points = self._prepare_sinkhorn_points(topology_targets)
                    sink_value = sinkhorn(pred_points, target_points).item()
                    sink_total += sink_value * batch_size

                if isinstance(metadata, list):
                    for idx, meta in enumerate(metadata):
                        if not isinstance(meta, dict):
                            continue
                        betti = meta.get("betti_numbers")
                        if isinstance(betti, dict) and betti:
                            target_sum = float(sum(betti.values()))
                            pred_sum = float(torch.relu(predictions[idx]).sum().item())
                            betti_sq_errors.append((pred_sum - target_sum) ** 2)

        if training_mode:
            model.train()

        metrics: Dict[str, float] = {}
        if total > 0:
            metrics["vector_mse"] = mse_total / total
            if sinkhorn is not None:
                metrics["sinkhorn_distance"] = sink_total / total
        if betti_sq_errors:
            metrics["betti_sum_rmse"] = (sum(betti_sq_errors) / len(betti_sq_errors)) ** 0.5
        return metrics


def _load_teacher_cache(config: TopologyTrainingConfig) -> Dict[str, Dict[str, torch.Tensor]]:
    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    path = config.teacher_cache
    if path is None:
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            vector = record.get("vector") or record.get("values")
            if vector is None:
                continue
            tensor = torch.tensor(vector, dtype=torch.float32)
            entry: Dict[str, torch.Tensor] = {"vector": tensor}
            diagrams = record.get("persistence_diagrams")
            if isinstance(diagrams, dict):
                flat_points: List[List[float]] = []
                for values in diagrams.values():
                    if not isinstance(values, list):
                        continue
                    for pair in values:
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            flat_points.append([float(pair[0]), float(pair[1])])
                if flat_points:
                    entry["diagram_points"] = torch.tensor(flat_points, dtype=torch.float32)
            keys: List[str] = []
            if config.teacher_match_field in record:
                keys.append(str(record[config.teacher_match_field]))
            if record.get("source_path"):
                resolved = str(Path(record["source_path"]).resolve())
                keys.extend([record["source_path"], resolved])
            if record.get("id"):
                keys.append(str(record["id"]))
            for key in keys:
                if key:
                    cache[key] = entry
    return cache


def run_training(config: TrainingConfig) -> None:
    _ensure_dir(config.runtime.output_dir)
    torch.manual_seed(config.runtime.seed)

    tokenizer = create_tokenizer(config)
    augmentor = TopologyAugmentor()
    train_dataset, eval_dataset = build_datasets(config.data, tokenizer, augmentor)

    model = _load_base_model(config, for_training=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, _lora_config(config))

    topology_head = None
    teacher_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    data_collator = _topology_data_collator
    topology_config = config.topology

    if topology_config is not None and len(train_dataset) > 0:
        topology_dim = int(train_dataset[0]["topology_vector"].shape[-1])
        hidden_size = model.config.hidden_size if hasattr(model, "config") else topology_dim
        base_param = next(model.parameters())
        topology_head = torch.nn.Linear(hidden_size, topology_dim, dtype=base_param.dtype)
        topology_head.to(base_param.device)
        model.add_module("topology_head", topology_head)
        if topology_config.teacher_cache is not None and topology_config.teacher_cache.exists():
            teacher_cache = _load_teacher_cache(topology_config)

    training_args = _training_arguments(config)
    trainer = TopologyAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        topology_config=topology_config,
        topology_head=topology_head,
        teacher_cache=teacher_cache,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(config.runtime.output_dir)

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        with (config.runtime.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


def evaluate_model(config: TrainingConfig, model_path: Optional[Path | str] = None) -> dict:
    model_dir = Path(model_path) if model_path else config.runtime.output_dir
    tokenizer = create_tokenizer(config)
    augmentor = TopologyAugmentor()
    _, eval_dataset = build_datasets(config.data, tokenizer, augmentor)
    if eval_dataset is None:
        raise ValueError("Evaluation dataset is not configured")

    base_model = _load_base_model(config, for_training=False)
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    training_args = _training_arguments(config)
    topology_head = getattr(model, "topology_head", None)
    teacher_cache = None
    if config.topology and config.topology.teacher_cache and config.topology.teacher_cache.exists():
        teacher_cache = _load_teacher_cache(config.topology)

    trainer = TopologyAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=_topology_data_collator,
        topology_config=config.topology,
        topology_head=topology_head,
        teacher_cache=teacher_cache,
    )
    metrics = trainer.evaluate()
    if isinstance(trainer, TopologyAwareTrainer):
        alignment_metrics = trainer.compute_alignment_metrics(trainer.get_eval_dataloader())
        metrics.update({f"topology_{k}": v for k, v in alignment_metrics.items()})
        paraphrase_metrics = compute_paraphrase_stability(model, tokenizer, config.topology)
        metrics.update(paraphrase_metrics)
    return metrics


def _point_cloud_from_vector(vector: torch.Tensor, max_points: int) -> torch.Tensor:
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
    if max_points and vector.size(1) > max_points:
        device = vector.device
        indices = torch.linspace(0, vector.size(1) - 1, steps=max_points, device=device)
        indices = indices.round().long()
        vector = vector.index_select(1, indices)
    return vector.unsqueeze(-1)


def _encode_topology_vector(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    topology_config: Optional[TopologyTrainingConfig],
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]
        if topology_config and topology_config.projection == "cls":
            pooled = hidden_states[:, 0]
        else:
            pooled = hidden_states.mean(dim=1)
        head = getattr(model, "topology_head", None)
        if head is None:
            raise ValueError("Model is missing topology_head; re-run training with topology loss enabled.")
        vector = head(pooled)
    return vector.squeeze(0)


def _default_paraphrase_pairs() -> List[tuple[str, str]]:
    return [
        ("The cat sat on the mat.", "On the mat sat the cat."),
        ("A torus has two fundamental loops.", "The doughnut-shaped torus carries two independent cycles."),
        ("Spheres have no holes.", "There are zero holes in a sphere."),
    ]


def compute_paraphrase_stability(
    model: torch.nn.Module,
    tokenizer,
    topology_config: Optional[TopologyTrainingConfig],
    threshold: float = 0.2,
) -> Dict[str, float]:
    if topology_config is None or not hasattr(model, "topology_head"):
        return {}

    device = next(model.parameters()).device
    blur = topology_config.sinkhorn_blur
    scaling = topology_config.sinkhorn_scaling
    p = topology_config.sinkhorn_p
    max_points = topology_config.max_sinkhorn_points

    sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, scaling=scaling, backend="auto")

    distances: List[float] = []
    stable = 0

    with torch.no_grad():
        for original, paraphrase in _default_paraphrase_pairs():
            vec_a = _encode_topology_vector(model, tokenizer, original, topology_config, device)
            vec_b = _encode_topology_vector(model, tokenizer, paraphrase, topology_config, device)

            points_a = _point_cloud_from_vector(vec_a, max_points)
            points_b = _point_cloud_from_vector(vec_b, max_points)

            distance = float(sinkhorn(points_a, points_b).item())
            distances.append(distance)
            if distance <= threshold:
                stable += 1

    if not distances:
        return {}

    mean_distance = sum(distances) / len(distances)
    max_distance = max(distances)
    stability = stable / len(distances)

    return {
        "paraphrase_stability": stability,
        "paraphrase_mean_distance": mean_distance,
        "paraphrase_max_distance": max_distance,
    }

