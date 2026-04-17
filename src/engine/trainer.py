from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.datasets import SUPPORTED_DATASETS
from src.datasets.manifest_builder import (
    build_output_paths,
    iter_manifest_jsonl,
    read_json,
    read_text_lines,
    sha256_of_file,
    write_json,
)
from src.losses import LossConfig
from src.models import (
    ChebyKANConfig,
    FormalHashModel,
    GraphRefineConfig,
    PredictorConfig,
    SemanticTreeConfig,
)

from .checkpoint_io import load_checkpoint
from .logger import read_jsonl


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _resolve_path(base_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _sample_id_digest_from_manifest(manifest_filtered_path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    line_count = 0
    with manifest_filtered_path.open("r", encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"sample_id is missing or invalid in manifest_filtered.jsonl at index {line_count - 1}"
                )
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\n")
    return line_count, digest.hexdigest()


def _ensure_cuda_protocol(device: str, dtype: str) -> torch.device:
    if device != "cuda:0":
        raise ValueError(f"Formal Stage 4 device must be cuda:0, got {device}.")
    if dtype != "float32":
        raise ValueError(f"Formal Stage 4 dtype must be float32, got {dtype}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Stage 4 requires cuda:0, but CUDA is unavailable.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Formal Stage 4 requires cuda:0, but no CUDA device is visible.")
    return torch.device(device)


def _ensure_finite_tensor(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


@dataclass(frozen=True)
class TrainConfig:
    stage_name: str
    processed_root: Path
    outputs_root: Path
    feature_cache_id: str
    semantic_set_id: str
    device: str
    dtype: str
    amp_enabled: bool
    multi_gpu: bool
    run_name_prefix: str
    seed: int
    batch_size: int
    num_epochs: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    shuffle: bool
    log_every: int
    best_metric: str
    validation_batch_size: int
    preferred_checkpoint: str
    model_chebykan: Dict[str, Any]
    model_tree: Dict[str, Any]
    model_graph: Dict[str, Any]
    hash_head: Dict[str, Any]
    loss: Dict[str, Any]
    config_path: Path

    def build_output_dir(self, dataset_name: str, run_name: str) -> Path:
        return (self.outputs_root / dataset_name / run_name).resolve()

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "paths": {
                "processed_root": str(self.processed_root),
                "outputs_root": str(self.outputs_root),
            },
            "inputs": {
                "feature_cache_id": self.feature_cache_id,
                "semantic_set_id": self.semantic_set_id,
            },
            "runtime": {
                "device": self.device,
                "dtype": self.dtype,
                "amp_enabled": self.amp_enabled,
                "multi_gpu": self.multi_gpu,
            },
            "training": {
                "run_name_prefix": self.run_name_prefix,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "num_workers": self.num_workers,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "grad_clip_norm": self.grad_clip_norm,
                "shuffle": self.shuffle,
                "log_every": self.log_every,
                "best_metric": self.best_metric,
            },
            "validation": {
                "batch_size": self.validation_batch_size,
                "preferred_checkpoint": self.preferred_checkpoint,
            },
            "model_chebykan": self.model_chebykan,
            "model_tree": self.model_tree,
            "model_graph": self.model_graph,
            "hash_head": self.hash_head,
            "loss": self.loss,
        }


@dataclass(frozen=True)
class Stage4InputContext:
    dataset_name: str
    manifest_filtered_path: Path
    manifest_filtered_sha256: str
    sample_id_order_sha256: str
    filtered_count: int
    feature_cache_dir: Path
    semantic_cache_dir: Path
    X_I_path: Path
    X_T_path: Path
    S_path: Path
    stage2_meta: Dict[str, Any]
    stage2_validator: Dict[str, Any]
    stage3_meta: Dict[str, Any]
    stage3_validator: Dict[str, Any]
    train_ids: list[str]
    train_indices: list[int]
    feature_dim_image: int
    feature_dim_text: int


class FormalTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        X_I_path: Path,
        X_T_path: Path,
        train_indices: Sequence[int],
        train_ids: Sequence[str],
    ) -> None:
        self.X_I = np.load(X_I_path, mmap_mode="r")
        self.X_T = np.load(X_T_path, mmap_mode="r")
        self.train_indices = list(train_indices)
        self.train_ids = list(train_ids)
        if self.X_I.shape[0] != self.X_T.shape[0]:
            raise ValueError("X_I and X_T row counts do not match.")
        if len(self.train_indices) != len(self.train_ids):
            raise ValueError("train_indices and train_ids length mismatch.")

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        manifest_index = int(self.train_indices[index])
        return {
            "sample_id": self.train_ids[index],
            "manifest_index": manifest_index,
            "X_I": torch.from_numpy(self.X_I[manifest_index].astype(np.float32, copy=True)),
            "X_T": torch.from_numpy(self.X_T[manifest_index].astype(np.float32, copy=True)),
        }


def load_train_config(project_root: Path, config_path: Path) -> TrainConfig:
    payload = load_yaml(config_path)
    config = TrainConfig(
        stage_name=str(payload["stage_name"]),
        processed_root=_resolve_path(project_root, str(payload["paths"]["processed_root"])),
        outputs_root=_resolve_path(project_root, str(payload["paths"]["outputs_root"])),
        feature_cache_id=str(payload["inputs"]["feature_cache_id"]),
        semantic_set_id=str(payload["inputs"]["semantic_set_id"]),
        device=str(payload["runtime"]["device"]),
        dtype=str(payload["runtime"]["dtype"]),
        amp_enabled=bool(payload["runtime"]["amp_enabled"]),
        multi_gpu=bool(payload["runtime"]["multi_gpu"]),
        run_name_prefix=str(payload["training"]["run_name_prefix"]),
        seed=int(payload["training"]["seed"]),
        batch_size=int(payload["training"]["batch_size"]),
        num_epochs=int(payload["training"]["num_epochs"]),
        num_workers=int(payload["training"]["num_workers"]),
        learning_rate=float(payload["training"]["learning_rate"]),
        weight_decay=float(payload["training"]["weight_decay"]),
        grad_clip_norm=float(payload["training"]["grad_clip_norm"]),
        shuffle=bool(payload["training"]["shuffle"]),
        log_every=int(payload["training"]["log_every"]),
        best_metric=str(payload["training"]["best_metric"]),
        validation_batch_size=int(payload["validation"]["batch_size"]),
        preferred_checkpoint=str(payload["validation"]["preferred_checkpoint"]),
        model_chebykan=load_yaml(_resolve_path(project_root, str(payload["configs"]["model_chebykan"]))),
        model_tree=load_yaml(_resolve_path(project_root, str(payload["configs"]["model_tree"]))),
        model_graph=load_yaml(_resolve_path(project_root, str(payload["configs"]["model_graph"]))),
        hash_head=load_yaml(_resolve_path(project_root, str(payload["configs"]["hash_head"]))),
        loss=load_yaml(_resolve_path(project_root, str(payload["configs"]["loss"]))),
        config_path=config_path,
    )
    if config.amp_enabled:
        raise ValueError("Formal Stage 4 forbids mixed precision.")
    if config.multi_gpu:
        raise ValueError("Formal Stage 4 forbids multi-GPU execution.")
    if config.feature_cache_id != "clip_vit_b32_formal_v1":
        raise ValueError("Formal Stage 4 feature_cache_id must be clip_vit_b32_formal_v1.")
    if config.semantic_set_id != "semantic_relation_highsignal_v1":
        raise ValueError("Formal Stage 4 semantic_set_id must be semantic_relation_highsignal_v1.")
    return config


def _manifest_sample_ids(manifest_filtered_path: Path) -> list[str]:
    sample_ids: list[str] = []
    for record in iter_manifest_jsonl(manifest_filtered_path):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("manifest_filtered.jsonl contains an invalid sample_id.")
        sample_ids.append(sample_id)
    return sample_ids


def _load_stage4_input_context(config: TrainConfig, dataset_name: str) -> Stage4InputContext:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset for Stage 4: {dataset_name}")
    stage1_paths = build_output_paths(config.processed_root, dataset_name)
    manifest_meta = read_json(stage1_paths.manifest_meta_path)
    stage2_dir = config.processed_root / dataset_name / "feature_cache" / config.feature_cache_id
    stage3_dir = config.processed_root / dataset_name / "semantic_cache" / config.semantic_set_id
    required_paths = (
        stage1_paths.manifest_filtered_path,
        stage1_paths.train_ids_path,
        stage2_dir / "X_I.npy",
        stage2_dir / "X_T.npy",
        stage2_dir / "meta.json",
        stage2_dir / "validator_summary.json",
        stage3_dir / "S.npz",
        stage3_dir / "meta.json",
        stage3_dir / "validator_summary.json",
    )
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing required Stage 4 formal input: {required_path}")

    filtered_count = int(manifest_meta["counts"]["filtered_record_count"])
    observed_count, observed_order_sha256 = _sample_id_digest_from_manifest(stage1_paths.manifest_filtered_path)
    observed_manifest_sha256 = sha256_of_file(stage1_paths.manifest_filtered_path)
    if observed_count != filtered_count:
        raise RuntimeError("Stage 4 manifest count drift detected.")

    stage2_meta = read_json(stage2_dir / "meta.json")
    stage2_validator = read_json(stage2_dir / "validator_summary.json")
    stage3_meta = read_json(stage3_dir / "meta.json")
    stage3_validator = read_json(stage3_dir / "validator_summary.json")
    if bool(stage2_validator.get("validator_passed")) is not True:
        raise RuntimeError(f"Stage 2 validator did not pass for {dataset_name}.")
    if bool(stage3_validator.get("validator_passed")) is not True:
        raise RuntimeError(f"Stage 3 validator did not pass for {dataset_name}.")
    if observed_order_sha256 != stage2_meta.get("sample_id_order_sha256"):
        raise RuntimeError("Stage 4 manifest order hash does not match Stage 2.")
    if observed_order_sha256 != stage3_meta.get("sample_id_order_sha256"):
        raise RuntimeError("Stage 4 manifest order hash does not match Stage 3.")
    if observed_manifest_sha256 != stage2_meta.get("manifest_filtered_sha256"):
        raise RuntimeError("Stage 4 manifest sha256 does not match Stage 2.")
    if observed_manifest_sha256 != stage3_meta.get("input_manifest_filtered_sha256"):
        raise RuntimeError("Stage 4 manifest sha256 does not match Stage 3.")
    if int(stage2_meta.get("filtered_count")) != filtered_count:
        raise RuntimeError("Stage 2 filtered_count drift detected.")
    if int(stage3_meta.get("filtered_count")) != filtered_count:
        raise RuntimeError("Stage 3 filtered_count drift detected.")
    if stage2_meta.get("feature_set_id") != config.feature_cache_id:
        raise RuntimeError("Stage 2 feature_set_id does not match Stage 4 config.")
    if stage3_meta.get("semantic_set_id") != config.semantic_set_id:
        raise RuntimeError("Stage 3 semantic_set_id does not match Stage 4 config.")
    if stage2_meta.get("device") != "cuda:0":
        raise RuntimeError("Stage 2 meta device is not cuda:0.")
    if stage3_meta.get("feature_meta", {}).get("stage_2_manifest_filtered_sha256") != observed_manifest_sha256:
        raise RuntimeError("Stage 3 feature_meta manifest sha256 does not match Stage 1.")
    if stage3_meta.get("feature_meta", {}).get("stage_2_sample_id_order_sha256") != observed_order_sha256:
        raise RuntimeError("Stage 3 feature_meta sample_id_order_sha256 does not match Stage 1.")

    train_ids = read_text_lines(stage1_paths.train_ids_path)
    manifest_sample_ids = _manifest_sample_ids(stage1_paths.manifest_filtered_path)
    manifest_index_by_id = {sample_id: index for index, sample_id in enumerate(manifest_sample_ids)}
    if len(manifest_index_by_id) != len(manifest_sample_ids):
        raise RuntimeError("manifest_filtered.jsonl contains duplicate sample_id values.")
    train_indices: list[int] = []
    for sample_id in train_ids:
        if sample_id not in manifest_index_by_id:
            raise RuntimeError(f"train_ids.txt contains sample_id not present in manifest: {sample_id}")
        train_indices.append(int(manifest_index_by_id[sample_id]))

    feature_dim_image = int(stage2_meta["feature_dim_image"])
    feature_dim_text = int(stage2_meta["feature_dim_text"])
    if feature_dim_image != int(config.model_chebykan["input_dim_image"]):
        raise RuntimeError("model_chebykan.input_dim_image does not match Stage 2 feature_dim_image.")
    if feature_dim_text != int(config.model_chebykan["input_dim_text"]):
        raise RuntimeError("model_chebykan.input_dim_text does not match Stage 2 feature_dim_text.")
    return Stage4InputContext(
        dataset_name=dataset_name,
        manifest_filtered_path=stage1_paths.manifest_filtered_path,
        manifest_filtered_sha256=observed_manifest_sha256,
        sample_id_order_sha256=observed_order_sha256,
        filtered_count=filtered_count,
        feature_cache_dir=stage2_dir.resolve(),
        semantic_cache_dir=stage3_dir.resolve(),
        X_I_path=(stage2_dir / "X_I.npy").resolve(),
        X_T_path=(stage2_dir / "X_T.npy").resolve(),
        S_path=(stage3_dir / "S.npz").resolve(),
        stage2_meta=stage2_meta,
        stage2_validator=stage2_validator,
        stage3_meta=stage3_meta,
        stage3_validator=stage3_validator,
        train_ids=train_ids,
        train_indices=train_indices,
        feature_dim_image=feature_dim_image,
        feature_dim_text=feature_dim_text,
    )


def build_model(config: TrainConfig) -> FormalHashModel:
    chebykan_config = ChebyKANConfig(
        input_dim_image=int(config.model_chebykan["input_dim_image"]),
        input_dim_text=int(config.model_chebykan["input_dim_text"]),
        d_z=int(config.model_chebykan["d_z"]),
        polynomial_order=int(config.model_chebykan["polynomial_order"]),
        hidden_dims=tuple(int(value) for value in config.model_chebykan["hidden_dims"]),
        normalize_inputs=bool(config.model_chebykan["normalize_inputs"]),
        input_clip_value=float(config.model_chebykan["input_clip_value"]),
        eps=float(config.model_chebykan["eps"]),
    )
    tree_config = SemanticTreeConfig(
        tree_depth=int(config.model_tree["tree_depth"]),
        prototype_counts=tuple(int(value) for value in config.model_tree["prototype_counts"]),
        feature_dim=int(config.model_tree["feature_dim"]),
        dropout=float(config.model_tree["dropout"]),
        eps=float(config.model_tree["eps"]),
    )
    graph_config = GraphRefineConfig(
        input_dim=int(config.model_graph["input_dim"]),
        f_dim=int(config.model_graph["f_dim"]),
        hash_bits=int(config.hash_head["hash_bits"]),
        k_neighbors=int(config.model_graph["k_neighbors"]),
        propagation_steps=int(config.model_graph["propagation_steps"]),
        self_loop_weight=float(config.model_graph["self_loop_weight"]),
        residual_weight=float(config.model_graph["residual_weight"]),
        similarity_eps=float(config.model_graph["similarity_eps"]),
    )
    if chebykan_config.d_z != tree_config.feature_dim:
        raise RuntimeError("ChebyKAN d_z must match SemanticTree feature_dim.")
    if tree_config.feature_dim != graph_config.input_dim:
        raise RuntimeError("SemanticTree feature_dim must match GraphRefine input_dim.")
    return FormalHashModel(
        chebykan_config=chebykan_config,
        tree_config=tree_config,
        graph_config=graph_config,
        predictor_config=PredictorConfig(relation_eps=float(config.hash_head["relation_eps"])),
    )


def build_loss_config(config: TrainConfig) -> LossConfig:
    return LossConfig(
        alpha=float(config.loss["alpha"]),
        lambda_sem=float(config.loss["lambda_sem"]),
        lambda_pair=float(config.loss["lambda_pair"]),
        lambda_q=float(config.loss["lambda_q"]),
        lambda_bal=float(config.loss["lambda_bal"]),
        bce_eps=float(config.loss["bce_eps"]),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_sparse_supervision(S_path: Path) -> csr_matrix:
    matrix = load_npz(S_path)
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()
    return matrix


def _gather_batch_S(S_matrix: csr_matrix, manifest_indices: Sequence[int]) -> np.ndarray:
    if len(set(int(value) for value in manifest_indices)) != len(manifest_indices):
        raise RuntimeError("Batch manifest indices must be unique.")
    batch_matrix = S_matrix[manifest_indices][:, manifest_indices]
    if not isinstance(batch_matrix, csr_matrix):
        batch_matrix = batch_matrix.tocsr()
    dense = batch_matrix.toarray().astype(np.float32, copy=False)
    if not np.isfinite(dense).all():
        raise RuntimeError("Batch S gather produced NaN or Inf.")
    return dense


def _make_batch_tensors(batch: dict[str, Any], S_matrix: csr_matrix, device: torch.device) -> dict[str, Tensor]:
    manifest_indices = [int(value) for value in batch["manifest_index"].tolist()]
    X_I = batch["X_I"].to(device=device, dtype=torch.float32, non_blocking=False)
    X_T = batch["X_T"].to(device=device, dtype=torch.float32, non_blocking=False)
    S_batch = torch.from_numpy(_gather_batch_S(S_matrix, manifest_indices)).to(device=device, dtype=torch.float32)
    _ensure_finite_tensor("X_I", X_I)
    _ensure_finite_tensor("X_T", X_T)
    _ensure_finite_tensor("S", S_batch)
    return {"X_I": X_I, "X_T": X_T, "S": S_batch}


def _float_metrics(loss_outputs: Dict[str, Tensor]) -> Dict[str, float]:
    metric_names = ("L_IT", "L_II", "L_TT", "L_sem", "L_pair", "L_q", "L_bal", "loss_total")
    return {name: float(loss_outputs[name].detach().item()) for name in metric_names}


def _accumulate_metrics(accumulator: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value)


def _average_metrics(accumulator: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: float(value / float(count)) for key, value in accumulator.items()}


def _default_run_name(config: TrainConfig, dataset_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.run_name_prefix}_{dataset_name}_{timestamp}"


def _select_checkpoint_path(run_dir: Path, preferred_checkpoint: str) -> Path:
    preferred_path = run_dir / preferred_checkpoint
    if preferred_path.is_file():
        return preferred_path
    fallback_path = run_dir / "last.pt"
    if fallback_path.is_file():
        return fallback_path
    raise FileNotFoundError(f"Neither {preferred_path} nor {fallback_path} exists.")


def _write_validator_section(run_dir: Path, section_name: str, summary: Dict[str, Any]) -> None:
    validator_summary_path = run_dir / "validator_summary.json"
    payload: Dict[str, Any] = {}
    if validator_summary_path.is_file():
        payload = read_json(validator_summary_path)
    payload[section_name] = summary
    write_json(validator_summary_path, payload)


def run_formal_training(
    config: TrainConfig,
    dataset_name: str,
    run_name: str | None = None,
) -> Dict[str, Any]:
    from torch.nn.utils import clip_grad_norm_

    from src.losses import compute_total_loss

    device = _ensure_cuda_protocol(config.device, config.dtype)
    input_context = _load_stage4_input_context(config, dataset_name)
    if len(input_context.train_indices) == 0:
        raise RuntimeError("train_ids.txt is empty.")
    if len(input_context.train_indices) % config.batch_size != 0:
        raise RuntimeError("Formal Stage 4 requires train_count to be divisible by batch_size.")
    if config.batch_size <= int(config.model_graph["k_neighbors"]):
        raise RuntimeError("batch_size must be greater than graph k_neighbors.")

    run_name = run_name or _default_run_name(config, dataset_name)
    run_dir = config.build_output_dir(dataset_name, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot = config.to_snapshot()
    write_json(run_dir / "config_snapshot.json", config_snapshot)
    S_matrix = _load_sparse_supervision(input_context.S_path)
    dataset = FormalTrainDataset(
        X_I_path=input_context.X_I_path,
        X_T_path=input_context.X_T_path,
        train_indices=input_context.train_indices,
        train_ids=input_context.train_ids,
    )
    _set_seed(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=False,
        generator=generator,
    )
    model = build_model(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_config = build_loss_config(config)
    from .checkpoint_io import save_checkpoint
    from .logger import JsonlLogger

    logger = JsonlLogger(run_dir / "train_log.jsonl")
    best_metric_value: float | None = None
    best_epoch = 0
    global_step = 0
    first_batch_model_summary: Dict[str, Any] | None = None
    epoch_summaries: list[Dict[str, Any]] = []

    for epoch_index in range(1, config.num_epochs + 1):
        model.train()
        epoch_metric_sums: Dict[str, float] = {}
        step_count = 0
        for batch_index, batch in enumerate(dataloader, start=1):
            batch_tensors = _make_batch_tensors(batch, S_matrix, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_tensors["X_I"], batch_tensors["X_T"])
            loss_outputs = compute_total_loss(
                P_IT=outputs["P_IT"],
                P_II=outputs["P_II"],
                P_TT=outputs["P_TT"],
                S=batch_tensors["S"],
                H_I=outputs["H_I"],
                H_T=outputs["H_T"],
                config=loss_config,
            )
            loss_outputs["loss_total"].backward()
            grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm).item())
            if not np.isfinite(grad_norm):
                raise RuntimeError("Gradient norm is NaN or Inf.")
            optimizer.step()

            metrics = _float_metrics(loss_outputs)
            metrics["grad_norm"] = grad_norm
            _accumulate_metrics(epoch_metric_sums, metrics)
            global_step += 1
            step_count += 1
            if first_batch_model_summary is None:
                first_batch_model_summary = outputs["model_summary"]
            logger.log(
                {
                    "event": "train_step",
                    "dataset": dataset_name,
                    "run_name": run_name,
                    "epoch": epoch_index,
                    "step": batch_index,
                    "global_step": global_step,
                    "batch_size": int(batch_tensors["X_I"].shape[0]),
                    **metrics,
                }
            )

        if step_count == 0:
            raise RuntimeError("Training produced zero steps.")
        epoch_metrics = _average_metrics(epoch_metric_sums, step_count)
        epoch_summary = {
            "event": "train_epoch",
            "dataset": dataset_name,
            "run_name": run_name,
            "epoch": epoch_index,
            "step_count": step_count,
            **epoch_metrics,
        }
        logger.log(epoch_summary)
        epoch_summaries.append(epoch_summary)
        checkpoint_payload = {
            "dataset": dataset_name,
            "run_name": run_name,
            "epoch": epoch_index,
            "global_step": global_step,
            "device": config.device,
            "dtype": config.dtype,
            "config_snapshot": config_snapshot,
            "epoch_metrics": epoch_metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_summary": first_batch_model_summary,
        }
        save_checkpoint(run_dir / "last.pt", checkpoint_payload)
        current_metric = float(epoch_metrics["loss_total"])
        if best_metric_value is None or current_metric < best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch_index
            save_checkpoint(run_dir / "best.pt", checkpoint_payload)

    training_summary = {
        "stage_name": config.stage_name,
        "dataset": dataset_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "device": config.device,
        "dtype": config.dtype,
        "feature_cache_id": config.feature_cache_id,
        "semantic_set_id": config.semantic_set_id,
        "manifest_filtered_sha256": input_context.manifest_filtered_sha256,
        "sample_id_order_sha256": input_context.sample_id_order_sha256,
        "filtered_count": input_context.filtered_count,
        "train_count": len(input_context.train_indices),
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "best_metric_name": config.best_metric,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "model_summary": first_batch_model_summary,
        "epoch_summaries": epoch_summaries,
    }
    write_json(run_dir / "training_summary.json", training_summary)
    return training_summary


def build_model_output_validator_summary(
    config: TrainConfig,
    dataset_name: str,
    run_name: str,
) -> Dict[str, Any]:
    from src.losses import compute_total_loss

    device = _ensure_cuda_protocol(config.device, config.dtype)
    input_context = _load_stage4_input_context(config, dataset_name)
    run_dir = config.build_output_dir(dataset_name, run_name)
    checkpoint_path = _select_checkpoint_path(run_dir, config.preferred_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device=config.device)
    model = build_model(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S_matrix = _load_sparse_supervision(input_context.S_path)
    batch_indices = input_context.train_indices[: config.validation_batch_size]
    if len(batch_indices) != config.validation_batch_size:
        raise RuntimeError("validation.batch_size exceeds the available train sample count.")
    dataset = FormalTrainDataset(
        X_I_path=input_context.X_I_path,
        X_T_path=input_context.X_T_path,
        train_indices=batch_indices,
        train_ids=input_context.train_ids[: config.validation_batch_size],
    )
    batch = {
        "manifest_index": torch.tensor(batch_indices, dtype=torch.int64),
        "X_I": torch.stack([dataset[index]["X_I"] for index in range(len(dataset))], dim=0),
        "X_T": torch.stack([dataset[index]["X_T"] for index in range(len(dataset))], dim=0),
    }
    batch_tensors = _make_batch_tensors(batch, S_matrix, device)
    with torch.inference_mode():
        outputs = model(batch_tensors["X_I"], batch_tensors["X_T"])
        loss_outputs = compute_total_loss(
            P_IT=outputs["P_IT"],
            P_II=outputs["P_II"],
            P_TT=outputs["P_TT"],
            S=batch_tensors["S"],
            H_I=outputs["H_I"],
            H_T=outputs["H_T"],
            config=build_loss_config(config),
        )
    B_ok = bool(
        torch.all((outputs["B_I"] == -1.0) | (outputs["B_I"] == 1.0)).item()
        and torch.all((outputs["B_T"] == -1.0) | (outputs["B_T"] == 1.0)).item()
    )
    H_range_ok = bool(
        torch.all((outputs["H_I"] > -1.0) & (outputs["H_I"] < 1.0)).item()
        and torch.all((outputs["H_T"] > -1.0) & (outputs["H_T"] < 1.0)).item()
    )
    loss_metrics = _float_metrics(loss_outputs)
    summary = {
        "dataset": dataset_name,
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "device": config.device,
        "dtype": config.dtype,
        "Z_I_shape": [int(value) for value in outputs["Z_I"].shape],
        "Z_T_shape": [int(value) for value in outputs["Z_T"].shape],
        "Y_I_shape": [int(value) for value in outputs["Y_I"].shape],
        "Y_T_shape": [int(value) for value in outputs["Y_T"].shape],
        "H_I_shape": [int(value) for value in outputs["H_I"].shape],
        "H_T_shape": [int(value) for value in outputs["H_T"].shape],
        "P_IT_shape": [int(value) for value in outputs["P_IT"].shape],
        "P_II_shape": [int(value) for value in outputs["P_II"].shape],
        "P_TT_shape": [int(value) for value in outputs["P_TT"].shape],
        "H_range_ok": H_range_ok,
        "B_binary_ok": B_ok,
        "loss_metrics": loss_metrics,
        "model_summary": outputs["model_summary"],
        "validator_passed": bool(
            H_range_ok
            and B_ok
            and all(np.isfinite(value) for value in loss_metrics.values())
            and checkpoint["device"] == "cuda:0"
        ),
    }
    _write_validator_section(run_dir, "model_outputs", summary)
    return summary


def build_training_output_validator_summary(
    config: TrainConfig,
    dataset_name: str,
    run_name: str,
) -> Dict[str, Any]:
    _ensure_cuda_protocol(config.device, config.dtype)
    run_dir = config.build_output_dir(dataset_name, run_name)
    required_paths = {
        "config_snapshot": run_dir / "config_snapshot.json",
        "train_log": run_dir / "train_log.jsonl",
        "last_checkpoint": run_dir / "last.pt",
        "best_checkpoint": run_dir / "best.pt",
        "training_summary": run_dir / "training_summary.json",
    }
    exists = {name: bool(path.is_file()) for name, path in required_paths.items()}
    if not all(exists.values()):
        missing = [name for name, value in exists.items() if not value]
        raise FileNotFoundError(f"Stage 4 training outputs are missing: {missing}")
    config_snapshot = read_json(required_paths["config_snapshot"])
    training_summary = read_json(required_paths["training_summary"])
    train_log_rows = read_jsonl(required_paths["train_log"])
    checkpoint = load_checkpoint(required_paths["best_checkpoint"], device=config.device)
    loss_fields = ("L_IT", "L_II", "L_TT", "L_sem", "L_pair", "L_q", "L_bal", "loss_total")
    finite_train_log = True
    for row in train_log_rows:
        if row.get("event") != "train_step":
            continue
        for field in loss_fields:
            value = row.get(field)
            finite_train_log = bool(finite_train_log and value is not None and np.isfinite(float(value)))
    config_protocol_ok = bool(
        config_snapshot.get("inputs", {}).get("feature_cache_id") == "clip_vit_b32_formal_v1"
        and config_snapshot.get("inputs", {}).get("semantic_set_id") == "semantic_relation_highsignal_v1"
        and config_snapshot.get("runtime", {}).get("device") == "cuda:0"
        and config_snapshot.get("runtime", {}).get("dtype") == "float32"
    )
    summary_protocol_ok = bool(
        training_summary.get("device") == "cuda:0"
        and training_summary.get("dtype") == "float32"
        and training_summary.get("dataset") == dataset_name
        and training_summary.get("run_name") == run_name
    )
    summary = {
        "dataset": dataset_name,
        "run_name": run_name,
        "device": config.device,
        "dtype": config.dtype,
        "required_files_present": exists,
        "train_log_row_count": len(train_log_rows),
        "finite_train_log": bool(finite_train_log),
        "config_protocol_ok": bool(config_protocol_ok),
        "training_summary_protocol_ok": bool(summary_protocol_ok),
        "checkpoint_device": checkpoint.get("device"),
        "checkpoint_dtype": checkpoint.get("dtype"),
        "validator_passed": bool(
            finite_train_log
            and config_protocol_ok
            and summary_protocol_ok
            and checkpoint.get("device") == "cuda:0"
            and checkpoint.get("dtype") == "float32"
        ),
    }
    _write_validator_section(run_dir, "training_outputs", summary)
    return summary


__all__ = [
    "SUPPORTED_DATASETS",
    "TrainConfig",
    "build_loss_config",
    "build_model",
    "build_model_output_validator_summary",
    "build_training_output_validator_summary",
    "load_train_config",
    "run_formal_training",
]
