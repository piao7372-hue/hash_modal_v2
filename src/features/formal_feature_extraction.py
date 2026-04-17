from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPModel, CLIPTokenizerFast

from src.datasets import SUPPORTED_DATASETS
from src.datasets.manifest_builder import (
    build_output_paths,
    iter_manifest_jsonl,
    read_json,
    sha256_of_file,
    write_json,
)


@dataclass(frozen=True)
class FeatureCachePaths:
    dataset_root: Path
    feature_cache_dir: Path
    x_i_path: Path
    x_t_path: Path
    meta_path: Path
    validator_summary_path: Path


@dataclass(frozen=True)
class FormalFeatureExtractionConfig:
    stage_name: str
    processed_root: Path
    feature_set_id: str
    model_name: str
    model_local_path: Optional[Path]
    image_size: int
    resize_mode: str
    interpolation: str
    crop_mode: str
    clip_mean: Tuple[float, float, float]
    clip_std: Tuple[float, float, float]
    tokenizer_name: str
    max_length: int
    padding: str
    truncation: bool
    return_attention_mask: bool
    dtype: str
    device: str
    image_batch_size: int
    text_batch_size: int
    feature_cache_dirname: str
    x_i_filename: str
    x_t_filename: str
    meta_filename: str
    validator_summary_filename: str
    expected_feature_dim: int
    l2_norm_atol: float
    sample_order_source: str

    def build_feature_cache_paths(self, dataset_name: str) -> FeatureCachePaths:
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError("Unsupported dataset for Stage 2: {0}".format(dataset_name))
        dataset_root = (self.processed_root / dataset_name).resolve()
        feature_cache_dir = dataset_root / self.feature_cache_dirname / self.feature_set_id
        return FeatureCachePaths(
            dataset_root=dataset_root,
            feature_cache_dir=feature_cache_dir,
            x_i_path=feature_cache_dir / self.x_i_filename,
            x_t_path=feature_cache_dir / self.x_t_filename,
            meta_path=feature_cache_dir / self.meta_filename,
            validator_summary_path=feature_cache_dir / self.validator_summary_filename,
        )

    @property
    def resolved_model_local_path(self) -> Optional[str]:
        if self.model_local_path is None:
            return None
        return str(self.model_local_path)


@dataclass(frozen=True)
class ManifestContext:
    manifest_filtered_path: Path
    manifest_filtered_sha256: str
    filtered_count_expected: int
    filtered_count_observed: int
    sample_id_order_sha256: str


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping: {0}".format(path))
    return payload


def load_feature_extraction_config(project_root: Path, config_path: Path) -> FormalFeatureExtractionConfig:
    payload = load_yaml(config_path)
    model_local_path_value = payload["model"].get("model_local_path")
    model_local_path: Optional[Path] = None
    if model_local_path_value is not None:
        candidate_path = Path(model_local_path_value)
        if not candidate_path.is_absolute():
            candidate_path = (project_root / candidate_path).resolve()
        model_local_path = candidate_path

    return FormalFeatureExtractionConfig(
        stage_name=str(payload["stage_name"]),
        processed_root=(project_root / payload["paths"]["processed_root"]).resolve(),
        feature_set_id=str(payload["model"]["feature_set_id"]),
        model_name=str(payload["model"]["model_name"]),
        model_local_path=model_local_path,
        image_size=int(payload["preprocessing"]["image_size"]),
        resize_mode=str(payload["preprocessing"]["resize_mode"]),
        interpolation=str(payload["preprocessing"]["interpolation"]),
        crop_mode=str(payload["preprocessing"]["crop_mode"]),
        clip_mean=tuple(float(value) for value in payload["preprocessing"]["clip_mean"]),
        clip_std=tuple(float(value) for value in payload["preprocessing"]["clip_std"]),
        tokenizer_name=str(payload["tokenizer"]["tokenizer_name"]),
        max_length=int(payload["tokenizer"]["max_length"]),
        padding=str(payload["tokenizer"]["padding"]),
        truncation=bool(payload["tokenizer"]["truncation"]),
        return_attention_mask=bool(payload["tokenizer"]["return_attention_mask"]),
        dtype=str(payload["runtime"]["dtype"]),
        device=str(payload["runtime"]["device"]),
        image_batch_size=int(payload["runtime"]["image_batch_size"]),
        text_batch_size=int(payload["runtime"]["text_batch_size"]),
        feature_cache_dirname=str(payload["outputs"]["feature_cache_dirname"]),
        x_i_filename=str(payload["outputs"]["x_i_filename"]),
        x_t_filename=str(payload["outputs"]["x_t_filename"]),
        meta_filename=str(payload["outputs"]["meta_filename"]),
        validator_summary_filename=str(payload["outputs"]["validator_summary_filename"]),
        expected_feature_dim=int(payload["validation"]["expected_feature_dim"]),
        l2_norm_atol=float(payload["validation"]["l2_norm_atol"]),
        sample_order_source=str(payload["validation"]["sample_order_source"]),
    )


def _sample_id_digest_from_manifest(manifest_filtered_path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    line_count = 0
    for line_count, record in enumerate(iter_manifest_jsonl(manifest_filtered_path), start=1):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                "sample_id is missing or invalid in manifest_filtered.jsonl at index {0}".format(
                    line_count - 1,
                )
            )
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return line_count, digest.hexdigest()


def inspect_manifest_context(
    config: FormalFeatureExtractionConfig,
    dataset_name: str,
) -> ManifestContext:
    stage1_paths = build_output_paths(config.processed_root, dataset_name)
    if not stage1_paths.manifest_filtered_path.is_file():
        raise FileNotFoundError(
            "Stage 2 input manifest is missing: {0}".format(stage1_paths.manifest_filtered_path)
        )
    if not stage1_paths.manifest_meta_path.is_file():
        raise FileNotFoundError(
            "Stage 1 manifest_meta.json is missing: {0}".format(stage1_paths.manifest_meta_path)
        )
    manifest_meta = read_json(stage1_paths.manifest_meta_path)
    filtered_count_expected = int(manifest_meta["counts"]["filtered_record_count"])
    filtered_count_observed, sample_id_order_sha256 = _sample_id_digest_from_manifest(
        stage1_paths.manifest_filtered_path
    )
    if filtered_count_observed != filtered_count_expected:
        raise ValueError(
            "filtered_count mismatch for {0}: manifest_meta={1}, manifest_filtered.jsonl={2}".format(
                dataset_name,
                filtered_count_expected,
                filtered_count_observed,
            )
        )
    return ManifestContext(
        manifest_filtered_path=stage1_paths.manifest_filtered_path,
        manifest_filtered_sha256=sha256_of_file(stage1_paths.manifest_filtered_path),
        filtered_count_expected=filtered_count_expected,
        filtered_count_observed=filtered_count_observed,
        sample_id_order_sha256=sample_id_order_sha256,
    )


def build_image_transform(config: FormalFeatureExtractionConfig) -> Compose:
    if config.resize_mode != "shortest_edge":
        raise ValueError("Unsupported resize_mode for formal Stage 2: {0}".format(config.resize_mode))
    if config.interpolation != "bicubic":
        raise ValueError(
            "Unsupported interpolation for formal Stage 2: {0}".format(config.interpolation)
        )
    if config.crop_mode != "center_crop":
        raise ValueError("Unsupported crop_mode for formal Stage 2: {0}".format(config.crop_mode))
    return Compose(
        [
            Resize(config.image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((config.image_size, config.image_size)),
            ToTensor(),
            Normalize(mean=config.clip_mean, std=config.clip_std),
        ]
    )


def _resolve_model_source(config: FormalFeatureExtractionConfig) -> str:
    if config.model_local_path is not None:
        if not config.model_local_path.exists():
            raise FileNotFoundError(
                "Configured model_local_path does not exist: {0}".format(config.model_local_path)
            )
        return str(config.model_local_path)
    return config.model_name


def load_formal_clip_model_and_tokenizer(
    config: FormalFeatureExtractionConfig,
) -> Tuple[CLIPModel, CLIPTokenizerFast]:
    if config.dtype != "float32":
        raise ValueError("Formal Stage 2 dtype must be float32, got {0}".format(config.dtype))
    if config.device != "cuda:0":
        raise ValueError("Formal Stage 2 device must be cuda:0, got {0}".format(config.device))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Stage 2 requires cuda:0, but CUDA is unavailable.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Formal Stage 2 requires cuda:0, but no CUDA device is visible.")

    model_source = _resolve_model_source(config)
    tokenizer_source = (
        str(config.model_local_path) if config.model_local_path is not None else config.tokenizer_name
    )
    model = CLIPModel.from_pretrained(model_source, use_safetensors=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_source)
    model = model.to(torch.device(config.device))
    model.eval()
    return model, tokenizer


def _iter_manifest_records(manifest_filtered_path: Path) -> Iterator[Dict[str, Any]]:
    for record_index, record in enumerate(iter_manifest_jsonl(manifest_filtered_path)):
        if "sample_id" not in record:
            raise ValueError("Manifest record missing sample_id at index {0}".format(record_index))
        yield record


def _validate_record_for_image(record: Dict[str, Any], record_index: int) -> Tuple[str, Path]:
    sample_id = record["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Invalid sample_id at manifest index {0}".format(record_index))
    image_path_value = record.get("image_path")
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(
            "image_path is missing or invalid for sample_id={0}".format(sample_id)
        )
    image_path = Path(image_path_value)
    if not image_path.is_file():
        raise FileNotFoundError(
            "image_path does not exist for sample_id={0}: {1}".format(sample_id, image_path)
        )
    return sample_id, image_path


def _validate_record_for_text(record: Dict[str, Any], record_index: int) -> Tuple[str, str]:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Invalid sample_id at manifest index {0}".format(record_index))
    if "text_source" not in record:
        raise ValueError("text_source is missing for sample_id={0}".format(sample_id))
    text_source = record["text_source"]
    if not isinstance(text_source, str):
        raise ValueError("text_source must be a string for sample_id={0}".format(sample_id))
    if text_source == "":
        raise ValueError("text_source must not be empty for sample_id={0}".format(sample_id))
    return sample_id, text_source


def _normalize_feature_batch(features: Tensor) -> np.ndarray:
    normalized = F.normalize(features, p=2.0, dim=-1)
    return normalized.detach().cpu().float().numpy().astype(np.float32, copy=False)


def _ensure_expected_feature_shape(
    feature_batch: np.ndarray,
    expected_rows: int,
    expected_dim: int,
    modality_name: str,
) -> None:
    expected_shape = (expected_rows, expected_dim)
    if feature_batch.shape != expected_shape:
        raise ValueError(
            "{0} feature shape mismatch: {1} != {2}".format(
                modality_name,
                feature_batch.shape,
                expected_shape,
            )
        )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def extract_image_features_to_npy(
    config: FormalFeatureExtractionConfig,
    manifest_context: ManifestContext,
    model: CLIPModel,
    output_path: Path,
) -> None:
    transform = build_image_transform(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / "{0}.tmp.npy".format(output_path.stem)
    _remove_if_exists(temp_path)
    feature_array = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(manifest_context.filtered_count_expected, config.expected_feature_dim),
    )

    batch_tensors: List[Tensor] = []
    row_start = 0
    observed_row_count = 0
    observed_digest = hashlib.sha256()

    with torch.inference_mode():
        for record_index, record in enumerate(
            _iter_manifest_records(manifest_context.manifest_filtered_path)
        ):
            sample_id, image_path = _validate_record_for_image(record, record_index)
            observed_digest.update(sample_id.encode("utf-8"))
            observed_digest.update(b"\n")
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                batch_tensors.append(transform(rgb_image))
            observed_row_count += 1

            if len(batch_tensors) == config.image_batch_size:
                image_tensor = torch.stack(batch_tensors, dim=0).to(torch.device(config.device))
                feature_batch = _normalize_feature_batch(model.get_image_features(pixel_values=image_tensor))
                _ensure_expected_feature_shape(
                    feature_batch=feature_batch,
                    expected_rows=len(batch_tensors),
                    expected_dim=config.expected_feature_dim,
                    modality_name="image",
                )
                feature_array[row_start : row_start + len(batch_tensors)] = feature_batch
                row_start += len(batch_tensors)
                batch_tensors = []

        if batch_tensors:
            image_tensor = torch.stack(batch_tensors, dim=0).to(torch.device(config.device))
            feature_batch = _normalize_feature_batch(model.get_image_features(pixel_values=image_tensor))
            _ensure_expected_feature_shape(
                feature_batch=feature_batch,
                expected_rows=len(batch_tensors),
                expected_dim=config.expected_feature_dim,
                modality_name="image",
            )
            feature_array[row_start : row_start + len(batch_tensors)] = feature_batch
            row_start += len(batch_tensors)

    if observed_row_count != manifest_context.filtered_count_expected:
        raise ValueError(
            "image feature count mismatch: {0} != {1}".format(
                observed_row_count,
                manifest_context.filtered_count_expected,
            )
        )
    if row_start != manifest_context.filtered_count_expected:
        raise ValueError(
            "image feature write count mismatch: {0} != {1}".format(
                row_start,
                manifest_context.filtered_count_expected,
            )
        )
    if observed_digest.hexdigest() != manifest_context.sample_id_order_sha256:
        raise ValueError("Image feature extraction detected sample order drift.")

    feature_array.flush()
    del feature_array
    temp_path.replace(output_path)


def extract_text_features_to_npy(
    config: FormalFeatureExtractionConfig,
    manifest_context: ManifestContext,
    model: CLIPModel,
    tokenizer: CLIPTokenizerFast,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / "{0}.tmp.npy".format(output_path.stem)
    _remove_if_exists(temp_path)
    feature_array = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=np.float32,
        shape=(manifest_context.filtered_count_expected, config.expected_feature_dim),
    )

    batch_texts: List[str] = []
    row_start = 0
    observed_row_count = 0
    observed_digest = hashlib.sha256()

    with torch.inference_mode():
        for record_index, record in enumerate(
            _iter_manifest_records(manifest_context.manifest_filtered_path)
        ):
            sample_id, text_source = _validate_record_for_text(record, record_index)
            observed_digest.update(sample_id.encode("utf-8"))
            observed_digest.update(b"\n")
            batch_texts.append(text_source)
            observed_row_count += 1

            if len(batch_texts) == config.text_batch_size:
                encoded = tokenizer(
                    batch_texts,
                    padding=config.padding,
                    truncation=config.truncation,
                    max_length=config.max_length,
                    return_attention_mask=config.return_attention_mask,
                    return_tensors="pt",
                )
                feature_batch = _normalize_feature_batch(
                    model.get_text_features(
                        input_ids=encoded["input_ids"].to(torch.device(config.device)),
                        attention_mask=encoded["attention_mask"].to(torch.device(config.device)),
                    )
                )
                _ensure_expected_feature_shape(
                    feature_batch=feature_batch,
                    expected_rows=len(batch_texts),
                    expected_dim=config.expected_feature_dim,
                    modality_name="text",
                )
                feature_array[row_start : row_start + len(batch_texts)] = feature_batch
                row_start += len(batch_texts)
                batch_texts = []

        if batch_texts:
            encoded = tokenizer(
                batch_texts,
                padding=config.padding,
                truncation=config.truncation,
                max_length=config.max_length,
                return_attention_mask=config.return_attention_mask,
                return_tensors="pt",
            )
            feature_batch = _normalize_feature_batch(
                model.get_text_features(
                    input_ids=encoded["input_ids"].to(torch.device(config.device)),
                    attention_mask=encoded["attention_mask"].to(torch.device(config.device)),
                )
            )
            _ensure_expected_feature_shape(
                feature_batch=feature_batch,
                expected_rows=len(batch_texts),
                expected_dim=config.expected_feature_dim,
                modality_name="text",
            )
            feature_array[row_start : row_start + len(batch_texts)] = feature_batch
            row_start += len(batch_texts)

    if observed_row_count != manifest_context.filtered_count_expected:
        raise ValueError(
            "text feature count mismatch: {0} != {1}".format(
                observed_row_count,
                manifest_context.filtered_count_expected,
            )
        )
    if row_start != manifest_context.filtered_count_expected:
        raise ValueError(
            "text feature write count mismatch: {0} != {1}".format(
                row_start,
                manifest_context.filtered_count_expected,
            )
        )
    if observed_digest.hexdigest() != manifest_context.sample_id_order_sha256:
        raise ValueError("Text feature extraction detected sample order drift.")

    feature_array.flush()
    del feature_array
    temp_path.replace(output_path)


def build_meta_payload(
    config: FormalFeatureExtractionConfig,
    dataset_name: str,
    manifest_context: ManifestContext,
) -> Dict[str, Any]:
    return {
        "stage": config.stage_name,
        "feature_set_id": config.feature_set_id,
        "dataset": dataset_name,
        "filtered_count": manifest_context.filtered_count_expected,
        "feature_dim_image": config.expected_feature_dim,
        "feature_dim_text": config.expected_feature_dim,
        "model_name": config.model_name,
        "model_local_path": config.resolved_model_local_path,
        "image_size": config.image_size,
        "resize_mode": config.resize_mode,
        "interpolation": config.interpolation,
        "crop_mode": config.crop_mode,
        "tokenizer_name": config.tokenizer_name,
        "max_length": config.max_length,
        "padding": config.padding,
        "truncation": config.truncation,
        "image_batch_size": config.image_batch_size,
        "text_batch_size": config.text_batch_size,
        "dtype": config.dtype,
        "device": config.device,
        "l2_normalized": True,
        "manifest_filtered_path": str(manifest_context.manifest_filtered_path),
        "manifest_filtered_sha256": manifest_context.manifest_filtered_sha256,
        "sample_order_source": config.sample_order_source,
        "runtime_completed": True,
        "failure_reason": None,
        "sample_id_order_sha256": manifest_context.sample_id_order_sha256,
    }


def _chunk_slices(total_rows: int, chunk_size: int = 4096) -> Iterator[slice]:
    for start in range(0, total_rows, chunk_size):
        yield slice(start, min(total_rows, start + chunk_size))


def _check_all_finite(array: np.ndarray) -> bool:
    total_rows = int(array.shape[0])
    for row_slice in _chunk_slices(total_rows):
        if not np.isfinite(array[row_slice]).all():
            return False
    return True


def _check_l2_norm(array: np.ndarray, atol: float) -> bool:
    total_rows = int(array.shape[0])
    for row_slice in _chunk_slices(total_rows):
        norms = np.linalg.norm(array[row_slice], axis=1)
        if not np.allclose(norms, np.ones_like(norms), atol=atol, rtol=0.0):
            return False
    return True


def validate_feature_cache(
    config: FormalFeatureExtractionConfig,
    dataset_name: str,
) -> Dict[str, Any]:
    manifest_context = inspect_manifest_context(config, dataset_name)
    cache_paths = config.build_feature_cache_paths(dataset_name)

    x_i_exists = cache_paths.x_i_path.is_file()
    x_t_exists = cache_paths.x_t_path.is_file()
    meta_exists = cache_paths.meta_path.is_file()

    x_i_shape: Optional[List[int]] = None
    x_t_shape: Optional[List[int]] = None
    feature_dim_image: Optional[int] = None
    feature_dim_text: Optional[int] = None
    dtype_ok = False
    order_consistency_passed = False
    nan_check_passed = False
    inf_check_passed = False
    l2_norm_check_passed = False
    meta_consistency_passed = False
    failure_reason: Optional[str] = None

    x_i_array: Optional[np.ndarray] = None
    x_t_array: Optional[np.ndarray] = None
    meta_payload: Optional[Dict[str, Any]] = None

    if x_i_exists:
        x_i_array = np.load(cache_paths.x_i_path, mmap_mode="r")
        x_i_shape = [int(value) for value in x_i_array.shape]
        if len(x_i_shape) == 2:
            feature_dim_image = x_i_shape[1]

    if x_t_exists:
        x_t_array = np.load(cache_paths.x_t_path, mmap_mode="r")
        x_t_shape = [int(value) for value in x_t_array.shape]
        if len(x_t_shape) == 2:
            feature_dim_text = x_t_shape[1]

    if meta_exists:
        meta_payload = read_json(cache_paths.meta_path)

    if not x_i_exists:
        failure_reason = "Missing X_I.npy."
    elif not x_t_exists:
        failure_reason = "Missing X_T.npy."
    elif not meta_exists:
        failure_reason = "Missing meta.json."
    elif x_i_shape != [manifest_context.filtered_count_expected, config.expected_feature_dim]:
        failure_reason = "X_I.npy shape mismatch."
    elif x_t_shape != [manifest_context.filtered_count_expected, config.expected_feature_dim]:
        failure_reason = "X_T.npy shape mismatch."
    else:
        dtype_ok = bool(
            x_i_array is not None
            and x_t_array is not None
            and x_i_array.dtype == np.float32
            and x_t_array.dtype == np.float32
        )
        if not dtype_ok:
            failure_reason = "Feature cache dtype must be float32."

    if failure_reason is None and meta_payload is not None:
        order_consistency_passed = bool(
            meta_payload.get("manifest_filtered_path") == str(manifest_context.manifest_filtered_path)
            and meta_payload.get("manifest_filtered_sha256") == manifest_context.manifest_filtered_sha256
            and meta_payload.get("sample_order_source") == config.sample_order_source
            and meta_payload.get("sample_id_order_sha256") == manifest_context.sample_id_order_sha256
        )
        if not order_consistency_passed:
            failure_reason = "Manifest order consistency check failed."

    if failure_reason is None and x_i_array is not None and x_t_array is not None:
        finite_x_i = _check_all_finite(x_i_array)
        finite_x_t = _check_all_finite(x_t_array)
        nan_check_passed = finite_x_i and finite_x_t
        inf_check_passed = finite_x_i and finite_x_t
        if not finite_x_i or not finite_x_t:
            failure_reason = "Feature cache contains NaN or Inf."

    if failure_reason is None and x_i_array is not None and x_t_array is not None:
        l2_norm_check_passed = _check_l2_norm(x_i_array, config.l2_norm_atol) and _check_l2_norm(
            x_t_array,
            config.l2_norm_atol,
        )
        if not l2_norm_check_passed:
            failure_reason = "L2 norm check failed."

    if failure_reason is None and meta_payload is not None:
        meta_consistency_passed = bool(
            meta_payload.get("stage") == config.stage_name
            and meta_payload.get("feature_set_id") == config.feature_set_id
            and meta_payload.get("dataset") == dataset_name
            and int(meta_payload.get("filtered_count")) == manifest_context.filtered_count_expected
            and int(meta_payload.get("feature_dim_image")) == config.expected_feature_dim
            and int(meta_payload.get("feature_dim_text")) == config.expected_feature_dim
            and meta_payload.get("model_name") == config.model_name
            and meta_payload.get("model_local_path") == config.resolved_model_local_path
            and int(meta_payload.get("image_size")) == config.image_size
            and meta_payload.get("resize_mode") == config.resize_mode
            and meta_payload.get("interpolation") == config.interpolation
            and meta_payload.get("crop_mode") == config.crop_mode
            and meta_payload.get("tokenizer_name") == config.tokenizer_name
            and int(meta_payload.get("max_length")) == config.max_length
            and meta_payload.get("padding") == config.padding
            and bool(meta_payload.get("truncation")) == config.truncation
            and int(meta_payload.get("image_batch_size")) == config.image_batch_size
            and int(meta_payload.get("text_batch_size")) == config.text_batch_size
            and meta_payload.get("dtype") == config.dtype
            and meta_payload.get("device") == config.device
            and bool(meta_payload.get("l2_normalized")) is True
            and bool(meta_payload.get("runtime_completed")) is True
            and meta_payload.get("failure_reason") is None
        )
        if not meta_consistency_passed:
            failure_reason = "meta.json is inconsistent with the formal Stage 2 protocol."

    filtered_count_observed = 0
    if x_i_shape is not None:
        filtered_count_observed = x_i_shape[0]
    elif x_t_shape is not None:
        filtered_count_observed = x_t_shape[0]

    validator_passed = failure_reason is None
    return {
        "dataset": dataset_name,
        "feature_set_id": config.feature_set_id,
        "filtered_count_expected": manifest_context.filtered_count_expected,
        "filtered_count_observed": filtered_count_observed,
        "x_i_exists": x_i_exists,
        "x_t_exists": x_t_exists,
        "meta_exists": meta_exists,
        "x_i_shape": x_i_shape,
        "x_t_shape": x_t_shape,
        "feature_dim_image": feature_dim_image,
        "feature_dim_text": feature_dim_text,
        "dtype_ok": dtype_ok,
        "order_consistency_passed": order_consistency_passed,
        "nan_check_passed": nan_check_passed,
        "inf_check_passed": inf_check_passed,
        "l2_norm_check_passed": l2_norm_check_passed,
        "meta_consistency_passed": meta_consistency_passed,
        "validator_passed": validator_passed,
        "failure_reason": failure_reason,
    }


def write_validator_summary(
    config: FormalFeatureExtractionConfig,
    dataset_name: str,
    summary: Dict[str, Any],
) -> None:
    cache_paths = config.build_feature_cache_paths(dataset_name)
    write_json(cache_paths.validator_summary_path, summary)


def run_formal_feature_extraction(
    config: FormalFeatureExtractionConfig,
    dataset_name: str,
) -> Dict[str, Any]:
    manifest_context = inspect_manifest_context(config, dataset_name)
    cache_paths = config.build_feature_cache_paths(dataset_name)
    cache_paths.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_formal_clip_model_and_tokenizer(config)
    extract_image_features_to_npy(config, manifest_context, model, cache_paths.x_i_path)
    extract_text_features_to_npy(config, manifest_context, model, tokenizer, cache_paths.x_t_path)
    meta_payload = build_meta_payload(config, dataset_name, manifest_context)
    write_json(cache_paths.meta_path, meta_payload)

    validator_summary = validate_feature_cache(config, dataset_name)
    write_validator_summary(config, dataset_name, validator_summary)
    if not bool(validator_summary["validator_passed"]):
        raise ValueError(
            "Stage 2 validator failed for {0}: {1}".format(
                dataset_name,
                validator_summary["failure_reason"],
            )
        )

    return {
        "stage": config.stage_name,
        "dataset": dataset_name,
        "feature_set_id": config.feature_set_id,
        "filtered_count": manifest_context.filtered_count_expected,
        "model_name": config.model_name,
        "model_local_path": config.resolved_model_local_path,
        "device": config.device,
        "dtype": config.dtype,
        "x_i_path": str(cache_paths.x_i_path),
        "x_t_path": str(cache_paths.x_t_path),
        "meta_path": str(cache_paths.meta_path),
        "validator_summary_path": str(cache_paths.validator_summary_path),
        "manifest_filtered_path": str(manifest_context.manifest_filtered_path),
        "manifest_filtered_sha256": manifest_context.manifest_filtered_sha256,
        "sample_id_order_sha256": manifest_context.sample_id_order_sha256,
        "x_i_shape": [manifest_context.filtered_count_expected, config.expected_feature_dim],
        "x_t_shape": [manifest_context.filtered_count_expected, config.expected_feature_dim],
        "validator_passed": True,
    }


__all__ = [
    "FeatureCachePaths",
    "FormalFeatureExtractionConfig",
    "ManifestContext",
    "SUPPORTED_DATASETS",
    "inspect_manifest_context",
    "load_feature_extraction_config",
    "load_formal_clip_model_and_tokenizer",
    "run_formal_feature_extraction",
    "validate_feature_cache",
    "write_validator_summary",
]
