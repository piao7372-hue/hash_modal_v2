from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import gc
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from src.datasets import SUPPORTED_DATASETS
from src.datasets.manifest_builder import (
    build_output_paths,
    read_json,
    read_text_lines,
    sha256_of_file,
    write_json,
)
from src.features import inspect_manifest_context

from .ann_utils import HnswBuildParams
from .confidence import build_confidence_from_probabilities, compute_confidence
from .direct_support import (
    build_cross_modal_candidate_support,
    compute_direct_support,
    validate_feature_inputs,
)
from .final_supervision import (
    assert_same_support,
    build_S,
    build_S_tilde,
    build_final_support,
    restrict_to_final_support,
)
from .structural_support import build_intra_modal_profiles, compute_structural_support


@dataclass(frozen=True)
class SemanticCachePaths:
    dataset_root: Path
    semantic_cache_dir: Path
    A_path: Path
    R_path: Path
    S_tilde_path: Path
    C_path: Path
    S_path: Path
    meta_path: Path
    validator_summary_path: Path


@dataclass(frozen=True)
class FormalSemanticRelationConfig:
    processed_root: Path
    semantic_set_id: str
    protocol_name: str
    protocol_source: str
    feature_cache_id: str
    ann_backend: str
    direct_topk: int
    intra_topk: int
    final_topk: int
    hnsw_M: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    relation_lambda: float
    tau: float
    dtype: str
    cache_dirname: str
    validator_entry: str
    l2_norm_atol: float

    def build_cache_paths(self, dataset_name: str) -> SemanticCachePaths:
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError("Unsupported dataset for Stage 3: {0}".format(dataset_name))
        dataset_root = (self.processed_root / dataset_name).resolve()
        semantic_cache_dir = dataset_root / self.cache_dirname / self.semantic_set_id
        return SemanticCachePaths(
            dataset_root=dataset_root,
            semantic_cache_dir=semantic_cache_dir,
            A_path=semantic_cache_dir / "A.npz",
            R_path=semantic_cache_dir / "R.npz",
            S_tilde_path=semantic_cache_dir / "S_tilde.npz",
            C_path=semantic_cache_dir / "C.npz",
            S_path=semantic_cache_dir / "S.npz",
            meta_path=semantic_cache_dir / "meta.json",
            validator_summary_path=semantic_cache_dir / "validator_summary.json",
        )

    @property
    def ann_params(self) -> HnswBuildParams:
        return HnswBuildParams(
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
        )


@dataclass(frozen=True)
class Stage1SemanticContext:
    manifest_filtered_path: Path
    manifest_filtered_sha256: str
    sample_id_order_sha256: str
    filtered_count: int
    query_ids_path: Path
    retrieval_ids_path: Path
    train_ids_path: Path
    query_count: int
    retrieval_count: int
    train_count: int


@dataclass(frozen=True)
class Stage2FeatureContext:
    feature_cache_dir: Path
    X_I_path: Path
    X_T_path: Path
    meta_path: Path
    validator_summary_path: Path
    meta_payload: Dict[str, Any]
    validator_payload: Dict[str, Any]
    input_feature_meta_sha256: str


@dataclass(frozen=True)
class SemanticRelationArtifacts:
    A: csr_matrix
    R: csr_matrix
    S_tilde: csr_matrix
    C: csr_matrix
    S: csr_matrix
    meta: Dict[str, Any]


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping: {0}".format(path))
    return payload


def load_semantic_relation_config(
    project_root: Path,
    config_path: Path,
) -> FormalSemanticRelationConfig:
    payload = load_yaml(config_path)
    return FormalSemanticRelationConfig(
        processed_root=(project_root / payload["paths"]["processed_root"]).resolve(),
        semantic_set_id=str(payload["semantic_set_id"]),
        protocol_name=str(payload["protocol_name"]),
        protocol_source=str(payload["protocol_source"]),
        feature_cache_id=str(payload["input"]["feature_cache_id"]),
        ann_backend=str(payload["support"]["ann_backend"]),
        direct_topk=int(payload["support"]["direct_topk"]),
        intra_topk=int(payload["support"]["intra_topk"]),
        final_topk=int(payload["support"]["final_topk"]),
        hnsw_M=int(payload["support"]["hnsw_M"]),
        hnsw_ef_construction=int(payload["support"]["hnsw_ef_construction"]),
        hnsw_ef_search=int(payload["support"]["hnsw_ef_search"]),
        relation_lambda=float(payload["relation"]["lambda"]),
        tau=float(payload["relation"]["tau"]),
        dtype=str(payload["runtime"]["dtype"]),
        cache_dirname=str(payload["outputs"]["cache_dir"]),
        validator_entry=str(payload["validation"]["validator_entry"]),
        l2_norm_atol=float(payload["validation"]["l2_norm_atol"]),
    )


def _sample_id_digest_from_manifest(manifest_filtered_path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    line_count = 0
    with manifest_filtered_path.open("r", encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            record = json.loads(line)
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


def inspect_stage1_semantic_context(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
) -> Stage1SemanticContext:
    output_paths = build_output_paths(config.processed_root, dataset_name)
    if not output_paths.manifest_filtered_path.is_file():
        raise FileNotFoundError(
            "Stage 3 input manifest is missing: {0}".format(output_paths.manifest_filtered_path)
        )
    if not output_paths.query_ids_path.is_file():
        raise FileNotFoundError("Stage 3 query_ids.txt is missing: {0}".format(output_paths.query_ids_path))
    if not output_paths.retrieval_ids_path.is_file():
        raise FileNotFoundError(
            "Stage 3 retrieval_ids.txt is missing: {0}".format(output_paths.retrieval_ids_path)
        )
    if not output_paths.train_ids_path.is_file():
        raise FileNotFoundError("Stage 3 train_ids.txt is missing: {0}".format(output_paths.train_ids_path))

    manifest_context = inspect_manifest_context(
        type("Stage1Adapter", (), {"processed_root": config.processed_root})(),
        dataset_name,
    )
    observed_count, observed_order_sha256 = _sample_id_digest_from_manifest(output_paths.manifest_filtered_path)
    if observed_count != manifest_context.filtered_count_expected:
        raise ValueError("Stage 3 manifest count drift detected.")
    if observed_order_sha256 != manifest_context.sample_id_order_sha256:
        raise ValueError("Stage 3 manifest sample_id order drift detected.")

    query_ids = read_text_lines(output_paths.query_ids_path)
    retrieval_ids = read_text_lines(output_paths.retrieval_ids_path)
    train_ids = read_text_lines(output_paths.train_ids_path)

    return Stage1SemanticContext(
        manifest_filtered_path=output_paths.manifest_filtered_path,
        manifest_filtered_sha256=manifest_context.manifest_filtered_sha256,
        sample_id_order_sha256=manifest_context.sample_id_order_sha256,
        filtered_count=manifest_context.filtered_count_expected,
        query_ids_path=output_paths.query_ids_path,
        retrieval_ids_path=output_paths.retrieval_ids_path,
        train_ids_path=output_paths.train_ids_path,
        query_count=len(query_ids),
        retrieval_count=len(retrieval_ids),
        train_count=len(train_ids),
    )


def inspect_stage2_feature_context(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
) -> Stage2FeatureContext:
    feature_cache_dir = (
        config.processed_root / dataset_name / "feature_cache" / config.feature_cache_id
    ).resolve()
    X_I_path = feature_cache_dir / "X_I.npy"
    X_T_path = feature_cache_dir / "X_T.npy"
    meta_path = feature_cache_dir / "meta.json"
    validator_summary_path = feature_cache_dir / "validator_summary.json"
    for required_path in (X_I_path, X_T_path, meta_path, validator_summary_path):
        if not required_path.is_file():
            raise FileNotFoundError("Stage 2 formal input is missing: {0}".format(required_path))

    meta_payload = read_json(meta_path)
    validator_payload = read_json(validator_summary_path)
    if bool(validator_payload.get("validator_passed")) is not True:
        raise ValueError("Stage 2 validator did not pass for dataset {0}.".format(dataset_name))

    return Stage2FeatureContext(
        feature_cache_dir=feature_cache_dir,
        X_I_path=X_I_path,
        X_T_path=X_T_path,
        meta_path=meta_path,
        validator_summary_path=validator_summary_path,
        meta_payload=meta_payload,
        validator_payload=validator_payload,
        input_feature_meta_sha256=sha256_of_file(meta_path),
    )


def load_stage2_feature_arrays(stage2_context: Stage2FeatureContext) -> tuple[np.ndarray, np.ndarray]:
    X_I = np.load(stage2_context.X_I_path, mmap_mode="r")
    X_T = np.load(stage2_context.X_T_path, mmap_mode="r")
    return X_I, X_T


def _matrix_stats(matrix: csr_matrix) -> Dict[str, Any]:
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / float(matrix.shape[0] * matrix.shape[1])),
    }


def build_semantic_meta(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
    stage1_context: Stage1SemanticContext,
    stage2_context: Stage2FeatureContext,
    paths: SemanticCachePaths,
    A: csr_matrix,
    R: csr_matrix,
    S_tilde: csr_matrix,
    C: csr_matrix,
    S: csr_matrix,
    feature_dim: int,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "semantic_set_id": config.semantic_set_id,
        "protocol_name": config.protocol_name,
        "protocol_source": config.protocol_source,
        "feature_cache_id": config.feature_cache_id,
        "input_manifest_filtered_path": str(stage1_context.manifest_filtered_path),
        "input_manifest_filtered_sha256": stage1_context.manifest_filtered_sha256,
        "input_feature_cache_dir": str(stage2_context.feature_cache_dir),
        "input_feature_meta_sha256": stage2_context.input_feature_meta_sha256,
        "sample_id_order_sha256": stage1_context.sample_id_order_sha256,
        "filtered_count": stage1_context.filtered_count,
        "feature_dim": feature_dim,
        "direct_topk": config.direct_topk,
        "intra_topk": config.intra_topk,
        "final_topk": config.final_topk,
        "lambda": config.relation_lambda,
        "tau": config.tau,
        "ann_backend": config.ann_backend,
        "hnsw_M": config.hnsw_M,
        "hnsw_ef_construction": config.hnsw_ef_construction,
        "hnsw_ef_search": config.hnsw_ef_search,
        "R_realization": "topk_sparse_profile_cosine_v1",
        "confidence_realization": "sparse_bidirectional_softmax_v1",
        "feature_meta": {
            "stage_2_manifest_filtered_sha256": stage2_context.meta_payload.get("manifest_filtered_sha256"),
            "stage_2_sample_id_order_sha256": stage2_context.meta_payload.get("sample_id_order_sha256"),
            "stage_2_validator_passed": stage2_context.validator_payload.get("validator_passed"),
            "stage_2_feature_set_id": stage2_context.meta_payload.get("feature_set_id"),
            "stage_2_filtered_count": stage2_context.meta_payload.get("filtered_count"),
            "stage_2_dtype": stage2_context.meta_payload.get("dtype"),
        },
        "output_files": {
            "A": str(paths.A_path),
            "R": str(paths.R_path),
            "S_tilde": str(paths.S_tilde_path),
            "C": str(paths.C_path),
            "S": str(paths.S_path),
            "meta": str(paths.meta_path),
            "validator_summary": str(paths.validator_summary_path),
        },
        "A_nnz": int(A.nnz),
        "R_nnz": int(R.nnz),
        "S_tilde_nnz": int(S_tilde.nnz),
        "C_nnz": int(C.nnz),
        "S_nnz": int(S.nnz),
        "sparsity_density": float(S.nnz / float(S.shape[0] * S.shape[1])),
        "dtype": config.dtype,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }


def run_formal_semantic_relation(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
) -> SemanticRelationArtifacts:
    if config.ann_backend != "hnswlib":
        raise ValueError("Stage 3 ann_backend must be hnswlib.")
    if config.dtype != "float32":
        raise ValueError("Stage 3 dtype must be float32.")

    stage1_context = inspect_stage1_semantic_context(config, dataset_name)
    stage2_context = inspect_stage2_feature_context(config, dataset_name)
    paths = config.build_cache_paths(dataset_name)

    X_I, X_T = load_stage2_feature_arrays(stage2_context)
    filtered_count, feature_dim = validate_feature_inputs(
        X_I=X_I,
        X_T=X_T,
        filtered_count=stage1_context.filtered_count,
        l2_norm_atol=config.l2_norm_atol,
    )
    if int(stage2_context.meta_payload.get("filtered_count")) != filtered_count:
        raise ValueError("Stage 2 filtered_count does not match Stage 3 inputs.")
    if stage2_context.meta_payload.get("manifest_filtered_sha256") != stage1_context.manifest_filtered_sha256:
        raise ValueError("Stage 2 manifest hash does not match Stage 1 manifest hash.")
    if stage2_context.meta_payload.get("sample_id_order_sha256") != stage1_context.sample_id_order_sha256:
        raise ValueError("Stage 2 sample order hash does not match Stage 1.")

    candidate_support = build_cross_modal_candidate_support(
        X_I=X_I,
        X_T=X_T,
        direct_topk=config.direct_topk,
        ann_params=config.ann_params,
    )
    A_candidate = compute_direct_support(X_I, X_T, candidate_support)

    image_profiles = build_intra_modal_profiles(X_I, config.intra_topk, config.ann_params)
    text_profiles = build_intra_modal_profiles(X_T, config.intra_topk, config.ann_params)
    R_candidate = compute_structural_support(candidate_support, image_profiles, text_profiles)
    S_tilde_candidate = build_S_tilde(A_candidate, R_candidate, config.relation_lambda)

    del A_candidate
    del R_candidate
    gc.collect()

    P_I_to_T, P_T_to_I, _ = compute_confidence(S_tilde_candidate, config.tau)
    C_candidate = build_confidence_from_probabilities(P_I_to_T, P_T_to_I)
    S_candidate = build_S(C_candidate, S_tilde_candidate)
    final_support = build_final_support(S_candidate, config.final_topk)

    A = compute_direct_support(X_I, X_T, final_support)
    R = compute_structural_support(final_support, image_profiles, text_profiles)
    S_tilde = build_S_tilde(A, R, config.relation_lambda)

    P_I_to_T_final = restrict_to_final_support(P_I_to_T, final_support)
    P_T_to_I_final = restrict_to_final_support(P_T_to_I, final_support)
    C = build_confidence_from_probabilities(P_I_to_T_final, P_T_to_I_final)
    S = build_S(C, S_tilde)

    assert_same_support(A, R)
    assert_same_support(A, S_tilde)
    assert_same_support(A, C)
    assert_same_support(A, S)

    paths.semantic_cache_dir.mkdir(parents=True, exist_ok=True)
    meta = build_semantic_meta(
        config=config,
        dataset_name=dataset_name,
        stage1_context=stage1_context,
        stage2_context=stage2_context,
        paths=paths,
        A=A,
        R=R,
        S_tilde=S_tilde,
        C=C,
        S=S,
        feature_dim=feature_dim,
    )
    return SemanticRelationArtifacts(A=A, R=R, S_tilde=S_tilde, C=C, S=S, meta=meta)


def write_semantic_cache(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
    artifacts: SemanticRelationArtifacts,
) -> None:
    paths = config.build_cache_paths(dataset_name)
    paths.semantic_cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(paths.A_path, artifacts.A.astype(np.float32))
    save_npz(paths.R_path, artifacts.R.astype(np.float32))
    save_npz(paths.S_tilde_path, artifacts.S_tilde.astype(np.float32))
    save_npz(paths.C_path, artifacts.C.astype(np.float32))
    save_npz(paths.S_path, artifacts.S.astype(np.float32))
    write_json(paths.meta_path, artifacts.meta)


def load_semantic_cache_matrices(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
) -> Dict[str, csr_matrix]:
    paths = config.build_cache_paths(dataset_name)
    required_paths = {
        "A": paths.A_path,
        "R": paths.R_path,
        "S_tilde": paths.S_tilde_path,
        "C": paths.C_path,
        "S": paths.S_path,
    }
    matrices: Dict[str, csr_matrix] = {}
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError("Stage 3 cache file is missing: {0}".format(path))
        matrix = load_npz(path)
        if not isinstance(matrix, csr_matrix):
            matrix = matrix.tocsr()
        matrices[name] = matrix
    return matrices


def _matrix_support_signature(matrix: csr_matrix) -> tuple[bytes, bytes]:
    return matrix.indptr.tobytes(), matrix.indices.tobytes()


def _check_value_range(matrix: csr_matrix, lower: float, upper: float, strict_lower: bool = False) -> bool:
    if matrix.nnz == 0:
        return False
    values = matrix.data
    if strict_lower:
        return bool(np.all((values > lower) & (values <= upper)))
    return bool(np.all((values >= lower) & (values <= upper)))


def _check_no_nan_or_inf(matrices: Iterable[csr_matrix]) -> tuple[bool, bool]:
    nan_passed = True
    inf_passed = True
    for matrix in matrices:
        nan_passed = nan_passed and (not np.isnan(matrix.data).any())
        inf_passed = inf_passed and (not np.isinf(matrix.data).any())
    return nan_passed, inf_passed


def build_validator_summary(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
) -> Dict[str, Any]:
    stage1_context = inspect_stage1_semantic_context(config, dataset_name)
    stage2_context = inspect_stage2_feature_context(config, dataset_name)
    paths = config.build_cache_paths(dataset_name)

    if not paths.meta_path.is_file():
        raise FileNotFoundError("Stage 3 meta.json is missing: {0}".format(paths.meta_path))

    matrices = load_semantic_cache_matrices(config, dataset_name)
    meta = read_json(paths.meta_path)

    A = matrices["A"].tocsr()
    R = matrices["R"].tocsr()
    S_tilde = matrices["S_tilde"].tocsr()
    C = matrices["C"].tocsr()
    S = matrices["S"].tocsr()

    csr_check_passed = all(isinstance(matrix, csr_matrix) for matrix in (A, R, S_tilde, C, S))
    dtype_check_passed = all(matrix.dtype == np.float32 for matrix in (A, R, S_tilde, C, S))
    support_signature = _matrix_support_signature(A)
    support_consistency_passed = all(
        _matrix_support_signature(matrix) == support_signature for matrix in (R, S_tilde, C, S)
    )
    matrix_shape = [int(A.shape[0]), int(A.shape[1])]
    if A.shape != (stage1_context.filtered_count, stage1_context.filtered_count):
        raise ValueError("Saved Stage 3 matrix shape does not match filtered_count.")

    nan_check_passed, inf_check_passed = _check_no_nan_or_inf((A, R, S_tilde, C, S))
    value_range_passed = (
        _check_value_range(A, 0.0, 1.0)
        and _check_value_range(R, 0.0, 1.0)
        and _check_value_range(S_tilde, 0.0, 1.0)
        and _check_value_range(S, 0.0, 1.0)
        and _check_value_range(C, 0.0, 1.0, strict_lower=True)
    )

    expected_S_tilde = build_S_tilde(A, R, config.relation_lambda)
    expected_S = build_S(C, S_tilde)
    formula_S_tilde_passed = np.allclose(expected_S_tilde.data, S_tilde.data, atol=1.0e-6, rtol=0.0)
    formula_S_passed = np.allclose(expected_S.data, S.data, atol=1.0e-6, rtol=0.0)

    diagonal_coverage_passed = True
    for row_index in range(stage1_context.filtered_count):
        row_start = int(A.indptr[row_index])
        row_end = int(A.indptr[row_index + 1])
        row_cols = A.indices[row_start:row_end]
        diagonal_coverage_passed = diagonal_coverage_passed and bool(np.any(row_cols == row_index))

    reconstructed_support = build_final_support(S, config.final_topk)
    support_matches_topk_union = support_consistency_passed and np.array_equal(
        reconstructed_support.indptr,
        A.indptr,
    ) and np.array_equal(reconstructed_support.indices, A.indices)

    manifest_sha256_matches_meta = (
        meta.get("input_manifest_filtered_sha256") == stage1_context.manifest_filtered_sha256
    )
    sample_id_order_matches_meta = (
        meta.get("sample_id_order_sha256") == stage1_context.sample_id_order_sha256
    )
    feature_meta_matches = bool(
        meta.get("input_feature_meta_sha256") == stage2_context.input_feature_meta_sha256
        and meta.get("feature_cache_id") == config.feature_cache_id
        and meta.get("feature_meta", {}).get("stage_2_validator_passed") is True
        and meta.get("feature_meta", {}).get("stage_2_manifest_filtered_sha256")
        == stage2_context.meta_payload.get("manifest_filtered_sha256")
        and meta.get("feature_meta", {}).get("stage_2_sample_id_order_sha256")
        == stage2_context.meta_payload.get("sample_id_order_sha256")
    )

    required_meta_keys = {
        "dataset",
        "semantic_set_id",
        "protocol_name",
        "protocol_source",
        "feature_cache_id",
        "input_manifest_filtered_path",
        "input_manifest_filtered_sha256",
        "input_feature_cache_dir",
        "input_feature_meta_sha256",
        "sample_id_order_sha256",
        "filtered_count",
        "feature_dim",
        "direct_topk",
        "intra_topk",
        "final_topk",
        "lambda",
        "tau",
        "ann_backend",
        "hnsw_M",
        "hnsw_ef_construction",
        "hnsw_ef_search",
        "R_realization",
        "confidence_realization",
        "output_files",
        "A_nnz",
        "R_nnz",
        "S_tilde_nnz",
        "C_nnz",
        "S_nnz",
        "sparsity_density",
        "dtype",
        "created_at",
    }
    meta_fields_complete = required_meta_keys.issubset(set(meta.keys()))
    validator_passed = all(
        (
            csr_check_passed,
            dtype_check_passed,
            support_consistency_passed,
            value_range_passed,
            formula_S_tilde_passed,
            formula_S_passed,
            manifest_sha256_matches_meta,
            sample_id_order_matches_meta,
            feature_meta_matches,
            nan_check_passed,
            inf_check_passed,
            diagonal_coverage_passed,
            support_matches_topk_union,
            meta_fields_complete,
        )
    )

    return {
        "dataset": dataset_name,
        "semantic_set_id": config.semantic_set_id,
        "filtered_count": stage1_context.filtered_count,
        "matrix_shape": matrix_shape,
        "support_consistency_passed": support_consistency_passed,
        "value_range_passed": value_range_passed,
        "formula_S_tilde_passed": formula_S_tilde_passed,
        "formula_S_passed": formula_S_passed,
        "manifest_sha256_matches_meta": manifest_sha256_matches_meta,
        "sample_id_order_matches_meta": sample_id_order_matches_meta,
        "feature_meta_matches": feature_meta_matches,
        "csr_check_passed": csr_check_passed,
        "dtype_check_passed": dtype_check_passed,
        "nan_check_passed": nan_check_passed,
        "inf_check_passed": inf_check_passed,
        "diagonal_coverage_passed": diagonal_coverage_passed,
        "validator_passed": validator_passed,
        "support_matches_topk_union": support_matches_topk_union,
        "meta_fields_complete": meta_fields_complete,
    }


def write_semantic_validator_summary(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
    summary: Dict[str, Any],
) -> None:
    paths = config.build_cache_paths(dataset_name)
    write_json(paths.validator_summary_path, summary)


def build_run_summary(
    config: FormalSemanticRelationConfig,
    dataset_name: str,
    artifacts: SemanticRelationArtifacts,
) -> Dict[str, Any]:
    paths = config.build_cache_paths(dataset_name)
    return {
        "dataset": dataset_name,
        "semantic_set_id": config.semantic_set_id,
        "protocol_name": config.protocol_name,
        "A": _matrix_stats(artifacts.A),
        "R": _matrix_stats(artifacts.R),
        "S_tilde": _matrix_stats(artifacts.S_tilde),
        "C": _matrix_stats(artifacts.C),
        "S": _matrix_stats(artifacts.S),
        "semantic_cache_dir": str(paths.semantic_cache_dir),
        "meta_path": str(paths.meta_path),
    }
