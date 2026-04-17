# Stage 3 Freeze Archive

Date: 2026-04-17
Stage: `stage_3_semantic_relation`
Status: `frozen_for_archive`

## Scope

This archive freezes the formal Stage 3 semantic relation code, configs, tests, documentation, and the small formal Stage 3 artifacts that must remain in Git:

- `configs/semantic_relation.yaml`
- `scripts/run_semantic_relation.py`
- `scripts/validate_semantic_cache.py`
- `src/semantic/`
- `tests/test_imports.py`
- `docs/stage3_freeze_archive.md`
- `docs/stage3_artifact_hashes.txt`
- `data/processed/<dataset>/semantic_cache/semantic_relation_highsignal_v1/meta.json`
- `data/processed/<dataset>/semantic_cache/semantic_relation_highsignal_v1/validator_summary.json`

The large sparse matrices are intentionally excluded from Git:

- `A.npz`
- `R.npz`
- `S_tilde.npz`
- `C.npz`
- `S.npz`

Their SHA256 digests are frozen in `docs/stage3_artifact_hashes.txt`.

## Stage 3 Protocol Identity

- `semantic_set_id = semantic_relation_highsignal_v1`
- `protocol_name = semantic_relation_highsignal_v1`
- `protocol_source = project_defined_formal_stage3`

## Frozen Parameters

- `feature_cache_id = clip_vit_b32_formal_v1`
- `ann_backend = hnswlib`
- `direct_topk = 200`
- `intra_topk = 100`
- `final_topk = 50`
- `hnsw_M = 32`
- `hnsw_ef_construction = 200`
- `hnsw_ef_search = 256`
- `lambda = 0.7`
- `tau = 0.07`
- `dtype = float32`
- `R_realization = topk_sparse_profile_cosine_v1`
- `confidence_realization = sparse_bidirectional_softmax_v1`

## Final Dataset Statistics

### MIRFlickr-25K

- `shape = [20015, 20015]`
- `nnz = 1369340`
- `density = 0.003418220746131633`
- `validator_passed = true`

### MSCOCO

- `shape = [123287, 123287]`
- `nnz = 8480556`
- `density = 0.0005579429015839454`
- `validator_passed = true`

### NUS-WIDE

- `shape = [186577, 186577]`
- `nnz = 13714231`
- `density = 0.0003939628304017828`
- `validator_passed = true`

## Revalidation Snapshot

The current formal Stage 3 artifacts were rechecked immediately before freeze. All three datasets passed:

- `validator_passed`
- `support_consistency_passed`
- `formula_S_tilde_passed`
- `formula_S_passed`
- `manifest_sha256_matches_meta`
- `sample_id_order_matches_meta`

## Dependency Record

- `hnswlib` is a formal Stage 3 dependency.
- `hnswlib` is not a temporary dependency and not a debug-only dependency.
- Stage 3 formal execution in the `deeplearning` environment required installing `hnswlib` into `C:\Users\ASVS\anaconda3\envs\deeplearning`.
- The package was installed from `conda-forge` because `pip install hnswlib` failed on this Windows environment due to missing local MSVC build tools.

## Git Archival Strategy

- Code, configs, tests, freeze docs, and small JSON artifacts are archived in Git.
- Large Stage 3 sparse matrices remain local-only and do not enter Git by default.
- Their frozen SHA256 digests, byte sizes, shapes, and nnz counts are archived in `docs/stage3_artifact_hashes.txt`.

## Freeze Boundary

- This archive freezes only Stage 3 semantic relation artifacts.
- It does not modify Stage 1 or Stage 2 frozen protocols.
- It does not authorize any Stage 4 model, loss, training, evaluation, graph block, or hash-head implementation.
