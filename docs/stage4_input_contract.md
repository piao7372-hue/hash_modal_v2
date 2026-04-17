# Stage 4 Input Contract

Date: 2026-04-17
Status: `audit_only`
Scope: `freeze_compatibility_contract_only`

## Purpose

This document fixes the compatibility contract that all Stage 4 and later implementations must obey when consuming the already frozen Stage 1, Stage 2, and Stage 3 artifacts.

This document does not authorize any Stage 4 implementation.

## Frozen Inputs That Later Stages May Consume

### Stage 1

The only allowed frozen Stage 1 inputs are:

- `manifest_filtered.jsonl`
- `query_ids.txt`
- `retrieval_ids.txt`
- `train_ids.txt`

### Stage 2

The only allowed frozen Stage 2 inputs are:

- `X_I.npy`
- `X_T.npy`
- `meta.json`
- `validator_summary.json`

### Stage 3

The only allowed frozen Stage 3 inputs are:

- `S.npz`
- `meta.json`
- `validator_summary.json`

Later stages may inspect `A.npz`, `R.npz`, `S_tilde.npz`, and `C.npz` only for Stage 3 audit purposes. They are not the formal supervision target for later training stages.

## Formal Supervision Object

- The only formal semantic supervision object that later stages may use is `S`.
- `A` is not a later-stage supervision target.
- `R` is not a later-stage supervision target.
- `S_tilde` is not a later-stage supervision target.
- `C` is not a later-stage supervision target.

## Explicitly Forbidden Direct Consumption

The following are forbidden:

- Using `A` as the final supervision target.
- Using `R` as the final supervision target.
- Using `S_tilde` as the final supervision target.
- Using `C` as the final supervision target.
- Recomputing Stage 2 features instead of consuming frozen `X_I.npy` and `X_T.npy`.
- Rewriting Stage 1 manifest files or split files.
- Treating the Stage 3 sparse semantic matrix as if it were a dense full-label matrix and wiring it directly into later code without sparse-aware handling.

## Order Contract

- Stage 4 and every later stage must inherit the Stage 1 `manifest_filtered.jsonl` line order.
- Every later stage must validate `sample_id_order_sha256`.
- Every later stage must validate `manifest_filtered_sha256`.
- No later stage may reorder samples unless it explicitly builds and validates an index remapping layer.
- Any batch index, sparse gather index, or sample-id lookup must map back to the frozen Stage 1 manifest order.

## Split Contract

- The only legal split names are `train`, `query`, and `retrieval`.
- Later training code may use only `train`.
- Later evaluation code must use the `query` and `retrieval` split contract.
- Query or retrieval samples must not leak into training supervision.
- The full filtered set must not be treated as the default training set unless a later-stage protocol explicitly documents and validates a new split contract. This audit does not authorize such a change.

## Sparse Semantic Supervision Contract

- `S.npz` is a sparse high-signal supervision matrix.
- If later stages need batch-level supervision, they must gather batch submatrices from `S` by batch indices aligned to the frozen manifest order.
- Later stages must not densify the full `S` matrix by default.
- Later stages must not drop sparse support and replace it with a dense all-zero matrix.
- Later stages must not rewrite `S` into a different semantic object for convenience.
- Later stages must preserve the meaning that stored nonzero entries are real-valued supervision scores in `[0, 1]`.

## Symbol Contract

### Frozen Current Symbols

- `A`
- `R`
- `S̃`
- `C`
- `S`
- `X_I`
- `X_T`

### Reserved Later-Stage Symbols

- `H_I`
- `H_T`
- `B_I`
- `B_T`
- later loss-related symbols

### Symbol Rules

- Later stages must not rename `R` to `H`.
- Later stages must not rename `S_tilde` back to `Se`.
- Later stages must not mix `S_tilde`, `Se`, and `S_hat` in comments or code.
- Later-stage symbols such as `H_I`, `H_T`, `B_I`, and `B_T` belong to later-stage model internals and must not be backfilled into Stage 3 semantics.

## Configuration Adaptation Contract

Every later-stage config must explicitly carry:

- `feature_cache_id = clip_vit_b32_formal_v1`
- `semantic_set_id = semantic_relation_highsignal_v1`

The following are forbidden:

- Scanning directories and guessing the latest feature cache.
- Scanning directories and guessing the latest semantic cache.
- Inferring artifact identity from directory names without explicit config fields.
- Filling missing IDs automatically.

## Minimum Later-Stage Validation Requirements

Before later-stage code starts model work, it must validate:

- Stage 1 `manifest_filtered_sha256` matches the later-stage config expectation.
- Stage 1 `sample_id_order_sha256` matches Stage 2 and Stage 3 metadata.
- Stage 2 `validator_summary.json` reports `validator_passed = true`.
- Stage 3 `validator_summary.json` reports `validator_passed = true`.
- The loaded `S.npz` shape matches the Stage 1 filtered count.
- Any gathered batch submatrix of `S` uses indices aligned to the frozen manifest order.

## Freeze Boundary

This contract is compatibility documentation only.

It does not implement:

- Stage 4 model code
- losses
- training loops
- evaluation
- ChebyKAN
- graph blocks
- hash heads
- later-stage caches
