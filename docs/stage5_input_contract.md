# Stage 5 Input Contract

Date: 2026-04-17
Status: `freeze_contract_only`
Scope: `stage5_must_consume_stage4_freeze_without_redefinition`

This document does not implement Stage 5.

It freezes what a future Stage 5 implementation is allowed to consume from the already validated Stage 4 run.

## Stage 5 Default Source Of Truth

The only validated Stage 4 run currently authorized as Stage 5 input is:

- `dataset = mirflickr25k`
- `run_name = stage4_formal_v1_mirflickr25k_run01`
- `run_dir = outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01`

## Stage 5 Allowed Inputs

### Inherited Frozen Inputs

Stage 5 must continue to inherit the same frozen Stage 1, Stage 2, and Stage 3 identities:

- `feature_cache_id = clip_vit_b32_formal_v1`
- `semantic_set_id = semantic_relation_highsignal_v1`
- `manifest_filtered.jsonl`
- `query_ids.txt`
- `retrieval_ids.txt`
- `train_ids.txt`
- `X_I.npy`
- `X_T.npy`
- `S.npz`

Stage 5 is not authorized to replace any of those identities.

### Stage 4 Inputs Stage 5 May Consume

- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/best.pt`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/config_snapshot.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/training_summary.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/validator_summary.json`

### Audit-Only Stage 4 Files

- `last.pt`
- `train_log.jsonl`

These may be inspected for audit or debugging, but they are not the default formal Stage 5 consumption path.

## Stage 5 Default Checkpoint Rule

- The default checkpoint for Stage 5 must be `best.pt`.
- `last.pt` is not the default Stage 5 checkpoint.
- A future Stage 5 implementation must not silently switch from `best.pt` to `last.pt`.

## Stage 5 Required Preflight Checks

Before any Stage 5 code runs, it must validate:

- `config_snapshot.json` exists.
- `training_summary.json` exists.
- `validator_summary.json` exists.
- `best.pt` exists.
- `config_snapshot.runtime.device == cuda:0`
- `config_snapshot.runtime.dtype == float32`
- `config_snapshot.inputs.feature_cache_id == clip_vit_b32_formal_v1`
- `config_snapshot.inputs.semantic_set_id == semantic_relation_highsignal_v1`
- `training_summary.dataset == mirflickr25k`
- `training_summary.run_name == stage4_formal_v1_mirflickr25k_run01`
- `validator_summary.model_outputs.validator_passed == true`
- `validator_summary.training_outputs.validator_passed == true`

## Stage 5 Meaning Of `best_metric_value`

Stage 5 must treat the Stage 4 field:

- `best_metric_name = loss_total`
- `best_metric_value = 2.399960253238678`

as a Stage 4 checkpoint-selection loss statistic only.

Stage 5 must not reinterpret this field as:

- retrieval mAP
- precision
- recall
- any encode-time or eval-time metric

## Stage 5 Forbidden Actions

- Do not scan `outputs/train/` and guess the newest run.
- Do not replace `best.pt` with another checkpoint implicitly.
- Do not mutate `config_snapshot.json`, `training_summary.json`, or `validator_summary.json`.
- Do not regenerate Stage 2 features.
- Do not regenerate Stage 3 semantic matrices.
- Do not rewrite Stage 1 manifest order or split files.
- Do not fall back to CPU if `cuda:0` is unavailable.
- Do not infer artifact identity from directory names without explicit config fields.

## Stage 5 Split Contract

- Any future encode path must preserve Stage 1 manifest row order.
- Any future eval path must obey `query_ids.txt` and `retrieval_ids.txt`.
- Stage 5 must not leak `train_ids.txt` into retrieval evaluation.

## Freeze Boundary

- This is a Stage 5 input contract only.
- It does not implement encode.
- It does not implement eval.
- It does not change the frozen Stage 4 run.
