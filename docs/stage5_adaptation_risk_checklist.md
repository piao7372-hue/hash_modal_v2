# Stage 5 Adaptation Risk Checklist

Date: 2026-04-17
Status: `freeze_checklist_only`
Scope: `stage5_planning_without_stage5_implementation`

This checklist is a freeze-time planning artifact only. It does not implement Stage 5.

## 1. Wrong Run Selection Risk

- Risk: Stage 5 consumes a different Stage 4 run than the frozen validated run.
- Trigger: Code scans `outputs/train/` and picks the newest directory.
- Consequence: Encode or eval can drift to an unvalidated checkpoint.
- Prevention: Pin `dataset = mirflickr25k` and `run_name = stage4_formal_v1_mirflickr25k_run01` explicitly.

## 2. Wrong Checkpoint Selection Risk

- Risk: Stage 5 consumes `last.pt` instead of `best.pt`.
- Trigger: Implementation treats both checkpoints as interchangeable.
- Consequence: Stage 5 results no longer correspond to the frozen formal checkpoint selection rule.
- Prevention: Default strictly to `best.pt`; allow `last.pt` only under explicit audit-only override.

## 3. `best_metric_value` Misread As mAP Risk

- Risk: Stage 5 interprets `best_metric_value` as retrieval quality.
- Trigger: The field name is read without checking `best_metric_name`.
- Consequence: Future reports can falsely claim Stage 4 already measured mAP.
- Prevention: Require Stage 5 code and docs to read `best_metric_name` and document that `best_metric_value` here is `loss_total`, not mAP.

## 4. Validator Bypass Risk

- Risk: Stage 5 loads the checkpoint without checking the Stage 4 validators.
- Trigger: Implementation treats checkpoint existence as sufficient.
- Consequence: Encode or eval can proceed from a run whose structure or device contract was never validated.
- Prevention: Refuse Stage 5 execution unless both `model_outputs.validator_passed` and `training_outputs.validator_passed` are `true`.

## 5. Device Drift Risk

- Risk: Stage 5 falls back to CPU or another device.
- Trigger: `cuda:0` is unavailable and code silently degrades.
- Consequence: Runtime behavior diverges from the frozen formal environment.
- Prevention: Keep the same fail-fast `cuda:0` requirement used by Stage 4.

## 6. Manifest Order Drift Risk

- Risk: Encoded outputs no longer align with Stage 1 manifest order.
- Trigger: Stage 5 iterates splits or datasets without preserving the canonical manifest index map.
- Consequence: Retrieval codes and metadata point at the wrong samples.
- Prevention: Carry forward the Stage 1 `sample_id -> manifest index` mapping and validate order hashes before any encode output is written.

## 7. Split Leakage Risk

- Risk: Future eval mixes `train`, `query`, and `retrieval`.
- Trigger: Stage 5 treats the full filtered set as one undifferentiated encode/eval pool.
- Consequence: Reported retrieval performance becomes contaminated and not comparable.
- Prevention: Keep `query_ids.txt` and `retrieval_ids.txt` as the only legal evaluation split definition.

## 8. Feature/Semantic Identity Drift Risk

- Risk: Stage 5 encodes or evaluates against the wrong `feature_cache_id` or `semantic_set_id`.
- Trigger: Config fields are omitted and the code guesses artifact identity.
- Consequence: Stage 5 no longer corresponds to the frozen Stage 4 training basis.
- Prevention: Require explicit equality checks against `clip_vit_b32_formal_v1` and `semantic_relation_highsignal_v1`.

## 9. Checkpoint Mutation Risk

- Risk: Stage 5 overwrites or amends the frozen Stage 4 run directory.
- Trigger: Later code writes new files back into the same run folder without a separate namespace.
- Consequence: The frozen run ceases to be a stable archival reference.
- Prevention: Treat `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/` as immutable.

## 10. Binary/Continuous Semantics Drift Risk

- Risk: Stage 5 confuses `H_I/H_T` and `B_I/B_T` semantics when exporting or evaluating.
- Trigger: A future encode path uses post-sign codes where continuous outputs are required, or vice versa, without an explicit protocol decision.
- Consequence: Downstream evaluation becomes incomparable across runs.
- Prevention: Stage 5 must explicitly document whether it consumes continuous `H`-derived outputs or binary `B` codes and must not infer that choice implicitly from Stage 4 training internals.

## Freeze Boundary

- This checklist is for Stage 5 planning only.
- It does not implement `run_encode.py`.
- It does not implement `run_eval.py`.
- It does not modify the frozen Stage 4 run.
