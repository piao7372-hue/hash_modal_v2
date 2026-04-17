# Stage 4 Freeze Archive

Date: 2026-04-17
Stage: `stage_4_train`
Status: `frozen_for_archive`

## Scope

This archive freezes only the formal Stage 4 training implementation, its required configs, its validators, its freeze documents, and the small Stage 4 run artifacts that must remain in Git.

This archive does not authorize any Stage 5 implementation.

## Stage 4 Protocol Identity

- `stage_name = stage_4_train`
- Formal chain: `(X_I, X_T) -> (Z_I, Z_T) -> (Y_I, Y_T) -> (H_I, H_T) -> (B_I, B_T)`
- Formal prediction objects induced from continuous `H_I, H_T`: `Psi = {P_IT, P_II, P_TT}`
- Final supervision object: `S`
- `feature_cache_id = clip_vit_b32_formal_v1`
- `semantic_set_id = semantic_relation_highsignal_v1`
- `runtime.device = cuda:0`
- `runtime.dtype = float32`

## Only Allowed Frozen Inputs Consumed By Stage 4 Training

### Stage 1

- `data/processed/<dataset>/manifest/manifest_filtered.jsonl`
- `data/processed/<dataset>/splits/query_ids.txt`
- `data/processed/<dataset>/splits/retrieval_ids.txt`
- `data/processed/<dataset>/splits/train_ids.txt`

### Stage 2

- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/X_I.npy`
- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/X_T.npy`
- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/meta.json`
- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/validator_summary.json`

### Stage 3

- `data/processed/<dataset>/semantic_cache/semantic_relation_highsignal_v1/S.npz`
- `data/processed/<dataset>/semantic_cache/semantic_relation_highsignal_v1/meta.json`
- `data/processed/<dataset>/semantic_cache/semantic_relation_highsignal_v1/validator_summary.json`

The final Stage 4 semantic training target is only `S`.

The following are not formal Stage 4 loss targets:

- `A`
- `R`
- `S_tilde`
- `C`

## Frozen Formal Configuration

- `model_chebykan.input_dim_image = 512`
- `model_chebykan.input_dim_text = 512`
- `model_chebykan.d_z = 256`
- `model_chebykan.polynomial_order = 3`
- `model_chebykan.hidden_dims = [512, 256]`
- `model_tree.tree_depth = 2`
- `model_tree.prototype_counts = [64, 32]`
- `model_tree.feature_dim = 256`
- `model_graph.input_dim = 256`
- `model_graph.f_dim = 128`
- `model_graph.k_neighbors = 7`
- `model_graph.propagation_steps = 1`
- `hash_head.hash_bits = 64`
- `loss.alpha = 0.5`
- `loss.lambda_sem = 1.0`
- `loss.lambda_pair = 0.2`
- `loss.lambda_q = 0.1`
- `loss.lambda_bal = 0.01`
- `training.batch_size = 50`
- `training.num_epochs = 1`
- `training.learning_rate = 0.001`
- `training.weight_decay = 0.0001`
- `training.grad_clip_norm = 5.0`
- `training.seed = 0`

## Real Formal Run Frozen In This Archive

- `dataset = mirflickr25k`
- `run_name = stage4_formal_v1_mirflickr25k_run01`
- `device = cuda:0`
- `dtype = float32`
- `batch_size = 50`
- `num_epochs = 1`
- `train_count = 5000`
- `global_step = 100`

## Real Successful Command Record

### Training

Command:

```powershell
& "C:\Users\ASVS\anaconda3\envs\deeplearning\python.exe" scripts\run_train.py --dataset mirflickr25k --config configs\train.yaml --run-name stage4_formal_v1_mirflickr25k_run01
```

Result:

- `exit_code = 0`
- `best_metric_name = loss_total`
- `best_metric_value = 2.399960253238678`
- `run_dir = outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01`

### Model Output Validation

Command:

```powershell
& "C:\Users\ASVS\anaconda3\envs\deeplearning\python.exe" scripts\validate_model_outputs.py --dataset mirflickr25k --config configs\train.yaml --run-name stage4_formal_v1_mirflickr25k_run01
```

Result:

- `exit_code = 0`
- `validator_passed = true`
- `H_range_ok = true`
- `B_binary_ok = true`
- `P_IT_shape = [50, 50]`
- `P_II_shape = [50, 50]`
- `P_TT_shape = [50, 50]`

### Training Output Validation

Command:

```powershell
& "C:\Users\ASVS\anaconda3\envs\deeplearning\python.exe" scripts\validate_training_outputs.py --dataset mirflickr25k --config configs\train.yaml --run-name stage4_formal_v1_mirflickr25k_run01
```

Result:

- `exit_code = 0`
- `validator_passed = true`
- `required_files_present.best_checkpoint = true`
- `required_files_present.last_checkpoint = true`
- `required_files_present.training_summary = true`
- `required_files_present.validator_summary = true`
- `checkpoint_device = cuda:0`
- `checkpoint_dtype = float32`

## Output Directory Structure

Frozen run directory:

- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/config_snapshot.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/train_log.jsonl`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/last.pt`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/best.pt`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/training_summary.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/validator_summary.json`

## Git Archival Strategy

### Files That Enter Git

- `configs/model_chebykan.yaml`
- `configs/model_tree.yaml`
- `configs/model_graph.yaml`
- `configs/hash_head.yaml`
- `configs/loss.yaml`
- `configs/train.yaml`
- `src/models/`
- `src/losses/`
- `src/engine/`
- `scripts/run_train.py`
- `scripts/validate_model_outputs.py`
- `scripts/validate_training_outputs.py`
- `tests/test_imports.py`
- `docs/stage4_freeze_archive.md`
- `docs/stage4_artifact_hashes.txt`
- `docs/stage5_input_contract.md`
- `docs/stage5_adaptation_risk_checklist.md`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/config_snapshot.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/training_summary.json`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/validator_summary.json`

### Files That Do Not Enter Git By Default

- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/best.pt`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/last.pt`
- `outputs/train/mirflickr25k/stage4_formal_v1_mirflickr25k_run01/train_log.jsonl`
- `data/processed/*/feature_cache/*/X_I.npy`
- `data/processed/*/feature_cache/*/X_T.npy`
- `data/processed/*/semantic_cache/*/*.npz`
- `data/raw/`

The local-only Stage 4 run artifacts are frozen by SHA256 in `docs/stage4_artifact_hashes.txt`.

## `best_metric_value` Meaning

`best_metric_value` is not mAP.

For this frozen Stage 4 run:

- `best_metric_name = loss_total`
- `best_metric_value = 2.399960253238678`

Its exact meaning is:

- the epoch-level average of `loss_total`
- measured on the training run recorded in `training_summary.json`
- used only as the Stage 4 checkpoint-selection metric

This archive does not rename the stored field in `training_summary.json`, because the run artifact is being frozen as-produced. The ambiguity is resolved by freezing `best_metric_name = loss_total` alongside this explanation.

Stage 5 and any later stage must not interpret `best_metric_value` as retrieval mAP.

## Artifact Hash Record

The following Stage 4 artifacts have frozen SHA256 digests in `docs/stage4_artifact_hashes.txt`:

- `best.pt`
- `last.pt`
- `config_snapshot.json`
- `training_summary.json`
- `validator_summary.json`
- `train_log.jsonl`

## Freeze Boundary

- This archive freezes Stage 4 training only.
- It does not authorize `run_encode.py`.
- It does not authorize `run_eval.py`.
- It does not authorize any Stage 5 code.
- It does not change Stage 1, Stage 2, or Stage 3 frozen identities.
