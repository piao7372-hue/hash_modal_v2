# Stage 2 Freeze Archive

Date: 2026-04-17
Stage: `stage_2_feature_extraction`
Status: `frozen_for_archive`

## Scope

This archive freezes the formal Stage 2 feature extraction code, configs, tests, documentation, and small formal Stage 2 artifacts:

- `configs/feature_extraction.yaml`
- `scripts/run_feature_extraction.py`
- `scripts/validate_feature_cache.py`
- `src/features/`
- `tests/test_imports.py`
- `docs/stage2_freeze_archive.md`
- `docs/stage2_artifact_hashes.txt`
- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/meta.json`
- `data/processed/<dataset>/feature_cache/clip_vit_b32_formal_v1/validator_summary.json`

The large feature matrices are intentionally excluded from Git and remain local-only.

## Protocol Sources

Stage 2 is frozen under the combined authority of:

- `跨模态哈希项目工程化实现文档_正式版_预处理修订版.docx`
- `跨模态哈希项目工程化实现文档_预处理修订版.pdf`
- `Stage2_正式特征编码协议补充说明.docx`
- `Stage2_正式特征编码协议补充说明.pdf`

## Frozen Protocol

- `feature_set_id = clip_vit_b32_formal_v1`
- `model_name = openai/clip-vit-base-patch32`
- `image_size = 224`
- `resize_mode = shortest_edge`
- `interpolation = bicubic`
- `crop_mode = center_crop`
- `max_length = 77`
- `image_batch_size = 64`
- `text_batch_size = 256`
- `dtype = float32`
- `device = cuda:0`
- `l2_normalized = true`

## Final Dataset Shapes

### MIRFlickr-25K

- `filtered_count = 20015`
- `X_I.shape = [20015, 512]`
- `X_T.shape = [20015, 512]`

### NUS-WIDE

- `filtered_count = 186577`
- `X_I.shape = [186577, 512]`
- `X_T.shape = [186577, 512]`

### MSCOCO

- `filtered_count = 123287`
- `X_I.shape = [123287, 512]`
- `X_T.shape = [123287, 512]`

## Validation Summary

- `compileall`: passed
- `tests/test_imports.py`: passed
- MIR feature validator: passed
- NUS-WIDE feature validator: passed
- MSCOCO feature validator: passed
- `dtype = float32`: passed for all three datasets
- `NaN check`: passed for all three datasets
- `Inf check`: passed for all three datasets
- `L2 norm check`: passed for all three datasets
- `manifest hash consistency`: passed for all three datasets
- `sample order digest consistency`: passed for all three datasets

## Local-Only Large Artifacts

The following large feature caches are intentionally excluded from Git. Their paths, shapes, sizes, and SHA256 digests are recorded in `docs/stage2_artifact_hashes.txt`.

- `data/processed/mirflickr25k/feature_cache/clip_vit_b32_formal_v1/X_I.npy`
- `data/processed/mirflickr25k/feature_cache/clip_vit_b32_formal_v1/X_T.npy`
- `data/processed/nuswide/feature_cache/clip_vit_b32_formal_v1/X_I.npy`
- `data/processed/nuswide/feature_cache/clip_vit_b32_formal_v1/X_T.npy`
- `data/processed/mscoco/feature_cache/clip_vit_b32_formal_v1/X_I.npy`
- `data/processed/mscoco/feature_cache/clip_vit_b32_formal_v1/X_T.npy`

## Notes

- Stage 2 inputs are inherited directly from Stage 1 frozen `manifest_filtered.jsonl`.
- Stage 2 does not modify `sample_id`, `text_source`, `label_vector`, or Stage 1 splits.
- Stage 3 and later stages remain out of scope for this archive.
