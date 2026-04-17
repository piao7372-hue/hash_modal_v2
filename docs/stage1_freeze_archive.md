# Stage 1 Freeze Archive

Date: 2026-04-17
Stage: `stage_1_preprocess`
Status: `frozen_for_archive`

## Scope

This archive freezes the Stage 1 preprocessing code, configs, tests, and the formal Stage 1 small artifacts:

- `configs/`
- `scripts/`
- `src/`
- `tests/`
- `docs/`
- `data/processed/<dataset>/reports/`
- `data/processed/<dataset>/splits/`
- `data/processed/<dataset>/manifest/manifest_meta.json`

The raw datasets are not archived in Git.

## Final Dataset Counts

### MIRFlickr-25K

- `raw = 25000`
- `filtered = 20015`
- `query = 2000`
- `retrieval = 18015`
- `train = 5000`
- `train subset retrieval = true`
- `filtered_empty_text_rows = 0`
- `empty_text_rows_removed = 2128`
- `protocol_name = pragmatic_high_signal_v1`
- `protocol_source = project_defined_not_ra_literal`

### NUS-WIDE

- `raw = 269648`
- `filtered = 186577`
- `query = 2000`
- `retrieval = 184577`
- `train = 5000`
- `train subset retrieval = true`

### MSCOCO

- `raw = 123287`
- `filtered = 123287`
- `query = 2000`
- `retrieval = 121287`
- `train = 5000`
- `train subset retrieval = true`

## Validation Summary

- `compileall`: passed
- `tests/test_imports.py`: passed
- MIR validator: passed
- NUS-WIDE validator: passed
- MSCOCO validator: passed
- rerun consistency: passed for all three datasets

## Local-Only Large Artifacts

The following large manifest files are intentionally excluded from Git. They remain on the local machine under `data/processed/` and are recorded in `docs/stage1_artifact_hashes.txt`.

- `data/processed/mirflickr25k/manifest/manifest_raw.jsonl`
- `data/processed/mirflickr25k/manifest/manifest_filtered.jsonl`
- `data/processed/nuswide/manifest/manifest_raw.jsonl`
- `data/processed/nuswide/manifest/manifest_filtered.jsonl`
- `data/processed/mscoco/manifest/manifest_raw.jsonl`
- `data/processed/mscoco/manifest/manifest_filtered.jsonl`

## Notes

- Stage 2 and later stages are out of scope for this archive.
- MIRFlickr-25K is frozen under the project-defined pragmatic preprocessing protocol, not as an RA-literal reproduction.

