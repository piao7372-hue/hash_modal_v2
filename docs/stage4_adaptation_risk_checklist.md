# Stage 4 Adaptation Risk Checklist

Date: 2026-04-17
Status: `audit_only`
Scope: `compatibility_risk_checklist_only`

This checklist identifies the main compatibility failures that can damage later retrieval mAP or break the frozen Stage 1, 2, and 3 contracts.

## 1. Order Misalignment Risk

- Risk description: Stage 1, Stage 2, and Stage 3 sample order diverges, so features, semantic supervision, and split membership no longer refer to the same sample rows.
- Trigger condition: A later stage shuffles rows, rebuilds its own indexing, or loads artifacts without checking manifest hashes and order hashes.
- mAP consequence: Semantic supervision is applied to the wrong samples, which corrupts learning targets and can collapse retrieval quality.
- Code-level prevention: Require explicit validation of `manifest_filtered_sha256` and `sample_id_order_sha256` before training or evaluation starts.
- Validator-level check: Add a later-stage validator that compares Stage 1, Stage 2, and Stage 3 hashes and rejects any run with mismatched order metadata.

## 2. Supervision Target Misuse Risk

- Risk description: Later code uses `A`, `R`, `S_tilde`, or `C` as the final target instead of using `S`.
- Trigger condition: Engineers wire the wrong Stage 3 artifact into later losses because intermediate matrices appear semantically meaningful.
- mAP consequence: The final bidirectional confidence calibration is lost, which increases semantic noise and weakens high-signal supervision.
- Code-level prevention: Expose only `S` through the later-stage training data interface and keep intermediate Stage 3 matrices out of the formal training path.
- Validator-level check: Require later-stage config and logs to record that the semantic target path is `S.npz`, not any other Stage 3 matrix.

## 3. Sparse Semantic Densification Risk

- Risk description: Full `S` is converted to a dense matrix for convenience.
- Trigger condition: A later implementation uses `.toarray()`, `todense()`, or equivalent full densification on the full Stage 3 matrix.
- mAP consequence: Memory pressure grows sharply, sparse support semantics get blurred, and downstream code may start treating missing edges as explicit zero labels.
- Code-level prevention: Keep `S` in sparse form and gather only batch-aligned submatrices.
- Validator-level check: Add runtime assertions that the full formal `S` object remains sparse and log any dense conversion attempt as a hard failure.

## 4. Split Leakage Risk

- Risk description: Query or retrieval samples leak into training supervision.
- Trigger condition: Later training code defaults to the full filtered set or merges split files without enforcement.
- mAP consequence: Offline evaluation becomes contaminated, inflated, and untrustworthy; learned representations can overfit evaluation samples.
- Code-level prevention: Make the training data loader consume only `train_ids.txt` and make the evaluation code consume only `query_ids.txt` and `retrieval_ids.txt`.
- Validator-level check: Add set-intersection checks proving that query does not intersect retrieval incorrectly and that training does not absorb query.

## 5. Symbol Drift Risk

- Risk description: Old names such as `Se`, `H`, or `S_hat` return and blur the frozen meaning of Stage 3 symbols.
- Trigger condition: Later code comments, configs, or docs are written from stale drafts or older experiments.
- mAP consequence: Engineers can connect the wrong tensors or losses because symbol meanings drift across files.
- Code-level prevention: Keep a single symbol table in docs and enforce the Stage 3 naming contract in code review.
- Validator-level check: Add static text checks over later-stage source files for banned symbol spellings and fail if they appear in the semantic supervision path.

## 6. Configuration Drift Risk

- Risk description: Later stages omit `feature_cache_id` or `semantic_set_id` and guess artifacts implicitly.
- Trigger condition: A config is written with partial fields and code scans directories for the latest output.
- mAP consequence: A run can silently consume the wrong frozen artifacts, producing irreproducible supervision and unstable results.
- Code-level prevention: Require both IDs explicitly in every later-stage config and reject missing values.
- Validator-level check: Add config-schema validation that fails when `feature_cache_id` or `semantic_set_id` is absent or mismatched.

## 7. Value-Range Assumption Risk

- Risk description: Later code assumes `S` is a boolean matrix or a `{-1, +1}` target instead of a sparse real-valued matrix in `[0, 1]`.
- Trigger condition: Loss code is ported from binary-pair supervision assumptions without adapting to Stage 3 semantics.
- mAP consequence: Supervision strength is distorted, calibrated confidence is destroyed, and optimization can move in the wrong direction.
- Code-level prevention: Document and assert that gathered semantic targets from `S` remain float-valued in `[0, 1]`.
- Validator-level check: Add range checks over gathered batch supervision and reject sign-flipped or thresholded substitutions unless a later formal protocol explicitly authorizes them.

## 8. Diagonal Edge Loss Risk

- Risk description: Later batch gathering or submatrix slicing drops the guaranteed diagonal high-confidence edges preserved by Stage 3.
- Trigger condition: Batch gather code filters support or remaps rows and columns incorrectly.
- mAP consequence: Paired self-alignment anchors disappear, weakening the strongest direct supervision signal.
- Code-level prevention: Preserve diagonal entries whenever a same-batch image-text submatrix is gathered from `S`.
- Validator-level check: Add batch-level diagnostics proving that in-batch diagonal entries remain present when the corresponding samples are included.

## 9. Batch Gather Index Misalignment Risk

- Risk description: Later code gathers a submatrix from `S` with batch indices that do not match Stage 1 manifest positions.
- Trigger condition: A dataloader uses local batch order, shuffled subset order, or sample-id lists without converting them back to frozen global indices.
- mAP consequence: The wrong supervision block is gathered, which silently poisons semantic training.
- Code-level prevention: Build and validate one canonical `sample_id -> manifest index` map and use it for every sparse gather.
- Validator-level check: Add batch trace tests that compare gathered indices against expected manifest positions for known sample IDs.

## 10. Memory Strategy Risk

- Risk description: Later code loads the full Stage 3 supervision into dense memory because it is simpler to code.
- Trigger condition: An implementation chooses convenience over sparse-aware batch access.
- mAP consequence: Large datasets such as NUS-WIDE become memory-bound, which pushes developers toward ad hoc truncation or lossy shortcuts that damage supervision quality.
- Code-level prevention: Keep Stage 3 loading sparse-first and expose only sparse gather utilities to later code.
- Validator-level check: Add stress tests on the largest frozen dataset and reject later implementations that require full dense materialization.

## 11. Stage 2 Feature Recompute Risk

- Risk description: Later stages recompute CLIP features instead of consuming frozen `X_I` and `X_T`.
- Trigger condition: A later pipeline tries to keep all preprocessing inside one training script.
- mAP consequence: Feature drift breaks reproducibility and decouples Stage 3 supervision from the actual feature basis it was built on.
- Code-level prevention: Load only the frozen Stage 2 feature cache referenced by explicit `feature_cache_id`.
- Validator-level check: Compare later-stage feature-cache metadata against the frozen Stage 2 `meta.json` and fail on mismatch.

## 12. Stage 3 Semantic Cache Identity Drift Risk

- Risk description: Later stages consume a semantic cache that is not `semantic_relation_highsignal_v1`.
- Trigger condition: Artifact paths are inferred automatically or overwritten by ad hoc experiments.
- mAP consequence: Later training may use a different semantic definition than the frozen one, invalidating comparisons.
- Code-level prevention: Require explicit `semantic_set_id` in later-stage config and resolve exactly one matching cache.
- Validator-level check: Compare later-stage config identity against Stage 3 `meta.json` and fail on any mismatch.

## Freeze Boundary

This checklist is an audit artifact only. It does not implement any later-stage model, loss, training loop, evaluation, graph block, ChebyKAN component, hash head, or cache.
