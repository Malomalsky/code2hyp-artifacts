# CodeNet Python800 Stage A confirmatory test, RunPod launch v1

Date: 2026-07-29

Role: single registered confirmatory test opening after frozen validation
selection.

## Validation Input

Validation input directory on the RunPod pod:

`/workspace/code2hyp_test/validation/codenet_python800_stage_a_validation_runpod_v1/`

The validation selection was sealed on the pod before test launch:

- Selected cell: `HEE_c3_true_LCA`
- Selected active curvature: `3.0`
- `registered_seed_set_complete = true`
- `all_seed_results_match_their_seals = true`
- `selection_recomputed_from_frozen_rule = true`
- `validation_only = true`

## Test Runner

- Test pod id: `cdm4x58qmsgqiz`
- Pod accelerator available: NVIDIA GeForce RTX 3090
- Computation device used by runner v4: CPU
- Cost at creation: `$0.22/hr`
- Remote runner directory:
  `/workspace/code2hyp_test/code2hyp_stage_a_test_runner_v4/`
- Runner commit: `6b479a441c603f2bd7331df1a7316d7d1c18e5a2`
- Runner tag: `codenet-stage-a-test-runner-v4`

The device description was corrected after a source-level audit on 2026-08-20;
see `reports/codenet_python800_stage_a_compute_device_correction_2026-08-20.md`.

The runner was cloned cleanly on the pod and checked out to the immutable test
tag. The first copied local worktree was not used because its `.git` file
referred to a local macOS worktree path.

## Data Paths

- Source tree:
  `/workspace/code2hyp_test/source/Project_CodeNet_Python800/`
- D5 metadata index:
  `/workspace/code2hyp_test/code2hyp_stage_a_test_runner_v4/data/codenet_python800_d5_metadata/d5_metadata_index.jsonl`
- Test output directory:
  `/workspace/code2hyp_test/test_runpod/codenet_python800_stage_a_test_v1/`
- Logs:
  `/workspace/code2hyp_test/test_runpod/codenet_python800_stage_a_test_v1/logs/`

## Opening Status

The first launch created `test_opening_receipt.json` and then failed before
reading test metadata because the clean Git clone did not contain the large
untracked `d5_metadata_index.jsonl`. The D5 metadata directory was then copied
to the expected path and the same transaction was resumed. This is consistent
with the frozen crash policy because the receipt identity is reused and no
test retrieval metric had been computed before the retry.

The resumed transaction created:

- `test_opening_receipt.json`
- `test_materialization_manifest.json`
- `test_programs.jsonl`
- `test_source_ast_index.jsonl`
- `test_source_ast_summary.json`

The test source audit materialized 3,088 query programs and 3,088 gallery
programs across 386 duplicate-closed test clusters.

## Running Seeds

The following seed sessions were launched in tmux:

- `c2h_test_20260711`
- `c2h_test_20260712`
- `c2h_test_20260713`
- `c2h_test_20260714`
- `c2h_test_20260715`
- `c2h_test_20260716`
- `c2h_test_20260717`
- `c2h_test_20260718`
- `c2h_test_20260719`
- `c2h_test_20260720`

Seed `20260714` initially failed while reading a concurrently rewritten
validation selection seal. The seal was recomputed once, then seed `20260714`
was relaunched. This happened before seed-level test distance matrices were
written for that seed.

At launch, all running seeds had reached the first planned test cell:

`EEE_true_LCA`

The run is resumable by seed and by query shard. Shards use the pattern:

`seed_<seed>_<cell>_q<start>_<stop>.pt`

Final seed matrices use the pattern:

`seed_<seed>_<cell>_test_distances.pt`

## Expected Follow-Up

After all ten seed JSON files are complete, run:

```bash
uv run python scripts/seal_codenet_stage_a_test_seed.py \
  outputs/codenet_python800_stage_a_test_v1/seed_20260711_test.json \
  --validation-selection-seal \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json
```

for every registered seed, then seal the final report:

```bash
uv run python scripts/seal_codenet_stage_a_confirmatory_report.py \
  outputs/codenet_python800_stage_a_test_v1/confirmatory_test_report.json \
  --validation-selection \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record.json \
  --validation-selection-seal \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json
```
