# Code2Hyp Artifacts

This repository contains reproducibility artifacts for the Code2Hyp research line.
It intentionally does not include the manuscript text.

Author: Ivan A. Kosyanenko  
ORCID: <https://orcid.org/0009-0009-1804-9412>

## Scope

The current artifact package studies task-level source-code retrieval with abstract syntax tree (AST) path objects. The main representation is an LCA-anchored path object: a terminal-to-terminal AST path is represented by the product of its least common ancestor, source endpoint and target endpoint. Programs are compared either as finite path measures or through a validation-selected multiview kernel that combines a clean LCA-path view with lexical and AST-count views.

The repository is not framed as a universal state-of-the-art benchmark. The supported claim is narrower: LCA anchoring is a useful AST path-object design principle, and its practical retrieval value is positive but corpus-dependent inside a validation-controlled multiview kernel.

## Repository Contents

- `geometry_profile_research/` contains the implementation of AST extraction, LCA-product path objects, structural distances, multiview retrieval, reporting utilities and the command-line tool.
- `scripts/` contains experiment runners, summarizers and figure-building utilities.
- `tests/` contains unit and integration tests for the public research code.
- `outputs/` contains JSON result artifacts.
- `reports/` contains generated Markdown reports used to audit and interpret the experiments.
- `figures/` contains PNG and PDF versions of the generated figures.
- `artifact_tools/build_figures.py` regenerates the figures from released JSON outputs.
- `data_manifests/` contains materialization manifests for the BugNet Python and Digital Teaching Assistant subsets. The raw datasets are not stored in this repository.

Some earlier Java/code2seq artifacts are retained for provenance, but they are not the primary evidence for the current artifact package.

## Datasets

The released experiments use two public Python corpora.

1. BugNet Python slice.

   Source: Hugging Face dataset `alexjercan/bugnet`, Python train split. The materialized corpus used in the released experiments contains 32 task groups and 512 accepted Python programs. The manifest is stored in:

   ```text
   data_manifests/bugnet_python_train_pass_16x32_manifest.json
   ```

2. Digital Teaching Assistant Python subset.

   Source: Zenodo Digital Teaching Assistant dataset, DOI `10.5281/zenodo.7799971`. The materialized Python subset used in the released experiments contains 11 task groups. The manifest is stored in:

   ```text
   data_manifests/dta_zenodo_balanced64_manifest.json
   ```

The raw corpora are intentionally excluded from git. The released JSON outputs are sufficient to regenerate the reported result tables and figures.

## Environment

The project was developed for Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

Optional neural/legacy experiments require:

```bash
.venv/bin/python -m pip install -r requirements-ml.txt
```

## Quick Tool Check

Run the core test suite:

```bash
.venv/bin/python -m pytest -q tests/test_code2hyp_tool.py \
  tests/test_raw_ast_geometry.py \
  tests/test_code2hyp_hybrid_retrieval_script.py \
  tests/test_summarize_path_sampling_sensitivity.py
```

Run all tests:

```bash
.venv/bin/python -m pytest -q
```

Use the installed command-line entry point:

```bash
.venv/bin/code2hyp --help
```

## Main Result Files

The main result tables and figures are based on these result objects:

```text
outputs/final_confirmatory_representation_benchmark_2026-06-28.json
outputs/task_retrieval_simple_baselines_2026-06-28.json
outputs/code2hyp_hybrid_task_retrieval_lca_kernel_nested_tokenast_margin001_2026-06-28.json
outputs/code2hyp_hybrid_task_level_contrasts_lca_kernel_nested_tokenast_margin001_2026-06-28.json
outputs/code2hyp_path_sampling_sensitivity_2026-06-28.json
outputs/code2hyp_label_mode_sensitivity_2026-06-28.json
outputs/code2hyp_label_mode_task_level_contrasts_2026-06-28.json
outputs/code2hyp_explainability_case_bugnet_2026-06-28.json
```

The corresponding interpretive reports are:

```text
reports/final_confirmatory_representation_benchmark_2026-06-28.md
reports/task_retrieval_simple_baselines_2026-06-28.md
reports/code2hyp_hybrid_task_retrieval_lca_kernel_nested_tokenast_margin001_2026-06-28.md
reports/code2hyp_hybrid_task_level_contrasts_lca_kernel_nested_tokenast_margin001_2026-06-28.md
reports/code2hyp_path_sampling_sensitivity_2026-06-28.md
reports/code2hyp_label_mode_sensitivity_2026-06-28.md
reports/code2hyp_label_mode_task_level_contrasts_2026-06-28.md
reports/code2hyp_explainability_case_bugnet_2026-06-28.md
```

## Rebuild Figures

Regenerate the figures from the released outputs:

```bash
.venv/bin/python artifact_tools/build_figures.py
```

The script writes PNG and PDF files to:

```text
figures/
```

Current generated figures:

```text
figures/figure01_code2hyp_architecture.png
figures/figure02_main_results.png
figures/figure03_geometry_diagnostics.png
figures/figure04_distance_levels.png
```

## Main Reproduction Commands

The commands below assume that the raw corpora have been materialized under the paths recorded in `data_manifests/`. If the raw corpora are absent, use the released JSON files in `outputs/` to regenerate the figures and inspect the reported results.

Structural-only representation benchmark:

```bash
.venv/bin/python scripts/summarize_confirmatory_benchmark.py \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260625.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260626.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260627.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260625.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260626.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260627.json \
  --output reports/final_confirmatory_representation_benchmark_reproduced.md \
  --json-output outputs/final_confirmatory_representation_benchmark_reproduced.json
```

Task-level lexical and AST baselines:

```bash
.venv/bin/python scripts/run_task_retrieval_baselines.py \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260625.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260626.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260627.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260625.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260626.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260627.json \
  --output reports/task_retrieval_simple_baselines_reproduced.md \
  --json-output outputs/task_retrieval_simple_baselines_reproduced.json
```

Validation-selected multiview retrieval:

```bash
.venv/bin/python scripts/run_code2hyp_hybrid_retrieval.py \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260625.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260626.json \
  --input bugnet_python outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260627.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260625.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260626.json \
  --input dta_zenodo outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260627.json \
  --path-selection-policy lca_depth_stratified \
  --lca-view code2hyp_path_signature_kernel \
  --weight-grid-mode expanded \
  --lca-selection-margin 0.01 \
  --output reports/code2hyp_hybrid_task_retrieval_reproduced.md \
  --json-output outputs/code2hyp_hybrid_task_retrieval_reproduced.json
```

Task-level paired contrasts:

```bash
.venv/bin/python scripts/summarize_hybrid_baseline_contrasts.py \
  --hybrid outputs/code2hyp_hybrid_task_retrieval_lca_kernel_nested_tokenast_margin001_2026-06-28.json \
  --simple outputs/task_retrieval_simple_baselines_2026-06-28.json \
  --output reports/code2hyp_hybrid_task_level_contrasts_lca_kernel_nested_tokenast_margin001_reproduced.md \
  --json-output outputs/code2hyp_hybrid_task_level_contrasts_lca_kernel_nested_tokenast_margin001_reproduced.json
```

## CodeNet Python800 Registered Evaluation

The repository includes the fail-closed data pipeline for the registered
evaluation on Project CodeNet Python800:

- global D0-D2 source/token/alpha-AST duplicate components;
- D3 MinHash/LSH candidate generation with exact token-5-gram Jaccard checks;
- statement-based and official-map D4 problem checks;
- privacy-preserving author metadata and D5 attrition diagnostics;
- a machine-readable Stage A design, readiness checker and immutable
  registration record;
- a deterministic HMAC-SHA256 cluster split derived from the first NIST
  Randomness Beacon pulse after registration.

The official Python800 object was verified by byte count, MD5/ETag and
SHA-256. The complete 8.34 GB Project CodeNet archive was independently
validated before extracting the official `identical_problem_clusters` map.
The map contains 153 clusters; 89 intersect Python800, each through exactly
one problem, so it introduces no additional within-benchmark edge. The final
audit covers 240,000 accepted Python programs and retains 213,550 programs in
773 duplicate-closed problem clusters. The preregistered power threshold is
764 clusters. The design was archived before randomization at
[Zenodo, DOI 10.5281/zenodo.21371188](https://doi.org/10.5281/zenodo.21371188).
The registered NIST pulse deterministically assigns 290 clusters to training,
97 to validation and 386 to the sealed test split. The assignment SHA-256 is
`5d8456569a11673ab2705b3606de358a26249b9fb1fc447f7c07cdce1b7b8f58`.

The cluster release contains assignments only. The subsequent frozen sampling
rule has now materialized 18,560 training programs and a validation set of 776
queries plus 776 gallery programs. Test program identifiers and relevance
labels remain unopened, and no CodeNet retrieval metric has been computed.

The registered design fixes user-distinct sample sizes but does not specify
how to order multiple programs submitted by the same user. Before sampling or
computing validation metrics, that execution detail was frozen in
`configs/codenet_python800_stage_a_sampling_protocol_v1.json`. It uses
domain-separated HMAC-SHA256 ordering and explicitly forbids materializing
test program identifiers before the single test unseal. The selected
train/validation samples contain 944 overlapping users across different
problem clusters. This is reported as an authorship-confounding diagnostic;
the registered primary estimand does not remove users globally because that
operation would eliminate most eligible training data.

Whole-program AST parsing and bounded path selection are frozen separately in
`configs/codenet_python800_stage_a_ast_path_protocol_v1.json`. The protocol
uses at most 64 unique terminal-to-terminal paths per program, stratified by
the exact LCA depth. It indexes the implicit pair space without constructing
all leaf pairs. Programs with fewer than 64 possible pairs are retained and
contribute every available pair once; they are neither excluded nor padded by
duplicate paths. The implementation was tagged before the selected-source
audit as `codenet-stage-a-path-sampler-v1`.

Run the released data tests:

```bash
uv sync --frozen --extra dev
uv run pytest -q \
  tests/test_codenet_eligibility.py \
  tests/test_codenet_d3.py \
  tests/test_codenet_statement_d4.py \
  tests/test_codenet_d5_metadata.py \
  tests/test_codenet_d5_attrition.py \
  tests/test_codenet_stage_a_readiness.py \
  tests/test_codenet_split.py \
  tests/test_codenet_sampling.py \
  tests/test_codenet_ast_audit.py
```

The official-map and power gates now pass. The readiness command still fails
closed in an uncommitted worktree; it passes only from a clean immutable
artifact commit:

```bash
uv run python scripts/check_codenet_stage_a_readiness.py
```

The complete audit narrative is in
`reports/codenet_python800_pre_split_eligibility_2026-07-11.md`.

Re-derive and audit the registered cluster split:

```bash
uv run python scripts/build_codenet_python800_split.py
uv run python scripts/check_codenet_stage_a_split.py
```

After reconstructing the ignored D5 metadata index from the official CodeNet
metadata, materialize and audit train/validation program sampling:

```bash
uv run python scripts/build_codenet_python800_program_sampling.py
uv run python scripts/check_codenet_stage_a_program_sampling.py
```

After placing the official Python800 source tree locally, reproduce the
selected-source and AST path audit:

```bash
uv run python scripts/audit_codenet_stage_a_selected_sources.py \
  --source-root data/external_raw/codenet_python800_extracted/Project_CodeNet_Python800
```

The split manifest is in
`data/codenet_python800_stage_a_split/split_manifest.json`; the independent
audit result is in `reports/codenet_stage_a_split_audit.json`.
The program sample manifest is in
`data/codenet_python800_stage_a_program_sampling/program_sampling_manifest.json`;
its audit is in `reports/codenet_stage_a_program_sampling_audit.json`.
The selected-source audit covers 20,112 programs with no decode or parse
failure. The median AST contains 184 nodes and 106 leaves; the largest contains
2,750 nodes and 1,652 leaves, corresponding to 1,363,726 possible terminal
pairs. Nine programs have fewer than 64 possible pairs and follow the frozen
small-program rule. The portable per-program index and manifest are in
`data/codenet_python800_stage_a_selected_source_ast/`.

The model, calibration and analysis choices were frozen before the first
validation retrieval score in
`configs/codenet_python800_stage_a_model_analysis_protocol_v1.json`. The
protocol fixes a shared structural-only encoder, the true-LCA versus
zero-anchor contrast, the matched Euclidean/near-zero/active-curvature cells,
full-gallery `MAP@8`, ten model seeds and the future problem-cluster bootstrap.
It excludes the Frechet-control hypothesis because no certified global solver
for the unoriented endpoint quotient is available.

Rebuild the 4,096 train-only calibration pairs and run the independent
numerical Gate 0:

```bash
uv run python scripts/build_codenet_stage_a_calibration_pairs.py
uv run python scripts/run_codenet_stage_a_gate0.py
```

The calibration set contains 2,048 within-cluster and 2,048 cross-cluster
pairs and never reads validation or test programs. Gate 0 checks the full
regularized OT objective against POT, scalar/batched equivalence, marginal
residuals, the standard Poincare near-zero limit, endpoint-reversal
invariance and an autograd/finite-difference gradient agreement.

Run the resumable validation experiment after supplying the local official
source tree:

```bash
uv run python scripts/run_codenet_stage_a_validation.py \
  --source-root /absolute/path/to/Project_CodeNet_Python800
```

Per-seed checkpoints, distance matrices and validation summaries are written
under `outputs/codenet_python800_stage_a_validation_v1/`, which is intentionally
not tracked. The validation selection record is produced only after all ten
registered seeds complete. This command does not materialize test program IDs
or test relevance labels.

Verify and seal each completed validation seed before aggregating the ten-seed
selection record:

```bash
uv run python scripts/seal_codenet_stage_a_validation_seed.py \
  outputs/codenet_python800_stage_a_validation_v1/seed_20260711_validation.json
```

The seal checks the frozen protocol and calibration hashes, numerical Gate 0,
the exact seven-cell design, validation cardinalities, checkpoint and distance
matrix hashes, and the three fail-closed test-access flags. It records the
immutable validation-runner commit and does not read the official source tree.

## Claim Boundary

Safe claim:

> LCA-anchored AST path objects are useful structural units, and a validation-selected multiview kernel can exploit the LCA-path view when it is supported by training folds.

Unsafe claim:

> Negative curvature or LCA anchoring universally improves all code retrieval settings.

The released results show a positive LCA-view contribution on BugNet Python and a zero-LCA fallback on the DTA subset. This is the intended interpretation: the method is useful as a controlled structural view, not as an unconditional replacement for lexical or pretrained semantic models.

The five-seed BugNet Gate A matrix is exploratory. It supports an LCA-role
signal under the pilot budget, while active hyperbolic curvature does not
improve the matched Euclidean control. It is not a substitute for the sealed
CodeNet experiment.
