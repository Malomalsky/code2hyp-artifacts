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

## CodeNet Java Stage B Pre-split Audit

The independent Java sampling frame is derived from the full Project CodeNet
1.0.0 archive, not from Java250. Java250 is unsuitable as a new confirmation
set because 242 of its 250 tasks overlap the opened Python800 Stage A frame.
The full metadata audit instead identified 847 independent Java components
with at least 16 accepted programs from 16 users.

After Java parsing and exact D0-D2 deduplication, 50,178 canonical programs
remain. The primary D3 rule (`exact token-5-gram set-Jaccard >= 0.90`) retains
48,149 programs in 566 duplicate-closed clusters. D4 applies the official
identical-problem map, checks normalized statement identity, and excludes an
entire cluster whenever an official HTML statement is unavailable. The final
pre-split frame has 531 evaluation-eligible clusters, 123 of which also meet
the obsolete preliminary threshold of 64 programs and 16 users. That rule was
not sampler-compatible because program selection is user-distinct. The frozen
pre-registration design instead requires 32 programs from 32 users. It has 253
train-eligible clusters and a full Hamilton `3:1:4` allocation of
`199/66/266`, with no clusters discarded. No Java retrieval labels or metrics
were opened while producing these artifacts.

The pre-specified D3 sensitivity yields `107/36/143` at threshold `0.80` and
`256/85/342` at threshold `0.95`; the primary threshold remains `0.90`
regardless of these counts. A location-shift power precheck at 266 test
clusters gives marginal power `0.9982` for `delta=0.01 MAP@8` under twice the
Stage A reference variance; the conservative two-contrast lower bound is
`0.9964`. This is a planning assumption, not Java retrieval evidence. The
machine-readable primary records are:

- `data/codenet_java_stage_b_eligibility_d0_d3_v1/d3_manifest.json`;
- `data/codenet_java_stage_b_eligibility_d4_train32_v2/d4_manifest.json`;
- `reports/codenet_java_stage_b_sampling_design_v2.json`;
- `reports/codenet_java_stage_b_power_precheck_train32_v2.json`.

The accepted-source tar and candidate inventory, D0-D2 inventory, D3 index,
MinHash arrays and user-bearing D5 index are deterministically reconstructable
from the official archive and intentionally excluded from Git. Their hashes
remain pinned in the design and compact manifests.

### Stage B execution status

At the public registration time, Java retrieval metrics had not been computed
and the Stage B split had not been generated. The design is archived at
[Zenodo, DOI 10.5281/zenodo.22028886](https://doi.org/10.5281/zenodo.22028886)
and in `configs/codenet_java_stage_b_draft_v1.json`. It fixes 20 independently trained
models (ten `label_depth_prefix`, ten `label_only`), prefix-only curvature
selection on 66 validation clusters, seven test cells on 266 clusters, and a
20,000-resample cluster bootstrap for H_B1-H_B4. The validation and test
pipelines recompute every metric from the stored `float64` distance matrix and
fail closed on any hash, AST, cardinality or provenance mismatch.
The registered execution device is CUDA. Both runners perform a deterministic
CUDA preflight before expensive loading and, for the test runner, before the
single test opening. The runtime records the actual device name, compute
capability, CUDA version, cuBLAS workspace setting and disabled TF32 state.
The frozen workload contains 180 validation matrices and 70 test matrices,
or 367,168,000 query-gallery measure pairs in total. Final `float64` matrices
alone require 2,937,344,000 bytes. Because transport evaluation is `float64`,
the official run must be benchmarked on an HPC-class CUDA device before test
opening; a consumer GPU must not be selected from FP32 throughput alone.
Exact hardware identity is recorded, while bitwise equality across different
GPU architectures is not claimed.

The first NIST Randomness Beacon pulse strictly after the Zenodo creation time
was chain 2, pulse 1911175 (`2026-08-20T11:57:00Z`). It deterministically
produced 199 train, 66 validation and 266 test clusters; the cluster-assignment
SHA-256 is `43031390aac5052c5958ae5b1b47006ffa5f83a34b8d358a53d7a06dbe4800d9`.
Only train and validation programs were then sampled: 6,368 train programs and
528 query plus 528 gallery programs for validation. All 7,424 opened sources
passed the registered SHA-256 and raw-AST audit, and 4,096 train-only
calibration pairs were frozen. The transfer unit is the duplicate-closed task
cluster, not the developer identity; 318 users occur in both train and
validation tasks, while one-program-per-user sampling is enforced within each
cluster and validation query/gallery users are disjoint. No Java retrieval
metric has yet been computed, and no test program identifier or relevance
label has been materialized.

Run the pre-registration audit with:

```bash
uv run python scripts/check_codenet_java_stage_b_readiness.py
```

The implementation is frozen at
`c419f6418056b618ce373ebd6fafe6601ff51566`; both Stage B runner tags point to
that commit. The protocol pins the public CUDA image by its registry digest:

```text
ghcr.io/malomalsky/code2hyp-stage-b@sha256:f58fec9b2a2a1f3d58f462596486f6dca1e1a29c5d303f083d145fb19bea4204
```

The image was built from the frozen commit in GitHub Actions with SBOM and
maximum-mode provenance. Its build receipt is stored in
`reports/codenet_java_stage_b_container_v1.json`. This protocol commit remains
separate from the implementation commit to avoid a self-referential Git hash.

Build the frozen `linux/amd64` runtime from the implementation commit with:

```bash
docker buildx build --platform linux/amd64 \
  --file Containerfile.stage-b \
  --build-arg IMPLEMENTATION_COMMIT=<implementation-commit> \
  --tag code2hyp-stage-b:<implementation-commit> \
  --load .
```

The Docker context excludes all local data and outputs; registered inputs are
mounted read-only at execution time. The canonical experiment must use the
digest above rather than a mutable image tag.

For an official validation run, keep the runtime, source checkout, inputs, and
outputs separate. The image supplies the pinned CUDA environment, while the
read-only checkout lets the runner verify the signed implementation tag. The
local validation-only input archive is described by
`reports/codenet_java_stage_b_validation_input_bundle_v1.json`; it contains
7,424 train/validation sources and no test program identifiers.

```bash
git clone https://github.com/Malomalsky/code2hyp-artifacts.git stage-b-runner
git -C stage-b-runner checkout --detach codenet-java-stage-b-validation-runner-v1
tar -xzf code2hyp_stage_b_validation_inputs_v1.tar.gz
mkdir -p stage-b-validation-output

docker run --rm --gpus all \
  --mount type=bind,src="$PWD/stage-b-runner",dst=/run/code2hyp,readonly \
  --mount type=bind,src="$PWD/code2hyp_stage_b_validation_inputs_v1",dst=/inputs,readonly \
  --mount type=bind,src="$PWD/stage-b-validation-output",dst=/outputs \
  --workdir /run/code2hyp \
  ghcr.io/malomalsky/code2hyp-stage-b@sha256:f58fec9b2a2a1f3d58f462596486f6dca1e1a29c5d303f083d145fb19bea4204 \
  /workspace/code2hyp/.venv/bin/python scripts/run_codenet_java_stage_b_validation.py \
  --design /inputs/configs/codenet_java_stage_b_draft_v1.json \
  --registration /inputs/registrations/codenet_java_stage_b_registration_v1.json \
  --sampling-manifest /inputs/data/codenet_java_stage_b_program_sampling_v1/program_sampling_manifest.json \
  --train-programs /inputs/data/codenet_java_stage_b_program_sampling_v1/train_programs.jsonl \
  --validation-programs /inputs/data/codenet_java_stage_b_program_sampling_v1/validation_programs.jsonl \
  --calibration-manifest /inputs/data/codenet_java_stage_b_calibration_pairs_v1/calibration_pair_manifest.json \
  --calibration-pairs /inputs/data/codenet_java_stage_b_calibration_pairs_v1/calibration_pairs.jsonl \
  --selected-ast-manifest /inputs/data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_manifest.json \
  --ast-index /inputs/data/codenet_java_stage_b_selected_source_ast_v1/selected_source_ast_index.jsonl \
  --source-root /inputs/sources \
  --output-dir /outputs
```

Before starting the container, verify the transferred archive against SHA-256
`c2b38980e54649973199db2fed461d0117cca283d976c662083b80979f701666`.
After extraction, `shasum -a 256 -c MANIFEST.sha256` verifies every packaged
input. Do not mount the full Java candidate source tree into validation.

The earlier Python Stage A RunPod jobs used CPU computation even though their
pods exposed an RTX 3090. This hardware-description correction is documented in
`reports/codenet_python800_stage_a_compute_device_correction_2026-08-20.md` and
does not alter the sealed Stage A matrices or metrics.

Only after the finalized design is publicly registered and the first later
NIST Randomness Beacon pulse is recorded, execute the fixed sequence:

```bash
uv run python scripts/build_codenet_java_stage_b_split.py
uv run python scripts/build_codenet_java_stage_b_program_sampling.py
uv run python scripts/audit_codenet_java_stage_b_selected_sources.py
uv run python scripts/build_codenet_java_stage_b_calibration_pairs.py
uv run python scripts/run_codenet_java_stage_b_validation.py
uv run python scripts/seal_codenet_java_stage_b_validation.py
uv run python scripts/run_codenet_java_stage_b_test.py
```

The final command writes the single test-opening receipt before reading a test
metadata row, rechecks every selected Java source against its pre-split D0 and
AST identity, evaluates only the seven registered cells, recomputes H_B1-H_B4,
and seals the complete report. A partial run can resume only the same receipt
and seed identities. No result in this section should be described as Java
evidence until that sequence has completed.

## CodeNet Python800 Registered Execution

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

Runner v1 stopped before its first validation metric when one self-transport
batch retained a `1.738e-7` marginal residual against the frozen `1e-7`
threshold. The incident is preserved in
`reports/codenet_python800_stage_a_numerical_incident_2026-07-15.json`.
The frozen numerical addendum introduces one standard nonnegative rank-one
marginal rounding step without changing the tolerance, model, cells or
estimand. The exact failed batch is reproduced by:

```bash
uv run python scripts/reproduce_codenet_stage_a_transport_incident.py \
  --source-root /absolute/path/to/Project_CodeNet_Python800
```

On that batch, rounding reduces the maximum residual from `1.738e-7` to
`6.939e-18`; the relative full-objective shift is between `3.51e-6` and
`4.83e-6`. Gate 0 v2 additionally checks the rounded batched gradient against
finite differences.

Validation-runner v3 then computed the first complete distance matrix but
stopped before persisting that matrix or any retrieval metric. The evaluator
had keyed relevance by the original CodeNet `problem_id`, whereas the
registered independent unit is the duplicate-closed `cluster_id`. One of the
97 validation clusters contains the equivalent source problems `p02388` and
`p02859`; all clusters nevertheless contain exactly 8 queries and 8 gallery
items. The incident and pre-metric correction are recorded in
`reports/codenet_python800_stage_a_relevance_identity_incident_2026-07-15.json`
and `configs/codenet_python800_stage_a_relevance_identity_addendum_v1.json`.
Runner v4 changes only the relevance and macro-aggregation key to
`cluster_id`; the split, model, distances, ranking, cutoff and hypotheses are
unchanged.

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

Validation progress can be inspected without reading any test-facing artifact:

```bash
uv run python scripts/check_codenet_stage_a_validation_progress.py
```

Verify and seal each completed validation seed before aggregating the ten-seed
selection record:

```bash
uv run python scripts/seal_codenet_stage_a_validation_seed.py \
  outputs/codenet_python800_stage_a_validation_v1/seed_20260711_validation.json
```

The seal checks the frozen protocol and calibration hashes, numerical Gate 0,
the relevance-identity addendum, the exact seven-cell design, validation
cardinalities, checkpoint and distance-matrix hashes, and the three fail-closed
test-access flags. It independently recomputes every retrieval summary from
the stored `float64` matrix and frozen validation `cluster_id` metadata. It
records the immutable validation-runner commit and does not read the official
source tree.

After all ten seed seals exist, verify the frozen curvature-selection rule and
bind the selection record to every sealed input:

```bash
uv run python scripts/seal_codenet_stage_a_validation_selection.py \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record.json
```

The single test-opening transaction is specified independently in
`configs/codenet_python800_stage_a_test_execution_protocol_v1.json`, frozen
during validation and before curvature selection. It requires the complete
validation-selection seal, writes an immutable opening receipt before parsing
any test metadata row, and permits a crashed process to resume only the same
receipt identity. The transaction applies the registered user-distinct HMAC
rule to 386 test clusters, materializes 3,088 queries and 3,088 gallery
programs, and repeats the frozen AST/path audit before computing a metric.

From the immutable `codenet-stage-a-test-runner-v4` worktree, perform the one
test opening and the complete evaluation with:

```bash
uv run python scripts/run_codenet_stage_a_test.py \
  --source-root /absolute/path/to/Project_CodeNet_Python800 \
  --d5-index /absolute/path/to/d5_metadata_index.jsonl \
  --validation-output-dir /absolute/path/to/codenet_python800_stage_a_validation_v1 \
  --output-dir /absolute/path/to/codenet_python800_stage_a_test_v1
```

Each test seed reuses the exact checkpoint, coordinate scales, role weights
and Sinkhorn epsilon bound to its sealed validation result. No parameter is
recalibrated on test. All seven planned cells are evaluated regardless of
sign, after which the frozen 20,000-resample problem-cluster bootstrap reports
H1 and the two-component H3 intersection-union decision. Test outputs remain
untracked because they contain the once-opened program identifiers.

Independently recompute and seal each completed test seed from its stored
distance matrices:

```bash
uv run python scripts/seal_codenet_stage_a_test_seed.py \
  outputs/codenet_python800_stage_a_test_v1/seed_20260711_test.json \
  --validation-selection-seal \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json
```

After all ten test-seed seals exist, recompute the complete seed-to-problem
aggregation and cluster bootstrap and seal the final report:

```bash
uv run python scripts/seal_codenet_stage_a_confirmatory_report.py \
  outputs/codenet_python800_stage_a_test_v1/confirmatory_test_report.json \
  --validation-selection \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record.json \
  --validation-selection-seal \
  outputs/codenet_python800_stage_a_validation_v1/validation_selection_record_seal.json
```

The separately frozen train-only scope audit quantifies how often the
`label_only` encoder input identifies distinct AST nodes, true-LCA anchors and
complete unoriented path objects with the same signature:

```bash
uv run python scripts/audit_codenet_stage_a_representation_identifiability.py \
  --source-root /absolute/path/to/Project_CodeNet_Python800
```

This diagnostic is descriptive, reads no validation manifest or test IDs, and
does not modify the registered curvature-selection rule.

Across the 18,560 training programs, the frozen `label_only` input maps 90.15%
of additional node identities, 61.90% of additional true-LCA identities and
45.44% of additional selected path objects onto already observed signatures
(micro collision rates). These values measure input equivalence, not learned
embedding quality. They narrow the Stage A interpretation and motivate a
separately controlled context-aware encoder in a future stage; they do not
authorize changing the registered Stage A model after validation begins.

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
