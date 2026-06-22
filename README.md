# Code2Hyp Artifacts

This repository contains the reproducibility artifacts for the manuscript:

**Code2Hyp: Product-Manifold Representations of Terminal-to-Terminal AST Paths**

Author: Ivan A. Kosyanenko  
ORCID: <https://orcid.org/0009-0009-1804-9412>

## Current revision, 2026-06-22

The current revision reframes the study as a controlled geometry analysis of
terminal-to-terminal AST-label path contexts, not as a broad state-of-the-art
claim for method-name prediction.

The most relevant files for the current revision are:

```text
code2hyp_branch_product_revision_report.md
reports/code2hyp_b62_multi_metric_validation_5k_3seeds_summary.md
reports/code2seq_path_geometry_audit_smoke_multi_metric.md
```

The current key method variants are:

```text
B60: order-aware branch-sequence product model for prefix-trie structural fidelity.
B62: B60 branch-sequence product model with a multi-metric structural objective.
B63: product-bias multi-metric control.
B64: Euclidean multi-metric control.
B65: L1 multi-metric control.
```

Under the 5k/3-seed medium validation protocol, B60 is the strongest
prefix-trie specialist, while B62 is the strongest cross-metric structural
candidate across prefix-tree, edit-distance, and Jaccard-bigram diagnostics.
Downstream F1 is reported as a sanity check rather than as the primary claim.

## Scope

The artifacts support a controlled Code2Hyp study on Java method-name subtoken
prediction using the official code2seq Java-small preprocessed corpus. The
study evaluates code2vec-style AST-path models with Euclidean, L1,
near-Euclidean, hyperbolic, and product-manifold structural channels. The main
claim is not universal state-of-the-art performance on method-name prediction.
The main supported claim is that representation geometry strongly affects
structural fidelity of AST-label path contexts, while downstream F1 depends on
the interaction between lexical signals, structural supervision, and the
attention/decoder architecture.

## Repository contents

- `geometry_profile_research/` -- implementation of Code2Hyp variants, Euclidean controls, structural diagnostics, reporting utilities, and data loaders.
- `scripts/` -- experiment runners, result aggregation scripts, paired-effect reports, and plotting scripts.
- `tests/` -- unit and integration tests for the research code.
- `outputs/` -- raw JSON outputs from the local-budget experiments.
- `reports/` -- generated Markdown reports used to audit and interpret the results.
- `manuscript_tools/build_manuscript_figures.py` -- script that generates the manuscript figures from JSON outputs.
- `manuscript_figures/` -- PNG and PDF versions of the figures used in the manuscript.
- `data/code2seq_java_small/DATASET_MANIFEST.md` -- dataset manifest and download information. The dataset itself is not included.

## Dataset

The experiments use the public code2seq Java-small preprocessed corpus:

- source repository: <https://github.com/tech-srl/code2seq>
- dataset archive: <https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz>

The archive is intentionally not stored in this repository. Place the extracted dataset under:

```text
data/code2seq_java_small/java-small/
```

The expected split files are:

```text
data/code2seq_java_small/java-small/java-small.train.c2s
data/code2seq_java_small/java-small/java-small.val.c2s
data/code2seq_java_small/java-small/java-small.test.c2s
```

## Environment

The project was developed for Python 3.12.

Install the minimal development environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -r requirements-ml.txt
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Rebuild manuscript figures

The manuscript figures can be regenerated from the released JSON outputs:

```bash
.venv/bin/python manuscript_tools/build_manuscript_figures.py
```

Generated files are written to `manuscript_figures/` when the script is run from this repository.

## Main result files

The earlier manuscript tables and figures are based on the following JSON outputs:

```text
outputs/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_original_plus_euclidean_controls.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_record_obfuscated_resumable_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_structural_only_resumable_with_stress.json
```

The main interpretive report is:

```text
reports/code2hyp_final_research_summary.md
```

The current post-review and multi-metric artifacts are:

```text
outputs/code2hyp_postreview_benchmark_25k_5epochs_5seeds_with_b49_l1_and_geometry_diagnostics.json
outputs/code2hyp_b60_branch_sequence_validation_5k_3seeds.json
outputs/code2hyp_b62_multi_metric_validation_5k_3seeds.json
outputs/code2seq_path_geometry_audit_smoke_multi_metric.json
reports/code2hyp_postreview_benchmark_25k_5epochs_5seeds_with_b49_l1_and_geometry_diagnostics.md
reports/code2hyp_b62_multi_metric_validation_5k_3seeds_summary.md
reports/code2seq_path_geometry_audit_smoke_multi_metric.md
```

## Main benchmark command

The main local-budget benchmark can be reproduced with:

```bash
.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303,404,505 \
  --max-positive-weight 7.0 \
  --variants B39_code2vec_context_transform_baseline,B36_code2hyp_product_frechet_neighbor,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress_reproduced.json
```

For a quick executable check without the full benchmark cost:

```bash
.venv/bin/python scripts/run_code2hyp_smoke.py \
  --seed 17 \
  --output outputs/code2hyp_smoke_report_reproduced.json
```

## Claim boundary

Safe claim:

> Product/hyperbolic Code2Hyp variants improve structural fidelity of
> AST-label path representations under controlled local-budget protocols.

Unsafe claim:

> Code2Hyp universally outperforms Euclidean structural baselines on method-name prediction.

The released results show a Pareto trade-off. B60 is the strongest
prefix-trie structural-fidelity model. B62 is the strongest cross-metric
structural model. Some Euclidean or L1 controls remain competitive on
downstream F1, so downstream prediction and structural adequacy must be
reported separately.
