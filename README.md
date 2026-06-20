# Code2Hyp Artifacts

This repository contains the reproducibility artifacts for the manuscript:

**Code2Hyp: Hyperbolic AST-Path Representations for Structurally Faithful Code Embeddings**

Author: Ivan A. Kosyanenko  
ORCID: <https://orcid.org/0009-0009-1804-9412>

## Scope

The artifacts support the controlled Code2Hyp study on Java method-name subtoken prediction using the official code2seq Java-small preprocessed corpus. The study evaluates a code2vec-style AST-path model with a Euclidean lexical channel and a hyperbolic structural channel. The main claim is not universal state-of-the-art performance on method-name prediction. The main supported claim is that hyperbolic product geometry substantially improves structural faithfulness of AST-path representations while downstream F1 depends on the interaction between lexical and structural signals.

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

The main manuscript tables and figures are based on the following JSON outputs:

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

> Code2Hyp improves structural faithfulness of AST-path representations and improves the matched Euclidean baseline in the original lexical condition.

Unsafe claim:

> Code2Hyp universally outperforms Euclidean structural baselines on method-name prediction.

The released results show that Euclidean metric/tree controls remain stronger on downstream F1, while Code2Hyp is substantially stronger on AST-distance Spearman, normalized stress, and local neighbor preservation.
