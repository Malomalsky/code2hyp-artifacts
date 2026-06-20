# Code2Hyp 25k validation campaign

Date: 2026-06-20.

## Objective

The campaign validates the current Code2Hyp research claim under a larger local
budget and with stronger controls:

```text
1. original lexical condition;
2. record-level lexical obfuscation;
3. structural-only stress;
4. Euclidean structural baselines;
5. extra random seeds for the main original-condition comparison.
```

The goal is scientific falsification, not cherry-picking. A claim is retained
only if it survives the relevant control.

## Shared setup

```text
corpus: official code2seq Java-small preprocessed corpus
task: method-name subtoken prediction
train records: 25000
test records loaded: 8192
test records after known-target filtering: 6642
epochs: 5
batch size: 128
max contexts: 30
max AST path length: 8
path encoder: GRU
representation transform: identity
metric: target-subtoken micro precision/recall/F1
structural diagnostics: AST Spearman, normalized stress, Overlap@3
```

## Runs

Original, 3 seeds:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B39_code2vec_context_transform_baseline,B36_code2hyp_product_frechet_neighbor,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_resumable_with_stress.json
```

Record-obfuscated lexical control:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --lexical-ablation record_obfuscated \
  --variants B39_code2vec_context_transform_baseline,B36_code2hyp_product_frechet_neighbor,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_record_obfuscated_resumable_with_stress.json
```

Structural-only stress:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --lexical-ablation structural_only \
  --variants B39_code2vec_context_transform_baseline,B36_code2hyp_product_frechet_neighbor,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_structural_only_resumable_with_stress.json
```

Euclidean structural baselines:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B6_euclidean_metric_code2vec,B14_bounded_euclidean_metric_code2vec,B_tree_euclidean_lca_bias \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_euclidean_structural_baselines_resumable_with_stress.json
```

Extra seeds for original main variants:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 404,505 \
  --max-positive-weight 7.0 \
  --variants B39_code2vec_context_transform_baseline,B36_code2hyp_product_frechet_neighbor,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_extra_seeds_404_505_resumable_with_stress.json
```

## Main results

Original 5-seed comparison:

```text
B36 - B39 on F1:
  mean delta = +0.0295
  95% bootstrap CI = [+0.0250, +0.0332]
  direction = +/5 -/0

B44 - B39 on AST Spearman:
  mean delta = +1.3215
  95% bootstrap CI = [+1.3131, +1.3309]
  direction = +/5 -/0

B39 - B44 on normalized stress:
  mean delta = +0.7567
  95% bootstrap CI = [+0.7438, +0.7702]
  direction = +/5 -/0
```

Lexical controls:

```text
record_obfuscated:
  B36 - B39 F1 = -0.0062, CI [-0.0220, +0.0056]
  B44 - B39 Spearman = +1.3264, CI [+1.3245, +1.3288]

structural_only:
  B36 - B39 F1 = +0.0001, CI [-0.0090, +0.0086]
  B44 - B39 Spearman = +1.3154, CI [+1.3034, +1.3289]
  B39 - B44 stress = +0.7522, CI [+0.7304, +0.7776]
```

Euclidean controls:

```text
B6 Euclidean metric:
  F1 = 0.2014 +/- 0.0010
  AST Spearman = 0.1678 +/- 0.0028
  normalized stress = 0.4344 +/- 0.0012

Btree Euclidean LCA-bias:
  F1 = 0.2023 +/- 0.0006
  AST Spearman = 0.1562 +/- 0.0008
  normalized stress = 0.4403 +/- 0.0022

B44 structural-bias Code2Hyp:
  F1 = 0.1671 +/- 0.0085
  AST Spearman = 0.9778 +/- 0.0021
  normalized stress = 0.0621 +/- 0.0028
```

## Interpretation

The validation campaign supports three claims and rejects one overclaim.

Supported:

```text
1. Against the matched B39 baseline, B36 improves original-condition F1 over
   five seeds.
2. B44 is a strong structural-faithfulness model across original,
   record-obfuscated and structural-only settings.
3. Hyperbolic product geometry substantially improves AST-distance Spearman,
   normalized stress and local neighbor overlap.
```

Rejected or restricted:

```text
1. The original F1 gain is not a pure geometry-only effect.
2. Code2Hyp does not currently dominate all Euclidean structural controls on
   downstream F1.
```

Manuscript-safe statement:

```text
Code2Hyp provides structurally faithful hyperbolic AST-path representations.
Downstream F1 gains appear under a matched code2vec-style baseline and original
lexical information, while stronger Euclidean metric/tree controls remain
competitive or superior on F1. Therefore predictive performance and structural
faithfulness should be reported as separate outcomes.
```

## Artifacts

```text
reports/code2hyp_final_research_summary.md
figures/code2hyp_final_controls_25k.png
figures/code2hyp_final_controls_25k.pdf
outputs/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_record_obfuscated_resumable_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_structural_only_resumable_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_euclidean_structural_baselines_resumable_with_stress.json
```
