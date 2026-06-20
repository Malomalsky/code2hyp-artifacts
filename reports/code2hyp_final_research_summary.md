# Code2Hyp final research summary

Date: 2026-06-20.

## 1. Frozen research question

This workspace investigates whether a code2vec-style AST-path model benefits
from hyperbolic structural representations.

Controlled task:

```text
Java method-name subtoken prediction on the official code2seq Java-small
preprocessed corpus.
```

The study is not a full-budget SOTA claim against code2seq. It is a controlled
geometry-aware extension of code2vec-style AST-path modeling under an explicitly
bounded local budget.

## 2. Current strongest experimental setup

Final local budget used for the current validation campaign:

```text
train split: first 25000 Java-small training examples
test split loaded: 8192 examples
test split after known-target filtering: 6642 examples
epochs: 5
batch size: 128
model seeds: 101, 202, 303, 404, 505 for the main original-condition comparison
max contexts per method: 30
max AST path length: 8
path encoder: GRU
target metric: target-subtoken micro precision/recall/F1
structural diagnostics: AST-distance Spearman, normalized stress, Overlap@3
```

Primary variants:

```text
B39_code2vec_context_transform_baseline
  matched Euclidean code2vec-style baseline.

B36_code2hyp_product_frechet_neighbor
  main downstream-performance Code2Hyp candidate.

B40_code2hyp_context_transform_frechet
  conservative Frechet structural candidate.

B44_code2hyp_context_transform_product_bias_frechet
  main structural-faithfulness Code2Hyp candidate.
```

Additional Euclidean structural controls:

```text
B6_euclidean_metric_code2vec
  Euclidean distance-based attention control.

B14_bounded_euclidean_metric_code2vec
  bounded Euclidean distance-based attention control.

B_tree_euclidean_lca_bias
  explicit tree/LCA attention-bias control.
```

## 3. Main 25k original-condition result

Sources:

```text
outputs/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress.json
reports/code2hyp_test_benchmark_25k_5epochs_5seeds_original_main_variants_with_stress.md
reports/code2hyp_test_benchmark_25k_5seeds_original_b36_vs_b39_f1_paired_effects.md
reports/code2hyp_test_benchmark_25k_5seeds_original_b44_vs_b39_spearman_paired_effects.md
reports/code2hyp_test_benchmark_25k_5seeds_original_b39_vs_code2hyp_stress_paired_effects.md
```

Mean over seeds:

| Variant | F1 | AST Spearman | Normalized stress | Overlap@3 | n seeds |
|---|---:|---:|---:|---:|---:|
| B39 matched baseline | 0.1578 +/- 0.0049 | -0.3437 +/- 0.0108 | 0.8188 +/- 0.0187 | 0.3345 +/- 0.0168 | 5 |
| B36 product-Frechet + neighbor | 0.1872 +/- 0.0012 | 0.7037 +/- 0.1119 | 0.1995 +/- 0.0251 | 0.8534 +/- 0.0317 | 5 |
| B40 context-transform + Frechet | 0.1757 +/- 0.0128 | 0.6508 +/- 0.0296 | 0.2220 +/- 0.0050 | 0.7840 +/- 0.0137 | 3 |
| B44 structural-bias attention | 0.1671 +/- 0.0085 | 0.9778 +/- 0.0021 | 0.0621 +/- 0.0028 | 0.9599 +/- 0.0035 | 5 |

Paired effects:

```text
B36 - B39 on F1:
  mean delta = +0.0295
  bootstrap CI over seeds = [+0.0250, +0.0332]
  direction = +/5 -/0

B44 - B39 on AST Spearman:
  mean delta = +1.3215
  bootstrap CI over seeds = [+1.3131, +1.3309]
  direction = +/5 -/0

B39 - B44 on normalized stress:
  mean delta = +0.7567
  bootstrap CI over seeds = [+0.7438, +0.7702]
  direction = +/5 -/0
```

Interpretation:

```text
B36 is the best matched-baseline downstream-performance candidate in the
original lexical condition.

B44 is the best structural-faithfulness candidate. It is not the best F1 model,
but it most cleanly preserves AST structural distances.
```

## 4. Lexical-control results

Sources:

```text
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_record_obfuscated_resumable_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_structural_only_resumable_with_stress.json
reports/code2hyp_test_benchmark_25k_record_obfuscated_f1_paired_effects.md
reports/code2hyp_test_benchmark_25k_structural_only_f1_paired_effects.md
reports/code2hyp_test_benchmark_25k_structural_only_spearman_paired_effects.md
reports/code2hyp_test_benchmark_25k_structural_only_stress_paired_effects.md
```

Record-obfuscated condition:

```text
B36 - B39 on F1:
  mean delta = -0.0062
  bootstrap CI = [-0.0220, +0.0056]
  direction = +/2 -/1

B44 - B39 on AST Spearman:
  mean delta = +1.3264
  bootstrap CI = [+1.3245, +1.3288]
  direction = +/3 -/0
```

Structural-only condition:

```text
B36 - B39 on F1:
  mean delta = +0.0001
  bootstrap CI = [-0.0090, +0.0086]
  direction = +/2 -/1

B44 - B39 on AST Spearman:
  mean delta = +1.3154
  bootstrap CI = [+1.3034, +1.3289]
  direction = +/3 -/0

B39 - B44 on normalized stress:
  mean delta = +0.7522
  bootstrap CI = [+0.7304, +0.7776]
  direction = +/3 -/0
```

Interpretation:

```text
The original-condition F1 gain is not a pure geometry-only effect. It depends
on lexical information interacting with structural modeling.

The structural-faithfulness effect remains large under lexical weakening and
under structural-only stress. This is the most robust current Code2Hyp result.
```

## 5. Euclidean structural controls

Sources:

```text
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_euclidean_structural_baselines_resumable_with_stress.json
outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_original_plus_euclidean_controls.json
reports/code2hyp_test_benchmark_25k_5epochs_3seeds_euclidean_structural_baselines_resumable_with_stress.md
reports/code2hyp_test_benchmark_25k_original_b36_vs_euclidean_f1_paired_effects.md
reports/code2hyp_test_benchmark_25k_original_b44_vs_euclidean_f1_paired_effects.md
reports/code2hyp_test_benchmark_25k_original_b44_vs_euclidean_spearman_paired_effects.md
reports/code2hyp_test_benchmark_25k_original_euclidean_vs_code2hyp_stress_paired_effects.md
```

Mean over seeds:

| Variant | F1 | AST Spearman | Normalized stress | Overlap@3 |
|---|---:|---:|---:|---:|
| B6 Euclidean metric | 0.2014 +/- 0.0010 | 0.1678 +/- 0.0028 | 0.4344 +/- 0.0012 | 0.4294 +/- 0.0053 |
| B14 bounded Euclidean metric | 0.1690 +/- 0.0016 | 0.1427 +/- 0.0021 | 0.5956 +/- 0.0052 | 0.2974 +/- 0.0018 |
| Btree Euclidean LCA-bias | 0.2023 +/- 0.0006 | 0.1562 +/- 0.0008 | 0.4403 +/- 0.0022 | 0.4402 +/- 0.0055 |

Paired effects:

```text
B36 - B6 on F1:
  mean delta = -0.0141
  direction = +/0 -/3

B36 - B14 on F1:
  mean delta = +0.0182
  direction = +/3 -/0

B36 - Btree on F1:
  mean delta = -0.0150
  direction = +/0 -/3

B44 - B6 on AST Spearman:
  mean delta = +0.8098
  direction = +/3 -/0

B6 - B44 on normalized stress:
  mean delta = +0.3721
  direction = +/3 -/0
```

Interpretation:

```text
The Euclidean metric and explicit tree/LCA controls are strong downstream F1
baselines. Current Code2Hyp variants do not dominate them on F1.

The hyperbolic structural-bias variant dominates them on structural-faithfulness
metrics. Therefore the defensible novelty is not "hyperbolic always gives the
best F1"; it is "hyperbolic product geometry gives much more faithful
hierarchical structural representations, while downstream utility depends on
the attention architecture."
```

## 6. Final defensible claims

Claim 1: matched-baseline downstream utility.

```text
Under the fixed local Java-small budget, B36 improves the matched
code2vec-context-transform baseline B39 in the original lexical condition by
+2.95 F1 points over five seeds.
```

Claim 2: robust structural faithfulness.

```text
Hyperbolic Code2Hyp variants, especially B44, preserve AST structural distances
much better than the matched baseline and Euclidean structural controls. This is
shown by AST-distance Spearman, normalized stress and local neighbor overlap.
```

Claim 3: lexical interaction boundary.

```text
The F1 improvement in the original condition should be reported as a
lexical-structural interaction, not as a pure geometry-only effect. When lexical
identity is weakened, F1 gains disappear or become low-power, while structural
metrics remain strong.
```

Claim 4: no one-model-dominates-all conclusion.

```text
B36 is the matched-baseline performance candidate. B44 is the structural
faithfulness candidate. B6/Btree are strong Euclidean performance controls.
The current result is a Pareto-style result, not a universal dominance result.
```

## 7. Final figure

Final figure:

```text
figures/code2hyp_final_controls_25k.png
figures/code2hyp_final_controls_25k.pdf
```

Reading:

```text
Panel A:
  B36 improves F1 in the original condition, but the advantage weakens under
  record-obfuscation and structural-only stress.

Panels B-D:
  Hyperbolic structural variants, especially B44, consistently dominate B39 on
  structural-faithfulness diagnostics.
```

## 8. Open limitations

The current study remains bounded by:

```text
1. local 25000-example training budget rather than full Java-small training;
2. test evaluation capped at 8192 loaded examples;
3. known-target filtering during evaluation;
4. no full-budget direct comparison against published code2seq results;
5. no claim that geometry alone explains downstream F1;
6. no current claim that Code2Hyp dominates all Euclidean structural controls
   on F1.
```

These limitations do not invalidate the controlled result. They define the next
experimental stage.

## 9. Recommended manuscript framing

Suggested title:

```text
Code2Hyp: Hyperbolic AST-Path Representations for Structurally Faithful Code
Embeddings
```

Suggested core conclusion:

```text
On a controlled Java-small method-name prediction benchmark, Code2Hyp shows
that hyperbolic AST-path representations can substantially improve structural
faithfulness of code embeddings. The strongest downstream model and the
strongest structural-faithfulness model are different, and strong Euclidean
metric controls remain competitive or superior on F1. This motivates reporting
predictive metrics together with explicit geometric diagnostics for code
representation models.
```
