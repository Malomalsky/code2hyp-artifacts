# B35 Code2Hyp code2vec-replacement pilot

Дата: 2026-06-16.

## Статус

Это быстрый real-data sanity pilot, а не финальное подтверждение для статьи.
Цель прогона - проверить, ведет ли себя новый code2vec-compatible кандидат
`B35_code2hyp_product_frechet_adaptive` осмысленно на реальном Java-small
path-context корпусе.

## Команды

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 2 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --structural-regularizer rank \
  --variants B1_euclidean,B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B10_factorized_product_code2vec,B16_factorized_product_three_metric_rank,B35_code2hyp_product_frechet_adaptive,B_tree_euclidean_lca_bias \
  --output outputs/code2hyp_b35_code2vec_replacement_original_512_2epochs_3seeds.json

./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 2 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --structural-regularizer rank \
  --lexical-ablation structural_only \
  --variants B1_euclidean,B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B10_factorized_product_code2vec,B16_factorized_product_three_metric_rank,B35_code2hyp_product_frechet_adaptive,B_tree_euclidean_lca_bias \
  --output outputs/code2hyp_b35_code2vec_replacement_structural_only_512_2epochs_3seeds.json
```

## Original

Validation records after known-target filtering: 53.

| Variant | F1 | Spearman | Overlap@1 | Overlap@3 | Structural loss | Curvature | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1_euclidean | 0.1780 | -0.1365 | 0.3082 | 0.3824 | 0.4078 | 1.0000 | 176768 |
| B4_hyperbolic_code2vec | 0.1894 | 0.6134 | 0.2752 | 0.4549 | 0.0832 | 1.0000 | 176768 |
| B8_hyperbolic_frechet_code2vec | 0.1970 | 0.6115 | 0.2774 | 0.4535 | 0.0834 | 1.0000 | 176768 |
| B10_factorized_product_code2vec | 0.1894 | 0.1612 | 0.2642 | 0.3429 | 0.1927 | 1.0000 | 176768 |
| B16_factorized_product_three_metric_rank | 0.2083 | -0.0079 | 0.2721 | 0.3462 | 0.2562 | 1.0000 | 176771 |
| B35_code2hyp_product_frechet_adaptive | 0.1932 | 0.0404 | 0.2655 | 0.3477 | 0.2214 | 0.9773 | 176772 |
| B_tree_euclidean_lca_bias | 0.1705 | 0.0528 | 0.4075 | 0.5431 | 0.1957 | 1.0000 | 176772 |

B35 learned channel weights:

```text
seed 101: [1.0065, 0.9751, 1.0149]
seed 202: [0.9756, 0.9801, 1.0004]
seed 303: [0.9771, 0.9739, 1.0052]
```

Interpretation:

```text
B35 is not the best original-regime model in this small pilot.
B16 has the highest F1, B8 is stronger than B35 by F1 and global structural
Spearman, and B_tree has the highest local AST-neighborhood overlap.

However B35 is still above B1, B10 and B_tree by F1 while adding only four
parameters over B1: three metric weights plus trainable curvature.
```

## Structural only

Validation records after known-target filtering: 53.

| Variant | F1 | Spearman | Overlap@1 | Overlap@3 | Structural loss | Curvature | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1_euclidean | 0.1932 | -0.1214 | 0.3365 | 0.3957 | 0.4324 | 1.0000 | 70816 |
| B4_hyperbolic_code2vec | 0.2008 | 0.1319 | 0.2862 | 0.3650 | 0.1899 | 1.0000 | 70816 |
| B8_hyperbolic_frechet_code2vec | 0.2008 | 0.1319 | 0.2862 | 0.3644 | 0.1900 | 1.0000 | 70816 |
| B10_factorized_product_code2vec | 0.2159 | 0.2082 | 0.2538 | 0.3480 | 0.1821 | 1.0000 | 70816 |
| B16_factorized_product_three_metric_rank | 0.2159 | -0.0247 | 0.2576 | 0.3468 | 0.2473 | 1.0000 | 70819 |
| B35_code2hyp_product_frechet_adaptive | 0.2197 | 0.0630 | 0.2630 | 0.3427 | 0.2161 | 0.9763 | 70820 |
| B_tree_euclidean_lca_bias | 0.2083 | -0.1189 | 0.3095 | 0.3812 | 0.4244 | 1.0000 | 70820 |

B35 learned channel weights:

```text
seed 101: [1.0000, 0.9758, 1.0000]
seed 202: [1.0000, 0.9792, 1.0000]
seed 303: [1.0000, 0.9767, 1.0000]
```

Interpretation:

```text
This is the most promising B35 signal so far: under structural_only stress,
B35 has the highest F1 among the tested variants.

The result is not yet a final article claim because the validation set is small
and the run is short. But it supports the direction that a code2vec-compatible
product geometry can improve predictive quality when lexical information is
suppressed.

At the same time, B35 is not the best local-neighborhood-preserving model:
B_tree has higher Overlap@1/Overlap@3. Therefore the next model improvement
should target local AST-neighborhood preservation without giving up B35's F1.
```

## Current scientific conclusion

Defensible wording:

```text
B35 is the cleanest current implementation of Code2Hyp as a direct analogue of
code2vec: it preserves AST path contexts and replaces only the geometry and
aggregation of the AST-path component. In a small Java-small pilot it is not a
universal winner, but it becomes the best F1 variant in the structural_only
stress regime, which justifies a larger confirmatory experiment.
```

Not defensible yet:

```text
B35 proves that hyperbolic code2vec is better than code2vec.
B35 is the final architecture for the paper.
B35 solves local structural neighborhood preservation.
```

## Next experiment

Run a larger preregistered comparison:

```text
train_limit: 4096 or more
val_limit: 1024 or more
epochs: 3-5
seeds: 5 if compute allows
variants: B1, B4, B8, B10, B16, B35, B_tree, plus B29/B31/B34 as path-attention controls
regimes: original, structural_only, optionally obfuscated
```

