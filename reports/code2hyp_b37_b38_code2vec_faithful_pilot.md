# B37/B38 code2vec-faithful Code2Hyp pilot

Дата: 2026-06-16.

## Purpose

Цель этого шага - отделить две разные гипотезы:

```text
H1. Minimal replacement:
    сохранить оригинальную code2vec-постановку и заменить только геометрию
    AST-path канала.

H2. Product-metric replacement:
    сохранить path-context вход, но заменить attention на product-metric
    attention по Euclidean x Hyperbolic x Euclidean factors.
```

Это разделение важно, потому что B35/B36 уже являются сильными кандидатами, но
их attention отличается от классического code2vec сильнее, чем необходимо.

## What code2vec does

code2vec представляет метод как множество path contexts:

```text
c_i = (s_i, p_i, t_i),
```

где `s_i` и `t_i` - terminal tokens, а `p_i` - AST path между ними.
Модель кодирует context, считает attention over contexts и агрегирует их в
один code vector для предсказания method-name subtokens.

Reference:

```text
Alon et al. code2vec: Learning Distributed Representations of Code.
POPL 2019 / arXiv:1803.09473.
https://arxiv.org/abs/1803.09473
```

## B37: minimal hyperbolic replacement

Variant:

```text
B37_code2hyp_code2vec_attention_frechet
```

Architecture:

```text
start token:
  u_i = E_s(s_i) in R^d_s

AST path:
  z_i = exp_0^c(g_p(p_i)) in H_c^d_p
  ell_i = log_0^c(z_i) in T_0 H_c^d_p

end token:
  v_i = E_t(t_i) in R^d_t
```

code2vec-style context used for attention:

```text
x_i = concat(u_i, ell_i, v_i)
a_i = softmax_i(<x_i, q>)
```

Aggregation:

```text
u_bar = sum_i a_i u_i
v_bar = sum_i a_i v_i
z_bar = argmin_z sum_i a_i d_H(z, z_i)^2
h = concat(u_bar, log_0^c(z_bar), v_bar)
```

Thus B37 preserves the original dot-product path-context attention form and
changes only the geometry/aggregation of the AST-path channel.

Parameter cost:

```text
B37 = B1 + 1 trainable curvature parameter.
```

## B38: B37 with local-neighborhood regularization

Variant:

```text
B38_code2hyp_code2vec_attention_neighbor
```

B38 keeps B37 architecture and fixes:

```text
structural_regularizer = neighbor_distribution
```

with:

```text
q(j | i) = softmax_j(-d_AST(i, j) / tau_T)
p(j | i) = softmax_j(-d_model(i, j) / tau_E)
L_neighbor = mean_i KL(q(. | i) || p(. | i)).
```

## Pilot commands

Original:

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
  --variants B37_code2hyp_code2vec_attention_frechet \
  --output outputs/code2hyp_b37_code2vec_attention_rank_original_512_2epochs_3seeds.json

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
  --variants B38_code2hyp_code2vec_attention_neighbor \
  --output outputs/code2hyp_b38_code2vec_attention_neighbor_original_512_2epochs_3seeds.json
```

Structural-only:

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
  --lexical-ablation structural_only \
  --variants B37_code2hyp_code2vec_attention_frechet \
  --output outputs/code2hyp_b37_code2vec_attention_rank_structural_only_512_2epochs_3seeds.json

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
  --lexical-ablation structural_only \
  --variants B38_code2hyp_code2vec_attention_neighbor \
  --output outputs/code2hyp_b38_code2vec_attention_neighbor_structural_only_512_2epochs_3seeds.json
```

## Results

All numbers are means over seeds 101, 202, 303.

| Variant | Regularizer | Regime | F1 | Spearman | Overlap@1 | Overlap@3 | Curvature | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B1 Euclidean code2vec | rank | original | 0.1780 | -0.1365 | 0.3082 | 0.3824 | 1.0000 | 176768 |
| B35 product metric | rank | original | 0.1932 | 0.0404 | 0.2655 | 0.3477 | 0.9773 | 176772 |
| B36 product metric | neighbor_distribution | original | 0.1894 | 0.1527 | 0.2708 | 0.3611 | 0.9764 | 176772 |
| B37 faithful attention | rank | original | 0.1856 | 0.0911 | 0.2702 | 0.3509 | 0.9775 | 176769 |
| B38 faithful attention | neighbor_distribution | original | 0.1856 | 0.1330 | 0.2755 | 0.3595 | 0.9772 | 176769 |
| B1 Euclidean code2vec | rank | structural_only | 0.1932 | -0.1214 | 0.3365 | 0.3957 | 1.0000 | 70816 |
| B35 product metric | rank | structural_only | 0.2197 | 0.0630 | 0.2630 | 0.3427 | 0.9763 | 70820 |
| B36 product metric | neighbor_distribution | structural_only | 0.2197 | 0.2188 | 0.2655 | 0.3569 | 0.9750 | 70820 |
| B37 faithful attention | rank | structural_only | 0.1932 | 0.1658 | 0.2683 | 0.3546 | 0.9765 | 70817 |
| B38 faithful attention | neighbor_distribution | structural_only | 0.1932 | 0.1994 | 0.2645 | 0.3596 | 0.9763 | 70817 |

## Interpretation

Supported by this pilot:

```text
B37/B38 are the cleanest code2vec-faithful hyperbolic replacements: they keep
the original dot-product attention over path contexts and add only trainable
curvature to B1.
```

```text
B37/B38 improve structural Spearman relative to B1 in both original and
structural_only regimes, but do not yet improve F1 over B35/B36.
```

```text
The neighbor-distribution regularizer consistently improves B37 structural
diagnostics in original and structural_only regimes.
```

Not supported yet:

```text
It is not correct to claim that the most faithful code2vec replacement is
already the best predictive model. In this pilot, product-metric B35/B36 have
better F1.
```

Best current scientific framing:

```text
The research now has two defensible tracks:

1. B37/B38 - minimal code2vec-faithful hyperbolic replacement.
   Strong for architectural purity and ablation logic.

2. B35/B36 - stronger product-metric Code2Hyp modification.
   Stronger in F1 in the small pilot, but less minimal relative to code2vec.
```

Next confirmatory experiment should report both tracks rather than pretending
they answer the same research question.
