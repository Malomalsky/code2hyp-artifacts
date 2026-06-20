# B39/B40/B41 code2vec context-transform Code2Hyp pilot

Дата: 2026-06-16.

## Purpose

Предыдущие варианты B37/B38 сохраняли code2vec-style attention over contexts,
но не воспроизводили важный слой исходной архитектуры code2vec:

```text
h_i = tanh(W [start_i; path_i; end_i] + b)
a_i = softmax_i(<h_i, q>)
v_code = sum_i a_i h_i
```

Поэтому B39/B40/B41 введены как более строгая проверка гипотезы:

```text
можно ли заменить именно AST-path канал code2vec на гиперболический канал,
не меняя входной объект path context и сохраняя context-transform attention.
```

## Variants

```text
B39_code2vec_context_transform_baseline
  Euclidean code2vec baseline:
  h_i = tanh(W concat(u_i, p_i, v_i) + b)
  a_i = softmax_i(<h_i, q>)
  representation = sum_i a_i h_i

B40_code2hyp_context_transform_frechet
  Code2Hyp context-transform candidate:
  z_i = exp_0^c(g_p(p_i)) in H_c
  ell_i = log_0^c(z_i)
  h_i = tanh(W concat(u_i, ell_i, v_i) + b)
  a_i = softmax_i(<h_i, q>)
  path aggregation = weighted Frechet mean of z_i
  representation = concat(weighted start mean, log_0^c(Frechet path mean), weighted end mean)

B41_code2hyp_context_transform_neighbor
  B40 + local AST-neighborhood distribution regularizer.
```

Parameter-cost control:

```text
B40 = B39 + 1 trainable curvature parameter.
B41 = B40 with a different structural regularizer, not a larger model.
```

## Commands

Original lexical+structural regime:

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
  --variants B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B41_code2hyp_context_transform_neighbor \
  --output outputs/code2hyp_b39_b40_b41_context_transform_original_512_2epochs_3seeds.json
```

Structural-only regime:

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
  --lexical-ablation structural_only \
  --variants B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B41_code2hyp_context_transform_neighbor \
  --output outputs/code2hyp_b39_b40_b41_context_transform_structural_only_512_2epochs_3seeds.json
```

## Results

All numbers are means over seeds 101, 202, 303.

| Variant | Regime | F1 | Accuracy | Spearman | Overlap@3 | Curvature | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| B39 Euclidean context-transform | original | 0.1705 | 0.1705 | -0.1419 | 0.3792 | 1.0000 | 186080 |
| B40 Code2Hyp context-transform Frechet | original | 0.1780 | 0.1780 | 0.2098 | 0.3486 | 0.9817 | 186081 |
| B41 B40 + neighbor distribution | original | 0.1780 | 0.1780 | 0.1840 | 0.3574 | 0.9777 | 186081 |
| B39 Euclidean context-transform | structural_only | 0.2386 | 0.2386 | -0.1298 | 0.3588 | 1.0000 | 80128 |
| B40 Code2Hyp context-transform Frechet | structural_only | 0.1742 | 0.1742 | 0.2345 | 0.3397 | 0.9790 | 80129 |
| B41 B40 + neighbor distribution | structural_only | 0.1742 | 0.1742 | 0.2127 | 0.3471 | 0.9789 | 80129 |

## Interpretation

Supported by this pilot:

```text
In the original regime, B40/B41 improve target-subtoken F1 over the stricter
B39 code2vec-context-transform baseline while adding only one trainable
curvature parameter.
```

```text
B40/B41 reverse the structural Spearman sign relative to B39 in both regimes.
This is the clearest current evidence that the hyperbolic AST-path channel
preserves tree-distance order better than the Euclidean context-transform
baseline.
```

```text
B41 improves Overlap@3 relative to B40 in both regimes, but does not improve
F1 in this small pilot. Its value is currently diagnostic/regularizing rather
than task-performance dominant.
```

Not supported yet:

```text
The structural-only F1 result does not support a broad claim that B40/B41 always
outperform Euclidean context-transform code2vec. B39 is stronger by F1 in the
small structural-only pilot.
```

```text
The correct claim is therefore narrower: hyperbolic AST-path replacement
improves structural order agreement, and in the original regime it also improves
F1 over the matched context-transform baseline. Larger runs are required before
claiming task-level dominance.
```

## Scientific role in the paper

B39/B40/B41 should be used to answer the direct reviewer-style question:

```text
What happens if we start from the actual code2vec context-transform pipeline
and replace the AST-path geometry with a hyperbolic one?
```

B35/B36 remain stronger product-metric candidates. B39/B40/B41 are the cleaner
mechanistic ablation family because they isolate the replacement of the AST-path
channel inside a closer code2vec pipeline.
