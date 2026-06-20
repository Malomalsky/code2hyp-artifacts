# Code2Hyp main candidates 4k pilot

Дата: 2026-06-16.

## Purpose

Этот прогон проверяет основные кандидаты Code2Hyp на более крупном subset,
чем ранние 512/128 exploratory pilots.

Проверяемый вопрос:

```text
Если взять code2vec-compatible постановку на Java-small AST path contexts,
какие варианты гиперболического AST-path канала дают выигрыш по downstream
method-name subtoken prediction и/или по structural faithfulness?
```

Сравниваемые варианты:

```text
B35_code2hyp_product_frechet_adaptive
  Product-space Code2Hyp: R_start x H_path x R_end, product-metric attention,
  trainable curvature, learned channel weights, Frechet AST-path aggregation.

B36_code2hyp_product_frechet_neighbor
  B35 + local AST-neighborhood distribution regularizer.

B39_code2vec_context_transform_baseline
  Euclidean code2vec baseline with h_i = tanh(W [start; path; end] + b).

B40_code2hyp_context_transform_frechet
  B39 with hyperbolic AST-path channel and Frechet path aggregation.

B44_code2hyp_context_transform_product_bias_frechet
  B39 with trainable hyperbolic product-metric structural bias:
  score_i = <h_i, q> + rho * product_score_i.
```

## Commands

Original regime:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 4000 \
  --val-limit 1024 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B35_code2hyp_product_frechet_adaptive,B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_main_candidates_original_4k_3epochs_3seeds.json
```

Structural-only regime:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 4000 \
  --val-limit 1024 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --lexical-ablation structural_only \
  --variants B35_code2hyp_product_frechet_adaptive,B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_main_candidates_structural_only_4k_3epochs_3seeds.json
```

Important dataset note:

```text
train records used: 4000
validation requested: 1024
validation records after known-target filtering: 637
seeds: 101, 202, 303
```

The filtered validation size must be reported because validation examples with
unknown target subtokens are removed by the real-data pilot pipeline.

## Results

All values are means over seeds 101, 202, 303. Standard deviations are reported
for F1 and structural Spearman.

### Original

| Variant | F1 | Spearman | Overlap@3 | Curvature | Bias rho | Params |
|---|---:|---:|---:|---:|---:|---:|
| B35 product-Frechet | 0.1495 +/- 0.0107 | 0.3191 +/- 0.3288 | 0.5841 | 0.9297 | 0.0000 | 735016 |
| B36 product+neighbor | 0.1516 +/- 0.0005 | 0.5856 +/- 0.0709 | 0.8043 | 0.9283 | 0.0000 | 735016 |
| B39 code2vec baseline | 0.1263 +/- 0.0147 | -0.3040 +/- 0.0145 | 0.4219 | 1.0000 | 0.0000 | 744324 |
| B40 hyp path-Frechet | 0.1374 +/- 0.0069 | 0.4036 +/- 0.2643 | 0.5704 | 0.9171 | 0.0000 | 744325 |
| B44 structural bias | 0.1269 +/- 0.0146 | 0.9014 +/- 0.0275 | 0.8711 | 0.8181 | 0.0987 | 744329 |

### Structural-only

| Variant | F1 | Spearman | Overlap@3 | Curvature | Bias rho | Params |
|---|---:|---:|---:|---:|---:|---:|
| B35 product-Frechet | 0.1338 +/- 0.0097 | 0.2878 +/- 0.3765 | 0.5804 | 0.9356 | 0.0000 | 200936 |
| B36 product+neighbor | 0.1296 +/- 0.0110 | 0.5495 +/- 0.0433 | 0.7905 | 0.8986 | 0.0000 | 200936 |
| B39 code2vec baseline | 0.1263 +/- 0.0147 | -0.3172 +/- 0.0124 | 0.4043 | 1.0000 | 0.0000 | 210244 |
| B40 hyp path-Frechet | 0.1335 +/- 0.0096 | 0.5817 +/- 0.0724 | 0.6900 | 0.9277 | 0.0000 | 210245 |
| B44 structural bias | 0.1263 +/- 0.0147 | 0.9317 +/- 0.0102 | 0.8938 | 0.8223 | 0.0998 | 210249 |

## Paired deltas vs B39

All deltas are paired by model seed.

### Original

| Variant | mean Delta F1 | mean Delta Spearman | mean Delta Overlap@3 |
|---|---:|---:|---:|
| B35 | +0.0233 | +0.6231 | +0.1622 |
| B36 | +0.0254 | +0.8896 | +0.3824 |
| B40 | +0.0112 | +0.7076 | +0.1485 |
| B44 | +0.0006 | +1.2054 | +0.4492 |

### Structural-only

| Variant | mean Delta F1 | mean Delta Spearman | mean Delta Overlap@3 |
|---|---:|---:|---:|
| B35 | +0.0076 | +0.6051 | +0.1761 |
| B36 | +0.0033 | +0.8667 | +0.3862 |
| B40 | +0.0072 | +0.8989 | +0.2857 |
| B44 | +0.0000 | +1.2490 | +0.4895 |

## Figure

```text
figures/code2hyp_main_candidates_4k_f1_spearman.png
figures/code2hyp_main_candidates_4k_f1_spearman.pdf
```

The figure plots validation F1 against structural Spearman. Bubble area is
proportional to structural Overlap@3.

## Interpretation

Supported by this 4k pilot:

```text
B36 is the strongest current downstream candidate in the original regime:
it has the best mean F1 and strong structural diagnostics.
```

```text
B44 is the strongest structural-faithfulness candidate:
it has the highest structural Spearman and Overlap@3 in both regimes.
```

```text
B40 is the strongest code2vec-context-transform replacement in structural-only
when balancing F1 and structural Spearman: it improves both over B39.
```

```text
All hyperbolic/product candidates turn structural Spearman from negative in
B39 to positive values in both original and structural-only regimes.
```

Not supported:

```text
B44 does not improve downstream F1 over B39 in this 4k pilot. Its value is
structural faithfulness and interpretability, not immediate task F1.
```

```text
With n = 3 seeds, this pilot should not be reported with p-values. The correct
statistical presentation is mean +/- standard deviation and paired seed deltas.
```

## Current scientific conclusion

The result supports a two-axis Code2Hyp claim:

```text
1. Product-space hyperbolic AST-path modeling can improve downstream
   method-name subtoken prediction relative to a matched Euclidean
   code2vec-context baseline.

2. A trainable hyperbolic structural-bias attention mechanism can strongly
   improve structural order preservation without changing the external
   code2vec-style representation interface.
```

The strongest current implementation path:

```text
Use B36 as the task-performance candidate.
Use B44 as the interpretable structural-faithfulness candidate.
Use B40 as the conservative code2vec-context-transform replacement.
```

Next required step:

```text
Run B35/B36/B39/B40/B44 on a larger subset or longer training schedule and
report confidence intervals. If the same pattern holds, the paper can be framed
around a task-performance/structural-faithfulness Pareto frontier rather than a
single winner.
```
