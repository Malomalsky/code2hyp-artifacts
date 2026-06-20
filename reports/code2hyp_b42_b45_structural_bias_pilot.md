# B42-B45 product-metric and structurally biased code2vec-context pilots

Дата: 2026-06-16.

## Purpose

B39-B41 проверили более точную code2vec-context-transform постановку:

```text
h_i = tanh(W [start_i; path_i; end_i] + b)
a_i = softmax_i(<h_i, q>)
```

Следующий вопрос:

```text
как встроить гиперболическую геометрию AST-path канала так, чтобы не разрушить
code2vec-compatible контекстный вектор?
```

Для этого реализованы две линии:

```text
B42/B43: заменить attention на product-metric attention в R x H x R,
         но итоговый code vector оставить code2vec-style weighted transformed
         context.

B44/B45: сохранить обычный code2vec semantic attention и добавить к нему
         обучаемый hyperbolic product-metric structural bias.
```

## Scientific motivation

`code2vec` моделирует метод как множество AST path contexts:

```text
c_i = (s_i, p_i, t_i),
```

где `p_i` является AST path. Поэтому гиперболическая часть должна относиться
не ко всему коду сразу, а именно к структурному path-каналу.

Опорные источники:

```text
Alon et al. code2vec: Learning Distributed Representations of Code.
POPL 2019 / arXiv:1803.09473.
https://arxiv.org/abs/1803.09473

Alon et al. code2seq: Generating Sequences from Structured Representations of Code.
ICLR 2019 / arXiv:1808.01400.
https://arxiv.org/abs/1808.01400

Nickel, Kiela. Poincare Embeddings for Learning Hierarchical Representations.
NeurIPS 2017.
https://arxiv.org/abs/1705.08039

Ganea, Becigneul, Hofmann. Hyperbolic Neural Networks.
NeurIPS 2018.
https://arxiv.org/abs/1805.09112

Skopek, Ganea, Becigneul. Mixed-curvature Variational Autoencoders.
ICLR 2020.
https://arxiv.org/abs/1911.08411
```

Conservative novelty framing:

```text
The novelty is not "the first hyperbolic model for code" as an absolute claim.
The defensible novelty is a controlled code2vec-compatible study that treats
AST-path representation geometry as an explicit experimental factor under
matched-capacity Euclidean, hyperbolic, product-metric and structural-bias
controls.
```

## Models

### B42/B43: product-metric attention with code2vec vector

For each context:

```text
u_i = E_s(s_i) in R^d_s
z_i = exp_0^c(g_p(p_i)) in H_c^d_p
v_i = E_t(t_i) in R^d_t

ell_i = log_0^c(z_i)
h_i = tanh(W [u_i; ell_i; v_i] + b)
```

Product-metric score:

```text
score_i = -(
    alpha_s ||u_i - q_s||^2
  + alpha_p d_H(z_i, q_p)^2
  + alpha_t ||v_i - q_t||^2
)

a_i = softmax_i(score_i)
v_code = sum_i a_i h_i
```

B43 adds `neighbor_distribution` regularization.

### B44/B45: hyperbolic structural bias for code2vec attention

B44 is more conservative than B42:

```text
semantic_score_i = <h_i, q>
product_score_i  = -(
    alpha_s ||u_i - q_s||^2
  + alpha_p d_H(z_i, q_p)^2
  + alpha_t ||v_i - q_t||^2
)

score_i = semantic_score_i + rho * product_score_i
a_i = softmax_i(score_i)
v_code = sum_i a_i h_i
```

where:

```text
rho > 0 is a trainable structural-bias weight.
```

Interpretation:

```text
B44 does not force the model to abandon code2vec attention. It gives the model
a learnable geometric correction that can be used if AST-path geometry is
useful.
```

B45 adds `neighbor_distribution` regularization.

## Commands

Original regime:

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
  --variants B42_code2hyp_product_context_transform_frechet,B43_code2hyp_product_context_transform_neighbor \
  --output outputs/code2hyp_b42_b43_product_context_transform_original_512_2epochs_3seeds.json

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
  --variants B44_code2hyp_context_transform_product_bias_frechet,B45_code2hyp_context_transform_product_bias_neighbor \
  --output outputs/code2hyp_b44_b45_context_product_bias_original_512_2epochs_3seeds.json
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
  --variants B42_code2hyp_product_context_transform_frechet,B43_code2hyp_product_context_transform_neighbor \
  --output outputs/code2hyp_b42_b43_product_context_transform_structural_only_512_2epochs_3seeds.json

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
  --variants B44_code2hyp_context_transform_product_bias_frechet,B45_code2hyp_context_transform_product_bias_neighbor \
  --output outputs/code2hyp_b44_b45_context_product_bias_structural_only_512_2epochs_3seeds.json
```

## Results

All numbers are means over seeds 101, 202, 303.

| Variant | Regime | F1 | Accuracy | Spearman | Overlap@3 | Curvature | Bias rho | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B39 Euclidean context-transform | original | 0.1705 | 0.1705 | -0.1419 | 0.3792 | 1.0000 | - | 186080 |
| B40 Hyperbolic path Frechet attention | original | 0.1780 | 0.1780 | 0.2098 | 0.3486 | 0.9817 | - | 186081 |
| B42 Product-metric attention | original | 0.1705 | 0.1705 | 0.1333 | 0.3466 | 1.0092 | - | 186084 |
| B44 Product-metric structural bias | original | 0.1705 | 0.1705 | 0.1355 | 0.3445 | 1.0077 | 0.0981 | 186085 |
| B39 Euclidean context-transform | structural_only | 0.2386 | 0.2386 | -0.1298 | 0.3588 | 1.0000 | - | 80128 |
| B40 Hyperbolic path Frechet attention | structural_only | 0.1742 | 0.1742 | 0.2345 | 0.3397 | 0.9790 | - | 80129 |
| B42 Product-metric attention | structural_only | 0.2386 | 0.2386 | 0.1611 | 0.3382 | 0.9974 | - | 80132 |
| B44 Product-metric structural bias | structural_only | 0.2386 | 0.2386 | 0.1971 | 0.3392 | 1.0027 | 0.0976 | 80133 |

Neighbor-distribution counterparts:

| Variant | Regime | F1 | Spearman | Overlap@3 | Notes |
|---|---|---:|---:|---:|---|
| B43 | original | 0.1705 | 0.1309 | 0.3508 | Improves Overlap@3 over B42 |
| B45 | original | 0.1705 | 0.1246 | 0.3471 | Improves Overlap@3 over B44 |
| B43 | structural_only | 0.2386 | 0.1776 | 0.3427 | Improves Spearman and Overlap@3 over B42 |
| B45 | structural_only | 0.2386 | 0.1812 | 0.3439 | Improves Overlap@3 over B44, Spearman below B44 |

## Interpretation

Supported:

```text
B42/B44 preserve the B39 task-level F1 in structural_only while changing
structural Spearman from negative to positive. This is a strong geometry
preservation signal without task-performance penalty in the lexically weakened
setting.
```

```text
B44 is currently the best conservative "tool" design: it keeps the original
code2vec-context vector and adds a trainable structural bias rather than fully
replacing semantic attention.
```

```text
B40 remains the best small-pilot original-regime model among B39-B45 by F1 and
Spearman. It is more aggressive: attention is computed from transformed
hyperbolic-path contexts rather than as a product-metric bias.
```

Not supported:

```text
B42-B45 do not improve original-regime F1 over B39 in the 512/128 two-epoch
pilot. They improve geometry diagnostics, not immediate task F1.
```

```text
The neighbor_distribution regularizer improves some local-neighborhood
diagnostics but is not consistently better by F1 or Spearman in this small
pilot.
```

## Current recommendation

For a paper/tool framing, keep three candidates:

```text
B40: aggressive hyperbolic path-Frechet context-transform model.
B44: conservative code2vec attention with trainable hyperbolic structural bias.
B35/B36: product-metric candidate family from the earlier line.
```

Recommended next experiment:

```text
Run B39, B40, B44, B35 and B36 on a larger Java-small subset with at least
3 seeds and enough epochs to test whether the structural-geometry advantage
survives beyond the 512-record pilot.
```

Claim boundary:

```text
At this stage, the strongest defensible statement is:

"Hyperbolic/product AST-path geometry consistently improves structural
distance-order agreement under matched-capacity code2vec-compatible controls.
In the small pilot, B40 also improves original-regime F1, while B44 preserves
structural-only F1 and improves structural order agreement."
```
