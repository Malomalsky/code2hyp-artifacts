# B36 local-neighborhood regularization pilot

Дата: 2026-06-16.

## Research question

Предыдущий B35-пилот показал полезный, но неполный результат:
`B35_code2hyp_product_frechet_adaptive` сохраняет наиболее чистую
code2vec-compatible постановку и в режиме `structural_only` даёт лучший F1 среди
проверенных B1/B4/B8/B10/B16/B35/B_tree, но хуже B_tree по локальному
AST-neighborhood Overlap@1/Overlap@3.

Вопрос этого шага:

```text
Можно ли сохранить predictive F1 B35 и одновременно усилить согласование
локальных AST-соседств внутри множества path contexts?
```

## Mathematical change

Для каждого метода и каждого path-context anchor `i` строятся две вероятностные
окрестности по всем другим context `j`.

AST-target distribution:

```text
q(j | i) = softmax_j(-d_AST(i, j) / tau_T).
```

Learned-geometry distribution:

```text
p(j | i) = softmax_j(-d_model(i, j) / tau_E).
```

Новый structural regularizer:

```text
L_neighbor = mean_i KL(q(. | i) || p(. | i)).
```

Здесь `d_AST` - tree distance между AST paths через longest common prefix,
`d_model` - расстояние между learned structural context representations:
евклидово расстояние для Euclidean embeddings, геодезическое расстояние
Пуанкаре для Poincare points и Lorentz-distance для Lorentz points.

Смысл:

```text
rank loss проверяет глобальный порядок парных расстояний;
neighbor-distribution loss проверяет локальную вероятностную окрестность
каждого AST path context.
```

Методологически это ближе к линии stochastic-neighbor objectives:
Hinton, Roweis (SNE, NeurIPS 2002), van der Maaten, Hinton (t-SNE, JMLR 2008),
Goldberger et al. (NCA, NeurIPS 2004). Важно: в текущем коде реализован не
t-SNE и не NCA, а прикладной KL-регуляризатор локального AST-соседства.

Sources:

```text
Hinton, Roweis. Stochastic Neighbor Embedding.
https://proceedings.neurips.cc/paper/2002/hash/6150ccc6069bea6b5716254057a194ef-Abstract.html

van der Maaten, Hinton. Visualizing Data using t-SNE.
https://www.jmlr.org/papers/v9/vandermaaten08a.html

Goldberger et al. Neighbourhood Components Analysis.
https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html
```

## Implementation

Files:

```text
geometry_profile_research/code2hyp_torch.py
geometry_profile_research/code2hyp_training.py
scripts/run_code2hyp_java_small_pilot.py
tests/test_code2hyp_torch_model.py
tests/test_code2hyp_training.py
tests/test_code2hyp_runner_cli.py
```

New API:

```text
batch_structural_neighbor_distribution_regularizer(...)
```

New CLI option:

```text
--structural-regularizer neighbor_distribution
```

Named experimental condition:

```text
B36_code2hyp_product_frechet_neighbor
  = B35_code2hyp_product_frechet_adaptive
  + structural_regularizer = neighbor_distribution
```

## Commands

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
  --structural-regularizer neighbor_distribution \
  --variants B36_code2hyp_product_frechet_neighbor \
  --output outputs/code2hyp_b36_neighbor_original_512_2epochs_3seeds.json
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
  --structural-regularizer neighbor_distribution \
  --lexical-ablation structural_only \
  --variants B36_code2hyp_product_frechet_neighbor \
  --output outputs/code2hyp_b36_neighbor_structural_only_512_2epochs_3seeds.json
```

## Results

All numbers are mean over seeds 101, 202, 303.

| Regime | Regularizer | F1 | Spearman | Overlap@1 | Overlap@3 | Structural loss | Curvature |
|---|---|---:|---:|---:|---:|---:|---:|
| original | rank | 0.1932 | 0.0404 | 0.2655 | 0.3477 | 0.2214 | 0.9773 |
| original | neighbor_distribution | 0.1894 | 0.1527 | 0.2708 | 0.3611 | 0.1973 | 0.9764 |
| structural_only | rank | 0.2197 | 0.0630 | 0.2630 | 0.3427 | 0.2161 | 0.9763 |
| structural_only | neighbor_distribution | 0.2197 | 0.2188 | 0.2655 | 0.3569 | 0.1842 | 0.9750 |

## Interpretation

Supported:

```text
The neighbor-distribution regularizer improves B35 structural diagnostics in
this small pilot. The strongest effect is on global structural Spearman and
Overlap@3; F1 is preserved in structural_only and only slightly lower in
original.
```

Not supported yet:

```text
It is not yet evidence that neighbor_distribution is universally better than
rank regularization. The validation set is small, the run is short, and the
comparison is restricted to B35.
```

Current best formulation for the paper direction:

```text
Code2Hyp can be made closer to code2vec not only architecturally, through a
product Euclidean-hyperbolic context space, but also metrically: the AST-path
channel can be regularized by a local neighborhood objective that directly
aligns learned structural neighborhoods with AST tree neighborhoods.
```

Next confirmatory experiment:

```text
variants: B1, B4, B8, B10, B16, B35-rank, B35-neighbor, B_tree
regimes: original, obfuscated, structural_only
train_limit: >= 4096
val_limit: >= 1024
epochs: 3-5
seeds: 5 if compute allows
primary metrics: F1, Overlap@1, Overlap@3
secondary metrics: Spearman, curvature stability, channel weights
```
