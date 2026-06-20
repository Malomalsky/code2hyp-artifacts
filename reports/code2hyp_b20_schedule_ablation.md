# Code2Hyp B20 schedule ablation

Date: 2026-06-15

## Question

This experiment checks whether the structural-rank objective for hyperbolic
AST-path message passing should be applied from the beginning of training or
introduced after a warmup epoch.

The comparison is intentionally narrow:

```text
B17 = hyperbolic AST-path message passing, no structural-rank loss
B18 = B17 + constant structural-rank loss
B19 = B17 + linear structural-rank schedule
B20 = B17 + delayed-linear structural-rank schedule
```

B20 is not a new architecture. It is a schedule ablation for the same
geometry-aware AST-path message-passing encoder.

## Schedule

For epoch index `t`, total epochs `T`, and base structural weight `lambda`,
B20 uses:

```text
lambda_t = lambda * t / (T - 1),  T > 1
```

For the current 2-epoch pilot with `lambda = 0.05`, the realized schedule is:

```text
epoch 0: 0.00
epoch 1: 0.05
```

This differs from B19 because the first epoch is completely unregularized by
the structural-rank objective. The experiment therefore asks whether the task
encoder should first learn the target-subtoken objective before structural
alignment is imposed.

## Protocol

Real data only:

- corpus: code2seq Java-small preprocessed split;
- train limit: 1024 records;
- validation limit: 256 records;
- max contexts per method: 30;
- path encoder: GRU;
- epochs: 2;
- batch size: 64;
- seeds: 101, 202, 303;
- validation after known-target filtering: 129 records;
- lexical regimes: original, obfuscated, structural-only.

The full all-variant B20 run was computationally expensive on the current
machine, so B20 was executed through a focused variant filter. This is not a
change in the model or data; it only restricts the experiment harness to the
variants needed for this schedule ablation.

## Reproduction

Original:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 1024 \
  --val-limit 256 \
  --max-contexts 30 \
  --path-encoder gru \
  --epochs 2 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --variants B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed \
  --output outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json
```

Obfuscated:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 1024 \
  --val-limit 256 \
  --max-contexts 30 \
  --path-encoder gru \
  --epochs 2 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --lexical-ablation obfuscated \
  --variants B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed \
  --output outputs/code2hyp_java_small_focused_b20_obfuscated_1k_3seeds.json
```

Structural-only:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 1024 \
  --val-limit 256 \
  --max-contexts 30 \
  --path-encoder gru \
  --epochs 2 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --lexical-ablation structural_only \
  --variants B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed \
  --output outputs/code2hyp_java_small_focused_b20_structural_only_1k_3seeds.json
```

Figures:

```bash
./.venv/bin/python scripts/plot_code2hyp_b4_pilot.py \
  --input outputs/code2hyp_java_small_focused_b20_original_1k_3seeds.json \
  --output-prefix figures/code2hyp_focused_b20_original_1k_metrics

./.venv/bin/python scripts/plot_code2hyp_lexical_ablation.py \
  --output-prefix figures/code2hyp_focused_b20_lexical_ablation_metrics

./.venv/bin/python scripts/plot_code2hyp_b20_tradeoff.py
```

## Results

Original and obfuscated are numerically identical under the current learned
embedding setup because endpoint obfuscation preserves token identity and no
external pretrained token semantics are used.

### Original / obfuscated

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.1715 | +0.3980 | 0.1181 | 0.5536 |
| B8 Poincare Frechet | 0.1715 | +0.4077 | 0.1176 | 0.5308 |
| B17 Hyperbolic path MP | 0.1425 | +0.3531 | 0.1180 | 0.4647 |
| B18 Hyperbolic path MP + rank | 0.1245 | +0.3553 | 0.1177 | 0.4698 |
| B19 Hyperbolic path MP + linear rank | 0.1577 | +0.3429 | 0.1183 | 0.4557 |
| B20 Hyperbolic path MP + delayed rank | 0.1577 | +0.3459 | 0.1181 | 0.5269 |

### Structural-only

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.1314 | -0.2225 | 0.2008 | 0.9306 |
| B8 Poincare Frechet | 0.1452 | -0.2430 | 0.2008 | 0.9086 |
| B17 Hyperbolic path MP | 0.1231 | -0.0341 | 0.1544 | 0.6636 |
| B18 Hyperbolic path MP + rank | 0.1010 | -0.0356 | 0.1472 | 0.5261 |
| B19 Hyperbolic path MP + linear rank | 0.1577 | -0.0263 | 0.1452 | 0.5515 |
| B20 Hyperbolic path MP + delayed rank | 0.1383 | -0.0016 | 0.1433 | 0.6361 |

## Paired diagnostics

Detailed paired tables:

- `reports/code2hyp_paired_effects_focused_b20_original_f1.md`
- `reports/code2hyp_paired_effects_focused_b20_original_spearman.md`
- `reports/code2hyp_paired_effects_focused_b20_structural_only_f1.md`
- `reports/code2hyp_paired_effects_focused_b20_structural_only_spearman.md`

Trade-off figure:

- `figures/code2hyp_b20_f1_spearman_tradeoff.png`
- `figures/code2hyp_b20_f1_spearman_tradeoff.pdf`

Key paired observations:

```text
Original: B4 - B20 = +0.0138 F1, +0.0522 Spearman.
Structural-only: B20 - B4 = +0.0069 F1, +0.2210 Spearman.
Structural-only: B20 Spearman is closest to zero among B4/B8/B17/B18/B19/B20.
```

With only three seeds, exact sign tests remain exploratory. The result should
not be presented as confirmatory statistical proof.

## Interpretation

B20 does not replace B4/B8 as the main model. In the original and obfuscated
regimes, B4/B8 remain stronger by F1 and global structural Spearman.

B20 is scientifically useful for a narrower reason:

```text
The first unregularized epoch helps B20 reach near-zero structural-only
Spearman, but the delayed schedule gives up F1 relative to B19.
```

Thus the current ordering is:

```text
B4/B8: best balanced full-context hyperbolic code2vec family.
B19: best F1-preserving message-passing schedule in structural-only stress.
B20: best structural-alignment schedule in structural-only stress, but not
     the best predictive schedule.
```

The next serious experiment should not introduce another ad hoc architecture.
It should run a pre-registered schedule sweep:

```text
constant, linear, delayed-linear, cosine, warmup-plus-decay
```

and select checkpoints by a Pareto rule over:

```text
target-subtoken F1,
AST-distance Spearman,
structural distance loss,
structural rank loss.
```

## Current claim boundary

Defensible claim:

```text
Hyperbolic AST-path message passing exposes a real task/structure trade-off.
The trade-off is schedule-sensitive: constant structural supervision
over-regularizes, linear annealing preserves F1 better, and delayed annealing
improves structural-only alignment more strongly.
```

Not defensible yet:

```text
B20 is a better final model than B4/B8.
B20 proves statistical superiority.
The schedule result generalizes beyond the 1k/256 pilot.
```
