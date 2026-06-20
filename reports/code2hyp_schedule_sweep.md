# Code2Hyp schedule sweep

Date: 2026-06-15

## Purpose

The previous B20 experiment showed that structural-rank regularization for
hyperbolic AST-path message passing is schedule-sensitive. This follow-up
checks whether that observation is robust across several schedules rather than
being a single B20 artifact.

The sweep is intentionally restricted to the same architecture:

```text
B17 = hyperbolic AST-path message passing, no structural-rank objective
B18 = B17 + constant structural-rank objective
B19 = B17 + linear structural-rank schedule
B20 = B17 + delayed-linear structural-rank schedule
B21 = B17 + cosine structural-rank schedule
B22 = B17 + warmup-decay structural-rank schedule
```

Thus B21/B22 are not new architectures. They are schedule controls.

## Schedules

For base weight `lambda = 0.05`:

| Variant | Schedule | 3 epochs | 5 epochs |
|---|---|---:|---:|
| B17 | none | `[0, 0, 0]` | `[0, 0, 0, 0, 0]` |
| B18 | constant | `[0.05, 0.05, 0.05]` | `[0.05, 0.05, 0.05, 0.05, 0.05]` |
| B19 | linear | `[0.0167, 0.0333, 0.05]` | `[0.01, 0.02, 0.03, 0.04, 0.05]` |
| B20 | delayed linear | `[0, 0.025, 0.05]` | `[0, 0.0125, 0.025, 0.0375, 0.05]` |
| B21 | cosine ramp | `[0, 0.025, 0.05]` | `[0, 0.0073, 0.025, 0.0427, 0.05]` |
| B22 | warmup-decay | `[0, 0.05, 0]` | `[0, 0.025, 0.05, 0.025, 0]` |

Important: with only three epochs, B20 and B21 are identical. Therefore a
five-epoch micro-sweep is included only to separate the schedule shapes.

## Protocol

Real data only:

- source corpus: code2seq Java-small preprocessed split;
- model seeds: 101, 202, 303;
- path encoder: GRU;
- max contexts: 20;
- structural regularizer: rank;
- regimes: original and structural-only.

Two pilot scales were used:

```text
Primary exploratory sweep:
  train-limit = 512
  val-limit = 128
  epochs = 3

Schedule-shape sensitivity:
  train-limit = 256
  val-limit = 64
  epochs = 5
```

The aborted larger run is intentionally not used as evidence:

```text
train-limit = 1024, val-limit = 256, epochs = 3
```

It was interrupted because runtime was too high for the current machine.

## Reproduction

Primary original sweep:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 20 \
  --path-encoder gru \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --structural-regularizer rank \
  --variants B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed,B21_hyperbolic_path_mp_rank_cosine,B22_hyperbolic_path_mp_rank_warmup_decay \
  --output outputs/code2hyp_schedule_sweep_original_512_3epochs_3seeds.json
```

Primary structural-only sweep:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 20 \
  --path-encoder gru \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --structural-regularizer rank \
  --lexical-ablation structural_only \
  --variants B4_hyperbolic_code2vec,B8_hyperbolic_frechet_code2vec,B17_hyperbolic_path_mp_code2vec,B18_hyperbolic_path_mp_struct_rank,B19_hyperbolic_path_mp_rank_annealed,B20_hyperbolic_path_mp_rank_delayed,B21_hyperbolic_path_mp_rank_cosine,B22_hyperbolic_path_mp_rank_warmup_decay \
  --output outputs/code2hyp_schedule_sweep_structural_only_512_3epochs_3seeds.json
```

Schedule-shape sensitivity uses the same commands with `--train-limit 256`,
`--val-limit 64`, and `--epochs 5`.

Figure:

```bash
./.venv/bin/python scripts/plot_code2hyp_schedule_sweep.py
```

## Results: 512 / 3 epochs

### Original

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.1705 | +0.5898 | 0.1014 | 1.7048 |
| B8 Frechet aggregation | 0.1742 | +0.5959 | 0.1008 | 1.7041 |
| B17 No rank schedule | 0.1818 | +0.4678 | 0.1195 | 0.3776 |
| B18 Constant rank | 0.1667 | +0.4455 | 0.1200 | 0.3898 |
| B19 Linear | 0.1591 | +0.4590 | 0.1205 | 0.2386 |
| B20 Delayed linear | 0.1780 | +0.4909 | 0.1188 | 0.2934 |
| B21 Cosine | 0.1780 | +0.4909 | 0.1188 | 0.2934 |
| B22 Warmup-decay | 0.1780 | +0.4614 | 0.1201 | 0.2963 |

### Structural only

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.1894 | +0.2326 | 0.1327 | 0.9162 |
| B8 Frechet aggregation | 0.1894 | +0.2316 | 0.1327 | 0.9192 |
| B17 No rank schedule | 0.1402 | -0.0488 | 0.1654 | 0.4250 |
| B18 Constant rank | 0.1136 | +0.0811 | 0.1574 | 0.3465 |
| B19 Linear | 0.1629 | +0.0269 | 0.1492 | 0.2061 |
| B20 Delayed linear | 0.1629 | -0.0010 | 0.1495 | 0.4402 |
| B21 Cosine | 0.1629 | -0.0010 | 0.1495 | 0.4402 |
| B22 Warmup-decay | 0.1477 | -0.0329 | 0.1547 | 0.2339 |

## Results: 256 / 5 epochs

This is a schedule-shape sensitivity check, not the main evidence base.

### Original

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.1057 | +0.5598 | 0.1106 | 0.6021 |
| B8 Frechet aggregation | 0.1057 | +0.5623 | 0.1101 | 0.5959 |
| B17 No rank schedule | 0.1870 | +0.4718 | 0.1310 | 0.2538 |
| B18 Constant rank | 0.1138 | +0.5470 | 0.1236 | 0.3325 |
| B19 Linear | 0.1545 | +0.5292 | 0.1278 | 0.2332 |
| B20 Delayed linear | 0.1789 | +0.5314 | 0.1284 | 0.2322 |
| B21 Cosine | 0.1707 | +0.5273 | 0.1284 | 0.2253 |
| B22 Warmup-decay | 0.1707 | +0.5280 | 0.1292 | 0.2514 |

### Structural only

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B4 Full-context Poincare | 0.0976 | +0.3573 | 0.1489 | 1.0548 |
| B8 Frechet aggregation | 0.0976 | +0.3572 | 0.1489 | 1.0596 |
| B17 No rank schedule | 0.0976 | +0.0735 | 0.1674 | 1.0813 |
| B18 Constant rank | 0.0650 | +0.1135 | 0.2028 | 0.6720 |
| B19 Linear | 0.1138 | +0.1473 | 0.1650 | 0.8947 |
| B20 Delayed linear | 0.0976 | +0.0971 | 0.1631 | 0.9435 |
| B21 Cosine | 0.0650 | +0.0857 | 0.1644 | 0.8990 |
| B22 Warmup-decay | 0.1138 | +0.1277 | 0.1664 | 0.9628 |

## Paired diagnostics

All paired diagnostics are exploratory because `n = 3` seeds:

- `reports/code2hyp_schedule_sweep_512_original_f1_vs_b17.md`
- `reports/code2hyp_schedule_sweep_512_original_spearman_vs_b17.md`
- `reports/code2hyp_schedule_sweep_512_structural_only_f1_vs_b17.md`
- `reports/code2hyp_schedule_sweep_512_structural_only_spearman_vs_b17.md`
- `reports/code2hyp_schedule_sweep_256_original_f1_vs_b17.md`
- `reports/code2hyp_schedule_sweep_256_original_spearman_vs_b17.md`
- `reports/code2hyp_schedule_sweep_256_structural_only_f1_vs_b17.md`
- `reports/code2hyp_schedule_sweep_256_structural_only_spearman_vs_b17.md`

Figure:

- `figures/code2hyp_schedule_sweep_f1_spearman.png`
- `figures/code2hyp_schedule_sweep_f1_spearman.pdf`

## Interpretation

The sweep strengthens the research argument by ruling out a premature B20
claim.

Main observations:

1. B20 is not a stable improvement over B19. In the 512/3 structural-only
   setting, B19 and B20 tie by F1, but B19 has better Spearman.
2. B21 is mathematically identical to B20 at three epochs and does not improve
   it at five epochs.
3. B22 warmup-decay is not dominant. It is competitive by F1 in the 256/5
   structural-only sensitivity check, but it does not beat B19 by Spearman.
4. B18 constant rank can improve structural Spearman, but it hurts F1. This
   supports the original over-regularization interpretation.
5. B19 linear scheduling is the best current compromise among the scheduled
   hyperbolic AST-path message-passing variants.
6. B4/B8 remain important controls. They can still dominate structural-only
   Spearman on small splits, so the stronger article claim must be comparative
   and bounded rather than "message passing always wins."

Current defensible claim:

```text
Structural-rank supervision for hyperbolic AST-path message passing is
schedule-sensitive. Constant supervision is structurally stricter but can
over-regularize F1; delayed/cosine/warmup-decay schedules do not dominate;
linear scheduling is the most stable current compromise.
```

Not defensible:

```text
B20/B21/B22 are better final models than B4/B8.
The schedule sweep proves statistical superiority.
The result is confirmatory rather than exploratory.
```

## Next step

The next scientifically strong run should be:

```text
train-limit = 1024
val-limit = 256
epochs = 5
variants = B4, B8, B17, B18, B19, B20, B21, B22
regimes = original, structural_only
selection = pre-registered Pareto rule over F1, Spearman, structural loss, rank loss
```

This is the first run large enough to separate schedules while staying close to
the previous B20 setting.
