# Article Results Snapshot

Source files:

- `reports/task_geometry_effect_sizes_limit50.csv`
- `reports/task_geometry_residual_effect_sizes_limit50.csv`
- `reports/task_geometry_permutation_tests_limit50.csv`

## Statistical Design

The task-level association is measured as:

```text
eta_squared_task = SS_between(task_id) / SS_total
```

Residual tests first remove linear size/growth controls:

```text
curvature_metric = beta_0 + beta_1 * node_count + beta_2 * ball_size_mean_r3 + epsilon
eta_squared_task_residual = SS_between(task_id, epsilon) / SS_total(epsilon)
```

Permutation test:

- null model: random permutation of `task_id` labels across programs;
- permutations: 5000;
- seed: 20260614;
- multiple testing correction: Holm over raw and residual tests.

## Raw Task-Level Effects

| Metric | eta_squared_task | permutation p | Holm p |
|---|---:|---:|---:|
| `node_count` | 0.6942 | 0.0002 | 0.0028 |
| `ball_size_mean_r3` | 0.3777 | 0.0002 | 0.0028 |
| `forman_negative_mass` | 0.5414 | 0.0002 | 0.0028 |
| `forman_positive_mass` | 0.4150 | 0.0002 | 0.0028 |
| `ollivier_mean` | 0.3686 | 0.0002 | 0.0028 |
| `ollivier_negative_mass` | 0.4142 | 0.0002 | 0.0028 |
| `ollivier_near_zero_mass` | 0.4142 | 0.0002 | 0.0028 |

## Residual Task-Level Effects

| Metric | covariate R^2 | residual eta_squared_task | permutation p | Holm p |
|---|---:|---:|---:|---:|
| `forman_negative_mass` | 0.0564 | 0.5317 | 0.0002 | 0.0028 |
| `forman_positive_mass` | 0.0564 | 0.3838 | 0.0002 | 0.0028 |
| `ollivier_mean` | 0.1763 | 0.2822 | 0.0002 | 0.0028 |
| `ollivier_negative_mass` | 0.1679 | 0.4182 | 0.0002 | 0.0028 |
| `ollivier_near_zero_mass` | 0.1679 | 0.4182 | 0.0002 | 0.0028 |

## Strict Claim

On the 550-program DTA AST atlas, local discrete-curvature distributions are associated with task type. The association remains visible after linear controls for AST size and local ball growth; permutation testing with Holm correction gives `p_Holm <= 0.0028` for the selected raw and residual descriptors.

## Boundary

This is structural evidence, not a downstream performance claim. It supports the existence of task-level geometry signal in AST curvature fractions; it does not by itself prove that curvature features improve retrieval or classification.
