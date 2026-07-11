# Code2Hyp LCA causal matrix summary

- Status: `pilot`
- Study stage: `pilot`
- Seeds: 5
- Tasks: 12
- Primary metric: MAP@2
- Task-cluster bootstrap resamples: 20000

## Cells

| Cell | MAP@R | Seed range |
|---|---:|---:|
| `EEE__depth_matched_shuffled__measure` | 0.0896 | 0.0312..0.1354 |
| `EEE__endpoint_only__measure` | 0.0625 | 0.0208..0.1042 |
| `EEE__full_path_no_explicit_lca__measure` | 0.0792 | 0.0625..0.1146 |
| `EEE__program_shuffled_lca__measure` | 0.0750 | 0.0521..0.1562 |
| `EEE__root_anchor__measure` | 0.0625 | 0.0208..0.1042 |
| `EEE__true_lca__measure` | 0.1562 | 0.1146..0.2292 |
| `EEE__zero_anchor__measure` | 0.0625 | 0.0208..0.1042 |
| `EEE_concat__true_lca__measure` | 0.1562 | 0.1146..0.2292 |
| `HEE__true_lca__measure` | 0.1542 | 0.1250..0.2188 |
| `HEE_near_zero__true_lca__measure` | 0.1562 | 0.1146..0.2292 |
| `HHH__true_lca__measure` | 0.1521 | 0.1042..0.2188 |

## Planned contrasts

| Contrast | Delta MAP@R | 95% task-bootstrap CI | +/=/- tasks | Exact sign p |
|---|---:|---:|---:|---:|
| `H1_true_lca_vs_zero_anchor` | +0.0938 | [+0.0542, +0.1500] | 11/1/0 | 0.0009766 |
| `true_lca_vs_endpoint_only` | +0.0938 | [+0.0542, +0.1479] | 11/1/0 | 0.0009766 |
| `true_lca_vs_program_shuffled_lca` | +0.0813 | [+0.0312, +0.1500] | 10/1/1 | 0.01172 |
| `true_lca_vs_full_path_pool` | +0.0771 | [+0.0479, +0.1083] | 10/2/0 | 0.001953 |
| `product_vs_equal_capacity_concat_identity` | +0.0000 | [+0.0000, +0.0000] | 0/12/0 | 1 |
| `HEE_vs_EEE` | -0.0021 | [-0.0125, +0.0083] | 2/8/2 | 1 |
| `HEE_vs_near_zero_HEE` | -0.0021 | [-0.0146, +0.0083] | 2/8/2 | 1 |
| `HEE_vs_HHH` | +0.0021 | [-0.0104, +0.0146] | 3/7/2 | 1 |

## Decision

- Gate A H1 positive direction: `True`.
- Gate A H1 CI excludes zero: `True`.
- Gate C pass: `False`. It requires HEE to beat both EEE and near-zero HEE with CIs above zero.
- Confirmatory claim allowed: `False`.
