# Multi-objective Code2Hyp selection

Objective:

`score = 0.700 * normalized(F1) + 0.300 * normalized(AST-distance Spearman)`

Best variant: `B31_hyperbolic_path_dual_attention_mp_soft_rank`

| Rank | Variant | Score | F1 mean | Spearman mean | Pareto | Note |
|---:|---|---:|---:|---:|---|---|
| 1 | B31_hyperbolic_path_dual_attention_mp_soft_rank | 0.7000 | 0.2008 | +0.1629 | yes | best; pareto |
| 2 | B32_lorentz_path_dual_attention_mp_soft_rank | 0.3000 | 0.1477 | +0.2925 | yes | pareto |

Interpretation boundary:

This is a selection diagnostic over already completed validation runs. It does not create a new trained model and must not be reported as a confirmatory statistical result without a preregistered larger run.
