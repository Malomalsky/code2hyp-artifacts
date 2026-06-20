# Multi-objective Code2Hyp selection

Objective:

`score = 0.500 * normalized(F1) + 0.500 * normalized(AST-distance Spearman)`

Best variant: `B29_hyperbolic_path_dual_attention_mp_separated`

| Rank | Variant | Score | F1 mean | Spearman mean | Pareto | Note |
|---:|---|---:|---:|---:|---|---|
| 1 | B29_hyperbolic_path_dual_attention_mp_separated | 0.9750 | 0.0982 | +0.2278 | yes | best; pareto |
| 2 | B34_hyperbolic_path_dual_attention_mp_adaptive_rank | 0.5215 | 0.1010 | +0.1430 | yes | pareto |
| 3 | B31_hyperbolic_path_dual_attention_mp_soft_rank | 0.0000 | 0.0456 | +0.1391 | no |  |

Interpretation boundary:

This is a selection diagnostic over already completed validation runs. It does not create a new trained model and must not be reported as a confirmatory statistical result without a preregistered larger run.
