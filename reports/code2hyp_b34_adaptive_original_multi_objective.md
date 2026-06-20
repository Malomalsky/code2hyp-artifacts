# Multi-objective Code2Hyp selection

Objective:

`score = 0.500 * normalized(F1) + 0.500 * normalized(AST-distance Spearman)`

Best variant: `B31_hyperbolic_path_dual_attention_mp_soft_rank`

| Rank | Variant | Score | F1 mean | Spearman mean | Pareto | Note |
|---:|---|---:|---:|---:|---|---|
| 1 | B31_hyperbolic_path_dual_attention_mp_soft_rank | 0.7500 | 0.2008 | +0.1629 | yes | best; pareto |
| 2 | B34_hyperbolic_path_dual_attention_mp_adaptive_rank | 0.4785 | 0.2008 | +0.1304 | no |  |
| 3 | B29_hyperbolic_path_dual_attention_mp_separated | 0.2500 | 0.2008 | +0.1030 | no |  |

Interpretation boundary:

This is a selection diagnostic over already completed validation runs. It does not create a new trained model and must not be reported as a confirmatory statistical result without a preregistered larger run.
