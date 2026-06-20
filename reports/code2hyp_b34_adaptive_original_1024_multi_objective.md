# Multi-objective Code2Hyp selection

Objective:

`score = 0.500 * normalized(F1) + 0.500 * normalized(AST-distance Spearman)`

Best variant: `B31_hyperbolic_path_dual_attention_mp_soft_rank`

| Rank | Variant | Score | F1 mean | Spearman mean | Pareto | Note |
|---:|---|---:|---:|---:|---|---|
| 1 | B31_hyperbolic_path_dual_attention_mp_soft_rank | 0.7540 | 0.0858 | +0.1396 | yes | best; pareto |
| 2 | B34_hyperbolic_path_dual_attention_mp_adaptive_rank | 0.5000 | 0.0775 | +0.1672 | yes | pareto |
| 3 | B29_hyperbolic_path_dual_attention_mp_separated | 0.4167 | 0.0844 | +0.1110 | no |  |

Interpretation boundary:

This is a selection diagnostic over already completed validation runs. It does not create a new trained model and must not be reported as a confirmatory statistical result without a preregistered larger run.
