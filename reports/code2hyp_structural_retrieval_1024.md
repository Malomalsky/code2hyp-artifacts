# Code2Hyp structural retrieval diagnostics

| Regime | Variant | n | F1 | Spearman | Overlap@1 | Overlap@3 |
|---|---|---:|---:|---:|---:|---:|
| Original | B1_euclidean | 3 | 0.0871 | -0.1971 | 0.3038 | 0.3459 |
| Original | B29_hyperbolic_path_dual_attention_mp_separated | 3 | 0.0844 | +0.1110 | 0.2655 | 0.4152 |
| Original | B31_hyperbolic_path_dual_attention_mp_soft_rank | 3 | 0.0858 | +0.1396 | 0.2769 | 0.4138 |
| Original | B34_hyperbolic_path_dual_attention_mp_adaptive_rank | 3 | 0.0775 | +0.1672 | 0.2769 | 0.3994 |
| Original | B4_hyperbolic_code2vec | 3 | 0.0954 | +0.3987 | 0.2834 | 0.4813 |
| Original | B8_hyperbolic_frechet_code2vec | 3 | 0.0954 | +0.3295 | 0.3104 | 0.5067 |
| Structural-only | B1_euclidean | 3 | 0.0705 | -0.1608 | 0.2988 | 0.3659 |
| Structural-only | B29_hyperbolic_path_dual_attention_mp_separated | 3 | 0.0982 | +0.2278 | 0.4699 | 0.4600 |
| Structural-only | B31_hyperbolic_path_dual_attention_mp_soft_rank | 3 | 0.0456 | +0.1391 | 0.4988 | 0.5054 |
| Structural-only | B34_hyperbolic_path_dual_attention_mp_adaptive_rank | 3 | 0.1010 | +0.1430 | 0.4393 | 0.4570 |
| Structural-only | B4_hyperbolic_code2vec | 3 | 0.0581 | -0.2143 | 0.3642 | 0.3900 |
| Structural-only | B8_hyperbolic_frechet_code2vec | 3 | 0.0692 | -0.1698 | 0.3632 | 0.3904 |

Interpretation boundary:

Overlap@k is a local structural diagnostic: for each AST path-context it checks whether the k nearest neighbors in the learned geometry fall inside the tie-tolerant set of k nearest AST neighbors. It complements global Spearman correlation and does not replace task F1.
Best Overlap@1 variant: `B31_hyperbolic_path_dual_attention_mp_soft_rank` (Structural-only, 0.4988).
Best Overlap@3 variant: `B8_hyperbolic_frechet_code2vec` (Original, 0.5067).
