# Code2Hyp Java-small benchmark summary

## Scope

This report separates external literature numbers from local Code2Hyp runs.
The external numbers are full Java-small literature baselines from code2seq Table 1.
The local numbers are controlled Code2Hyp runs with the training budget recorded below.

## Local run metadata

- evaluation split: `val`
- train records used: `1000`
- evaluation records loaded: `512`
- evaluation records after known-target filtering: `187`
- epochs: `2`
- batch size: `128`
- seeds: `[101, 202, 303]`
- metric: `target-subtoken micro precision/recall/F1 with top-k = true target subtoken count`

## External Java-small literature baselines

| Model | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---|
| ConvAttention | 50.25 | 24.62 | 33.05 | Allamanis et al. 2016; code2seq Table 1 |
| Paths+CRFs | 8.39 | 5.63 | 6.74 | Alon et al. 2018; code2seq Table 1 |
| code2vec | 18.51 | 18.74 | 18.62 | Alon et al. 2019; code2seq Table 1 |
| 2-layer BiLSTM, no token splitting | 32.40 | 20.40 | 25.03 | code2seq Table 1 |
| 2-layer BiLSTM | 42.63 | 29.97 | 35.20 | code2seq Table 1 |
| TreeLSTM | 40.02 | 31.84 | 35.46 | Tai et al. 2015; code2seq Table 1 |
| Transformer | 38.13 | 26.70 | 31.41 | Vaswani et al. 2017; code2seq Table 1 |
| code2seq | 50.64 | 37.40 | 43.02 | Alon et al. 2019; code2seq Table 1 |

## Local Code2Hyp controlled results

| Variant | Precision | Recall | F1 | Structural Spearman | Edit Spearman | Jaccard Spearman | Normalized stress | Edit stress | Jaccard stress | Overlap@3 | Exact Overlap@3 | Karcher residual | Radius max | Near-boundary rate | Curvature | rho | n seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B39 matched code2vec-style baseline | 6.57 +/- 3.34 | 6.57 +/- 3.34 | 6.57 +/- 3.34 | -0.1710 +/- 0.0196 | 0.6114 +/- 0.0122 | 0.6428 +/- 0.0100 | 0.6899 +/- 0.0032 | 0.4872 +/- 0.0036 | 0.5167 +/- 0.0045 | 0.3842 +/- 0.0070 | 0.3438 +/- 0.0065 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 6.57 +/- 3.41 | 6.57 +/- 3.41 | 6.57 +/- 3.41 | 0.2062 +/- 0.0228 | 0.5319 +/- 0.0163 | 0.2053 +/- 0.0275 | 0.4204 +/- 0.0065 | 0.3465 +/- 0.0051 | 0.3531 +/- 0.0052 | 0.3243 +/- 0.0038 | 0.2833 +/- 0.0050 | 0.0003 +/- 0.0000 | 0.9950 +/- 0.0008 | 0.7442 +/- 0.0101 | 1.0028 +/- 0.0017 | 0.0971 +/- 0.0008 | 3 |
| B60 branch-sequence product manifold | 6.81 +/- 3.81 | 6.81 +/- 3.81 | 6.81 +/- 3.81 | 0.5658 +/- 0.0058 | 0.4986 +/- 0.0475 | 0.4599 +/- 0.0442 | 0.2603 +/- 0.0056 | 0.3523 +/- 0.0106 | 0.2696 +/- 0.0075 | 0.6956 +/- 0.0214 | 0.6028 +/- 0.0244 | 0.0000 +/- 0.0000 | 0.4353 +/- 0.0572 | 0.0000 +/- 0.0000 | 1.0173 +/- 0.0011 | 0.1037 +/- 0.0005 | 3 |
| B62 branch-sequence product manifold + multi-metric loss | 6.81 +/- 3.81 | 6.81 +/- 3.81 | 6.81 +/- 3.81 | 0.4168 +/- 0.0354 | 0.7455 +/- 0.0450 | 0.6974 +/- 0.0317 | 0.3055 +/- 0.0120 | 0.2796 +/- 0.0200 | 0.2058 +/- 0.0042 | 0.6784 +/- 0.0096 | 0.5913 +/- 0.0139 | 0.0000 +/- 0.0000 | 0.4274 +/- 0.0537 | 0.0000 +/- 0.0000 | 1.0172 +/- 0.0012 | 0.1037 +/- 0.0005 | 3 |
| B63 product-bias manifold + multi-metric loss | 6.57 +/- 3.41 | 6.57 +/- 3.41 | 6.57 +/- 3.41 | 0.1945 +/- 0.0196 | 0.5546 +/- 0.0135 | 0.2352 +/- 0.0219 | 0.4225 +/- 0.0065 | 0.3411 +/- 0.0053 | 0.3492 +/- 0.0058 | 0.3255 +/- 0.0026 | 0.2840 +/- 0.0035 | 0.0003 +/- 0.0000 | 0.9947 +/- 0.0008 | 0.7386 +/- 0.0073 | 0.9870 +/- 0.0025 | 0.0971 +/- 0.0008 | 3 |
| B64 Euclidean context-transform + multi-metric loss | 6.57 +/- 3.34 | 6.57 +/- 3.34 | 6.57 +/- 3.34 | -0.1671 +/- 0.0193 | 0.6149 +/- 0.0115 | 0.6447 +/- 0.0093 | 0.6879 +/- 0.0032 | 0.4850 +/- 0.0035 | 0.5147 +/- 0.0044 | 0.3826 +/- 0.0046 | 0.3423 +/- 0.0047 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| B65 L1 context-transform + multi-metric loss | 6.57 +/- 3.34 | 6.57 +/- 3.34 | 6.57 +/- 3.34 | -0.1651 +/- 0.0196 | 0.6081 +/- 0.0121 | 0.6383 +/- 0.0083 | 0.6971 +/- 0.0044 | 0.4984 +/- 0.0049 | 0.5282 +/- 0.0065 | 0.3832 +/- 0.0021 | 0.3416 +/- 0.0030 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
