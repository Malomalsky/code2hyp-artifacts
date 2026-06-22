# Code2Hyp Java-small benchmark summary

## Scope

This report separates external literature numbers from local Code2Hyp runs.
The external numbers are full Java-small literature baselines from code2seq Table 1.
The local numbers are controlled Code2Hyp runs with the training budget recorded below.

## Local run metadata

- evaluation split: `test`
- train records used: `25000`
- evaluation records loaded: `8192`
- evaluation records after known-target filtering: `6642`
- epochs: `5`
- batch size: `128`
- seeds: `[101, 202, 303, 404, 505]`
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

| Variant | Precision | Recall | F1 | Structural Spearman | Normalized stress | Overlap@3 | Exact Overlap@3 | Karcher residual | Radius max | Near-boundary rate | Curvature | rho | n seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code2Hyp B36 product-Frechet + neighbor | 18.59 +/- 0.47 | 18.61 +/- 0.47 | 18.60 +/- 0.47 | 0.6634 +/- 0.1655 | 0.2062 +/- 0.0392 | 0.8481 +/- 0.0605 | 0.7486 +/- 0.0543 | 0.0048 +/- 0.0052 | 0.9999 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.8979 +/- 0.0505 | 0.0000 +/- 0.0000 | 5 |
| B39 matched code2vec-style baseline | 15.77 +/- 0.49 | 15.79 +/- 0.49 | 15.78 +/- 0.49 | -0.3567 +/- 0.0112 | 0.8218 +/- 0.0193 | 0.3378 +/- 0.0166 | 0.2923 +/- 0.0149 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 5 |
| Code2Hyp B44 structural-bias attention | 16.70 +/- 0.85 | 16.72 +/- 0.85 | 16.71 +/- 0.85 | 0.9775 +/- 0.0023 | 0.0622 +/- 0.0029 | 0.9603 +/- 0.0029 | 0.8473 +/- 0.0034 | 0.0001 +/- 0.0001 | 0.9990 +/- 0.0001 | 0.9988 +/- 0.0026 | 0.9476 +/- 0.0771 | 0.1065 +/- 0.0086 | 5 |
| B47 Euclidean context-transform + distance loss | 16.00 +/- 0.60 | 16.02 +/- 0.60 | 16.01 +/- 0.60 | 0.7603 +/- 0.0519 | 0.1980 +/- 0.0146 | 0.7778 +/- 0.0115 | 0.6803 +/- 0.0102 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 5 |
| B48 hyperbolic product-bias without structural loss | 16.74 +/- 0.77 | 16.76 +/- 0.77 | 16.75 +/- 0.77 | 0.1838 +/- 0.0292 | 0.3329 +/- 0.0061 | 0.4313 +/- 0.0240 | 0.3674 +/- 0.0230 | 0.0022 +/- 0.0011 | 0.9999 +/- 0.0000 | 0.9985 +/- 0.0022 | 0.9907 +/- 0.0250 | 0.1019 +/- 0.0053 | 5 |
| B49 same code path, near-Euclidean curvature | 16.07 +/- 0.55 | 16.09 +/- 0.55 | 16.08 +/- 0.55 | 0.8089 +/- 0.0347 | 0.1816 +/- 0.0088 | 0.7842 +/- 0.0051 | 0.6847 +/- 0.0038 | 0.0000 +/- 0.0000 | 0.0426 +/- 0.0034 | 0.0000 +/- 0.0000 | 0.0001 +/- 0.0000 | 0.0931 +/- 0.0006 | 5 |
| B50 L1 structural-distance baseline | 15.77 +/- 0.49 | 15.79 +/- 0.49 | 15.78 +/- 0.49 | -0.3855 +/- 0.0106 | 0.8582 +/- 0.0130 | 0.3480 +/- 0.0177 | 0.3019 +/- 0.0161 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 5 |
| B51 L1 structural-distance + distance loss | 16.26 +/- 0.45 | 16.27 +/- 0.45 | 16.27 +/- 0.45 | 0.8197 +/- 0.0376 | 0.1778 +/- 0.0134 | 0.7992 +/- 0.0083 | 0.7014 +/- 0.0070 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 5 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
