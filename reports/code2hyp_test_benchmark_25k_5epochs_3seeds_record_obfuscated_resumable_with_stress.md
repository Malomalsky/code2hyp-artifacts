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

| Variant | Precision | Recall | F1 | Structural Spearman | Normalized stress | Overlap@3 | Curvature | rho | n seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code2Hyp B36 product-Frechet + neighbor | 14.46 +/- 0.45 | 14.47 +/- 0.45 | 14.47 +/- 0.45 | 0.5222 +/- 0.0506 | 0.2530 +/- 0.0332 | 0.7804 +/- 0.0257 | 0.9005 +/- 0.0106 | 0.0000 +/- 0.0000 | 3 |
| B39 matched code2vec-style baseline | 15.08 +/- 1.58 | 15.09 +/- 1.58 | 15.09 +/- 1.58 | -0.3472 +/- 0.0031 | 0.8189 +/- 0.0052 | 0.3521 +/- 0.0155 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B40 context-transform + Frechet | 15.00 +/- 0.12 | 15.02 +/- 0.12 | 15.01 +/- 0.12 | 0.7613 +/- 0.0480 | 0.2034 +/- 0.0137 | 0.7773 +/- 0.0139 | 0.8836 +/- 0.0388 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 15.90 +/- 0.45 | 15.91 +/- 0.45 | 15.91 +/- 0.45 | 0.9792 +/- 0.0007 | 0.0613 +/- 0.0010 | 0.9583 +/- 0.0033 | 0.9228 +/- 0.0445 | 0.1121 +/- 0.0008 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
