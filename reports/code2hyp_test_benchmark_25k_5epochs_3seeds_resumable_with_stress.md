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
| Code2Hyp B36 product-Frechet + neighbor | 18.71 +/- 0.11 | 18.73 +/- 0.11 | 18.72 +/- 0.11 | 0.7198 +/- 0.1210 | 0.1949 +/- 0.0279 | 0.8626 +/- 0.0365 | 0.8475 +/- 0.0161 | 0.0000 +/- 0.0000 | 3 |
| B39 matched code2vec-style baseline | 15.79 +/- 0.69 | 15.80 +/- 0.70 | 15.80 +/- 0.70 | -0.3393 +/- 0.0097 | 0.8222 +/- 0.0230 | 0.3397 +/- 0.0141 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B40 context-transform + Frechet | 17.56 +/- 1.28 | 17.57 +/- 1.28 | 17.57 +/- 1.28 | 0.6508 +/- 0.0296 | 0.2220 +/- 0.0050 | 0.7840 +/- 0.0137 | 0.9099 +/- 0.0327 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 16.56 +/- 1.01 | 16.58 +/- 1.01 | 16.57 +/- 1.01 | 0.9776 +/- 0.0026 | 0.0624 +/- 0.0035 | 0.9602 +/- 0.0042 | 0.9399 +/- 0.0974 | 0.1119 +/- 0.0062 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
