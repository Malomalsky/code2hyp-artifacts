# Code2Hyp Java-small benchmark summary

## Scope

This report separates external literature numbers from local Code2Hyp runs.
The external numbers are full Java-small literature baselines from code2seq Table 1.
The local numbers are controlled Code2Hyp runs with the training budget recorded below.

## Local run metadata

- evaluation split: `val`
- train records used: `5000`
- evaluation records loaded: `2048`
- evaluation records after known-target filtering: `1083`
- epochs: `3`
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
| B60 branch-sequence product manifold | 7.97 +/- 8.28 | 7.97 +/- 8.29 | 7.97 +/- 8.28 | 0.9485 +/- 0.0016 | 0.3963 +/- 0.0087 | 0.1295 +/- 0.0049 | 0.0986 +/- 0.0027 | 0.3719 +/- 0.0024 | 0.3515 +/- 0.0009 | 0.8735 +/- 0.0146 | 0.7552 +/- 0.0191 | 0.0001 +/- 0.0000 | 0.9960 +/- 0.0005 | 0.4919 +/- 0.0099 | 1.2076 +/- 0.0241 | 0.1285 +/- 0.0117 | 3 |
| B62 branch-sequence product manifold + multi-metric loss | 7.87 +/- 8.36 | 7.87 +/- 8.36 | 7.87 +/- 8.36 | 0.5566 +/- 0.0138 | 0.8986 +/- 0.0061 | 0.7064 +/- 0.0126 | 0.2437 +/- 0.0024 | 0.2161 +/- 0.0009 | 0.1866 +/- 0.0075 | 0.7841 +/- 0.0064 | 0.6935 +/- 0.0076 | 0.0002 +/- 0.0001 | 0.9869 +/- 0.0109 | 0.7648 +/- 0.3106 | 1.1406 +/- 0.0415 | 0.1290 +/- 0.0160 | 3 |
| B63 product-bias manifold + multi-metric loss | 9.92 +/- 6.93 | 9.93 +/- 6.93 | 9.93 +/- 6.93 | 0.3604 +/- 0.0726 | 0.7635 +/- 0.0404 | 0.6095 +/- 0.0082 | 0.2880 +/- 0.0142 | 0.2781 +/- 0.0063 | 0.2117 +/- 0.0058 | 0.6923 +/- 0.0162 | 0.5984 +/- 0.0144 | 0.0024 +/- 0.0006 | 0.9998 +/- 0.0001 | 0.9973 +/- 0.0044 | 0.7699 +/- 0.0077 | 0.1030 +/- 0.0023 | 3 |
| B64 Euclidean context-transform + multi-metric loss | 8.09 +/- 7.72 | 8.09 +/- 7.72 | 8.09 +/- 7.72 | -0.1343 +/- 0.0499 | 0.5717 +/- 0.0446 | 0.5666 +/- 0.0203 | 0.5374 +/- 0.0460 | 0.3742 +/- 0.0461 | 0.3551 +/- 0.0508 | 0.3497 +/- 0.0176 | 0.3036 +/- 0.0155 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| B65 L1 context-transform + multi-metric loss | 9.96 +/- 7.25 | 9.96 +/- 7.26 | 9.96 +/- 7.25 | -0.1237 +/- 0.0270 | 0.6018 +/- 0.0139 | 0.6296 +/- 0.0169 | 0.5695 +/- 0.0569 | 0.3942 +/- 0.0503 | 0.3856 +/- 0.0656 | 0.3925 +/- 0.0339 | 0.3447 +/- 0.0297 | n/a | n/a | n/a | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
