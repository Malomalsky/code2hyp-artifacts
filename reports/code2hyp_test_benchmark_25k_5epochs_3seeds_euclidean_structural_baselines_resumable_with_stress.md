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
| B14_bounded_euclidean_metric_code2vec | 16.89 +/- 0.16 | 16.91 +/- 0.16 | 16.90 +/- 0.16 | 0.1427 +/- 0.0021 | 0.5956 +/- 0.0052 | 0.2974 +/- 0.0018 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| B6_euclidean_metric_code2vec | 20.13 +/- 0.10 | 20.15 +/- 0.10 | 20.14 +/- 0.10 | 0.1678 +/- 0.0028 | 0.4344 +/- 0.0012 | 0.4294 +/- 0.0053 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| B_tree_euclidean_lca_bias | 20.21 +/- 0.06 | 20.24 +/- 0.06 | 20.23 +/- 0.06 | 0.1562 +/- 0.0008 | 0.4403 +/- 0.0022 | 0.4402 +/- 0.0055 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
