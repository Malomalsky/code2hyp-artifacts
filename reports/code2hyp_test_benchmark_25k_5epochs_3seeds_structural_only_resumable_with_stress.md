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
| Code2Hyp B36 product-Frechet + neighbor | 16.01 +/- 0.59 | 16.03 +/- 0.59 | 16.02 +/- 0.59 | 0.6549 +/- 0.1428 | 0.2089 +/- 0.0353 | 0.8431 +/- 0.0479 | 0.8262 +/- 0.0558 | 0.0000 +/- 0.0000 | 3 |
| B39 matched code2vec-style baseline | 16.00 +/- 0.78 | 16.02 +/- 0.78 | 16.01 +/- 0.78 | -0.3350 +/- 0.0162 | 0.8111 +/- 0.0299 | 0.3507 +/- 0.0124 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B40 context-transform + Frechet | 16.44 +/- 0.51 | 16.45 +/- 0.51 | 16.45 +/- 0.51 | 0.7240 +/- 0.1514 | 0.2060 +/- 0.0507 | 0.7765 +/- 0.0990 | 0.9068 +/- 0.0642 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 16.32 +/- 0.80 | 16.34 +/- 0.80 | 16.33 +/- 0.80 | 0.9805 +/- 0.0009 | 0.0589 +/- 0.0015 | 0.9631 +/- 0.0020 | 0.9725 +/- 0.0252 | 0.1162 +/- 0.0051 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
