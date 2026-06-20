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
- seeds: `[404, 505]`
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
| Code2Hyp B36 product-Frechet + neighbor | 18.72 +/- 0.19 | 18.73 +/- 0.19 | 18.73 +/- 0.19 | 0.6794 +/- 0.1374 | 0.2065 +/- 0.0283 | 0.8397 +/- 0.0268 | 0.8740 +/- 0.0636 | 0.0000 +/- 0.0000 | 2 |
| B39 matched code2vec-style baseline | 15.74 +/- 0.09 | 15.76 +/- 0.09 | 15.75 +/- 0.09 | -0.3504 +/- 0.0115 | 0.8139 +/- 0.0159 | 0.3266 +/- 0.0227 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 2 |
| Code2Hyp B44 structural-bias attention | 16.90 +/- 0.85 | 16.92 +/- 0.85 | 16.91 +/- 0.85 | 0.9781 +/- 0.0017 | 0.0617 +/- 0.0024 | 0.9594 +/- 0.0034 | 0.9591 +/- 0.0661 | 0.0985 +/- 0.0010 | 2 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
