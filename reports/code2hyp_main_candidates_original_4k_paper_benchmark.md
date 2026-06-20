# Code2Hyp Java-small benchmark summary

## Scope

This report separates external literature numbers from local Code2Hyp runs.
The external numbers are full Java-small literature baselines from code2seq Table 1.
The local numbers are controlled Code2Hyp runs with the training budget recorded below.

## Local run metadata

- evaluation split: `unknown`
- train records used: `4000`
- evaluation records loaded: `1024`
- evaluation records after known-target filtering: `637`
- epochs: `3`
- batch size: `64`
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

| Variant | Precision | Recall | F1 | Structural Spearman | Overlap@3 | Curvature | rho | n seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B35_code2hyp_product_frechet_adaptive | 14.95 +/- 1.07 | 14.96 +/- 1.07 | 14.95 +/- 1.07 | 0.3191 +/- 0.3288 | 0.5841 +/- 0.1746 | 0.9297 +/- 0.0296 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B36 product-Frechet + neighbor | 15.16 +/- 0.05 | 15.17 +/- 0.05 | 15.16 +/- 0.05 | 0.5856 +/- 0.0709 | 0.8043 +/- 0.0094 | 0.9283 +/- 0.0123 | 0.0000 +/- 0.0000 | 3 |
| B39 matched code2vec-style baseline | 12.62 +/- 1.47 | 12.63 +/- 1.47 | 12.63 +/- 1.47 | -0.3040 +/- 0.0145 | 0.4219 +/- 0.0308 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B40 context-transform + Frechet | 13.74 +/- 0.69 | 13.75 +/- 0.69 | 13.74 +/- 0.69 | 0.4036 +/- 0.2643 | 0.5704 +/- 0.1962 | 0.9171 +/- 0.0207 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 12.68 +/- 1.46 | 12.69 +/- 1.46 | 12.69 +/- 1.46 | 0.9014 +/- 0.0275 | 0.8711 +/- 0.0112 | 0.8181 +/- 0.0259 | 0.0987 +/- 0.0017 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
