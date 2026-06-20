# Code2Hyp Java-small benchmark summary

## Scope

This report separates external literature numbers from local Code2Hyp runs.
The external numbers are full Java-small literature baselines from code2seq Table 1.
The local numbers are controlled Code2Hyp runs with the training budget recorded below.

## Local run metadata

- evaluation split: `test`
- train records used: `10000`
- evaluation records loaded: `4096`
- evaluation records after known-target filtering: `3123`
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
| Code2Hyp B36 product-Frechet + neighbor | 19.69 +/- 0.20 | 19.71 +/- 0.20 | 19.70 +/- 0.20 | 0.5215 +/- 0.0356 | 0.2501 +/- 0.0124 | 0.7912 +/- 0.0286 | 0.9087 +/- 0.0416 | 0.0000 +/- 0.0000 | 3 |
| B39 matched code2vec-style baseline | 16.79 +/- 0.93 | 16.82 +/- 0.93 | 16.81 +/- 0.93 | -0.3142 +/- 0.0201 | 0.8421 +/- 0.0084 | 0.3825 +/- 0.0231 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B40 context-transform + Frechet | 17.67 +/- 0.36 | 17.69 +/- 0.36 | 17.68 +/- 0.36 | 0.5864 +/- 0.0439 | 0.2420 +/- 0.0178 | 0.7379 +/- 0.0646 | 0.9326 +/- 0.0267 | 0.0000 +/- 0.0000 | 3 |
| Code2Hyp B44 structural-bias attention | 16.67 +/- 0.69 | 16.70 +/- 0.69 | 16.68 +/- 0.69 | 0.9592 +/- 0.0075 | 0.0890 +/- 0.0080 | 0.9257 +/- 0.0091 | 0.6815 +/- 0.0052 | 0.0966 +/- 0.0003 | 3 |

## Interpretation boundary

Do not compare the local subset run as a direct SOTA claim against full-budget literature models.
Use it to validate the instrument and decide whether a full Java-small run is worth executing.
A manuscript-level claim requires fixed hyperparameters, `--eval-split test`, full or explicitly
bounded training budget, paired seed analysis, and no post-hoc model selection on the test split.
