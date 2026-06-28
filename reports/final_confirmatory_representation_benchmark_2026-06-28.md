# Final confirmatory representation benchmark

Inputs:
- `dta_zenodo`: `outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260625.json`
- `dta_zenodo`: `outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260626.json`
- `dta_zenodo`: `outputs/dta_zenodo_balanced64_representation_ablation_euclidean_p1_seed20260627.json`
- `bugnet_python`: `outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260625.json`
- `bugnet_python`: `outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260626.json`
- `bugnet_python`: `outputs/bugnet_python_32tasks_representation_ablation_euclidean_p1_seed20260627.json`

Bootstrap samples: `5000`.

## Cell summaries

| Dataset | Geometry | Cost | Path object | Aggregation | Queries | MRR, 95% bootstrap CI | Recall@1, 95% bootstrap CI | Recall@5, 95% bootstrap CI | Mean rank, 95% bootstrap CI |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| bugnet_python | E | train_weighted_combined_p1p00 | lca_product | centroid | 384 | 0.3811 [0.3390, 0.4219] | 0.2969 [0.2500, 0.3438] | 0.4193 [0.3698, 0.4688] | 11.1302 [10.1302, 12.1484] |
| bugnet_python | E | train_weighted_combined_p1p00 | lca_product | measure | 384 | 0.3890 [0.3501, 0.4304] | 0.2969 [0.2500, 0.3464] | 0.4375 [0.3906, 0.4870] | 10.5990 [9.5625, 11.6068] |
| bugnet_python | E | train_weighted_combined_p1p00 | single_point | centroid | 384 | 0.3659 [0.3254, 0.4069] | 0.2839 [0.2396, 0.3281] | 0.4193 [0.3698, 0.4688] | 11.4219 [10.3958, 12.4349] |
| bugnet_python | E | train_weighted_combined_p1p00 | single_point | measure | 384 | 0.3699 [0.3304, 0.4093] | 0.2682 [0.2240, 0.3151] | 0.4427 [0.3932, 0.4896] | 11.0755 [10.0729, 12.1250] |
| dta_zenodo | E | train_weighted_combined_p1p00 | lca_product | centroid | 264 | 0.4777 [0.4347, 0.5213] | 0.2879 [0.2311, 0.3447] | 0.7576 [0.7083, 0.8068] | 3.8788 [3.5114, 4.2386] |
| dta_zenodo | E | train_weighted_combined_p1p00 | lca_product | measure | 264 | 0.4798 [0.4344, 0.5237] | 0.3106 [0.2538, 0.3636] | 0.6970 [0.6402, 0.7500] | 4.1364 [3.7576, 4.5341] |
| dta_zenodo | E | train_weighted_combined_p1p00 | single_point | centroid | 264 | 0.4277 [0.3865, 0.4710] | 0.2462 [0.1932, 0.2992] | 0.6061 [0.5455, 0.6629] | 4.5417 [4.1515, 4.9318] |
| dta_zenodo | E | train_weighted_combined_p1p00 | single_point | measure | 264 | 0.4818 [0.4374, 0.5252] | 0.3106 [0.2538, 0.3674] | 0.6515 [0.5947, 0.7045] | 4.1629 [3.7803, 4.5682] |

## Paired query contrasts

| Dataset | Geometry | Cost | Contrast | Paired queries | Delta MRR, 95% bootstrap CI | Delta Recall@1, 95% bootstrap CI | Delta Recall@5, 95% bootstrap CI | Delta mean rank, 95% bootstrap CI |
|---|---|---|---|---:|---:|---:|---:|---:|
| bugnet_python | E | train_weighted_combined_p1p00 | LCA-product measure - single-point measure | 384 | 0.0191 [0.0026, 0.0357] | 0.0286 [0.0052, 0.0521] | -0.0052 [-0.0312, 0.0182] | -0.4766 [-0.9688, 0.0339] |
| bugnet_python | E | train_weighted_combined_p1p00 | LCA-product centroid - single-point centroid | 384 | 0.0153 [0.0014, 0.0303] | 0.0130 [-0.0026, 0.0312] | 0.0000 [-0.0339, 0.0312] | -0.2917 [-0.8464, 0.2812] |
| bugnet_python | E | train_weighted_combined_p1p00 | LCA-product measure - LCA-product centroid | 384 | 0.0078 [-0.0080, 0.0236] | 0.0000 [-0.0208, 0.0208] | 0.0182 [-0.0078, 0.0443] | -0.5312 [-0.9896, -0.0521] |
| bugnet_python | E | train_weighted_combined_p1p00 | single-point measure - single-point centroid | 384 | 0.0040 [-0.0116, 0.0201] | -0.0156 [-0.0417, 0.0104] | 0.0234 [-0.0026, 0.0495] | -0.3464 [-0.6745, -0.0156] |
| dta_zenodo | E | train_weighted_combined_p1p00 | LCA-product measure - single-point measure | 264 | -0.0020 [-0.0247, 0.0220] | 0.0000 [-0.0379, 0.0379] | 0.0455 [0.0076, 0.0833] | -0.0265 [-0.2235, 0.1742] |
| dta_zenodo | E | train_weighted_combined_p1p00 | LCA-product centroid - single-point centroid | 264 | 0.0500 [0.0230, 0.0781] | 0.0417 [0.0076, 0.0795] | 0.1515 [0.0909, 0.2121] | -0.6629 [-0.9848, -0.3258] |
| dta_zenodo | E | train_weighted_combined_p1p00 | LCA-product measure - LCA-product centroid | 264 | 0.0021 [-0.0319, 0.0357] | 0.0227 [-0.0227, 0.0682] | -0.0606 [-0.1136, -0.0076] | 0.2576 [-0.0189, 0.5341] |
| dta_zenodo | E | train_weighted_combined_p1p00 | single-point measure - single-point centroid | 264 | 0.0541 [0.0263, 0.0846] | 0.0644 [0.0152, 0.1136] | 0.0455 [0.0076, 0.0871] | -0.3788 [-0.5530, -0.2045] |

## Paired task-level contrasts

| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta Recall@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |
|---|---|---:|---:|---:|---:|---:|
| bugnet_python | LCA-product measure - single-point measure | 32 | 384 | 0.0191 [0.0005, 0.0379] | -0.0052 [-0.0260, 0.0156] | 0.0021 |
| bugnet_python | LCA-product centroid - single-point centroid | 32 | 384 | 0.0153 [-0.0032, 0.0380] | 0.0000 [-0.0339, 0.0339] | 0.2153 |
| bugnet_python | LCA-product measure - LCA-product centroid | 32 | 384 | 0.0078 [-0.0119, 0.0255] | 0.0182 [-0.0182, 0.0495] | 0.3771 |
| bugnet_python | single-point measure - single-point centroid | 32 | 384 | 0.0040 [-0.0151, 0.0217] | 0.0234 [-0.0026, 0.0495] | 0.7201 |
| dta_zenodo | LCA-product measure - single-point measure | 11 | 264 | -0.0020 [-0.0333, 0.0260] | 0.0455 [-0.0303, 0.1061] | 1.0000 |
| dta_zenodo | LCA-product centroid - single-point centroid | 11 | 264 | 0.0500 [-0.0027, 0.1034] | 0.1515 [-0.0038, 0.3220] | 0.2266 |
| dta_zenodo | LCA-product measure - LCA-product centroid | 11 | 264 | 0.0021 [-0.0766, 0.0825] | -0.0606 [-0.2008, 0.0682] | 1.0000 |
| dta_zenodo | single-point measure - single-point centroid | 11 | 264 | 0.0541 [0.0147, 0.0931] | 0.0455 [-0.0265, 0.1364] | 0.3438 |

## Interpretation

The retrieval outcome is first computed per query. Because queries from the same task are dependent, the summary also reports task-level paired contrasts with deterministic non-parametric bootstrap over tasks and exact sign tests over task-level deltas.
The paired contrasts use only matched query identifiers within the same dataset, geometry and cost mode. This avoids mixing different train/query/gallery splits when comparing path-object representations.
bugnet_python / E / LCA-product measure - single-point measure: improves MRR by +0.0191 with 95% bootstrap CI [+0.0026, +0.0357] over 384 paired queries.
bugnet_python / E / LCA-product centroid - single-point centroid: improves MRR by +0.0153 with 95% bootstrap CI [+0.0014, +0.0303] over 384 paired queries.
bugnet_python / E / LCA-product measure - LCA-product centroid: improves MRR by +0.0078 with 95% bootstrap CI [-0.0080, +0.0236] over 384 paired queries.
bugnet_python / E / single-point measure - single-point centroid: improves MRR by +0.0040 with 95% bootstrap CI [-0.0116, +0.0201] over 384 paired queries.
dta_zenodo / E / LCA-product measure - single-point measure: changes MRR by -0.0020 with 95% bootstrap CI [-0.0247, +0.0220] over 264 paired queries.
dta_zenodo / E / LCA-product centroid - single-point centroid: improves MRR by +0.0500 with 95% bootstrap CI [+0.0230, +0.0781] over 264 paired queries.
dta_zenodo / E / LCA-product measure - LCA-product centroid: improves MRR by +0.0021 with 95% bootstrap CI [-0.0319, +0.0357] over 264 paired queries.
dta_zenodo / E / single-point measure - single-point centroid: improves MRR by +0.0541 with 95% bootstrap CI [+0.0263, +0.0846] over 264 paired queries.
