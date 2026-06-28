# Code2Hyp-Structural label-encoding sensitivity

This diagnostic evaluates whether the deterministic structural layer depends on the arbitrary scalar AST-label hash.
Configuration: max_paths=128; distance_mode=centroid_proxy; sinkhorn_iterations=128.

| Dataset | Label mode | Queries | Tasks | MRR | R@1 | R@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|
| bugnet_python | categorical | 384 | 32 | 0.3250 | 0.1979 | 0.4193 | 10.8620 |
| bugnet_python | none | 384 | 32 | 0.2823 | 0.1589 | 0.3672 | 10.6380 |
| bugnet_python | scalar_hash | 384 | 32 | 0.2906 | 0.1719 | 0.3724 | 11.0156 |
| dta_zenodo | categorical | 264 | 11 | 0.5100 | 0.3598 | 0.6591 | 3.9697 |
| dta_zenodo | none | 264 | 11 | 0.4717 | 0.2652 | 0.7424 | 3.9167 |
| dta_zenodo | scalar_hash | 264 | 11 | 0.4335 | 0.2576 | 0.6098 | 4.5417 |

Interpretation rule: if categorical labels and no-label controls preserve the qualitative ordering, the LCA-path object claim is not merely an artifact of scalar label hashing. If the scalar mode is uniquely strong, the manuscript should treat the label hash as a limitation rather than as part of the main method.
