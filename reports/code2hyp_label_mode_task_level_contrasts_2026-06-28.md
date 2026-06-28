# Code2Hyp label-encoding paired task-level contrasts

Input diagnostic: max_paths=128; distance_mode=centroid_proxy.

| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta R@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |
|---|---|---:|---:|---:|---:|---:|
| bugnet_python | categorical - scalar hash | 32 | 384 | 0.0344 [0.0037, 0.0664] | 0.0469 [-0.0104, 0.1094] | 0.5966 |
| dta_zenodo | categorical - scalar hash | 11 | 264 | 0.0765 [0.0309, 0.1387] | 0.0492 [-0.0227, 0.1250] | 0.0010 |
| bugnet_python | categorical - no label | 32 | 384 | 0.0428 [0.0067, 0.0815] | 0.0521 [-0.0156, 0.1276] | 0.3771 |
| dta_zenodo | categorical - no label | 11 | 264 | 0.0383 [-0.0457, 0.1267] | -0.0833 [-0.2159, 0.0833] | 0.5488 |
| bugnet_python | scalar hash - no label | 32 | 384 | 0.0084 [-0.0072, 0.0244] | 0.0052 [-0.0260, 0.0365] | 0.0294 |
| dta_zenodo | scalar hash - no label | 11 | 264 | -0.0382 [-0.0839, 0.0056] | -0.1326 [-0.2652, 0.0341] | 0.5488 |

Interpretation: positive categorical-minus-scalar deltas mean that the diagnostic structural signal does not rely on the arbitrary scalar AST-label hash.
