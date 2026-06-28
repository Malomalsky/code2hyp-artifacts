# Code2Hyp hybrid paired task-level contrasts

| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta Recall@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |
|---|---|---:|---:|---:|---:|---:|
| bugnet_python | Code2Hyp multiview selected - token bag | 32 | 384 | 0.0654 [0.0321, 0.1009] | 0.0469 [0.0104, 0.0859] | 0.0001 |
| dta_zenodo | Code2Hyp multiview selected - token bag | 11 | 264 | 0.0329 [-0.0044, 0.0768] | -0.0038 [-0.0455, 0.0303] | 0.2266 |
| bugnet_python | Code2Hyp multiview selected - AST node bag | 32 | 384 | 0.1330 [0.0812, 0.1866] | 0.1250 [0.0573, 0.1927] | 0.0005 |
| dta_zenodo | Code2Hyp multiview selected - AST node bag | 11 | 264 | 0.0609 [0.0346, 0.0870] | 0.0038 [-0.0265, 0.0341] | 0.0654 |
| bugnet_python | Code2Hyp multiview selected - LCA path signature | 32 | 384 | 0.1324 [0.0819, 0.1844] | 0.1172 [0.0547, 0.1771] | 0.0070 |
| dta_zenodo | Code2Hyp multiview selected - LCA path signature | 11 | 264 | 0.0775 [0.0038, 0.1521] | 0.0076 [-0.0379, 0.0606] | 0.0117 |
| bugnet_python | Code2Hyp multiview selected - multiview without LCA path view | 32 | 384 | 0.0358 [0.0132, 0.0595] | 0.0234 [-0.0052, 0.0547] | 0.0201 |
| dta_zenodo | Code2Hyp multiview selected - multiview without LCA path view | 11 | 264 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bugnet_python | Code2Hyp multiview selected - token+AST selected | 32 | 384 | 0.0593 [0.0206, 0.0999] | 0.0599 [0.0130, 0.1120] | 0.2153 |
| dta_zenodo | Code2Hyp multiview selected - token+AST selected | 11 | 264 | 0.0013 [-0.0054, 0.0081] | -0.0076 [-0.0265, 0.0076] | 1.0000 |
