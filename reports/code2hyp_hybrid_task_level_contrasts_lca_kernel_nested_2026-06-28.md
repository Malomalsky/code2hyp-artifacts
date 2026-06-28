# Code2Hyp hybrid paired task-level contrasts

| Dataset | Contrast | Tasks | Paired queries | Delta MRR, task-bootstrap 95% CI | Delta Recall@5, task-bootstrap 95% CI | Sign-test p for Delta MRR |
|---|---|---:|---:|---:|---:|---:|
| bugnet_python | Code2Hyp multiview selected - token bag | 32 | 384 | 0.0508 [0.0159, 0.0871] | 0.0365 [0.0052, 0.0651] | 0.0501 |
| dta_zenodo | Code2Hyp multiview selected - token bag | 11 | 264 | 0.0343 [-0.0029, 0.0765] | 0.0227 [-0.0038, 0.0568] | 0.5488 |
| bugnet_python | Code2Hyp multiview selected - AST node bag | 32 | 384 | 0.1184 [0.0651, 0.1729] | 0.1146 [0.0391, 0.1901] | 0.0005 |
| dta_zenodo | Code2Hyp multiview selected - AST node bag | 11 | 264 | 0.0623 [0.0309, 0.0943] | 0.0303 [0.0000, 0.0644] | 0.0215 |
| bugnet_python | Code2Hyp multiview selected - LCA path signature | 32 | 384 | 0.1178 [0.0672, 0.1683] | 0.1068 [0.0443, 0.1693] | 0.0070 |
| dta_zenodo | Code2Hyp multiview selected - LCA path signature | 11 | 264 | 0.0789 [0.0225, 0.1444] | 0.0341 [-0.0152, 0.0871] | 0.1094 |
| bugnet_python | Code2Hyp multiview selected - multiview without LCA path view | 32 | 384 | 0.0212 [-0.0079, 0.0515] | 0.0130 [-0.0260, 0.0495] | 0.0708 |
| dta_zenodo | Code2Hyp multiview selected - multiview without LCA path view | 11 | 264 | 0.0100 [-0.0017, 0.0242] | 0.0114 [-0.0076, 0.0341] | 0.4531 |
| bugnet_python | Code2Hyp multiview selected - token+AST selected | 32 | 384 | 0.0447 [0.0034, 0.0880] | 0.0495 [-0.0052, 0.1120] | 0.2153 |
| dta_zenodo | Code2Hyp multiview selected - token+AST selected | 11 | 264 | 0.0027 [-0.0216, 0.0340] | 0.0189 [-0.0152, 0.0530] | 1.0000 |
