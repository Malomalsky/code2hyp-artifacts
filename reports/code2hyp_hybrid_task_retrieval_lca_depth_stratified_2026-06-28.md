# Code2Hyp hybrid task-level retrieval benchmark

The variants keep LCA-anchored AST path objects and add IDF-weighted lexical/path-signature features to the comparison kernel.

| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | code2hyp_multiview_no_lca_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_multiview_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_path_signature_kernel | 384 | 32 | 3 | 0.4525 | 0.3229 | 0.5599 | 6.9740 |
| bugnet_python | code2hyp_path_signature_plus_tokens | 384 | 32 | 3 | 0.3964 | 0.2656 | 0.5260 | 8.3594 |
| bugnet_python | code2hyp_path_signature_shape_only | 384 | 32 | 3 | 0.3986 | 0.2578 | 0.5365 | 7.8307 |
| bugnet_python | token_ast_selected | 384 | 32 | 3 | 0.4696 | 0.3464 | 0.5833 | 6.8047 |
| dta_zenodo | code2hyp_multiview_no_lca_selected | 264 | 11 | 3 | 0.7453 | 0.5985 | 0.9356 | 1.9811 |
| dta_zenodo | code2hyp_multiview_selected | 264 | 11 | 3 | 0.7443 | 0.6023 | 0.9242 | 2.0303 |
| dta_zenodo | code2hyp_path_signature_kernel | 264 | 11 | 3 | 0.6949 | 0.5720 | 0.8712 | 2.5985 |
| dta_zenodo | code2hyp_path_signature_plus_tokens | 264 | 11 | 3 | 0.6764 | 0.5076 | 0.9129 | 2.3561 |
| dta_zenodo | code2hyp_path_signature_shape_only | 264 | 11 | 3 | 0.6857 | 0.5530 | 0.8636 | 2.6402 |
| dta_zenodo | token_ast_selected | 264 | 11 | 3 | 0.7526 | 0.6174 | 0.9280 | 2.0341 |
