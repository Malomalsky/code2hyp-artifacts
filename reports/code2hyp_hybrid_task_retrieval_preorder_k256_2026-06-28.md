# Code2Hyp hybrid task-level retrieval benchmark

The variants keep LCA-anchored AST path objects and add IDF-weighted lexical/path-signature features to the comparison kernel.

| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | code2hyp_multiview_no_lca_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_multiview_selected | 384 | 32 | 3 | 0.5017 | 0.3802 | 0.6198 | 6.5156 |
| bugnet_python | code2hyp_path_signature_kernel | 384 | 32 | 3 | 0.3692 | 0.2552 | 0.4688 | 10.1432 |
| bugnet_python | code2hyp_path_signature_plus_tokens | 384 | 32 | 3 | 0.3996 | 0.2656 | 0.5339 | 8.2839 |
| bugnet_python | code2hyp_path_signature_shape_only | 384 | 32 | 3 | 0.3265 | 0.2005 | 0.4505 | 10.3255 |
| bugnet_python | token_ast_selected | 384 | 32 | 3 | 0.4696 | 0.3464 | 0.5833 | 6.8047 |
| dta_zenodo | code2hyp_multiview_no_lca_selected | 264 | 11 | 3 | 0.7453 | 0.5985 | 0.9356 | 1.9811 |
| dta_zenodo | code2hyp_multiview_selected | 264 | 11 | 3 | 0.7461 | 0.6061 | 0.9242 | 2.0303 |
| dta_zenodo | code2hyp_path_signature_kernel | 264 | 11 | 3 | 0.5694 | 0.3902 | 0.8068 | 3.1629 |
| dta_zenodo | code2hyp_path_signature_plus_tokens | 264 | 11 | 3 | 0.6785 | 0.5114 | 0.9167 | 2.3447 |
| dta_zenodo | code2hyp_path_signature_shape_only | 264 | 11 | 3 | 0.5558 | 0.3674 | 0.8447 | 3.0152 |
| dta_zenodo | token_ast_selected | 264 | 11 | 3 | 0.7526 | 0.6174 | 0.9280 | 2.0341 |
