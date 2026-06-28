# Code2Hyp hybrid task-level retrieval benchmark

The variants keep LCA-anchored AST path objects and add IDF-weighted lexical/path-signature features to the comparison kernel.

| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | code2hyp_multiview_no_lca_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_multiview_selected | 384 | 32 | 3 | 0.5093 | 0.3906 | 0.6224 | 6.5104 |
| bugnet_python | code2hyp_path_signature_kernel | 384 | 32 | 3 | 0.3462 | 0.2318 | 0.4427 | 10.6146 |
| bugnet_python | code2hyp_path_signature_plus_tokens | 384 | 32 | 3 | 0.4019 | 0.2708 | 0.5339 | 8.2422 |
| bugnet_python | code2hyp_path_signature_shape_only | 384 | 32 | 3 | 0.3434 | 0.2161 | 0.4531 | 10.1849 |
| bugnet_python | token_ast_selected | 384 | 32 | 3 | 0.4696 | 0.3464 | 0.5833 | 6.8047 |
| dta_zenodo | code2hyp_multiview_no_lca_selected | 264 | 11 | 3 | 0.7453 | 0.5985 | 0.9356 | 1.9811 |
| dta_zenodo | code2hyp_multiview_selected | 264 | 11 | 3 | 0.7468 | 0.6061 | 0.9242 | 2.0265 |
| dta_zenodo | code2hyp_path_signature_kernel | 264 | 11 | 3 | 0.5274 | 0.3333 | 0.7841 | 3.4432 |
| dta_zenodo | code2hyp_path_signature_plus_tokens | 264 | 11 | 3 | 0.6796 | 0.5152 | 0.9129 | 2.3485 |
| dta_zenodo | code2hyp_path_signature_shape_only | 264 | 11 | 3 | 0.5455 | 0.3788 | 0.7879 | 3.3258 |
| dta_zenodo | token_ast_selected | 264 | 11 | 3 | 0.7526 | 0.6174 | 0.9280 | 2.0341 |
