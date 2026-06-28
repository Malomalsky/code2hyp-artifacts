# Code2Hyp hybrid task-level retrieval benchmark

The variants keep LCA-anchored AST path objects and add IDF-weighted lexical/path-signature features to the comparison kernel.

| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | code2hyp_multiview_no_lca_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_multiview_selected | 384 | 32 | 3 | 0.5101 | 0.3932 | 0.6250 | 6.5078 |
| bugnet_python | code2hyp_path_signature_kernel | 384 | 32 | 3 | 0.3112 | 0.2083 | 0.3932 | 11.2839 |
| bugnet_python | code2hyp_path_signature_plus_tokens | 384 | 32 | 3 | 0.4056 | 0.2760 | 0.5339 | 8.2083 |
| bugnet_python | code2hyp_path_signature_shape_only | 384 | 32 | 3 | 0.3064 | 0.1875 | 0.4115 | 11.0078 |
| bugnet_python | token_ast_selected | 384 | 32 | 3 | 0.4696 | 0.3464 | 0.5833 | 6.8047 |
| dta_zenodo | code2hyp_multiview_no_lca_selected | 264 | 11 | 3 | 0.7453 | 0.5985 | 0.9356 | 1.9811 |
| dta_zenodo | code2hyp_multiview_selected | 264 | 11 | 3 | 0.7461 | 0.6061 | 0.9242 | 2.0303 |
| dta_zenodo | code2hyp_path_signature_kernel | 264 | 11 | 3 | 0.5271 | 0.3523 | 0.7614 | 3.5189 |
| dta_zenodo | code2hyp_path_signature_plus_tokens | 264 | 11 | 3 | 0.6803 | 0.5152 | 0.9129 | 2.3333 |
| dta_zenodo | code2hyp_path_signature_shape_only | 264 | 11 | 3 | 0.5202 | 0.3409 | 0.8144 | 3.4280 |
| dta_zenodo | token_ast_selected | 264 | 11 | 3 | 0.7526 | 0.6174 | 0.9280 | 2.0341 |
