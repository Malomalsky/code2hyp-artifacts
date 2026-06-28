# Code2Hyp hybrid task-level retrieval benchmark

The variants keep LCA-anchored AST path objects and add IDF-weighted lexical/path-signature features to the comparison kernel.

| Dataset | Variant | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | code2hyp_multiview_no_lca_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_multiview_selected | 384 | 32 | 3 | 0.4931 | 0.3646 | 0.6198 | 6.5990 |
| bugnet_python | code2hyp_path_signature_kernel | 384 | 32 | 3 | 0.3811 | 0.2552 | 0.5104 | 9.1771 |
| bugnet_python | code2hyp_path_signature_plus_tokens | 384 | 32 | 3 | 0.3974 | 0.2656 | 0.5260 | 8.3438 |
| bugnet_python | code2hyp_path_signature_shape_only | 384 | 32 | 3 | 0.3597 | 0.2266 | 0.4948 | 9.2734 |
| bugnet_python | token_ast_selected | 384 | 32 | 3 | 0.4696 | 0.3464 | 0.5833 | 6.8047 |
| dta_zenodo | code2hyp_multiview_no_lca_selected | 264 | 11 | 3 | 0.7453 | 0.5985 | 0.9356 | 1.9811 |
| dta_zenodo | code2hyp_multiview_selected | 264 | 11 | 3 | 0.7443 | 0.6023 | 0.9242 | 2.0303 |
| dta_zenodo | code2hyp_path_signature_kernel | 264 | 11 | 3 | 0.6328 | 0.4621 | 0.8939 | 2.7083 |
| dta_zenodo | code2hyp_path_signature_plus_tokens | 264 | 11 | 3 | 0.6761 | 0.5076 | 0.9129 | 2.3598 |
| dta_zenodo | code2hyp_path_signature_shape_only | 264 | 11 | 3 | 0.6699 | 0.5076 | 0.9167 | 2.4962 |
| dta_zenodo | token_ast_selected | 264 | 11 | 3 | 0.7526 | 0.6174 | 0.9280 | 2.0341 |
