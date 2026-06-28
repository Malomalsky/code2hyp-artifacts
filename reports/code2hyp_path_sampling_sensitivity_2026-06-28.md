# Code2Hyp AST path-sampling sensitivity

This diagnostic checks whether the LCA-path view in the final multiview kernel depends on the terminal-pair sampling policy.

| Run | Dataset | Policy | LCA view | K | C2H-MV MRR | MV-noLCA MRR | Tok+AST MRR | C2H-MV - noLCA | Mean LCA weight |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| preorder_k64 | bugnet_python | preorder_first | path_signature_plus_tokens | 64 | 0.5101 | 0.4931 | 0.4696 | +0.0170 | 0.0667 |
| preorder_k64 | dta_zenodo | preorder_first | path_signature_plus_tokens | 64 | 0.7461 | 0.7453 | 0.7526 | +0.0008 | 0.1000 |
| preorder_k128 | bugnet_python | preorder_first | path_signature_plus_tokens | 128 | 0.5093 | 0.4931 | 0.4696 | +0.0162 | 0.0667 |
| preorder_k128 | dta_zenodo | preorder_first | path_signature_plus_tokens | 128 | 0.7468 | 0.7453 | 0.7526 | +0.0015 | 0.1000 |
| preorder_k256 | bugnet_python | preorder_first | path_signature_plus_tokens | 256 | 0.5017 | 0.4931 | 0.4696 | +0.0086 | 0.0333 |
| preorder_k256 | dta_zenodo | preorder_first | path_signature_plus_tokens | 256 | 0.7461 | 0.7453 | 0.7526 | +0.0008 | 0.1000 |
| hash_k128 | bugnet_python | hash_sorted | path_signature_plus_tokens | 128 | 0.4931 | 0.4931 | 0.4696 | +0.0000 | 0.0000 |
| hash_k128 | dta_zenodo | hash_sorted | path_signature_plus_tokens | 128 | 0.7443 | 0.7453 | 0.7526 | -0.0010 | 0.1000 |
| lca_strat_k128 | bugnet_python | lca_depth_stratified | path_signature_plus_tokens | 128 | 0.4931 | 0.4931 | 0.4696 | +0.0000 | 0.0000 |
| lca_strat_k128 | dta_zenodo | lca_depth_stratified | path_signature_plus_tokens | 128 | 0.7443 | 0.7453 | 0.7526 | -0.0010 | 0.1000 |

Interpretation: the earlier lexicalized LCA view is sensitive to terminal-pair sampling. This diagnostic motivates the final protocol, which separates the clean LCA-path signature from raw token evidence, uses LCA-depth-stratified sampling, and evaluates the LCA view inside a nested train-selected multiview grid.
