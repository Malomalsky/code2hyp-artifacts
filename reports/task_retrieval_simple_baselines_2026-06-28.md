# Task-level retrieval baseline benchmark

| Dataset | Baseline | Queries | Tasks | Seeds | MRR | Recall@1 | Recall@5 | Mean rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bugnet_python | ast_node_bag | 384 | 32 | 3 | 0.3959 | 0.2682 | 0.5182 | 8.1302 |
| bugnet_python | ast_path_bigram_bag | 384 | 32 | 3 | 0.3730 | 0.2578 | 0.4688 | 10.0547 |
| bugnet_python | random_expected | 384 | 32 | 3 | 0.1268 | 0.0312 | 0.1562 | 16.5000 |
| bugnet_python | token_bag | 384 | 32 | 3 | 0.4635 | 0.3281 | 0.5964 | 7.1823 |
| bugnet_python | token_shape_bag | 384 | 32 | 3 | 0.4200 | 0.2995 | 0.5365 | 7.9740 |
| dta_zenodo | ast_node_bag | 264 | 11 | 3 | 0.6930 | 0.5265 | 0.9167 | 2.2841 |
| dta_zenodo | ast_path_bigram_bag | 264 | 11 | 3 | 0.5903 | 0.4205 | 0.8371 | 2.9015 |
| dta_zenodo | random_expected | 264 | 11 | 3 | 0.2745 | 0.0909 | 0.4545 | 6.0000 |
| dta_zenodo | token_bag | 264 | 11 | 3 | 0.7210 | 0.5871 | 0.9242 | 2.1856 |
| dta_zenodo | token_shape_bag | 264 | 11 | 3 | 0.6586 | 0.5227 | 0.8523 | 2.8333 |
