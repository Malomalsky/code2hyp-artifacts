# AST-label path geometry audit

This audit describes the structural object available in the official
preprocessed code2seq Java-small corpus: serialized terminal-to-terminal
AST-label paths. It does not claim access to original AST node IDs.

## Configuration

- data_root: `data/code2seq_java_small/java-small`
- splits: `['train', 'val']`
- record_limit_per_split: `200`
- sample_seed: `777`
- max_contexts: `30`
- max_path_length: `8`
- tie_k: `3`
- hyperbolicity_samples: `100`

## Split-level summary

| Split | Contexts | Unique full paths | Unique truncated paths | Ambiguous truncated classes | Contexts in ambiguous classes | Within-record zero-distance pairs | Mean tie-set size@k | Mean four-point delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4387 | 1789 | 1622 | 114 | 0.0900388 | 0.0276834 | 4.54965 | 0 |
| val | 4738 | 1722 | 1641 | 67 | 0.0519206 | 0.0171184 | 4.28599 | 0 |

## Combined summary

- contexts: `9125`
- records_with_at_least_one_context: `400`
- unique_full_paths: `3166`
- unique_truncated_paths: `2923`
- full_path_collision_rate_pairwise: `0.00316941`
- truncated_path_collision_rate_pairwise: `0.00324321`
- ambiguous_truncated_class_rate: `0.0602121`
- contexts_in_ambiguous_truncated_classes_rate: `0.0723288`
- within_record_zero_distance_pair_rate: `0.0221973`
- within_record_ambiguous_zero_pair_rate: `0.00445294`
- prefix_distance_unique_levels_global: `16`
- prefix_distance_entropy_bits: `3.88447`
- metric_unique_levels_global: `{'prefix_tree': 16, 'edit': 9, 'jaccard_bigrams': 33}`
- metric_entropy_bits: `{'prefix_tree': 3.884470980550963, 'edit': 3.009685111831524, 'jaccard_bigrams': 3.9411033439752656}`
- cross_metric_spearman: `{'prefix_tree_vs_edit': 0.49615700934254925, 'prefix_tree_vs_jaccard_bigrams': 0.26787215407204834, 'edit_vs_jaccard_bigrams': 0.8260074162001955}`
- mean_unique_distance_levels_per_record: `11.9975`
- mean_tie_set_size_at_k: `4.41265`
- p95_tie_set_size_at_k: `9`
- mean_tie_expansion_ratio_at_k: `1.47088`
- four_point_hyperbolicity_by_metric: `{'prefix_tree': {'samples': 100, 'delta_mean': 0.0, 'delta_max': 0.0, 'delta_over_diameter_mean': 0.0, 'delta_over_diameter_max': 0.0}, 'edit': {'samples': 100, 'delta_mean': 0.415, 'delta_max': 1.5, 'delta_over_diameter_mean': 0.051875, 'delta_over_diameter_max': 0.1875}, 'jaccard_bigrams': {'samples': 100, 'delta_mean': 0.020406080031080027, 'delta_max': 0.11111111111111116, 'delta_over_diameter_mean': 0.020406080031080027, 'delta_over_diameter_max': 0.11111111111111116}}`
- four_point_delta_mean: `0`
- four_point_delta_max: `0`
- four_point_delta_over_diameter_max: `0`

## Most frequent within-record prefix distances

| Distance | Pair count | Share |
|---:|---:|---:|
| 11 | 12082 | 0.101702 |
| 12 | 11832 | 0.0995976 |
| 10 | 10561 | 0.0888988 |
| 9 | 9379 | 0.0789491 |
| 14 | 8691 | 0.0731578 |
| 16 | 8689 | 0.073141 |
| 7 | 8570 | 0.0721393 |
| 13 | 8274 | 0.0696476 |
| 15 | 7534 | 0.0634186 |
| 8 | 6704 | 0.0564319 |
| 2 | 6588 | 0.0554555 |
| 6 | 6562 | 0.0552366 |

## Cross-metric geometry

| Metric | Unique levels | Entropy, bits | Mean four-point delta | Max normalized delta |
|---|---:|---:|---:|---:|
| prefix_tree | 16 | 3.88447 | 0 | 0 |
| edit | 9 | 3.00969 | 0.415 | 0.1875 |
| jaccard_bigrams | 33 | 3.9411 | 0.0204061 | 0.111111 |

### Spearman correlation between path metrics

| Metric pair | Spearman rho |
|---|---:|
| prefix_tree_vs_edit | 0.496157 |
| prefix_tree_vs_jaccard_bigrams | 0.267872 |
| edit_vs_jaccard_bigrams | 0.826007 |

## Interpretation

The current structural target is best described as `truncated AST-label
prefix-trie distance`. The sampled four-point hyperbolicity is zero,
as expected for a tree metric on the prefix trie. This supports the
mathematical interpretation of B44 as preserving prefix-trie geometry,
not necessarily the full original AST path geometry.

The cross-metric block checks whether the chosen structural proxy
is interchangeable with edit and n-gram-overlap relations. Low
cross-metric correlations or different hyperbolicity profiles mean
that model claims must name the preserved relation explicitly.

If ambiguous truncated classes or within-record zero-distance pairs are
non-negligible, the metric must be treated as a proxy or pseudometric
over the original AST paths. A stronger follow-up needs node IDs and
full untruncated paths to evaluate endpoint, edge-overlap, Hausdorff,
and edit-distance metrics independently.
