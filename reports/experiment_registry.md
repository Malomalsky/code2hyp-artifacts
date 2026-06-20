# Experiment Registry

This file is generated from `outputs/*.json` by `scripts/build_experiment_artifacts.py`.

## Full Experiment Inventory

| experiment_file | experiment_type | role | article_use | records | baseline_kind | markov_weight | geometry_weight | feature_set | sample_seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dta_ast_pilot_limit3.json | ast_geometry_pilot | smoke_test | pipeline sanity check | 33 |  |  |  |  |  |
| dta_confirmatory_feature_sweep_transition_count_v20_t50_seed101.json | confirmatory_feature_sweep | confirmatory | feature-set control | 550 | transition_count |  |  |  | 101 |
| dta_confirmatory_feature_sweep_transition_count_v20_t50_seed202.json | confirmatory_feature_sweep | confirmatory | feature-set control | 550 | transition_count |  |  |  | 202 |
| dta_confirmatory_feature_sweep_transition_count_v20_t50_seed303.json | confirmatory_feature_sweep | confirmatory | feature-set control | 550 | transition_count |  |  |  | 303 |
| dta_confirmatory_residual_sweep_transition_count_v20_t50_seed101.json | confirmatory_residual_sweep | confirmatory | residual feature control | 550 | transition_count |  |  |  | 101 |
| dta_confirmatory_residual_sweep_transition_count_v20_t50_seed202.json | confirmatory_residual_sweep | confirmatory | residual feature control | 550 | transition_count |  |  |  | 202 |
| dta_confirmatory_residual_sweep_transition_count_v20_t50_seed303.json | confirmatory_residual_sweep | confirmatory | residual feature control | 550 | transition_count |  |  |  | 303 |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | confirmatory_split | confirmatory | primary evidence | 550 | transition_count | 0.9000 | 0.1000 | all | 101 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | confirmatory_split | confirmatory | primary evidence | 550 | transition_count | 0.9000 | 0.1000 | length_only | 101 |
| dta_feature_ablation_limit20_seed13_w085.json | method_comparison | exploratory | ablation / method design | 220 | flat_markov_jsd | 0.8500 | 0.1500 |  | 13 |
| dta_feature_ablation_limit50_seed13_w085.json | method_comparison | exploratory | ablation / method design | 550 | flat_markov_jsd | 0.8500 | 0.1500 |  | 13 |
| dta_feature_ablation_transition_count_limit50_seed13_w085.json | method_comparison | exploratory | ablation / method design | 550 | transition_count | 0.8500 | 0.1500 |  | 13 |
| dta_feature_ablation_transition_count_limit50_seed13_w090.json | method_comparison | exploratory | ablation / method design | 550 | transition_count | 0.9000 | 0.1000 |  | 13 |
| dta_feature_ablation_transition_count_limit50_seed13_w090_length_control.json | method_comparison | exploratory | ablation / method design | 550 | transition_count | 0.9000 | 0.1000 |  | 13 |
| dta_markov_baselines_limit50_seed13.json | markov_baseline_audit | exploratory | baseline selection | 550 | multiple |  |  |  | 13 |
| dta_multiseed_ablation_limit20_w085.json | multiseed_ablation | exploratory | robustness check | 660 | flat_markov_jsd | 0.8500 | 0.1500 |  |  |
| dta_multiseed_ablation_transition_count_limit20_w090.json | multiseed_ablation | exploratory | robustness check | 660 | transition_count | 0.9000 | 0.1000 |  |  |
| dta_retrieval_limit10_w08.json | method_comparison | exploratory | ablation / method design | 110 | flat_markov_jsd | 0.8000 | 0.2000 |  |  |
| dta_retrieval_limit20_w08.json | method_comparison | exploratory | ablation / method design | 220 | flat_markov_jsd | 0.8000 | 0.2000 |  |  |
| dta_weight_sweep_limit10.json | weight_sweep | exploratory | hyperparameter selection | 110 | flat_markov_jsd |  |  |  |  |
| dta_weight_sweep_limit20.json | weight_sweep | exploratory | hyperparameter selection | 220 | flat_markov_jsd |  |  |  |  |
| dta_weight_sweep_limit20_seed13.json | weight_sweep | exploratory | hyperparameter selection | 220 | flat_markov_jsd |  |  |  | 13 |
| dta_weight_sweep_transition_count_limit50_seed13.json | weight_sweep | exploratory | hyperparameter selection | 550 | transition_count |  |  |  | 13 |
| file_tree_pilot_sample.json | file_tree_smoke_test | smoke_test | auxiliary sanity check |  |  |  |  |  |  |

## Confirmatory Result

| experiment_file | method | top1_accuracy | mrr | map | recall@10 |
| --- | --- | --- | --- | --- | --- |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | baseline | 0.9855 | 0.9902 | 0.7463 | 0.1887 |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | candidate | 0.9891 | 0.9924 | 0.7540 | 0.1904 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | baseline | 0.9855 | 0.9902 | 0.7463 | 0.1887 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | candidate | 0.9891 | 0.9928 | 0.7743 | 0.1916 |

## Confirmatory Paired Deltas

| experiment_file | feature_set | metric | mean_delta | ci95_low | ci95_high | p_one_sided |
| --- | --- | --- | --- | --- | --- | --- |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | all | top1_accuracy | 0.0036 | 0.0000 | 0.0091 | 0.2420 |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | all | mrr | 0.0023 | 0.0001 | 0.0052 | 0.0620 |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | all | map | 0.0077 | 0.0048 | 0.0107 | 0.0002 |
| dta_confirmatory_split_transition_count_all_v20_t50_seed101.json | all | recall@10 | 0.0017 | 0.0007 | 0.0027 | 0.0004 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | length_only | top1_accuracy | 0.0036 | -0.0055 | 0.0127 | 0.3379 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | length_only | mrr | 0.0026 | -0.0021 | 0.0077 | 0.1412 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | length_only | map | 0.0280 | 0.0211 | 0.0351 | 0.0002 |
| dta_confirmatory_split_transition_count_length_only_v20_t50_seed101.json | length_only | recall@10 | 0.0029 | 0.0014 | 0.0045 | 0.0002 |

## Confirmatory Feature-Set Controls

| feature_set | metric | mean_delta | ci95_low | ci95_high | p_one_sided |
| --- | --- | --- | --- | --- | --- |
| length_only | map | 0.0280 | 0.0211 | 0.0351 | 0.0002 |
| length_only | recall@10 | 0.0029 | 0.0014 | 0.0045 | 0.0002 |
| size_depth | map | 0.0138 | 0.0099 | 0.0178 | 0.0002 |
| size_depth | recall@10 | 0.0023 | 0.0013 | 0.0035 | 0.0002 |
| branching | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| branching | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| metric_distortion | map | 0.0043 | 0.0013 | 0.0073 | 0.0034 |
| metric_distortion | recall@10 | 0.0005 | -0.0003 | 0.0013 | 0.1082 |
| shape | map | 0.0053 | 0.0022 | 0.0085 | 0.0008 |
| shape | recall@10 | 0.0016 | 0.0006 | 0.0026 | 0.0012 |
| all | map | 0.0077 | 0.0048 | 0.0107 | 0.0002 |
| all | recall@10 | 0.0017 | 0.0007 | 0.0027 | 0.0004 |
| length_only | map | 0.0281 | 0.0228 | 0.0335 | 0.0002 |
| length_only | recall@10 | 0.0035 | 0.0020 | 0.0050 | 0.0002 |
| size_depth | map | 0.0183 | 0.0149 | 0.0217 | 0.0002 |
| size_depth | recall@10 | 0.0030 | 0.0019 | 0.0042 | 0.0002 |
| branching | map | 0.0001 | -0.0003 | 0.0005 | 0.3909 |
| branching | recall@10 | 0.0000 | -0.0003 | 0.0004 | 0.4789 |
| metric_distortion | map | 0.0072 | 0.0054 | 0.0090 | 0.0002 |
| metric_distortion | recall@10 | 0.0013 | 0.0007 | 0.0020 | 0.0002 |
| shape | map | 0.0098 | 0.0072 | 0.0125 | 0.0002 |
| shape | recall@10 | 0.0021 | 0.0011 | 0.0031 | 0.0002 |
| all | map | 0.0136 | 0.0110 | 0.0163 | 0.0002 |
| all | recall@10 | 0.0026 | 0.0017 | 0.0036 | 0.0002 |
| length_only | map | 0.0252 | 0.0184 | 0.0322 | 0.0002 |
| length_only | recall@10 | 0.0017 | 0.0002 | 0.0033 | 0.0168 |
| size_depth | map | 0.0147 | 0.0108 | 0.0185 | 0.0002 |
| size_depth | recall@10 | 0.0016 | 0.0006 | 0.0026 | 0.0006 |
| branching | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| branching | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| metric_distortion | map | 0.0053 | 0.0025 | 0.0081 | 0.0002 |
| metric_distortion | recall@10 | 0.0004 | -0.0003 | 0.0011 | 0.1610 |
| shape | map | 0.0032 | 0.0022 | 0.0042 | 0.0002 |
| shape | recall@10 | 0.0004 | 0.0000 | 0.0009 | 0.0258 |
| all | map | 0.0087 | 0.0058 | 0.0118 | 0.0002 |
| all | recall@10 | 0.0014 | 0.0006 | 0.0023 | 0.0014 |

## Multi-Split Feature-Set Summary

| feature_set | metric | n_splits | split_seeds | mean_delta_mean | mean_delta_std | mean_delta_min | mean_delta_max | significant_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | map | 3 | 101,202,303 | 0.0100 | 0.0032 | 0.0077 | 0.0136 | 3 |
| all | recall@10 | 3 | 101,202,303 | 0.0019 | 0.0006 | 0.0014 | 0.0026 | 3 |
| branching | map | 3 | 101,202,303 | 0.0000 | 0.0000 | -0.0000 | 0.0001 | 0 |
| branching | recall@10 | 3 | 101,202,303 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| length_only | map | 3 | 101,202,303 | 0.0271 | 0.0017 | 0.0252 | 0.0281 | 3 |
| length_only | recall@10 | 3 | 101,202,303 | 0.0027 | 0.0009 | 0.0017 | 0.0035 | 3 |
| metric_distortion | map | 3 | 101,202,303 | 0.0056 | 0.0015 | 0.0043 | 0.0072 | 3 |
| metric_distortion | recall@10 | 3 | 101,202,303 | 0.0007 | 0.0005 | 0.0004 | 0.0013 | 1 |
| shape | map | 3 | 101,202,303 | 0.0061 | 0.0034 | 0.0032 | 0.0098 | 3 |
| shape | recall@10 | 3 | 101,202,303 | 0.0014 | 0.0008 | 0.0004 | 0.0021 | 3 |
| size_depth | map | 3 | 101,202,303 | 0.0156 | 0.0024 | 0.0138 | 0.0183 | 3 |
| size_depth | recall@10 | 3 | 101,202,303 | 0.0023 | 0.0007 | 0.0016 | 0.0030 | 3 |

## Confirmatory Residual Controls After Length

| feature_set | metric | mean_delta | ci95_low | ci95_high | p_one_sided |
| --- | --- | --- | --- | --- | --- |
| residual_size_depth | map | -0.0003 | -0.0008 | 0.0003 | 1.0000 |
| residual_size_depth | recall@10 | -0.0003 | -0.0007 | 0.0001 | 1.0000 |
| residual_branching | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| residual_branching | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_metric_distortion | map | -0.0000 | -0.0014 | 0.0014 | 1.0000 |
| residual_metric_distortion | recall@10 | -0.0001 | -0.0006 | 0.0004 | 1.0000 |
| residual_shape | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| residual_shape | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_all_nonlength | map | 0.0003 | 0.0000 | 0.0007 | 0.0168 |
| residual_all_nonlength | recall@10 | 0.0001 | -0.0002 | 0.0005 | 0.2509 |
| residual_size_depth | map | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_size_depth | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_branching | map | -0.0004 | -0.0009 | 0.0000 | 1.0000 |
| residual_branching | recall@10 | -0.0001 | -0.0005 | 0.0003 | 1.0000 |
| residual_metric_distortion | map | 0.0031 | 0.0021 | 0.0042 | 0.0002 |
| residual_metric_distortion | recall@10 | 0.0007 | 0.0002 | 0.0012 | 0.0022 |
| residual_shape | map | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_shape | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_all_nonlength | map | 0.0007 | 0.0004 | 0.0010 | 0.0002 |
| residual_all_nonlength | recall@10 | 0.0002 | -0.0000 | 0.0005 | 0.0622 |
| residual_size_depth | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| residual_size_depth | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_branching | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| residual_branching | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_metric_distortion | map | 0.0018 | 0.0008 | 0.0028 | 0.0010 |
| residual_metric_distortion | recall@10 | -0.0000 | -0.0005 | 0.0004 | 1.0000 |
| residual_shape | map | -0.0000 | -0.0000 | 0.0000 | 1.0000 |
| residual_shape | recall@10 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| residual_all_nonlength | map | 0.0005 | 0.0002 | 0.0008 | 0.0018 |
| residual_all_nonlength | recall@10 | 0.0001 | -0.0001 | 0.0004 | 0.2771 |

## Multi-Split Residual Summary After Length

| feature_set | metric | n_splits | split_seeds | mean_delta_mean | mean_delta_std | mean_delta_min | mean_delta_max | significant_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_all_nonlength | map | 3 | 101,202,303 | 0.0005 | 0.0002 | 0.0003 | 0.0007 | 3 |
| residual_all_nonlength | recall@10 | 3 | 101,202,303 | 0.0002 | 0.0001 | 0.0001 | 0.0002 | 0 |
| residual_branching | map | 3 | 101,202,303 | -0.0001 | 0.0003 | -0.0004 | -0.0000 | 0 |
| residual_branching | recall@10 | 3 | 101,202,303 | -0.0000 | 0.0000 | -0.0001 | 0.0000 | 0 |
| residual_metric_distortion | map | 3 | 101,202,303 | 0.0016 | 0.0016 | -0.0000 | 0.0031 | 2 |
| residual_metric_distortion | recall@10 | 3 | 101,202,303 | 0.0002 | 0.0005 | -0.0001 | 0.0007 | 1 |
| residual_shape | map | 3 | 101,202,303 | -0.0000 | 0.0000 | -0.0000 | 0.0000 | 0 |
| residual_shape | recall@10 | 3 | 101,202,303 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| residual_size_depth | map | 3 | 101,202,303 | -0.0001 | 0.0002 | -0.0003 | 0.0000 | 0 |
| residual_size_depth | recall@10 | 3 | 101,202,303 | -0.0001 | 0.0002 | -0.0003 | 0.0000 | 0 |

## Multi-Split Task-Level MAP Delta

| feature_set | task_label | metric | n_splits | mean_delta_mean | mean_delta_min | mean_delta_max |
| --- | --- | --- | --- | --- | --- | --- |
| all | 0 | map | 3 | 0.0049 | -0.0033 | 0.0161 |
| all | 1 | map | 3 | 0.0147 | 0.0078 | 0.0253 |
| all | 10 | map | 3 | 0.0300 | 0.0252 | 0.0350 |
| all | 2 | map | 3 | -0.0101 | -0.0148 | -0.0040 |
| all | 3 | map | 3 | 0.0408 | 0.0361 | 0.0464 |
| all | 4 | map | 3 | -0.0032 | -0.0110 | 0.0074 |
| all | 5 | map | 3 | 0.0034 | 0.0011 | 0.0056 |
| all | 6 | map | 3 | -0.0012 | -0.0053 | 0.0014 |
| all | 7 | map | 3 | 0.0134 | 0.0049 | 0.0265 |
| all | 8 | map | 3 | 0.0027 | -0.0022 | 0.0055 |
| all | 9 | map | 3 | 0.0148 | 0.0106 | 0.0178 |
| branching | 0 | map | 3 | -0.0011 | -0.0032 | 0.0000 |
| branching | 1 | map | 3 | -0.0007 | -0.0022 | 0.0000 |
| branching | 10 | map | 3 | 0.0003 | 0.0000 | 0.0009 |
| branching | 2 | map | 3 | -0.0004 | -0.0012 | 0.0000 |
| branching | 3 | map | 3 | 0.0013 | 0.0000 | 0.0040 |
| branching | 4 | map | 3 | 0.0004 | 0.0000 | 0.0011 |
| branching | 5 | map | 3 | -0.0003 | -0.0010 | -0.0000 |
| branching | 6 | map | 3 | -0.0002 | -0.0007 | 0.0000 |
| branching | 7 | map | 3 | 0.0007 | 0.0000 | 0.0020 |
| branching | 8 | map | 3 | 0.0002 | 0.0000 | 0.0007 |
| branching | 9 | map | 3 | 0.0001 | 0.0000 | 0.0003 |
| length_only | 0 | map | 3 | -0.0246 | -0.0299 | -0.0154 |
| length_only | 1 | map | 3 | 0.0176 | 0.0018 | 0.0413 |
| length_only | 10 | map | 3 | 0.1485 | 0.1315 | 0.1627 |
| length_only | 2 | map | 3 | -0.0104 | -0.0168 | -0.0050 |
| length_only | 3 | map | 3 | 0.0450 | 0.0400 | 0.0498 |
| length_only | 4 | map | 3 | -0.0151 | -0.0164 | -0.0130 |
| length_only | 5 | map | 3 | 0.0255 | 0.0187 | 0.0324 |
| length_only | 6 | map | 3 | -0.0131 | -0.0192 | -0.0082 |
| length_only | 7 | map | 3 | 0.0612 | 0.0519 | 0.0710 |
| length_only | 8 | map | 3 | 0.0468 | 0.0393 | 0.0539 |
| length_only | 9 | map | 3 | 0.0168 | 0.0120 | 0.0247 |
| metric_distortion | 0 | map | 3 | 0.0182 | 0.0056 | 0.0247 |
| metric_distortion | 1 | map | 3 | 0.0279 | 0.0233 | 0.0327 |
| metric_distortion | 10 | map | 3 | 0.0104 | 0.0097 | 0.0118 |
| metric_distortion | 2 | map | 3 | -0.0028 | -0.0075 | 0.0022 |
| metric_distortion | 3 | map | 3 | 0.0266 | 0.0182 | 0.0312 |
| metric_distortion | 4 | map | 3 | -0.0168 | -0.0280 | -0.0077 |
| metric_distortion | 5 | map | 3 | 0.0016 | -0.0000 | 0.0031 |
| metric_distortion | 6 | map | 3 | 0.0021 | -0.0008 | 0.0041 |
| metric_distortion | 7 | map | 3 | -0.0039 | -0.0104 | 0.0018 |
| metric_distortion | 8 | map | 3 | -0.0018 | -0.0082 | 0.0047 |
| metric_distortion | 9 | map | 3 | -0.0004 | -0.0012 | 0.0006 |
| shape | 0 | map | 3 | -0.0100 | -0.0131 | -0.0049 |
| shape | 1 | map | 3 | -0.0054 | -0.0172 | 0.0006 |
| shape | 10 | map | 3 | 0.0254 | 0.0074 | 0.0371 |
| shape | 2 | map | 3 | -0.0104 | -0.0195 | -0.0041 |
| shape | 3 | map | 3 | 0.0276 | 0.0122 | 0.0406 |
| shape | 4 | map | 3 | 0.0039 | 0.0021 | 0.0077 |
| shape | 5 | map | 3 | 0.0033 | 0.0006 | 0.0062 |
| shape | 6 | map | 3 | -0.0027 | -0.0034 | -0.0019 |
| shape | 7 | map | 3 | 0.0148 | 0.0058 | 0.0309 |
| shape | 8 | map | 3 | 0.0064 | 0.0021 | 0.0141 |
| shape | 9 | map | 3 | 0.0143 | 0.0081 | 0.0220 |
| size_depth | 0 | map | 3 | -0.0010 | -0.0072 | 0.0065 |
| size_depth | 1 | map | 3 | 0.0121 | 0.0033 | 0.0238 |
| size_depth | 10 | map | 3 | 0.0660 | 0.0646 | 0.0672 |
| size_depth | 2 | map | 3 | -0.0170 | -0.0241 | -0.0072 |
| size_depth | 3 | map | 3 | 0.0306 | 0.0256 | 0.0335 |
| size_depth | 4 | map | 3 | -0.0084 | -0.0164 | -0.0013 |
| size_depth | 5 | map | 3 | 0.0292 | 0.0207 | 0.0353 |
| size_depth | 6 | map | 3 | -0.0016 | -0.0051 | 0.0017 |
| size_depth | 7 | map | 3 | 0.0174 | -0.0007 | 0.0389 |
| size_depth | 8 | map | 3 | 0.0258 | 0.0220 | 0.0287 |
| size_depth | 9 | map | 3 | 0.0186 | 0.0143 | 0.0223 |

## Baseline Audit

| method | top1_accuracy | mrr | map | recall@10 |
| --- | --- | --- | --- | --- |
| flat_markov_jsd_divergence | 0.9600 | 0.9705 | 0.6596 | 0.1794 |
| flat_markov_jsd_distance | 0.9600 | 0.9705 | 0.6596 | 0.1794 |
| transition_count_jsd_divergence | 0.9836 | 0.9874 | 0.7418 | 0.1886 |
| transition_count_jsd_distance | 0.9836 | 0.9874 | 0.7418 | 0.1886 |
| rowwise_markov_jsd | 0.9636 | 0.9731 | 0.6447 | 0.1794 |
| ast_histogram_euclidean | 0.9000 | 0.9312 | 0.5001 | 0.1621 |

## Generated Figures

- `figures/fig01_markov_baseline_comparison.png`
- `figures/fig01_markov_baseline_comparison.pdf`
- `figures/fig02_transition_count_weight_sweep.png`
- `figures/fig02_transition_count_weight_sweep.pdf`
- `figures/fig03_feature_ablation_map.png`
- `figures/fig03_feature_ablation_map.pdf`
- `figures/fig04_confirmatory_delta_ci.png`
- `figures/fig04_confirmatory_delta_ci.pdf`
- `figures/fig05_confirmatory_feature_sweep.png`
- `figures/fig05_confirmatory_feature_sweep.pdf`
- `figures/fig06_confirmatory_residual_sweep.png`
- `figures/fig06_confirmatory_residual_sweep.pdf`
- `figures/fig07_task_level_map_delta.png`
- `figures/fig07_task_level_map_delta.pdf`

## Figure Interpretation

- `fig01`: transition-count JSD is the strongest non-geometric baseline and must be treated as the main comparator.
- `fig02`: the validation sweep selects a mostly Markov distance with a small geometry component, not a geometry-only model.
- `fig03`: geometry features add signal over the transition-count baseline; the effect size is modest.
- `fig04`: the confirmatory test supports a stable MAP and Recall@10 improvement, while Top1 and MRR remain weaker claims.
- `fig05`: the feature-set control summarizes mean deltas across confirmatory split seeds and checks whether the gain is explained by AST length/scale or by richer shape/distortion features.
- `fig06`: residual controls summarize mean deltas across confirmatory split seeds after regressing out length_only.
- `fig07`: task-level MAP deltas show whether the effect is broad across DTA tasks or concentrated in a few task groups.

## Interpretation Rule

Exploratory sweeps and ablations are used for method design. Article claims should be based on confirmatory validation/test splits and interpreted together with the feature-set control. The current strongest additional signal is length_only, not the full geometry profile.
