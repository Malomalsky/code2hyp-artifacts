from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from geometry_profile_research.code2hyp_experiments import (
    RealCode2HypPilotConfig,
    SyntheticComparisonConfig,
    run_real_code2hyp_pilot,
    run_synthetic_comparison,
)


class Code2HypExperimentTests(unittest.TestCase):
    def test_synthetic_comparison_reports_b1_and_b3_without_claiming_winner(self) -> None:
        result = run_synthetic_comparison(
            SyntheticComparisonConfig(
                examples=64,
                branches=4,
                epochs=8,
                batch_size=16,
                dataset_seed=31,
                model_seeds=(101,),
            )
        )

        self.assertEqual(result["experiment"], "synthetic_code2hyp_b1_b3_comparison")
        self.assertEqual(result["interpretation_status"], "sanity_check_not_scientific_claim")
        self.assertEqual(
            {run["variant"] for run in result["runs"]},
            {
                "B1_euclidean",
                "B2_product_fixed_curvature",
                "B3_product",
                "B4_hyperbolic_code2vec",
                "B4T_hyperbolic_code2vec_trainable_curvature",
                "B8_hyperbolic_frechet_code2vec",
                "B17_hyperbolic_path_mp_code2vec",
                "B18_hyperbolic_path_mp_struct_rank",
                "B19_hyperbolic_path_mp_rank_annealed",
                "B20_hyperbolic_path_mp_rank_delayed",
                "B21_hyperbolic_path_mp_rank_cosine",
                "B22_hyperbolic_path_mp_rank_warmup_decay",
                "B23_hyperbolic_path_attention_mp_code2vec",
                "B24_hyperbolic_path_attention_mp_rank_annealed",
                "B25_hyperbolic_path_depth_attention_mp_code2vec",
                "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                "B27_hyperbolic_path_attention_mp_monotone",
                "B28_hyperbolic_path_attention_mp_tree_distance",
                "B29_hyperbolic_path_dual_attention_mp_separated",
                "B30_hyperbolic_path_dual_attention_mp_rank_separated",
                "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                "B32_lorentz_path_dual_attention_mp_soft_rank",
                "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                "B9_lorentz_code2vec",
                "B15_lorentz_product_code2vec",
                "B10_factorized_product_code2vec",
                "B11_factorized_product_struct_rank",
                "B12_factorized_product_learned_metric_rank",
                "B16_factorized_product_three_metric_rank",
                "B35_code2hyp_product_frechet_adaptive",
                "B13_factorized_product_channel_mixer_rank",
                "B7_hyperbolic_attention_only",
                "B5_euclidean_struct_loss",
                "B6_euclidean_metric_code2vec",
                "B14_bounded_euclidean_metric_code2vec",
                "B_tree_euclidean_lca_bias",
            },
        )
        for run in result["runs"]:
            self.assertGreaterEqual(run["final_accuracy"], 0.0)
            self.assertLessEqual(run["final_accuracy"], 1.0)
            self.assertGreater(run["parameter_count"], 0)

    def test_synthetic_comparison_is_deterministic_for_fixed_seed(self) -> None:
        config = SyntheticComparisonConfig(
            examples=48,
            branches=3,
            epochs=4,
            batch_size=12,
            dataset_seed=37,
            model_seeds=(3,),
        )

        first = run_synthetic_comparison(config)
        second = run_synthetic_comparison(config)

        self.assertEqual(first["runs"], second["runs"])

    def test_real_code2hyp_pilot_uses_c2v_files_and_reports_multilabel_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.c2v"
            val_path = root / "val.c2v"
            train_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "to|string obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "hash|map self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "unknown|target self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_real_code2hyp_pilot(
                train_path,
                val_path,
                RealCode2HypPilotConfig(
                    train_limit=4,
                    val_limit=3,
                    epochs=2,
                    batch_size=2,
                    token_dim=8,
                    structural_dim=8,
                    structural_regularizer="rank",
                    lexical_ablation="structural_only",
                    use_positive_weighting=True,
                    max_positive_weight=7.0,
                    curvature=0.3,
                    model_seeds=(7,),
                ),
            )

        self.assertEqual(result["experiment"], "real_code2hyp_multilabel_pilot")
        self.assertEqual(result["interpretation_status"], "real_data_pilot_not_final_claim")
        self.assertEqual(result["dataset"]["train_records"], 4)
        self.assertEqual(result["dataset"]["validation_records_after_known_target_filter"], 2)
        self.assertEqual(result["dataset"]["lexical_ablation"], "structural_only")
        self.assertTrue(result["training"]["use_positive_weighting"])
        self.assertEqual(result["training"]["max_positive_weight"], 7.0)
        self.assertEqual(result["training"]["curvature"], 0.3)
        self.assertEqual(result["training"]["structural_regularizer_for_B5"], "rank")
        self.assertEqual(
            {run["variant"] for run in result["runs"]},
            {
                "B1_euclidean",
                "B2_product_fixed_curvature",
                "B3_product",
                "B4_hyperbolic_code2vec",
                "B4T_hyperbolic_code2vec_trainable_curvature",
                "B8_hyperbolic_frechet_code2vec",
                "B17_hyperbolic_path_mp_code2vec",
                "B18_hyperbolic_path_mp_struct_rank",
                "B19_hyperbolic_path_mp_rank_annealed",
                "B20_hyperbolic_path_mp_rank_delayed",
                "B21_hyperbolic_path_mp_rank_cosine",
                "B22_hyperbolic_path_mp_rank_warmup_decay",
                "B23_hyperbolic_path_attention_mp_code2vec",
                "B24_hyperbolic_path_attention_mp_rank_annealed",
                "B25_hyperbolic_path_depth_attention_mp_code2vec",
                "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                "B27_hyperbolic_path_attention_mp_monotone",
                "B28_hyperbolic_path_attention_mp_tree_distance",
                "B29_hyperbolic_path_dual_attention_mp_separated",
                "B30_hyperbolic_path_dual_attention_mp_rank_separated",
                "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                "B32_lorentz_path_dual_attention_mp_soft_rank",
                "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                "B9_lorentz_code2vec",
                "B15_lorentz_product_code2vec",
                "B10_factorized_product_code2vec",
                "B11_factorized_product_struct_rank",
                "B12_factorized_product_learned_metric_rank",
                "B16_factorized_product_three_metric_rank",
                "B35_code2hyp_product_frechet_adaptive",
                "B13_factorized_product_channel_mixer_rank",
                "B7_hyperbolic_attention_only",
                "B5_euclidean_struct_loss",
                "B6_euclidean_metric_code2vec",
                "B14_bounded_euclidean_metric_code2vec",
                "B_tree_euclidean_lca_bias",
            },
        )
        for run in result["runs"]:
            self.assertGreaterEqual(run["validation_f1"], 0.0)
            self.assertLessEqual(run["validation_f1"], 1.0)
            self.assertEqual(len(run["history"]), 2)
            self.assertIn("validation_structural_loss", run)
            self.assertIn("validation_structural_normalized_stress", run)
            self.assertIn("validation_structural_rank_loss", run)
            self.assertIn("validation_structural_spearman", run)
            self.assertIn("validation_structural_neighbor_overlap_at_1", run)
            self.assertIn("validation_structural_neighbor_overlap_at_3", run)
            self.assertIn("validation_structural_neighbor_recall_at_1", run)
            self.assertIn("validation_structural_neighbor_recall_at_3", run)
            self.assertGreaterEqual(run["validation_structural_loss"], 0.0)
            self.assertGreaterEqual(run["validation_structural_normalized_stress"], 0.0)
            self.assertGreaterEqual(run["validation_structural_rank_loss"], 0.0)
            self.assertGreaterEqual(run["validation_structural_spearman"], -1.0)
            self.assertLessEqual(run["validation_structural_spearman"], 1.0)
            self.assertGreaterEqual(run["validation_structural_neighbor_overlap_at_1"], 0.0)
            self.assertLessEqual(run["validation_structural_neighbor_overlap_at_1"], 1.0)
            self.assertGreaterEqual(run["validation_structural_neighbor_overlap_at_3"], 0.0)
            self.assertLessEqual(run["validation_structural_neighbor_overlap_at_3"], 1.0)
            self.assertEqual(
                run["validation_structural_neighbor_recall_at_1"],
                run["validation_structural_neighbor_overlap_at_1"],
            )
            self.assertEqual(
                run["validation_structural_neighbor_recall_at_3"],
                run["validation_structural_neighbor_overlap_at_3"],
            )
        b2_run = next(run for run in result["runs"] if run["variant"] == "B2_product_fixed_curvature")
        self.assertAlmostEqual(b2_run["curvature"], 0.3)
        b4_run = next(run for run in result["runs"] if run["variant"] == "B4_hyperbolic_code2vec")
        b4t_run = next(
            run for run in result["runs"] if run["variant"] == "B4T_hyperbolic_code2vec_trainable_curvature"
        )
        b8_run = next(run for run in result["runs"] if run["variant"] == "B8_hyperbolic_frechet_code2vec")
        b17_run = next(run for run in result["runs"] if run["variant"] == "B17_hyperbolic_path_mp_code2vec")
        b18_run = next(run for run in result["runs"] if run["variant"] == "B18_hyperbolic_path_mp_struct_rank")
        b19_run = next(run for run in result["runs"] if run["variant"] == "B19_hyperbolic_path_mp_rank_annealed")
        b20_run = next(run for run in result["runs"] if run["variant"] == "B20_hyperbolic_path_mp_rank_delayed")
        b21_run = next(run for run in result["runs"] if run["variant"] == "B21_hyperbolic_path_mp_rank_cosine")
        b22_run = next(run for run in result["runs"] if run["variant"] == "B22_hyperbolic_path_mp_rank_warmup_decay")
        b23_run = next(run for run in result["runs"] if run["variant"] == "B23_hyperbolic_path_attention_mp_code2vec")
        b24_run = next(run for run in result["runs"] if run["variant"] == "B24_hyperbolic_path_attention_mp_rank_annealed")
        b25_run = next(run for run in result["runs"] if run["variant"] == "B25_hyperbolic_path_depth_attention_mp_code2vec")
        b26_run = next(
            run for run in result["runs"] if run["variant"] == "B26_hyperbolic_path_depth_attention_mp_rank_annealed"
        )
        b27_run = next(run for run in result["runs"] if run["variant"] == "B27_hyperbolic_path_attention_mp_monotone")
        b28_run = next(
            run for run in result["runs"] if run["variant"] == "B28_hyperbolic_path_attention_mp_tree_distance"
        )
        b29_run = next(
            run for run in result["runs"] if run["variant"] == "B29_hyperbolic_path_dual_attention_mp_separated"
        )
        b30_run = next(
            run for run in result["runs"] if run["variant"] == "B30_hyperbolic_path_dual_attention_mp_rank_separated"
        )
        b31_run = next(
            run for run in result["runs"] if run["variant"] == "B31_hyperbolic_path_dual_attention_mp_soft_rank"
        )
        b32_run = next(
            run for run in result["runs"] if run["variant"] == "B32_lorentz_path_dual_attention_mp_soft_rank"
        )
        b34_run = next(
            run for run in result["runs"] if run["variant"] == "B34_hyperbolic_path_dual_attention_mp_adaptive_rank"
        )
        b9_run = next(run for run in result["runs"] if run["variant"] == "B9_lorentz_code2vec")
        b15_run = next(run for run in result["runs"] if run["variant"] == "B15_lorentz_product_code2vec")
        b10_run = next(run for run in result["runs"] if run["variant"] == "B10_factorized_product_code2vec")
        b11_run = next(run for run in result["runs"] if run["variant"] == "B11_factorized_product_struct_rank")
        b12_run = next(run for run in result["runs"] if run["variant"] == "B12_factorized_product_learned_metric_rank")
        b16_run = next(run for run in result["runs"] if run["variant"] == "B16_factorized_product_three_metric_rank")
        b35_run = next(run for run in result["runs"] if run["variant"] == "B35_code2hyp_product_frechet_adaptive")
        b13_run = next(run for run in result["runs"] if run["variant"] == "B13_factorized_product_channel_mixer_rank")
        b7_run = next(run for run in result["runs"] if run["variant"] == "B7_hyperbolic_attention_only")
        b6_run = next(run for run in result["runs"] if run["variant"] == "B6_euclidean_metric_code2vec")
        b14_run = next(run for run in result["runs"] if run["variant"] == "B14_bounded_euclidean_metric_code2vec")
        btree_run = next(run for run in result["runs"] if run["variant"] == "B_tree_euclidean_lca_bias")
        self.assertEqual(b4t_run["parameter_count"], b4_run["parameter_count"] + 1)
        self.assertGreater(b4t_run["curvature"], 0.0)
        self.assertEqual(b8_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b8_run["curvature"], 0.3)
        self.assertGreater(b17_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b17_run["curvature"], 0.3)
        self.assertEqual(b18_run["parameter_count"], b17_run["parameter_count"])
        self.assertAlmostEqual(b18_run["curvature"], 0.3)
        self.assertEqual(b19_run["parameter_count"], b17_run["parameter_count"])
        self.assertAlmostEqual(b19_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b19_run["history"]], [0.025, 0.05])
        self.assertEqual(b20_run["parameter_count"], b17_run["parameter_count"])
        self.assertAlmostEqual(b20_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b20_run["history"]], [0.0, 0.05])
        self.assertEqual(b21_run["parameter_count"], b17_run["parameter_count"])
        self.assertAlmostEqual(b21_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b21_run["history"]], [0.0, 0.05])
        self.assertEqual(b22_run["parameter_count"], b17_run["parameter_count"])
        self.assertAlmostEqual(b22_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b22_run["history"]], [0.05, 0.05])
        self.assertEqual(b23_run["parameter_count"], b17_run["parameter_count"] + 8)
        self.assertAlmostEqual(b23_run["curvature"], 0.3)
        self.assertEqual(b24_run["parameter_count"], b23_run["parameter_count"])
        self.assertAlmostEqual(b24_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b24_run["history"]], [0.025, 0.05])
        self.assertEqual(b25_run["parameter_count"], b23_run["parameter_count"] + 1)
        self.assertAlmostEqual(b25_run["curvature"], 0.3)
        self.assertEqual(b26_run["parameter_count"], b25_run["parameter_count"])
        self.assertAlmostEqual(b26_run["curvature"], 0.3)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b26_run["history"]], [0.025, 0.05])
        self.assertEqual(b27_run["parameter_count"], b23_run["parameter_count"])
        self.assertAlmostEqual(b27_run["curvature"], 0.3)
        self.assertEqual(b27_run["structural_regularizer"], "path_attention_monotone")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b27_run["history"]], [0.025, 0.05])
        self.assertEqual(b28_run["parameter_count"], b23_run["parameter_count"])
        self.assertAlmostEqual(b28_run["curvature"], 0.3)
        self.assertEqual(b28_run["structural_regularizer"], "path_attention_tree_distance")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b28_run["history"]], [0.025, 0.05])
        self.assertGreater(b29_run["parameter_count"], b23_run["parameter_count"])
        self.assertAlmostEqual(b29_run["curvature"], 0.3)
        self.assertEqual(b29_run["structural_regularizer"], "path_dual_attention_separation")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b29_run["history"]], [0.025, 0.05])
        self.assertEqual(b30_run["parameter_count"], b29_run["parameter_count"])
        self.assertAlmostEqual(b30_run["curvature"], 0.3)
        self.assertEqual(b30_run["structural_regularizer"], "path_dual_attention_separation_rank")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b30_run["history"]], [0.025, 0.05])
        self.assertEqual(b31_run["parameter_count"], b29_run["parameter_count"])
        self.assertAlmostEqual(b31_run["curvature"], 0.3)
        self.assertEqual(b31_run["structural_regularizer"], "path_dual_attention_separation_soft_rank")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b31_run["history"]], [0.025, 0.05])
        self.assertEqual(b32_run["parameter_count"], b29_run["parameter_count"])
        self.assertAlmostEqual(b32_run["curvature"], 0.3)
        self.assertEqual(b32_run["structural_regularizer"], "path_dual_attention_separation_soft_rank")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b32_run["history"]], [0.025, 0.05])
        self.assertEqual(b34_run["parameter_count"], b29_run["parameter_count"])
        self.assertAlmostEqual(b34_run["curvature"], 0.3)
        self.assertEqual(b34_run["structural_regularizer"], "path_dual_attention_separation_adaptive_rank")
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b34_run["history"]], [0.025, 0.05])
        self.assertEqual(b9_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b9_run["curvature"], 0.3)
        self.assertEqual(b15_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b15_run["curvature"], 0.3)
        self.assertEqual(b10_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b10_run["curvature"], 0.3)
        self.assertEqual(b11_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b11_run["curvature"], 0.3)
        self.assertEqual(b12_run["parameter_count"], b4_run["parameter_count"] + 2)
        self.assertAlmostEqual(b12_run["curvature"], 0.3)
        self.assertIn("factorized_metric_weights", b12_run)
        self.assertEqual(len(b12_run["factorized_metric_weights"]), 2)
        self.assertGreater(b12_run["factorized_metric_weights"][0], 0.0)
        self.assertGreater(b12_run["factorized_metric_weights"][1], 0.0)
        self.assertEqual(b16_run["parameter_count"], b4_run["parameter_count"] + 3)
        self.assertAlmostEqual(b16_run["curvature"], 0.3)
        self.assertIn("factorized_metric_weights", b16_run)
        self.assertEqual(len(b16_run["factorized_metric_weights"]), 3)
        self.assertGreater(b16_run["factorized_metric_weights"][0], 0.0)
        self.assertGreater(b16_run["factorized_metric_weights"][1], 0.0)
        self.assertGreater(b16_run["factorized_metric_weights"][2], 0.0)
        self.assertEqual(b35_run["parameter_count"], b16_run["parameter_count"] + 1)
        self.assertGreater(b35_run["curvature"], 0.0)
        self.assertIn("factorized_metric_weights", b35_run)
        self.assertEqual(len(b35_run["factorized_metric_weights"]), 3)
        self.assertGreater(b35_run["factorized_metric_weights"][0], 0.0)
        self.assertGreater(b35_run["factorized_metric_weights"][1], 0.0)
        self.assertGreater(b35_run["factorized_metric_weights"][2], 0.0)
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b35_run["history"]], [0.0, 0.05])
        expected_b13_extra = 2 * 24 * 4
        self.assertEqual(b13_run["parameter_count"], b4_run["parameter_count"] + expected_b13_extra)
        self.assertEqual(b13_run["factorized_channel_mixer_rank"], 4)
        self.assertEqual(b7_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b7_run["curvature"], 0.3)
        self.assertEqual(b6_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b6_run["curvature"], 0.3)
        self.assertEqual(b14_run["parameter_count"], b4_run["parameter_count"])
        self.assertAlmostEqual(b14_run["curvature"], 0.3)
        self.assertEqual(btree_run["parameter_count"], b6_run["parameter_count"] + 4)
        self.assertAlmostEqual(btree_run["curvature"], 0.3)

    def test_real_code2hyp_pilot_can_run_focused_variant_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.c2v"
            val_path = root / "val.c2v"
            train_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "to|string obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "hash|map self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_real_code2hyp_pilot(
                train_path,
                val_path,
                RealCode2HypPilotConfig(
                    train_limit=4,
                    val_limit=2,
                    epochs=2,
                    batch_size=2,
                    token_dim=8,
                    structural_dim=8,
                    curvature=0.3,
                    model_seeds=(7,),
                    variant_filter=(
                        "B4_hyperbolic_code2vec",
                        "B20_hyperbolic_path_mp_rank_delayed",
                    ),
                ),
            )

        self.assertEqual(
            [run["variant"] for run in result["runs"]],
            [
                "B4_hyperbolic_code2vec",
                "B20_hyperbolic_path_mp_rank_delayed",
            ],
        )
        b20_run = result["runs"][1]
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b20_run["history"]], [0.0, 0.05])

    def test_real_code2hyp_pilot_reports_product_attention_bias_weight_for_b44(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.c2v"
            val_path = root / "val.c2v"
            train_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "to|string obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "hash|map self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_real_code2hyp_pilot(
                train_path,
                val_path,
                RealCode2HypPilotConfig(
                    train_limit=4,
                    val_limit=2,
                    epochs=1,
                    batch_size=2,
                    token_dim=8,
                    structural_dim=8,
                    structural_regularizer="rank",
                    use_positive_weighting=True,
                    max_positive_weight=7.0,
                    model_seeds=(7,),
                    variant_filter=("B44_code2hyp_context_transform_product_bias_frechet",),
                ),
            )

        run = result["runs"][0]
        self.assertEqual(run["variant"], "B44_code2hyp_context_transform_product_bias_frechet")
        self.assertIn("product_attention_bias_weight", run)
        self.assertGreater(run["product_attention_bias_weight"], 0.0)

    def test_real_code2hyp_pilot_can_run_schedule_sweep_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.c2v"
            val_path = root / "val.c2v"
            train_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "to|string obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "hash|map self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_real_code2hyp_pilot(
                train_path,
                val_path,
                RealCode2HypPilotConfig(
                    train_limit=4,
                    val_limit=2,
                    epochs=3,
                    batch_size=2,
                    token_dim=8,
                    structural_dim=8,
                    curvature=0.3,
                    model_seeds=(7,),
                    variant_filter=(
                        "B17_hyperbolic_path_mp_code2vec",
                        "B21_hyperbolic_path_mp_rank_cosine",
                        "B22_hyperbolic_path_mp_rank_warmup_decay",
                    ),
                ),
            )

        self.assertEqual(
            [run["variant"] for run in result["runs"]],
            [
                "B17_hyperbolic_path_mp_code2vec",
                "B21_hyperbolic_path_mp_rank_cosine",
                "B22_hyperbolic_path_mp_rank_warmup_decay",
            ],
        )
        b17_run, b21_run, b22_run = result["runs"]
        self.assertEqual(b21_run["parameter_count"], b17_run["parameter_count"])
        self.assertEqual(b22_run["parameter_count"], b17_run["parameter_count"])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b21_run["history"]], [0.0, 0.025, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b22_run["history"]], [0.0, 0.05, 0.0])

    def test_real_code2hyp_pilot_can_run_path_attention_message_passing_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.c2v"
            val_path = root / "val.c2v"
            train_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "to|string obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                        "hash|map self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(
                    [
                        "to|lower|case obj,Name|Call,value",
                        "hash|code self,Name|Call,result",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_real_code2hyp_pilot(
                train_path,
                val_path,
                RealCode2HypPilotConfig(
                    train_limit=4,
                    val_limit=2,
                    epochs=3,
                    batch_size=2,
                    token_dim=8,
                    structural_dim=8,
                    curvature=0.3,
                    model_seeds=(7,),
                    structural_regularizer="rank",
                    variant_filter=(
                        "B17_hyperbolic_path_mp_code2vec",
                        "B23_hyperbolic_path_attention_mp_code2vec",
                        "B24_hyperbolic_path_attention_mp_rank_annealed",
                        "B25_hyperbolic_path_depth_attention_mp_code2vec",
                        "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                        "B27_hyperbolic_path_attention_mp_monotone",
                        "B28_hyperbolic_path_attention_mp_tree_distance",
                        "B29_hyperbolic_path_dual_attention_mp_separated",
                        "B30_hyperbolic_path_dual_attention_mp_rank_separated",
                        "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                        "B32_lorentz_path_dual_attention_mp_soft_rank",
                        "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
                    ),
                ),
            )

        self.assertEqual(
            [run["variant"] for run in result["runs"]],
            [
                "B17_hyperbolic_path_mp_code2vec",
                "B23_hyperbolic_path_attention_mp_code2vec",
                "B24_hyperbolic_path_attention_mp_rank_annealed",
                "B25_hyperbolic_path_depth_attention_mp_code2vec",
                "B26_hyperbolic_path_depth_attention_mp_rank_annealed",
                "B27_hyperbolic_path_attention_mp_monotone",
                "B28_hyperbolic_path_attention_mp_tree_distance",
                "B29_hyperbolic_path_dual_attention_mp_separated",
                "B30_hyperbolic_path_dual_attention_mp_rank_separated",
                "B31_hyperbolic_path_dual_attention_mp_soft_rank",
                "B32_lorentz_path_dual_attention_mp_soft_rank",
                "B34_hyperbolic_path_dual_attention_mp_adaptive_rank",
            ],
        )
        (
            b17_run,
            b23_run,
            b24_run,
            b25_run,
            b26_run,
            b27_run,
            b28_run,
            b29_run,
            b30_run,
            b31_run,
            b32_run,
            b34_run,
        ) = result["runs"]
        self.assertEqual(b23_run["parameter_count"], b17_run["parameter_count"] + 8)
        self.assertEqual(b24_run["parameter_count"], b23_run["parameter_count"])
        self.assertEqual(b25_run["parameter_count"], b23_run["parameter_count"] + 1)
        self.assertEqual(b26_run["parameter_count"], b25_run["parameter_count"])
        self.assertEqual(b27_run["parameter_count"], b23_run["parameter_count"])
        self.assertEqual(b28_run["parameter_count"], b23_run["parameter_count"])
        self.assertGreater(b29_run["parameter_count"], b23_run["parameter_count"])
        self.assertEqual(b30_run["parameter_count"], b29_run["parameter_count"])
        self.assertEqual(b31_run["parameter_count"], b29_run["parameter_count"])
        self.assertEqual(b32_run["parameter_count"], b29_run["parameter_count"])
        self.assertEqual(b34_run["parameter_count"], b29_run["parameter_count"])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b23_run["history"]], [0.0, 0.0, 0.0])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b24_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b25_run["history"]], [0.0, 0.0, 0.0])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b26_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b27_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b28_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b29_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b30_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b31_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b32_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertEqual([round(row["structural_loss_weight"], 4) for row in b34_run["history"]], [0.0167, 0.0333, 0.05])
        self.assertGreaterEqual(b27_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b28_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b29_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b30_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b31_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b32_run["history"][-1]["structural_loss"], 0.0)
        self.assertGreaterEqual(b34_run["history"][-1]["structural_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
