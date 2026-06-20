import unittest

from geometry_profile_research.analysis import geometry_profile_for_ast_source
from geometry_profile_research.geometry_features import geometry_feature_sets


class GeometryFeatureSetTests(unittest.TestCase):
    def test_geometry_feature_sets_partition_basic_and_metric_features(self):
        profile = geometry_profile_for_ast_source("def main(x):\n    return x + 1\n")

        feature_sets = geometry_feature_sets(profile)

        self.assertEqual(
            set(feature_sets),
            {"length_only", "size_depth", "branching", "metric_distortion", "all"},
        )
        self.assertEqual(set(feature_sets["length_only"]), {"log_node_count", "log_edge_count"})
        self.assertIn("log_node_count", feature_sets["size_depth"])
        self.assertIn("max_depth", feature_sets["size_depth"])
        self.assertIn("leaf_fraction", feature_sets["branching"])
        self.assertIn("branching_entropy", feature_sets["branching"])
        self.assertIn("hyperbolic_stress", feature_sets["metric_distortion"])
        self.assertIn("geometry_advantage", feature_sets["metric_distortion"])
        self.assertEqual(
            set(feature_sets["all"]),
            set(feature_sets["length_only"])
            | set(feature_sets["size_depth"])
            | set(feature_sets["branching"])
            | set(feature_sets["metric_distortion"]),
        )


if __name__ == "__main__":
    unittest.main()
