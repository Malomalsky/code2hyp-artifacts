from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_factor_matrix_from_manifest import (
    dry_run_command,
    item_scope_from_manifest,
    language_from_manifest,
    task_sources_from_manifest,
)


class RunFactorMatrixFromManifestTests(unittest.TestCase):
    def test_manifest_to_task_sources_and_dry_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            task_a = root / "task-a"
            task_b = root / "task-b"
            task_a.mkdir()
            task_b.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "language": "python",
                        "item_scope": "module",
                        "tasks": [
                            {"label": "task-a", "path": str(task_a)},
                            {"label": "task-b", "path": str(task_b)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            tasks = task_sources_from_manifest(manifest)
            args = Namespace(
                output=root / "out.json",
                benchmark_level="B_independent_solution",
                geometries="E,H_1e-4,H_1",
                path_object_modes="single_point,lca_product",
                method_aggregations="centroid,measure",
                dim=4,
                epochs=1,
                seed=20260625,
                side_weight=0.5,
                max_ball_fraction=0.25,
                encoder_policy="geometry_aware",
            )

            command = dry_run_command(args, tasks, language_from_manifest(manifest), item_scope_from_manifest(manifest))
            self.assertEqual(language_from_manifest(manifest), "python")
            self.assertEqual(item_scope_from_manifest(manifest), "module")
            self.assertEqual([task.label for task in tasks], ["task-a", "task-b"])
            self.assertIn("scripts/run_dta_factor_matrix.py", command)
            self.assertIn("--item-scope module", command)
            self.assertIn("--side-weight 0.5", command)
            self.assertIn("--max-ball-fraction 0.25", command)
            self.assertIn("--encoder-policy geometry_aware", command)
            self.assertIn("--task task-a", command)
            self.assertIn("--task task-b", command)

    def test_materialized_dta_manifest_with_task_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "materialized"
            (output_dir / "task-00").mkdir(parents=True)
            (output_dir / "task-01").mkdir(parents=True)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "output_dir": str(output_dir),
                        "tasks": ["00", "01"],
                    }
                ),
                encoding="utf-8",
            )

            tasks = task_sources_from_manifest(manifest)

            self.assertEqual(language_from_manifest(manifest), "python")
            self.assertEqual([task.label for task in tasks], ["task-00", "task-01"])
            self.assertEqual([task.source for task in tasks], [output_dir / "task-00", output_dir / "task-01"])


if __name__ == "__main__":
    unittest.main()
