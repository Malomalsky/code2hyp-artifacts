from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.interpret_raw_ast_orientation_probe import format_markdown, interpret_orientation_probe


def _run(
    *,
    root: Path,
    name: str,
    path_cost_orientation: str,
    ranks: tuple[int, int, int],
) -> dict[str, object]:
    query_records = [
        {
            "anchor_id": f"q{index}",
            "rank": rank,
            "margin": 1.0 / rank - 0.25,
            "positive_distance": float(rank),
            "nearest_negative_distance": float(rank) - 0.5,
        }
        for index, rank in enumerate(ranks)
    ]
    path = root / f"{name}.json"
    path.write_text(json.dumps({"query_records": query_records}), encoding="utf-8")
    return {
        "project": "toy",
        "node_input_mode": "label_only",
        "path_object_mode": "lca_product",
        "method_aggregation": "measure",
        "path_cost_orientation": path_cost_orientation,
        "geometry": "euclidean",
        "curvature": 1.0,
        "dim": 4,
        "seed": 11,
        "output_path": str(path),
    }


class InterpretRawASTOrientationProbeTests(unittest.TestCase):
    def test_reports_ordered_directed_when_unoriented_delta_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "runs": [
                            _run(root=root, name="directed", path_cost_orientation="directed", ranks=(1, 1, 1)),
                            _run(root=root, name="unoriented", path_cost_orientation="unoriented", ranks=(5, 5, 5)),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = interpret_orientation_probe(matrix, bootstrap_samples=25, seed=7)
            markdown = format_markdown(payload)

        self.assertEqual(payload["overall"], "ordered_directed_readout")
        self.assertEqual(payload["decisions"][0]["decision"], "ordered_directed_supported")
        self.assertIn("ordered_directed_supported", markdown)

    def test_reports_unoriented_quotient_when_unoriented_delta_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix = root / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "runs": [
                            _run(root=root, name="directed", path_cost_orientation="directed", ranks=(5, 5, 5)),
                            _run(root=root, name="unoriented", path_cost_orientation="unoriented", ranks=(1, 1, 1)),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = interpret_orientation_probe(matrix, bootstrap_samples=25, seed=7)

        self.assertEqual(payload["overall"], "unoriented_quotient_readout")
        self.assertEqual(payload["decisions"][0]["decision"], "unoriented_quotient_supported")


if __name__ == "__main__":
    unittest.main()
