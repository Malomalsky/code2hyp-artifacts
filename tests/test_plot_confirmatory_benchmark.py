from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_confirmatory_benchmark import plot_confirmatory_benchmark


def test_plot_confirmatory_benchmark_writes_png_and_pdf(tmp_path: Path) -> None:
    summary = {
        "paired_contrasts": [
            {
                "dataset": "bugnet_python",
                "geometry": "E",
                "cost_mode": "train_weighted_combined_p1p00",
                "contrast": "lca_product_measure_minus_single_point_measure",
                "label": "LCA-product measure - single-point measure",
                "paired_query_count": 384,
                "delta_mrr": 0.019,
                "bootstrap_ci": {"delta_mrr": [0.003, 0.036]},
            },
            {
                "dataset": "dta_zenodo",
                "geometry": "E",
                "cost_mode": "train_weighted_combined_p1p00",
                "contrast": "lca_product_measure_minus_single_point_measure",
                "label": "LCA-product measure - single-point measure",
                "paired_query_count": 264,
                "delta_mrr": -0.002,
                "bootstrap_ci": {"delta_mrr": [-0.025, 0.022]},
            },
        ]
    }
    input_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(summary), encoding="utf-8")

    outputs = plot_confirmatory_benchmark(input_path=input_path, output_prefix=tmp_path / "confirmatory")

    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)
