from __future__ import annotations

import json

import pytest

from scripts.summarize_path_sampling_sensitivity import summarize


def test_path_sampling_sensitivity_summarizes_delta_and_lca_weight(tmp_path):
    payload = {
        "max_paths": 128,
        "path_selection_policy": "lca_depth_stratified",
        "lca_view": "code2hyp_path_signature_kernel",
        "lca_selection_margin": 0.02,
        "cell_summaries": [
            {
                "dataset": "toy",
                "variant": "code2hyp_multiview_selected",
                "mrr": 0.60,
                "recall_at_5": 0.80,
                "selected_weights_by_seed": {
                    "1": {"path_signature_plus_tokens": 0.2},
                    "2": {"code2hyp_path_signature_kernel": 0.4},
                },
            },
            {
                "dataset": "toy",
                "variant": "code2hyp_multiview_no_lca_selected",
                "mrr": 0.55,
                "recall_at_5": 0.75,
            },
            {
                "dataset": "toy",
                "variant": "token_ast_selected",
                "mrr": 0.50,
                "recall_at_5": 0.70,
            },
        ],
    }
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = summarize([("lca_strat_k128", path)])
    row = result["rows"][0]

    assert row["label"] == "lca_strat_k128"
    assert row["path_selection_policy"] == "lca_depth_stratified"
    assert row["lca_view"] == "code2hyp_path_signature_kernel"
    assert row["lca_selection_margin"] == 0.02
    assert row["max_paths"] == 128
    assert row["multiview_minus_no_lca_mrr"] == pytest.approx(0.05)
    assert row["mean_selected_lca_weight"] == pytest.approx(0.2)
