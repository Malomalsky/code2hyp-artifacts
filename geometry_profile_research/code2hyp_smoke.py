from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .code2hyp_torch import Code2HypBatch, Code2HypTorchConfig, Code2HypTorchModel


def build_toy_code2hyp_batch() -> Code2HypBatch:
    return Code2HypBatch(
        start_tokens=torch.tensor([[1, 3, 1], [2, 4, 0]], dtype=torch.long),
        end_tokens=torch.tensor([[2, 4, 5], [3, 1, 0]], dtype=torch.long),
        ast_paths=torch.tensor(
            [
                [[1, 2, 3, 0], [2, 4, 0, 0], [3, 0, 0, 0]],
                [[1, 5, 6, 0], [6, 7, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.long,
        ),
        ast_path_mask=torch.tensor(
            [
                [[True, True, True, False], [True, True, False, False], [True, False, False, False]],
                [[True, True, True, False], [True, True, False, False], [False, False, False, False]],
            ],
            dtype=torch.bool,
        ),
        context_mask=torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool),
    )


def _variant_report(model: Code2HypTorchModel, batch: Code2HypBatch) -> dict[str, Any]:
    with torch.no_grad():
        output = model(batch)
    return {
        "parameter_count": model.parameter_count(),
        "logits_shape": list(output.logits.shape),
        "representation_shape": list(output.representation.shape),
        "attention_shape": list(output.attention.shape),
        "curvature": float(output.curvature.detach()),
        "attention_row_sums": [float(value) for value in output.attention.sum(dim=1).detach()],
        "structural_points_shape": None
        if output.structural_points is None
        else list(output.structural_points.shape),
    }


def build_code2hyp_smoke_report(seed: int = 13) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = Code2HypTorchConfig(
        token_vocab_size=16,
        ast_node_vocab_size=12,
        label_vocab_size=7,
        token_dim=4,
        structural_dim=5,
        trainable_curvature=True,
    )
    batch = build_toy_code2hyp_batch()
    euclidean = Code2HypTorchModel(config, variant="euclidean")
    product = Code2HypTorchModel(config, variant="product")

    euclidean_report = _variant_report(euclidean, batch)
    product_report = _variant_report(product, batch)
    parameter_delta = product_report["parameter_count"] - euclidean_report["parameter_count"]
    relative_overhead = parameter_delta / euclidean_report["parameter_count"]

    return {
        "report_type": "code2hyp_torch_smoke",
        "seed": seed,
        "batch_size": int(batch.start_tokens.shape[0]),
        "contexts_per_method": int(batch.start_tokens.shape[1]),
        "label_vocab_size": config.label_vocab_size,
        "variants": {
            "B1_euclidean": euclidean_report,
            "B3_product_trainable_curvature": product_report,
        },
        "parameter_delta_B3_minus_B1": parameter_delta,
        "relative_parameter_overhead_B3_vs_B1": relative_overhead,
        "interpretation": (
            "B1 and B3 share logits and representation shapes; B3 adds only "
            "one trainable curvature parameter in this smoke configuration."
        ),
    }


def write_code2hyp_smoke_report(output_path: str | Path, seed: int = 13) -> dict[str, Any]:
    report = build_code2hyp_smoke_report(seed=seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
