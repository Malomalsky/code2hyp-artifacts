from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_raw_ast_fgw_relation_ablation import RELATION_LABELS


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _relation_from_payload(path: Path, payload: dict[str, Any]) -> str:
    relation = payload.get("config", {}).get("structural_relation")
    if relation:
        return str(relation)
    if "alpha0p75" in path.stem and "32methods" in path.stem:
        return "endpoint"
    return path.stem


def _pair_key(record: dict[str, Any]) -> tuple[int, int]:
    left = int(record["left"])
    right = int(record["right"])
    return (left, right) if left <= right else (right, left)


def _load_relation_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    relation = _relation_from_payload(path, payload)
    pairs = {
        _pair_key(record): {
            "gw_structure": float(record["gw_structure"]),
            "fgw": float(record["fgw"]),
        }
        for record in payload.get("pairs", [])
    }
    if not pairs:
        raise ValueError(f"result has no pair records: {path}")
    return {
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, relation),
        "path": str(path),
        "method_count": int(payload.get("method_count", 0)),
        "pair_count": int(payload.get("pair_count", len(pairs))),
        "pairs": pairs,
    }


def analyze_cross_relation_transfer(input_paths: Sequence[Path], output_stem: Path) -> dict[str, Path]:
    results = [_load_relation_result(path) for path in input_paths]
    if len(results) < 2:
        raise ValueError("at least two relation results are required")

    rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, float]] = {}
    for source in results:
        source_relation = str(source["relation"])
        source_pairs = source["pairs"]
        matrix[source_relation] = {}
        for target in results:
            target_relation = str(target["relation"])
            target_pairs = target["pairs"]
            common_keys = sorted(set(source_pairs) & set(target_pairs))
            source_values = [source_pairs[key]["gw_structure"] for key in common_keys]
            target_values = [target_pairs[key]["fgw"] for key in common_keys]
            value = _spearman(source_values, target_values)
            matrix[source_relation][target_relation] = value
            rows.append(
                {
                    "source_relation": source_relation,
                    "source_label": source["relation_label"],
                    "target_relation": target_relation,
                    "target_label": target["relation_label"],
                    "spearman": value,
                    "common_pair_count": len(common_keys),
                    "source_path": source["path"],
                    "target_path": target["path"],
                }
            )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_name(output_stem.name + "_long.csv")
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")

    json_path.write_text(
        json.dumps({"spearman_matrix": matrix, "rows": rows}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    heatmap_frame = frame.pivot(index="source_label", columns="target_label", values="spearman")
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="seaborn.matrix")
        sns.heatmap(
            heatmap_frame,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Spearman correlation"},
            ax=ax,
        )
    ax.set_title("Cross-relation transfer: structural GW vs target FGW")
    ax.set_xlabel("Target FGW relation")
    ax.set_ylabel("Source structural GW relation")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"json": json_path, "csv": csv_path, "png": png_path, "pdf": pdf_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze cross-relation transfer for raw-AST FGW results.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-stem", type=Path, default=Path("figures/raw_ast_fgw_cross_relation_transfer"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = analyze_cross_relation_transfer(tuple(args.input), args.output_stem)
    print(json.dumps({key: str(value) for key, value in written.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
