from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.raw_ast_order_probe import (
    encode_order_probe_split,
    load_order_probe_records,
    run_order_probe,
    split_order_probe_records,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _summaries(results: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    grouped: dict[tuple[str, int], list[dict[str, int | float | str]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["model"]), int(result["dim"]))].append(result)
    rows: list[dict[str, int | float | str]] = []
    for (model, dim), items in sorted(grouped.items()):
        auc_values = [float(item["eval_auc"]) for item in items]
        accuracy_values = [float(item["eval_accuracy"]) for item in items]
        pos_energy = [float(item["positive_energy_mean"]) for item in items]
        neg_energy = [float(item["negative_energy_mean"]) for item in items]
        rows.append(
            {
                "model": model,
                "dim": dim,
                "seeds": len(items),
                "eval_auc_mean": mean(auc_values),
                "eval_auc_sd": stdev(auc_values) if len(auc_values) > 1 else 0.0,
                "eval_accuracy_mean": mean(accuracy_values),
                "eval_accuracy_sd": stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0,
                "positive_energy_mean": mean(pos_energy),
                "negative_energy_mean": mean(neg_energy),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a raw-AST partial-order probe with Poincare entailment cones and Euclidean order controls.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/java_raw_ast_order_relations_validation_callable_sample_100.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_ast_order_probe_results.json"))
    parser.add_argument("--dims", type=str, default="2,4,8")
    parser.add_argument("--seeds", type=str, default="101,202,303")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--cone-k", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_order_probe_records(args.input)
    split = split_order_probe_records(records)
    encoded = encode_order_probe_split(split)
    dims = _parse_ints(args.dims)
    seeds = _parse_ints(args.seeds)
    results = []
    for dim in dims:
        for seed in seeds:
            for model in ("poincare_cone", "euclidean_order"):
                result = run_order_probe(
                    encoded,
                    model=model,
                    dim=dim,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    margin=args.margin,
                    curvature=args.curvature,
                    cone_k=args.cone_k,
                )
                results.append(result.as_dict())
                print(
                    f"{model} dim={dim} seed={seed}: "
                    f"AUC={result.eval_auc:.4f}, acc={result.eval_accuracy:.4f}"
                )

    payload = {
        "input": str(args.input),
        "record_count": len(records),
        "node_count": encoded.node_count,
        "train_positive_count": len(split.train_positive),
        "train_negative_count": len(split.train_negative),
        "eval_positive_count": len(split.eval_positive),
        "eval_negative_count": len(split.eval_negative),
        "config": {
            "dims": dims,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "margin": args.margin,
            "curvature": args.curvature,
            "cone_k": args.cone_k,
        },
        "runs": results,
        "summary": _summaries(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
