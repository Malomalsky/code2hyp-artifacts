"""Portable clocks, completion boundaries and ranking for the fixed cost study."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ("D", "BM25", "D_BM25", "D_AST", "D_QwenReranker", "QwenEmbedding")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def select_queries(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["problem"]].append(i)
    return [min(groups[p], key=lambda i: (
        hashlib.sha256(("end-to-end-cost-20261014:" + rows[i]["id"]).encode()).hexdigest(),
        rows[i]["id"])) for p in sorted(groups)]


def order(scores, tie):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite retrieval scores")
    return np.lexsort((tie, -scores))


def rerank(dense_order, scores, tie):
    prefix = dense_order[:len(scores)]
    return np.concatenate((prefix[order(scores, tie[prefix])], dense_order[len(scores):]))


def metrics(ranking, relevant):
    hits = relevant[ranking]
    if hits.sum() != 8:
        raise ValueError("Expected eight relevant gallery items")
    first = int(np.flatnonzero(hits)[0]) + 1
    ap = np.sum(np.cumsum(hits[:8]) * hits[:8] / np.arange(1, 9)) / 8
    return np.array([ap, 1 / first, float(first <= 10)])


def validate_times(values, count):
    if values.shape != (3, count) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Require all three complete positive timing passes")


def summary(values):
    validate_times(values, values.shape[1])
    return {"mean_seconds": float(values.mean()), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, .95)),
            "pass_mean_seconds": values.mean(axis=1).tolist()}


def feature_bag(source, token_vocab):
    from geometry_profile_research.python_raw_ast import parse_python_ast_tree
    from geometry_profile_research.raw_ast import terminal_to_terminal_paths
    from frozen_helpers import path_keys, fragments
    from collections import Counter
    tree = parse_python_ast_tree(source)
    paths = terminal_to_terminal_paths(tree, max_paths=64, selection_policy="lca_depth_affine_sampled")
    if not paths:
        raise ValueError("Frozen cohort requires nonempty paths")
    bag = Counter()
    for path in paths:
        retained, _, _ = path_keys(tree, path, token_vocab)
        bag.update(fragments(retained))
    return bag
