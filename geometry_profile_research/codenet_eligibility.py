from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import platform
import sys
import tokenize
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasketch import MinHash


SCHEMA_VERSION = "codenet-python800-eligibility-v1"


@dataclass(frozen=True)
class CanonicalSource:
    text: str
    encoding: str
    decode_ok: bool
    decode_error: str | None


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_python_source(raw: bytes) -> CanonicalSource:
    """Decode a Python file and apply the preregistered D0 normalization."""

    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding, errors="strict")
    except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
        return CanonicalSource(
            text="",
            encoding="unknown",
            decode_ok=False,
            decode_error=type(exc).__name__,
        )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    normalized = "\n".join(lines).rstrip("\n")
    if normalized:
        normalized += "\n"
    return CanonicalSource(
        text=normalized,
        encoding=encoding,
        decode_ok=True,
        decode_error=None,
    )


def lexical_token_stream(source: str) -> tuple[str, ...]:
    """Return D1 tokens with comments and formatting removed."""

    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    tokens: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        if token.type == tokenize.ERRORTOKEN and token.string.isspace():
            continue
        tokens.append(f"{tokenize.tok_name[token.type]}:{token.string}")
    return tuple(tokens)


def token_ngrams(tokens: Sequence[str], *, width: int = 5) -> frozenset[tuple[str, ...]]:
    if width <= 0:
        raise ValueError("width must be positive")
    if len(tokens) < width:
        return frozenset()
    return frozenset(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def exact_jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left and not right:
        raise ValueError("Jaccard is undefined for two empty shingle sets")
    return len(left & right) / len(left | right)


def minhash_signature(
    shingles: Iterable[tuple[str, ...]],
    *,
    num_perm: int = 256,
    seed: int = 20260711,
) -> np.ndarray:
    encoded = ["\x1f".join(shingle).encode("utf-8") for shingle in shingles]
    if not encoded:
        raise ValueError("cannot create a MinHash signature for an empty shingle set")
    signature = MinHash(num_perm=num_perm, seed=seed, scheme="affine32")
    signature.update_batch(encoded)
    return np.asarray(signature.hashvalues, dtype=np.uint32)


class _AlphaAstCanonicalizer(ast.NodeTransformer):
    """Canonicalize identifiers by first occurrence and literals by type.

    The transform is deliberately syntax-only. It does not infer types or use
    task labels, so its output can be frozen before any benchmark split.
    """

    def __init__(self) -> None:
        self.identifiers: dict[str, str] = {}
        self.attributes: dict[str, str] = {}
        self.imports: dict[str, str] = {}

    def _identifier(self, value: str) -> str:
        if keyword.iskeyword(value):
            return value
        if value not in self.identifiers:
            self.identifiers[value] = f"ID_{len(self.identifiers)}"
        return self.identifiers[value]

    def _attribute(self, value: str) -> str:
        if value not in self.attributes:
            self.attributes[value] = f"ATTR_{len(self.attributes)}"
        return self.attributes[value]

    def _import(self, value: str) -> str:
        if value not in self.imports:
            self.imports[value] = f"IMPORT_{len(self.imports)}"
        return self.imports[value]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._identifier(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(
            ast.arg(
                arg=self._identifier(node.arg),
                annotation=self.visit(node.annotation) if node.annotation is not None else None,
                type_comment=None,
            ),
            node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr=self._attribute(node.attr),
                ctx=node.ctx,
            ),
            node,
        )

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        return ast.copy_location(
            ast.keyword(
                arg=self._attribute(node.arg) if node.arg is not None else None,
                value=self.visit(node.value),
            ),
            node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._identifier(node.name)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        return ast.copy_location(
            ast.alias(
                name=self._import(node.name),
                asname=self._identifier(node.asname) if node.asname is not None else None,
            ),
            node,
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value=_literal_placeholder(node.value)), node)


def alpha_normalized_ast(source: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, type_comments=True)
    return alpha_normalized_ast_tree(tree)


def alpha_normalized_ast_tree(tree: ast.AST) -> str:
    canonical = _AlphaAstCanonicalizer().visit(tree)
    ast.fix_missing_locations(canonical)
    return ast.dump(canonical, annotate_fields=True, include_attributes=False)


def _literal_placeholder(value: object) -> str:
    if value is None:
        return "<NONE>"
    if isinstance(value, bool):
        return "<BOOL>"
    if isinstance(value, bytes):
        return "<BYTES>"
    if isinstance(value, str):
        return "<STRING>"
    if isinstance(value, int):
        return "<INTEGER>"
    if isinstance(value, float):
        return "<FLOAT>"
    if isinstance(value, complex):
        return "<COMPLEX>"
    if value is Ellipsis:
        return "<ELLIPSIS>"
    return f"<{type(value).__name__.upper()}>"


def stable_sha256(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def analyze_python_file(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    if len(relative.parts) != 2:
        raise ValueError(f"expected <problem>/<submission>.py, got {relative}")
    raw = path.read_bytes()
    canonical = normalize_python_source(raw)
    record: dict[str, Any] = {
        "problem_id": relative.parts[0],
        "submission_id": path.stem,
        "source_relpath": relative.as_posix(),
        "source_bytes": len(raw),
        "normalized_bytes": len(canonical.text.encode("utf-8")),
        "encoding": canonical.encoding,
        "decode_ok": canonical.decode_ok,
        "decode_error": canonical.decode_error,
        "tokenize_ok": False,
        "tokenize_error": None,
        "parse_ok": False,
        "parse_error": None,
        "token_count": 0,
        "ast_node_count": 0,
        "d0_sha256": stable_sha256(canonical.text) if canonical.decode_ok else None,
        "d1_sha256": None,
        "d2_sha256": None,
    }
    if not canonical.decode_ok:
        return record
    try:
        tokens = lexical_token_stream(canonical.text)
        record["tokenize_ok"] = True
        record["token_count"] = len(tokens)
        record["d1_sha256"] = stable_sha256("\x1f".join(tokens))
    except (IndentationError, SyntaxError, tokenize.TokenError) as exc:
        record["tokenize_error"] = type(exc).__name__
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(canonical.text, type_comments=True)
        record["parse_ok"] = True
        record["ast_node_count"] = sum(1 for _ in ast.walk(tree))
        record["d2_sha256"] = stable_sha256(alpha_normalized_ast_tree(tree))
    except (IndentationError, SyntaxError, TabError, ValueError) as exc:
        record["parse_error"] = type(exc).__name__
    return record


def build_exact_duplicate_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_cluster_programs: int = 64,
    d4_min_shared_d2: int = 5,
    d4_min_fraction: float = 0.05,
) -> dict[str, Any]:
    """Build global D0-D2 components and preliminary D4 problem clusters."""

    dsu = DisjointSet(len(records))
    first_by_level: dict[str, dict[str, int]] = {
        "D0": {},
        "D1": {},
        "D2": {},
    }
    hash_key = {"D0": "d0_sha256", "D1": "d1_sha256", "D2": "d2_sha256"}
    for index, record in enumerate(records):
        for level, key in hash_key.items():
            digest = record.get(key)
            if not digest:
                continue
            previous = first_by_level[level].setdefault(str(digest), index)
            if previous != index:
                dsu.union(previous, index)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members_by_root[dsu.find(index)].append(index)

    canonical_by_index: dict[int, int] = {}
    duplicate_components: list[dict[str, Any]] = []
    for members in members_by_root.values():
        canonical = min(members, key=lambda item: str(records[item]["source_relpath"]))
        component_payload = "\n".join(sorted(str(records[item]["source_relpath"]) for item in members))
        component_id = f"exact-{stable_sha256(component_payload)[:20]}"
        for index in members:
            canonical_by_index[index] = canonical
        if len(members) <= 1:
            continue
        levels = []
        for level, key in hash_key.items():
            digest_counts = Counter(records[index].get(key) for index in members if records[index].get(key))
            if any(count >= 2 for count in digest_counts.values()):
                levels.append(level)
        duplicate_components.append(
            {
                "component_id": component_id,
                "size": len(members),
                "levels": levels,
                "canonical_source_relpath": records[canonical]["source_relpath"],
                "problem_ids": sorted({str(records[index]["problem_id"]) for index in members}),
                "members": sorted(str(records[index]["source_relpath"]) for index in members),
            }
        )

    retained_by_problem: Counter[str] = Counter()
    problem_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for index, record in enumerate(records):
        problem = str(record["problem_id"])
        stats = problem_stats[problem]
        stats["source_files"] += 1
        stats["decode_ok"] += int(bool(record.get("decode_ok")))
        stats["tokenize_ok"] += int(bool(record.get("tokenize_ok")))
        stats["parse_ok"] += int(bool(record.get("parse_ok")))
        retained = bool(record.get("parse_ok")) and canonical_by_index[index] == index
        if retained:
            retained_by_problem[problem] += 1
        elif record.get("parse_ok"):
            stats["exact_duplicates_removed"] += 1

    d2_problems: dict[str, set[str]] = defaultdict(set)
    unique_d2_by_problem: dict[str, set[str]] = defaultdict(set)
    for record in records:
        digest = record.get("d2_sha256")
        if digest:
            problem = str(record["problem_id"])
            d2_problems[str(digest)].add(problem)
            unique_d2_by_problem[problem].add(str(digest))
    shared_d2: Counter[tuple[str, str]] = Counter()
    for problems in d2_problems.values():
        ordered = sorted(problems)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                shared_d2[(left, right)] += 1

    problem_ids = sorted(problem_stats)
    problem_index = {problem: index for index, problem in enumerate(problem_ids)}
    problem_dsu = DisjointSet(len(problem_ids))
    d4_edges: list[dict[str, Any]] = []
    for (left, right), shared_count in sorted(shared_d2.items()):
        denominator = min(len(unique_d2_by_problem[left]), len(unique_d2_by_problem[right]))
        fraction = shared_count / denominator if denominator else 0.0
        if shared_count < d4_min_shared_d2 or fraction < d4_min_fraction:
            continue
        problem_dsu.union(problem_index[left], problem_index[right])
        d4_edges.append(
            {
                "left_problem_id": left,
                "right_problem_id": right,
                "shared_d2_programs": shared_count,
                "fraction_of_smaller_problem": fraction,
                "rule": "shared_D2",
            }
        )

    problems_by_cluster_root: dict[int, list[str]] = defaultdict(list)
    for problem in problem_ids:
        problems_by_cluster_root[problem_dsu.find(problem_index[problem])].append(problem)
    problem_clusters: list[dict[str, Any]] = []
    for problems in problems_by_cluster_root.values():
        ordered = sorted(problems)
        cluster_id = f"problem-{stable_sha256('|'.join(ordered))[:20]}"
        retained = sum(retained_by_problem[problem] for problem in ordered)
        problem_clusters.append(
            {
                "cluster_id": cluster_id,
                "problem_ids": ordered,
                "problem_count": len(ordered),
                "retained_programs_after_d0_d2": retained,
                "eligible_minimum_64": retained >= minimum_cluster_programs,
            }
        )
    problem_clusters.sort(key=lambda item: str(item["cluster_id"]))

    problem_summaries: list[dict[str, Any]] = []
    cluster_for_problem = {
        problem: str(cluster["cluster_id"])
        for cluster in problem_clusters
        for problem in cluster["problem_ids"]
    }
    for problem in problem_ids:
        stats = problem_stats[problem]
        source_files = stats["source_files"]
        parse_ok = stats["parse_ok"]
        retained = retained_by_problem[problem]
        problem_summaries.append(
            {
                "problem_id": problem,
                "problem_cluster_id": cluster_for_problem[problem],
                "source_files": source_files,
                "decode_ok": stats["decode_ok"],
                "tokenize_ok": stats["tokenize_ok"],
                "parse_ok": parse_ok,
                "parse_rate": parse_ok / source_files if source_files else 0.0,
                "exact_duplicates_removed": stats["exact_duplicates_removed"],
                "retained_programs_after_d0_d2": retained,
                "eligible_minimum_64": retained >= minimum_cluster_programs,
            }
        )

    parse_ok_total = sum(int(bool(record.get("parse_ok"))) for record in records)
    retained_total = sum(retained_by_problem.values())
    return {
        "canonical_index_by_record": canonical_by_index,
        "duplicate_components": sorted(duplicate_components, key=lambda item: str(item["component_id"])),
        "problem_summaries": problem_summaries,
        "d4_edges": d4_edges,
        "problem_clusters": problem_clusters,
        "summary": {
            "source_files": len(records),
            "parse_ok": parse_ok_total,
            "parse_rate": parse_ok_total / len(records) if records else 0.0,
            "retained_programs_after_d0_d2": retained_total,
            "exact_duplicates_removed": parse_ok_total - retained_total,
            "exact_duplicate_components": len(duplicate_components),
            "problem_count": len(problem_ids),
            "problem_cluster_count": len(problem_clusters),
            "multi_problem_cluster_count": sum(int(cluster["problem_count"] > 1) for cluster in problem_clusters),
            "eligible_problem_clusters_minimum_64": sum(
                int(bool(cluster["eligible_minimum_64"])) for cluster in problem_clusters
            ),
        },
    }


def environment_record() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "ast_module": ast.__name__,
        "tokenize_module": tokenize.__name__,
    }


def portable_manifest_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)
