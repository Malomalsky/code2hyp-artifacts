from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import javalang

from geometry_profile_research.codenet_eligibility import (
    build_exact_duplicate_audit,
    canonical_json_bytes,
    jsonl_bytes,
    normalize_java_source,
    stable_sha256,
)
from geometry_profile_research.java_raw_ast import parse_java_ast_tree
from geometry_profile_research.raw_ast import RawAstTree


_PRESERVED_SYNTAX_EDGES = {"operator", "prefix_operators", "postfix_operators", "modifiers", "kind"}


def analyze_java_source(raw: bytes, *, source_relpath: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_relpath": source_relpath,
        "source_bytes": len(raw),
        "decode_ok": False,
        "decode_error": None,
        "tokenize_ok": False,
        "tokenize_error": None,
        "parse_ok": False,
        "parse_error": None,
        "token_count": 0,
        "ast_node_count": 0,
        "d0_sha256": None,
        "d1_sha256": None,
        "d2_sha256": None,
    }
    canonical = normalize_java_source(raw)
    if not canonical.decode_ok:
        record["decode_error"] = canonical.decode_error
        return record
    source = canonical.text
    record["decode_ok"] = True
    record["d0_sha256"] = stable_sha256(source)
    try:
        tokens = tuple(f"{type(token).__name__}:{token.value}" for token in javalang.tokenizer.tokenize(source))
        record["tokenize_ok"] = True
        record["token_count"] = len(tokens)
        record["d1_sha256"] = stable_sha256("\x1f".join(tokens))
    except (javalang.tokenizer.LexerError, TypeError) as error:
        record["tokenize_error"] = type(error).__name__
    try:
        tree = parse_java_ast_tree(source)
        record["parse_ok"] = True
        record["ast_node_count"] = len(tree.labels)
        record["d2_sha256"] = stable_sha256(_canonical_raw_ast_signature(tree))
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError, TypeError, IndexError) as error:
        record["parse_error"] = type(error).__name__
    return record


def audit_candidates(
    *,
    candidate_archive: Path,
    candidate_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("schema_version") != "code2hyp-codenet-java-stage-b-candidates-v1":
        raise ValueError("unsupported Java Stage B candidate manifest")
    metadata_by_member = {
        str(row["candidate_archive_member"]): row
        for row in candidate_manifest["inventory"]
    }
    records = []
    seen = set()
    with tarfile.open(candidate_archive) as archive:
        for member in archive:
            if not member.isfile():
                continue
            metadata = metadata_by_member.get(member.name)
            if metadata is None:
                raise ValueError(f"candidate archive contains an unregistered member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read candidate member {member.name}")
            record = analyze_java_source(stream.read(), source_relpath=member.name)
            record.update(
                {
                    "problem_id": str(metadata["component_id"]),
                    "original_problem_id": str(metadata["problem_id"]),
                    "submission_id": str(metadata["submission_id"]),
                    "user_id": str(metadata["user_id"]),
                }
            )
            records.append(record)
            seen.add(member.name)
    missing = sorted(set(metadata_by_member) - seen)
    if missing:
        raise ValueError(f"candidate archive is missing {len(missing)} manifest members; first={missing[0]}")
    records.sort(key=lambda row: row["source_relpath"])

    audit = build_exact_duplicate_audit(records, minimum_cluster_programs=64)
    canonical_index = audit.pop("canonical_index_by_record")
    for index, record in enumerate(records):
        canonical = int(canonical_index[index])
        record["retained_after_d0_d2"] = bool(record["parse_ok"]) and canonical == index
        record["canonical_source_relpath"] = records[canonical]["source_relpath"]

    final_cluster_by_component = {
        component_id: str(cluster["cluster_id"])
        for cluster in audit["problem_clusters"]
        for component_id in cluster["problem_ids"]
    }
    users_by_cluster: dict[str, set[str]] = defaultdict(set)
    retained_by_cluster: dict[str, int] = defaultdict(int)
    for record in records:
        if not record["retained_after_d0_d2"]:
            continue
        cluster_id = final_cluster_by_component[str(record["problem_id"])]
        users_by_cluster[cluster_id].add(str(record["user_id"]))
        retained_by_cluster[cluster_id] += 1

    final_clusters = []
    for cluster in audit["problem_clusters"]:
        cluster_id = str(cluster["cluster_id"])
        retained = retained_by_cluster[cluster_id]
        users = len(users_by_cluster[cluster_id])
        final_clusters.append(
            {
                **cluster,
                "retained_programs_after_d0_d2": retained,
                "distinct_users_after_d0_d2": users,
                "eligible_evaluation_minimum_16": retained >= 16 and users >= 16,
                "eligible_train_minimum_64": retained >= 64 and users >= 16,
            }
        )

    summary = {
        **audit["summary"],
        "eligible_evaluation_clusters_minimum_16_users_16": sum(
            cluster["eligible_evaluation_minimum_16"] for cluster in final_clusters
        ),
        "eligible_train_clusters_minimum_64_users_16": sum(
            cluster["eligible_train_minimum_64"] for cluster in final_clusters
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "file_inventory.jsonl": jsonl_bytes(records),
        "preliminary_problem_clusters.jsonl": jsonl_bytes(final_clusters),
    }
    output_records = []
    for filename, content in outputs.items():
        path = output_dir / filename
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different Java eligibility artifact: {path}")
        path.write_bytes(content)
        output_records.append({"path": filename, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})

    manifest = {
        "schema_version": "code2hyp-codenet-java-stage-b-d0-d2-v1",
        "status": "preliminary_before_D3_and_statement_D4",
        "inputs": {
            "candidate_archive_sha256": _sha256_file(candidate_archive),
            "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        },
        "protocol": {
            "D0": "strict UTF-8; normalized line endings and trailing horizontal whitespace",
            "D1": "exact javalang token classes and values without comments or formatting",
            "D2": "ordered raw-AST structure; identifiers alpha-normalized by first occurrence; literals typed; operators and primitive types preserved",
            "D3": "pending MinHash candidate generation and exact token-shingle verification",
            "D4": "preliminary shared-D2 rule within the independent Java frame; statement equivalence against Stage A remains pending",
        },
        "summary": summary,
        "preliminary_d4_edges": audit["d4_edges"],
        "artifacts": output_records,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError(f"refusing to overwrite a different Java eligibility manifest: {manifest_path}")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


def _canonical_raw_ast_signature(tree: RawAstTree) -> str:
    identifiers: dict[str, str] = {}
    depths = {tree.root_id: 0}
    rows = []
    for node in tree.preorder():
        parent = tree.parent(node)
        if parent is not None:
            depths[node] = depths[parent] + 1
        attributes = tree.attributes.get(node, {})
        edge_type = str(attributes.get("edge_type", "root"))
        child_index = str(attributes.get("child_index", "0"))
        label = str(tree.labels.get(node, ""))
        terminal = ""
        if label == "TerminalToken":
            raw_value = str(attributes.get("terminal_token", ""))
            parent_label = str(tree.labels.get(parent, "")) if parent is not None else ""
            terminal = _canonical_terminal(raw_value, parent_label, edge_type, identifiers)
        rows.append(f"{depths[node]}\x1f{edge_type}\x1f{child_index}\x1f{label}\x1f{terminal}")
    return "\x1e".join(rows)


def _canonical_terminal(value: str, parent_label: str, edge_type: str, identifiers: dict[str, str]) -> str:
    if parent_label == "Literal" and edge_type == "value":
        return _literal_class(value)
    if edge_type in _PRESERVED_SYNTAX_EDGES or (parent_label == "BasicType" and edge_type == "name"):
        return f"syntax:{value}"
    return identifiers.setdefault(value, f"ID_{len(identifiers)}")


def _literal_class(value: str) -> str:
    lowered = value.lower()
    if lowered == "null":
        return "<NULL>"
    if lowered in {"true", "false"}:
        return "<BOOL>"
    if value.startswith("\""):
        return "<STRING>"
    if value.startswith("'"):
        return "<CHAR>"
    if re.search(r"[.eEpP]", value):
        return "<FLOAT>"
    return "<INTEGER>"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Java Stage B parse and exact D0-D2 eligibility gates.")
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_candidates(
        candidate_archive=args.candidate_archive,
        candidate_manifest_path=args.candidate_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
