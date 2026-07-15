from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from datasketch import MinHash, MinHashLSH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_profile_research.codenet_eligibility import (
    DisjointSet,
    canonical_json_bytes,
    exact_jaccard,
    jsonl_bytes,
    lexical_token_stream,
    minhash_signature,
    normalize_python_source,
    portable_manifest_path,
    stable_sha256,
    token_ngrams,
)


D3_SCHEMA_VERSION = "codenet-python800-d3-v1"


def _signature_worker(payload: tuple[str, str, int, int]) -> tuple[str, int, np.ndarray | None]:
    relative_path, root_value, num_perm, seed = payload
    root = Path(root_value)
    canonical = normalize_python_source((root / relative_path).read_bytes())
    if not canonical.decode_ok:
        return relative_path, 0, None
    tokens = lexical_token_stream(canonical.text)
    shingles = token_ngrams(tokens, width=5)
    if not shingles:
        return relative_path, 0, None
    return relative_path, len(shingles), minhash_signature(shingles, num_perm=num_perm, seed=seed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_and_hash(path: Path, content: bytes) -> dict[str, Any]:
    path.write_bytes(content)
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_d3_artifacts(
    *,
    input_root: Path,
    d0_d2_dir: Path,
    output_dir: Path,
    workers: int,
    num_perm: int = 256,
    bands: int = 32,
    rows_per_band: int = 8,
    minhash_seed: int = 20260711,
    jaccard_threshold: float = 0.90,
    minimum_cluster_programs: int = 64,
    max_records: int | None = None,
) -> dict[str, Any]:
    if bands * rows_per_band != num_perm:
        raise ValueError("bands * rows_per_band must equal num_perm")
    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    eligibility_manifest_path = d0_d2_dir / "eligibility_manifest.json"
    eligibility_manifest_sha = stable_sha256(eligibility_manifest_path.read_bytes())
    inventory: list[dict[str, Any]] = []
    with (d0_d2_dir / "file_inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("retained_after_d0_d2"):
                inventory.append(
                    {
                        "problem_id": str(record["problem_id"]),
                        "submission_id": str(record["submission_id"]),
                        "source_relpath": str(record["source_relpath"]),
                    }
                )
    inventory.sort(key=lambda item: item["source_relpath"])
    if max_records is not None:
        inventory = inventory[:max_records]
    if not inventory:
        raise ValueError("D0-D2 inventory contains no retained programs")

    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = output_dir / "minhash_signatures.npy"
    shingle_count_path = output_dir / "shingle_counts.npy"
    signatures = np.lib.format.open_memmap(
        signature_path,
        mode="w+",
        dtype=np.uint32,
        shape=(len(inventory), num_perm),
    )
    shingle_counts = np.lib.format.open_memmap(
        shingle_count_path,
        mode="w+",
        dtype=np.uint32,
        shape=(len(inventory),),
    )
    payloads = [
        (record["source_relpath"], str(input_root), num_perm, minhash_seed)
        for record in inventory
    ]
    if workers == 1:
        iterator = map(_signature_worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_signature_worker, payloads, chunksize=64)
    try:
        for index, (relative_path, shingle_count, signature) in enumerate(iterator):
            if relative_path != inventory[index]["source_relpath"]:
                raise RuntimeError("signature worker changed deterministic inventory order")
            shingle_counts[index] = shingle_count
            if signature is None:
                signatures[index, :] = np.iinfo(np.uint32).max
            else:
                signatures[index, :] = signature
            if (index + 1) % 10_000 == 0:
                print(f"signatures {index + 1}/{len(inventory)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    signatures.flush()
    shingle_counts.flush()

    @lru_cache(maxsize=4096)
    def source_shingles(relative_path: str) -> frozenset[tuple[str, ...]]:
        canonical = normalize_python_source((input_root / relative_path).read_bytes())
        if not canonical.decode_ok:
            return frozenset()
        return token_ngrams(lexical_token_stream(canonical.text), width=5)

    lsh = MinHashLSH(
        threshold=jaccard_threshold,
        num_perm=num_perm,
        params=(bands, rows_per_band),
        prepickle=False,
    )
    dsu = DisjointSet(len(inventory))
    candidate_pairs = 0
    verified_edges: list[dict[str, Any]] = []
    max_candidates_for_query = 0
    for index, record in enumerate(inventory):
        if int(shingle_counts[index]) == 0:
            continue
        signature = MinHash(
            num_perm=num_perm,
            seed=minhash_seed,
            scheme="affine32",
            hashvalues=np.asarray(signatures[index], dtype=np.uint32),
        )
        candidates = sorted(int(candidate) for candidate in lsh.query(signature))
        max_candidates_for_query = max(max_candidates_for_query, len(candidates))
        if candidates:
            current_shingles = source_shingles(record["source_relpath"])
            for candidate in candidates:
                candidate_pairs += 1
                other = inventory[candidate]
                similarity = exact_jaccard(current_shingles, source_shingles(other["source_relpath"]))
                if similarity + 1e-15 < jaccard_threshold:
                    continue
                dsu.union(candidate, index)
                verified_edges.append(
                    {
                        "left_index": candidate,
                        "right_index": index,
                        "left_source_relpath": other["source_relpath"],
                        "right_source_relpath": record["source_relpath"],
                        "left_problem_id": other["problem_id"],
                        "right_problem_id": record["problem_id"],
                        "set_jaccard": similarity,
                    }
                )
        lsh.insert(index, signature)
        if (index + 1) % 10_000 == 0:
            print(
                f"lsh {index + 1}/{len(inventory)} candidates={candidate_pairs} edges={len(verified_edges)}",
                flush=True,
            )

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(inventory)):
        members_by_root[dsu.find(index)].append(index)
    canonical_by_index: dict[int, int] = {}
    d3_components: list[dict[str, Any]] = []
    for members in members_by_root.values():
        canonical = min(members, key=lambda item: inventory[item]["source_relpath"])
        for index in members:
            canonical_by_index[index] = canonical
        if len(members) <= 1:
            continue
        member_paths = sorted(inventory[index]["source_relpath"] for index in members)
        d3_components.append(
            {
                "component_id": f"d3-{stable_sha256(chr(10).join(member_paths))[:20]}",
                "size": len(members),
                "canonical_source_relpath": inventory[canonical]["source_relpath"],
                "problem_ids": sorted({inventory[index]["problem_id"] for index in members}),
                "members": member_paths,
            }
        )
    d3_components.sort(key=lambda item: item["component_id"])

    preliminary_clusters = _read_jsonl(d0_d2_dir / "preliminary_problem_clusters.jsonl")
    all_problem_ids = sorted({record["problem_id"] for record in inventory})
    problem_index = {problem: index for index, problem in enumerate(all_problem_ids)}
    problem_dsu = DisjointSet(len(all_problem_ids))
    for cluster in preliminary_clusters:
        problems = [problem for problem in cluster["problem_ids"] if problem in problem_index]
        for problem in problems[1:]:
            problem_dsu.union(problem_index[problems[0]], problem_index[problem])
    for component in d3_components:
        problems = component["problem_ids"]
        for problem in problems[1:]:
            problem_dsu.union(problem_index[problems[0]], problem_index[problem])

    retained_by_problem: Counter[str] = Counter()
    for index, record in enumerate(inventory):
        if canonical_by_index[index] == index:
            retained_by_problem[record["problem_id"]] += 1
    problems_by_root: dict[int, list[str]] = defaultdict(list)
    for problem in all_problem_ids:
        problems_by_root[problem_dsu.find(problem_index[problem])].append(problem)
    final_problem_clusters: list[dict[str, Any]] = []
    for problems in problems_by_root.values():
        ordered = sorted(problems)
        retained = sum(retained_by_problem[problem] for problem in ordered)
        final_problem_clusters.append(
            {
                "cluster_id": f"problem-{stable_sha256('|'.join(ordered))[:20]}",
                "problem_ids": ordered,
                "problem_count": len(ordered),
                "retained_programs_after_d0_d3": retained,
                "eligible_minimum_64": retained >= minimum_cluster_programs,
            }
        )
    final_problem_clusters.sort(key=lambda item: item["cluster_id"])

    index_rows = [
        {
            "index": index,
            **record,
            "shingle_count": int(shingle_counts[index]),
            "retained_after_d0_d3": canonical_by_index[index] == index,
            "d3_canonical_source_relpath": inventory[canonical_by_index[index]]["source_relpath"],
        }
        for index, record in enumerate(inventory)
    ]
    summary = {
        "d0_d2_representatives": len(inventory),
        "programs_with_at_least_one_token_5gram": int(np.count_nonzero(shingle_counts)),
        "programs_without_token_5grams": int(np.count_nonzero(shingle_counts == 0)),
        "lsh_candidate_pairs": candidate_pairs,
        "verified_d3_edges": len(verified_edges),
        "d3_duplicate_components": len(d3_components),
        "d3_duplicates_removed": sum(len(members) - 1 for members in members_by_root.values()),
        "retained_programs_after_d0_d3": sum(retained_by_problem.values()),
        "max_lsh_candidates_for_one_query": max_candidates_for_query,
        "problem_count": len(all_problem_ids),
        "problem_cluster_count_after_exact_d4_and_cross_problem_d3": len(final_problem_clusters),
        "eligible_problem_clusters_minimum_64": sum(
            int(cluster["eligible_minimum_64"]) for cluster in final_problem_clusters
        ),
    }

    artifact_records = [
        _write_and_hash(output_dir / "d3_index.jsonl", jsonl_bytes(index_rows)),
        _write_and_hash(output_dir / "d3_near_duplicate_edges.jsonl", jsonl_bytes(verified_edges)),
        _write_and_hash(output_dir / "d3_duplicate_components.jsonl", jsonl_bytes(d3_components)),
        _write_and_hash(output_dir / "post_d3_problem_clusters.jsonl", jsonl_bytes(final_problem_clusters)),
        _write_and_hash(output_dir / "d3_summary.json", canonical_json_bytes(summary)),
    ]
    for binary_path in (signature_path, shingle_count_path):
        artifact_records.append(
            {
                "path": binary_path.name,
                "bytes": binary_path.stat().st_size,
                "sha256": stable_sha256(binary_path.read_bytes()),
            }
        )

    manifest = {
        "schema_version": D3_SCHEMA_VERSION,
        "experiment_role": (
            "full_pre_split_D3_eligibility_without_retrieval_metrics"
            if max_records is None
            else "computational_pilot_not_for_eligibility"
        ),
        "input": {
            "source_root": portable_manifest_path(input_root, project_root=PROJECT_ROOT),
            "d0_d2_manifest": portable_manifest_path(
                eligibility_manifest_path,
                project_root=PROJECT_ROOT,
            ),
            "d0_d2_manifest_sha256": eligibility_manifest_sha,
        },
        "protocol": {
            "representation": "set of exact consecutive D1 token 5-grams",
            "num_perm": num_perm,
            "minhash_scheme": "affine32",
            "bands": bands,
            "rows_per_band": rows_per_band,
            "minhash_seed": minhash_seed,
            "lsh_role": "candidate_generation_only",
            "acceptance_rule": f"exact set-Jaccard >= {jaccard_threshold:.2f}",
            "short_program_rule": "fewer than five D1 tokens bypass D3 and retain D0-D2 status",
            "minimum_cluster_programs": minimum_cluster_programs,
            "split_status": "not_generated",
            "retrieval_metrics_opened": False,
            "official_D4_status": "pending_full_CodeNet_derived_metadata",
        },
        "summary": summary,
        "gate_precheck": {
            "at_least_764_eligible_clusters_for_300_100_364": (
                summary["eligible_problem_clusters_minimum_64"] >= 764
            ),
            "final_eligibility": "pending_official_D4",
        },
        "artifacts": sorted(artifact_records, key=lambda item: item["path"]),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (output_dir / "d3_manifest.json").write_bytes(manifest_bytes)
    manifest_sha = stable_sha256(manifest_bytes)
    (output_dir / "d3_manifest.sha256").write_text(
        f"{manifest_sha}  d3_manifest.json\n",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the exact-verified D3 audit for CodeNet Python800.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--d0-d2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--num-perm", type=int, default=256)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--rows-per-band", type=int, default=8)
    parser.add_argument("--minhash-seed", type=int, default=20260711)
    parser.add_argument("--jaccard-threshold", type=float, default=0.90)
    parser.add_argument("--minimum-cluster-programs", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_d3_artifacts(
        input_root=args.input_root,
        d0_d2_dir=args.d0_d2_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        num_perm=args.num_perm,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        minhash_seed=args.minhash_seed,
        jaccard_threshold=args.jaccard_threshold,
        minimum_cluster_programs=args.minimum_cluster_programs,
        max_records=args.max_records,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={args.output_dir / 'd3_manifest.json'}")


if __name__ == "__main__":
    main()
