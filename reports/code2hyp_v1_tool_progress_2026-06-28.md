# Code2Hyp-v1 tool progress, 2026-06-28

## Reviewer direction implemented

The work has been moved from isolated experiments toward the two requested
products:

1. A representation claim built around object order:
   `AST node != AST path != method/program`.
2. A practical structural retrieval tool with `index`, `search`, `compare`,
   `explain` and `audit-geometry` commands.

The current implementation does not claim that negative curvature is the
primary effect. The primary representation is:

```text
AST node -> deterministic geometric point
AST path -> LCA-anchored product object (LCA, start, end)
program  -> finite probability measure over path objects
```

This follows the recommended formulation that a terminal-to-terminal AST path
should not be collapsed to a node-like single point and that a method/program
should not be collapsed to a centroid when structural motifs are multimodal.

## Implemented artifact

New public module:

```text
code2hyp/__init__.py
geometry_profile_research/code2hyp_tool.py
```

Public Python API:

```python
from code2hyp import Code2Hyp

model = Code2Hyp.load("code2hyp-v1")
index = model.index_directory("solutions/")
results = index.search("query.py", top_k=20)
explanation = model.explain_files("query.py", results[0].path)
```

CLI:

```bash
code2hyp index ./solutions --language python --output code2hyp-index.json
code2hyp search query.py --index code2hyp-index.json --top-k 20
code2hyp compare a.py b.py
code2hyp explain a.py b.py --top-k 10
code2hyp audit-geometry ./solutions
```

The implementation currently supports Python source files. It parses raw ASTs,
extracts deterministic terminal-to-terminal paths, builds LCA-product path
objects, stores a measure over these objects and compares programs by Sinkhorn
divergence over the product ground cost.

## Explainability

The `explain` command exposes the transport plan. For each selected path-pair it
returns:

```text
transport mass
local cost
query and candidate LCA node labels
query and candidate start/end terminal labels
query and candidate source spans
full query and candidate path descriptors
```

This is the practical distinction from ordinary embedding retrieval: Code2Hyp
can show which structural path motifs were matched between two programs.

## Geometry audit

The `audit-geometry` command reports:

```text
number of indexed entries
number of compared pairs
number of path objects
point-cost share
side-cost share
median positive full cost
```

This makes product-cost domination observable. If side-cost share dominates,
curvature effects cannot be interpreted as isolated geometric evidence. This
matches the reviewer recommendation to keep geometry diagnostics explicit.

## Current scientific status

The tool is a deterministic structural retrieval layer over the representation
developed in the experiments. It is not a newly trained neural model and should
not be described as a large-scale SOTA result.

Supported claim at this stage:

```text
Code2Hyp-v1 operationalizes the object-order-aware representation:
node as point, path as LCA-product object, program as measure over paths.
```

Claims that still require confirmatory validation:

```text
measure > centroid
LCA-product > single-point
active curvature > near-zero curvature
improvement on external CodeNet or POJ-style retrieval
hard-negative structural discrimination
```

The already implemented train-normalized product-cost mode should remain the
canonical metric for confirmatory runs, because it prevents side-feature
domination from hiding the geometric LCA/start/end channel.

## Verification

New tests:

```text
tests/test_code2hyp_tool.py
tests/test_code2hyp_cli.py
```

Relevant regression command:

```bash
.venv/bin/python -m pytest \
  tests/test_code2hyp_tool.py \
  tests/test_code2hyp_cli.py \
  tests/test_python_raw_ast_extractor.py \
  tests/test_raw_ast_geometry.py \
  tests/test_raw_ast_retrieval.py \
  tests/test_constant_curvature.py \
  tests/test_lca_path_measure.py \
  tests/test_gromov_wasserstein.py
```

Observed result:

```text
55 passed
```

Full project regression:

```text
.venv/bin/python -m pytest -q
502 passed, 3 subtests passed
```

## Next work

The next implementation step should not be another ad hoc B-variant. It should
extend the same tool and protocol in two directions:

1. Add Level C hard-negative evaluation with pairwise ranking accuracy:
   `Pr[D(query, positive) < D(query, hard_negative)]`.
2. Add an external CodeNet or POJ subset as a frozen confirmatory benchmark with
   task-level uncertainty as the unit of statistical inference.
