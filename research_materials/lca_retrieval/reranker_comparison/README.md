# Fixed ancestor masking and Qwen3-Reranker comparison

Download the numerical ZIP from the release
[lca-reranker-comparison-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-reranker-comparison-materials-v1).
The scripts in this directory are also included in that standalone ZIP.
No manuscript, extended report, raw program bodies or model weights are included.

## Offline reproduction

Extract the archive and run from its directory, with Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_reproduction.py
.venv/bin/python verify_results.py
```

This checks the manifest, computes dense scores from saved CodeRankEmbed vectors,
reconstructs retained/masked square-root AST features from counts, reconstructs
BM25 and its fixed fusion, reorders the same top-100 candidates using saved Qwen
pair scores, and recomputes 16,000 rankings and problem-level bootstrap intervals.
It checks both session records and the byte-identical 2,091-query resume prefix.
Results are written under `reproduced/`, not over frozen references.

The cohort has 400 problems, eight queries and eight gallery programs per problem.
Relevance means accepted solutions of the same problem; it is not independently
established semantic equivalence. These additional comparisons reuse an inspected
cohort. They are exploratory; no model, weight or method was selected using their
results. All contrast signs are retained.

MAP@8: CodeRankEmbed 0.30383; dense+BM25 0.31033; structural 0.33308;
masked structural 0.32915; Qwen3-Reranker-0.6B 0.57796. The structural method is
substantially worse than the learned reranker on the same candidates.

Neural inference is **not** replayed by this offline verifier. `source/` preserves
the actual fixed inference script and launcher, without corpus data. The original
corpora and pinned model are required to repeat parsing/inference. Private session
identifiers are provenance only, not a required reviewer account dependency.
All 3,200 query-processing times include both sessions; loading is per-session,
and first-session peak GPU memory was not recorded. M3 CPU ranking-only timings
from the earlier study cannot be divided by T4 pair-inference time to infer an
end-to-end speedup.

## Existing figures

The three unchanged numerical figures and their standalone reproduction are in
[lca-external-transfer-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-external-transfer-materials-v1).
Use that archive's `verify_results.py` and `reproduce_figures.py`; the learned
reranker is added as a table row, not as a hardware-confounded timing curve.

See `PROVENANCE.json` for original artifact hashes and numerical-export scope.
Source programs are not redistributed; this package is an export of numerical
research artifacts and reproduction instructions under the repository terms.
