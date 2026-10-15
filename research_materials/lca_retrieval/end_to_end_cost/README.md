# Retrieval quality and complete query-processing costs

Download the standalone numerical archive from
[lca-end-to-end-cost-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-end-to-end-cost-materials-v1).
This directory contains the same code and instructions; large numerical
files are release assets, not Git objects. No manuscript, editorial response,
declaration, original program bodies, credentials or pretrained weights are included.

## Reproduce the numerical results

Use Python 3.11 or newer, extract the archive, and run in its root directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_reproduction.py
.venv/bin/python verify_results.py
.venv/bin/python reproduce_figures.py
```

The verifier checks file and model-identity bindings, rebuilds BM25 from
term counts, reconstructs square-root-weighted AST similarities and dense
scores from saved embeddings, and reranks fixed candidates using saved Qwen
pair scores. It recomputes 22,400 query-method evaluations across both
sessions and checks all 8,400 raw timing records and their ranked results.
Full-cohort results use 3,200 queries; timing uses 400 prespecified queries
in three complete passes. Metrics are MAP@8, MRR and Success@10. Relevance
is the shared contest problem, not independently proved semantic equivalence.

Outputs are written to `reproduced/`, never over original measurements:

- `result.json`: separate-session quality, mean/median/p95 latency,
  pass means, startup, memory and projected workload costs.
- `cost_table.csv`: all measured systems with explicit session labels.
- `figure-1.tex`: 400-problem quality comparison and all paired differences.
- `figure-2.tex`: session-A quality/latency points and p95 endpoints.
- `figure-3.tex`: unchanged crossed-corpus transfer estimates and intervals.

Compile the standalone figure sources with XeLaTeX, for example:

```sh
xelatex -interaction=nonstopmode -halt-on-error -output-directory=reproduced reproduced/figure-1.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=reproduced reproduced/figure-2.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=reproduced reproduced/figure-3.tex
```

The figure generators reproduce the current numerical diagrams; the primary
problem-bootstrap interval is checked again. Transfer intervals are checked
against the previously verified summary. Their full independent numerical
replay remains in the immutable
[external-transfer release](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-external-transfer-materials-v1).
Earlier ablation and reranker comparisons retain their existing releases.
No manuscript text is required by these commands.

## Two sessions, not a pooled six-method timing experiment

Session A is the original private Kaggle T4 run. It measured BM25,
CodeRankEmbed+BM25, CodeRankEmbed+AST, Qwen3-Embedding and
CodeRankEmbed+Qwen3-Reranker on the same VM. All five timing files have
400 x 3 requests. Four full-quality workers completed; the reranker
quality replay stopped at an overly strict candidate-order assertion,
after its timing had finished. Dense-only never started. This history
and the absence of a global session-A completion receipt are preserved.

The author approved a narrower continuation rather than repeating all five
completed timing experiments. Session B measures CodeRankEmbed alone and
CodeRankEmbed+AST as a paired control on another free T4 VM. The original
method, weights, code, precision, batch settings, 400 query IDs and three
query orders are unchanged. The two workers run in their original relative
order (AST hybrid, then dense-only). Each has its own preparation and
memory records. Compare D versus D+AST within B; compare the original five
systems within A. Do not pool, rescale or equate timings across the two VMs.
The original six-method single-VM design was not completed.

Session B also replays full quality for all 3,200 reranker queries, with
each saved pair score attached to its original candidate ID. This fixes
an evaluation bookkeeping error, not the model: recomputed float32 dense
scores can permute nearly tied candidates. The replay reports any changes
in their order or membership and retains the historical fixed candidate
set. It is not new Qwen inference or retroactive completion of session A.

## What the clocks mean

Warm request time starts with an in-memory raw query and ends with ranked
IDs. It includes query tokenization, encoding, feature extraction, retrieval
and reranking as applicable, with CUDA synchronization. It does not reuse
cached query representations or pair scores. Network serving, installation,
evaluation and checkpoint writes are outside this clock. Gallery preparation,
loading, warmup and serialized-index size are recorded separately; downloads
are excluded from the reported ready-startup time.

Memory distinguishes process RSS, allocated CUDA memory and reserved CUDA
memory. Reranker memory includes its first-stage encoder. Predictions for
1/100/1,000/10,000 queries are startup plus measured mean latency, not new
workload experiments. These are sequential single-query measurements on a
3,200-program gallery, not throughput, energy or monetary-cost measurements.
The numerical replay verifies measurements; it cannot reproduce a particular
wall-clock duration on different hardware.

All outcome signs are retained. The structural hybrid improves the tested
CodeRankEmbed+BM25 configuration, but Qwen3-Embedding has substantially higher
retrieval quality at a higher latency and memory footprint. Neither broad
deployment superiority nor worldwide novelty follows from these costs.

## Restore sources for a new inference run

The optional reconstruction command requires `pyarrow`, the six original
CodeContests training shards listed in `data/input/original_selection_design.json`,
and enough disk space. Obtain them under the original dataset's terms from
[revision 802411c3010cb00d1b05bad57ca77365a3c699d6](https://huggingface.co/datasets/deepmind/code_contests/tree/802411c3010cb00d1b05bad57ca77365a3c699d6/data).
The command makes no network request or GPU launch:

```sh
.venv/bin/python -m pip install pyarrow==23.0.1
.venv/bin/python reconstruct_input.py --parquet-dir /path/to/shards --output /new/path/to/payload
```

It verifies every shard, applies the original source normalization, retrieves
the 6,400 already selected programs by SHA-256 and requires byte-identical
`programs.json` and original payload hashes. It does not reselect problems or
optimize parameters. Missing sources stop reconstruction rather than being replaced.

`source/` preserves the actual frozen benchmark, model code, protocols,
historical evaluator, continuation launcher and focused tests. The historical
six-method controller retains its documented failed assertion; it is not
silently patched in this archive. The continuation launcher is the executed
route for D/D+AST plus corrected cached reranker replay. New inference or
timing requires the reconstructed payload, original model revisions, a
Kaggle T4 environment and sufficient quota. Private kernel identifiers are
provenance, not a requirement to access the numerical archive.

The original source/model licenses remain applicable; this numerical export
does not grant rights to upstream source programs or weights. CodeRankEmbed,
Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B are credited in the recorded
model settings. Ivan Aleksandrovich Kosyanenko, MIREA Russian Technological
University; ORCID https://orcid.org/0009-0009-1804-9412. No external funding.
