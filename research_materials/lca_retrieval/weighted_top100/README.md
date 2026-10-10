# Weighted AST retrieval and candidate-limited reranking

Numerical reproduction package, 10 October 2026. Code, numerical data,
protocols and instructions only. No manuscript, article PDF, editorial
correspondence, original contest programs or pretrained model weights.

## Download and Run

Download `lca-weighted-top100-materials-v1.zip` and `SHA256SUMS.txt` from
[the versioned release](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-weighted-top100-materials-v1).
Alternatively, from a checkout of this repository:

```sh
python3 research_materials/lca_retrieval/download_materials.py --package weighted_top100 --output downloads
unzip downloads/weighted_top100/lca-weighted-top100-materials-v1.zip -d downloads/weighted_top100
cd downloads/weighted_top100/lca-weighted-top100-materials-v1
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_results.py
```

Use Python 3.12 with the pinned NumPy/SciPy versions. On Windows, extract
with the standard ZIP utility and use `.venv\Scripts\python.exe`. Do not use
`python -O`: the assertions are part of validation. No network, credentials,
GPU, model inference, training or original corpus is needed for numerical
replay after installing dependencies. Allow 4 GB RAM and 4 GB disk space,
including the archive, extraction and a temporary clean second extraction.
The complete verifier performs substantial CPU sorting; allow several minutes.
Original inputs are read-only. Outputs go into `reproduced/`.

## What Is Reproduced

| Evaluation | Cohort | Checks |
|---|---|---|
| Weighted structural comparison | 650 unused problems, 5,200 queries / gallery items | All 14 methods; every reconstructed score-matrix SHA-256; query AP@8, problem means, MRR, Success@10 and three paired intervals |
| Structural development summaries | 325 earlier development problems | Eight saved problem-by-weight grids; all five-fold choices and 12 descriptive contrasts |
| BM25 calibration | 325 earlier development problems | Every AP@8 at all 24 fixed weights; problem means and selection of alpha=0.10 |
| BM25 comparisons | 325 earlier validation problems; the later 650-problem cohort reused | Every saved top ten and all metrics for D, BM25, D+BM25; structural comparator AP@8 linked to the weighted replay |
| Candidate budgets | The same later 650 problems, exploratory | Nine rankings; K=50/100/500/full, first relevant ranks, top ten, candidate recall, coverage and gain retention |
| Top-100 validation | 400 unused problems, 3,200 queries / gallery items | All nine rankings, top ten, first relevant ranks, all metrics, candidate recall and the one primary paired interval |
| Runtime data | Five passes over 512 fixed queries | Every latency summary and component sum recomputed from measured samples |
| Three vector diagrams | The above results and immutable earlier references | Every coordinate, all 400 paired observations, histogram, cost/gain curves and earlier point estimates/intervals |

AP@8 divides by **eight relevant items**, including when fewer are retrieved
in the first eight ranks. MRR is computed from the first relevant item in the
complete ranking. Success@10 is the fraction with a relevant item in the
first ten. UTF-8 source IDs break score ties. Top-budget reranking changes
only the dense prefix and appends its unchanged tail. Eight queries belong
to each problem, so averaging problem means equals averaging query AP@8 here.

The portable verifier uses stable sorting after an independent source-ID sort
and a relevant-rank AP calculation rather than the original evaluator's
lexicographic sort / cumulative-hit implementation. The extracted scoring
kernels retain the original function bodies; their original file hashes and
line ranges are recorded in `PROVENANCE.json`. The four fragment transforms
are rebuilt from counts and gallery document frequencies before evaluation.
This does not retokenize or reparse the original contest programs.
The earlier structural development grids are checked from saved per-problem
values; their original full score matrices and conditional-residual pilot
are not replayed by this package.

## Main Numerical References

On the unused 400-problem cohort, MAP@8 is 0.3038324033 for dense retrieval,
0.3103338449 for D+BM25, and 0.3330842634 for root-TF fragments reranking
the dense top 100. The primary difference against D+BM25 is
0.0227504185, with paired problem-bootstrap 95% bounds
[0.0175350877, 0.0279257940]. All 20,000 resamples use PCG64 seed 20261010
and retain a problem's eight queries together. The weighted 650-problem
comparison uses 98.333333% marginal bounds for its three planned contrasts.
Neither bootstrap treats individual queries as independent sampling units.

The 400-problem gallery is smaller than the 650-problem galleries. Do not
compare their absolute scores as a temporal improvement, pool cohorts, or
interpret the later exploratory timing/BM25 analyses as new confirmatory data.
The protocols and unrounded outcomes, including losses, are included.

## Regenerate the Graphs

The full verifier also creates `reproduced/figure-1.tex` through
`figure-3.tex`, and `figure-data.json`. For figure-only replay:

```sh
.venv/bin/python reproduce_figures.py
xelatex -halt-on-error -interaction=nonstopmode -output-directory=reproduced reproduced/figure-1.tex
xelatex -halt-on-error -interaction=nonstopmode -output-directory=reproduced reproduced/figure-2.tex
xelatex -halt-on-error -interaction=nonstopmode -output-directory=reproduced reproduced/figure-3.tex
```

The optional rendering step requires XeLaTeX, `standalone`, `fontspec`,
`pgfplots` 1.18 and TeX Gyre Termes. It creates **figures only**, not an
article. TikZ sources are compared exactly against the checked reference
diagrams before rendering; PDF bytes can differ with TeX/font versions.
Figure 1 is the primary paired comparison on 400 unused problems. Figure 2
is the exploratory cost/gain analysis on 650 earlier problems. Figure 3
shows earlier structural-only comparisons from the existing releases. Its
numbers are checked here against immutable result JSONs; replaying their
underlying older score matrices requires those older release archives.

## Runtime and Inference Limits

Runtime statistics are reproducible from the included raw measurements;
the elapsed times themselves are not hardware-independent constants.
The original benchmark used one Apple M3 CPU thread, a resident 5,200-item
index, 512 fixed queries and five warmed passes. It excludes model encoding,
AST extraction, cold index loading and network serving. It is **not** a
latency measurement on the new 400-problem cohort. The original benchmark
code is preserved under `source/`; it is not invoked during numerical replay.

Dense scores are preserved as numerical intermediates to fix ordering near
float32 ties; the embeddings are also supplied and cosine reconstruction
is checked within absolute 1e-6. All metric checks use absolute 1e-12.
Score-file hashes, recorded first ranks, top ten and manifested bytes are
exact checks. The pinned dependency environment is the tested route; other
BLAS/SciPy versions can change floating-point accumulation and exact hashes.

The encoding receipts fix CodeRankEmbed revision
`3c4b60807d71f79b43f3c4363786d9493691f8b1`, CLS pooling, unit-normalized
768-dimensional float32 embeddings, 512 tokens, batch 16 and no query prefix.
All model-file hashes are preserved, but model inference is **not** rerun.
Selection receipts contain original source fingerprints, memberships and
input hashes, not the source programs. `source/` preserves historical
experimental runners and protocols; their original workspace paths remain
historical provenance, not the portable entry point. `verify_results.py`
is self-contained and does not import these runners.

## Attribution and Scope

Ivan Aleksandrovich Kosyanenko, MIREA Russian Technological University.
ORCID: https://orcid.org/0009-0009-1804-9412. No external funding.

Numerical features derive from the CodeContests training split at revision
`802411c3010cb00d1b05bad57ca77365a3c699d6`; see the
[original dataset repository](https://github.com/google-deepmind/code_contests)
and [pinned dataset](https://huggingface.co/datasets/deepmind/code_contests/tree/802411c3010cb00d1b05bad57ca77365a3c699d6).
The pretrained model is attributed to
[CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed).
This archive grants no license to the upstream corpus or model weights;
their terms remain those of their respective providers. No upstream weights
or original third-party programs are redistributed.

Earlier packages remain immutable:
[initial materials](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-retrieval-materials-v1)
and [prospective structural fusion](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-fresh-fusion-materials-v1).
This release is independent of ongoing Java Stage B and CPRet experiments:
it does not open the Java test, change the frozen experiments, or imply
that those unfinished studies are complete. It does not contain the wider
unreported conditional-residual pilot as a new confirmatory result.
