# Ancestor Retention in Structural Code Retrieval

Research code, numerical data and reproduction instructions for local AST
fragments, ancestor retention and candidate-limited code reranking.

Ivan Aleksandrovich Kosyanenko, Ph.D. (Engineering), MIREA Russian Technological
University. ORCID: https://orcid.org/0009-0009-1804-9412. No external funding.
The manuscript and third-party source programs are not included.

## Choose the Data Package

| Package | Contents | Download |
|---|---|---|
| Earlier analyses, v1 | CodeNet comparison, categorical path comparisons, weighting/order controls and exploratory fusion; 103 score matrices | [lca-retrieval-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-retrieval-materials-v1) |
| Prospective fusion, v1 | Fixed fusion on a separate 650-problem cohort; five score matrices, metadata and bootstrap indices | [lca-fresh-fusion-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-fresh-fusion-materials-v1) |
| Weighted and top-100, v1 | Weighted 650-problem comparison, BM25 development/comparators, candidate budgets, 400 unused problems, raw timing samples and three vector diagrams | [lca-weighted-top100-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-weighted-top100-materials-v1) |
| External transfer, v1 | Four corpus/retriever cells, Qwen development calibration, independent APPS/Codewars test including negative results, and three current vector diagrams | [lca-external-transfer-materials-v1](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-external-transfer-materials-v1) |

For the external comparisons, start with the
[external-transfer instructions](external_transfer/README.md). The latest
archive replays 41,712 crossed rankings, all development weights/folds and
4,800 APPS rankings from numerical inputs. It includes negative transfer and
regenerates all three current diagrams. This is an offline numerical replay,
not a repetition of corpus parsing, model inference or data collection.
Use `download_materials.py --package external_transfer` (now the default).

For the preceding neural/structural comparisons, use the
[weighted/top-100 instructions](weighted_top100/README.md). This compact,
independent ZIP includes embeddings, dense scores, sparse features, outcomes
and portable NumPy/SciPy code. It recomputes all packaged metrics and intervals
and regenerates the three figure sources without the manuscript. Original
corpus parsing and model inference are not rerun; these limits and the
hardware dependence of runtime are documented explicitly. Older archives
remain unchanged and are needed only to replay the older underlying scores.

The earlier package has two independent ZIPs sharing one directory; extract both.
The prospective package has one ZIP. Neither of the two older releases
contains the weighted/top-100 analyses. Do not pool scores across cohorts or treat the exploratory
fusion as prospective evidence. Large matrices are hosted as ordinary GitHub
Release assets, without Git LFS or paid storage.

## Reproduce the Prospective Evaluation

Python 3.12 or newer and NumPy are sufficient. Download the checked archive with
the standard-library script below, or download the ZIP and `SHA256SUMS.txt` from
the release page. The script verifies the complete downloaded file, not just its
name or the server's upload receipt.

```sh
python3 research_materials/lca_retrieval/download_materials.py --package prospective --output downloads
unzip downloads/prospective/lca-fresh-fusion-materials-v1.zip -d downloads/prospective
cd downloads/prospective/lca-fresh-fusion-materials-v1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_fresh_fusion.py data
```

On Windows, extract with the standard ZIP tool and use `py -3.12` and
`.venv\Scripts\python.exe`. Do not pass Python's `-O` option. The final command
prints `PASS` and writes `data/verified_results.json`, which is separate from
the immutable `reference_results.json`. No GPU, training, network connection
or original corpus is needed after download and dependency installation.
Allow approximately 3 GB of free disk space and 4 GB of available memory.

The verifier checks the SHA-256 manifest, reconstructs both fused score matrices
from their component rankings, and recomputes five sets of retrieval metrics and
the three paired bootstrap contrasts. Metric comparisons allow roundoff of
1e-12; hashes and saved fusion reconstruction are exact. Verification code is
also available in [fresh_fusion/](fresh_fusion/), byte-identical to the archive.
Run it from the extracted package, which supplies its manifested inputs.

To download the earlier analyses instead:

```sh
python3 research_materials/lca_retrieval/download_materials.py --package earlier --output downloads
unzip downloads/earlier/lca-retrieval-materials-v1-part1.zip -d downloads/earlier
unzip downloads/earlier/lca-retrieval-materials-v1-part2.zip -d downloads/earlier
cd downloads/earlier/lca-retrieval-materials-v1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_results.py
```

## Earlier Fusion Result

The prospective cohort contains 650 previously unused Codeforces problems,
5,200 queries and 5,200 gallery programs, with eight relevant items per query.
The rule uses reciprocal rank fusion with constant 60, equal component weights
and full one-based ranks. It was fixed before downloading the new cohort.

| Representation | MAP@8 |
|---|---:|
| Node counts | 0.07883 |
| Retained fragments | 0.07669 |
| Masked fragments | 0.06771 |
| Nodes + retained fragments | 0.09704 |
| Nodes + masked fragments | 0.09087 |

| Retained fusion minus | Difference | Adjusted interval |
|---|---:|---|
| Node counts | 0.01821 | [0.01430, 0.02216] |
| Retained fragments | 0.02035 | [0.01644, 0.02442] |
| Masked fusion | 0.00617 | [0.00486, 0.00751] |

The intervals use 20,000 paired problem resamples and 98.333333% marginal
percentile bounds, Bonferroni-adjusted for the three comparisons. They condition
on the selected gallery. The result supports complementary value of local
fragments and node counts, and ancestor retention in their combination, in this
Python contest-code sample. It is not a new fusion algorithm, semantic-equivalence
test, independent-team replication, or demonstrated superiority to learned
retrieval systems. No learned baseline was evaluated on this gallery.

## Earlier Provenance

The frozen implementation commit is `c419f6418056b618ce373ebd6fafe6601ff51566`.
CodeContests revision: `802411c3010cb00d1b05bad57ca77365a3c699d6`.
The archive preserves the original protocol, design/selection receipts, result
and file hashes. Historical runner paths describe the original workspace; the
NumPy verifier is the portable numerical-reproduction entry point. Source-to-
feature regeneration requires the original corpus, checkpoint and dependencies
and is not provided by the score-only archive.

On 3 October 2026, six deleted numerical files were regenerated from the sealed
selected programs and the original vocabulary checkpoint. All six file hashes
matched the original recorded SHA-256 exactly. There was no training, model
change, new sample, GPU use or change to the sealed result. The recovery receipt
is a separate release attachment, outside the unchanged scientific manifest.

Keep the verified release ZIP as the compact local backup. Do not remove the
only local copy until a complete download from the published release has passed
SHA-256 verification. This repository update does not complete the separate
CodeNet Stage A/B protocols or open the Java test set.
