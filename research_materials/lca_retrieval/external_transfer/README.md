# External transfer: numerical reproduction

Code, numerical intermediates and instructions by Ivan Aleksandrovich
Kosyanenko, MIREA Russian Technological University.
ORCID: https://orcid.org/0009-0009-1804-9412. No external funding.
No manuscript, manuscript PDF, editorial correspondence, raw corpus programs
or model weights are included. Prior releases are unchanged.

## Quick start

Download `lca-external-transfer-materials-v1.zip` and `SHA256SUMS.txt` from
[the versioned release](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-external-transfer-materials-v1).
Check the ZIP SHA-256 before extraction. Alternatively the repository's
`download_materials.py --package external_transfer --output downloads` checks
every downloaded asset automatically. Extract the ZIP and enter its directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest test_reproduction.py
.venv/bin/python verify_results.py
.venv/bin/python reproduce_figures.py
```

Python 3.12+; allow 4 GB available RAM and 4 GB free disk for the downloaded
ZIP, extracted inputs and outputs. Runtime is several minutes on a recent CPU.
No GPU, credentials, local training or network access is used after dependency
installation. Do not use Python `-O`: verification assertions must be enabled.
On Windows use `.venv\Scripts\python.exe` and your ZIP extractor.

The replay writes `reproduced/verification.json`, `figures.json`, and three
standalone vector-figure sources. To render them, install XeLaTeX, pgfplots,
standalone and TeX Gyre Termes, then run:

```sh
xelatex -halt-on-error -output-directory=reproduced reproduced/figure-1.tex
xelatex -halt-on-error -output-directory=reproduced reproduced/figure-2.tex
xelatex -halt-on-error -output-directory=reproduced reproduced/figure-3.tex
```

## What is recomputed

* Six fixed methods in all four CodeContests/CPRet x CodeRankEmbed/Qwen cells:
  dense, BM25, dense+BM25, top-100 nodes, full fragments and top-100 fragments.
  All 41,712 full rankings are compared exactly with the sealed arrays; query
  metrics and problem-macro metrics are recomputed independently.
* Both sets of crossed contrasts, including all four cells, within-corpus
  retriever differences, within-retriever corpus differences and their
  interaction. The original 20,000 paired problem resamples are reconstructed.
* Qwen calibration on the original 325 development problems: every candidate
  weight, all five fold choices, held-fold performance, control AP@8, the final
  weight 0.10 and its seal. CPRet and APPS are not used for selection.
* APPS: 200 introductory Codewars problems, 1,600 queries and 1,600 gallery
  programs. All 4,800 rankings, four metrics and both bootstrap contrasts are
  recomputed for dense, weight 0.25 and the development-selected weight 0.10.
* Three current diagrams: all 400 paired AP@8 observations and their histogram;
  candidate-budget quality and saved timing summaries; four crossed nDCG@10
  effects with their exploratory 95% intervals. Figure sources contain no
  article captions or surrounding prose. Earlier release figure 3 is historical
  and is deliberately not overwritten.

The metric tolerance is 1e-12; full rankings and integer counts must agree
exactly. The reranker tail and candidate budget are preserved. Saved float32
dense scores are retained to avoid changing historical near-ties across BLAS
implementations; cosines are independently recomputed from the saved normalized
embeddings and compared within 1e-6. APPS square-root features are reconstructed
from sparse counts. Crossed sparse features are the sealed extraction outputs;
their dot products and BM25 scores are recomputed. The package manifest and
original receipts bind inputs and results by SHA-256.

## Interpretation and coverage

This is numerical replay, not independent data collection or a new experiment.
The crossed design followed two already-known diagonal outcomes and is
exploratory. It shows a positive structural increment for CodeRankEmbed on
CodeContests but negative transfer for Qwen on CPRet. The subsequent APPS
test also remains below dense Qwen after development-only weight selection.
Negative results are included unchanged. The crossed interaction interval
contains zero; neither interaction nor equivalence is established.

CPRet uses the published positive judgments. Other candidates receive zero
evaluation gain under its retrieval metric convention; they are **not asserted
to be manually verified irrelevant**. Eight entirely unjudged queries in the
full upstream corpus are outside this judged Python subset. CodeContests and
APPS use problem identity, not a proof of semantic equivalence. The APPS result
is limited to the selected introductory Codewars tasks, not all APPS.

The first two diagrams use byte-identical top-100/cost result files from the
[weighted/top-100 release](https://github.com/Malomalsky/code2hyp-artifacts/releases/tag/lca-weighted-top100-materials-v1).
This release cross-checks their primary paired observations and recomputes the
saved timing summaries. Replay the earlier package as well to reconstruct all
older weighted, BM25 and candidate-budget rankings, including all nine budget
conditions. The two earliest releases are needed for the earlier structural
ablation and RRF analyses. No ongoing Java study is included.

Neural inference, raw-corpus parsing, cohort selection from original downloads,
and fresh wall-clock benchmarking are **not** rerun by this archive. The
`source/` files preserve original experiment code/protocols for provenance;
their historical workspace paths are not portable entry points. Use the
top-level scripts for offline replay. Original source-to-embedding regeneration
requires upstream corpora, pinned models and the extraction environment.
Historical timings cannot be reproduced exactly on different hardware.

## Sources and terms

* [CPRet-data](https://huggingface.co/datasets/coldchair16/CPRet-data),
  coldchair16 and the CPRet authors, dataset card: CC BY-NC 4.0.
  CPRet-derived numerical features, embeddings and the positive-judgment subset
  retain attribution and noncommercial restrictions; see
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  Changes are limited to subset selection and the documented numerical transforms.
* [CodeContests](https://github.com/google-deepmind/code_contests), DeepMind:
  use the upstream terms for the dataset and underlying programs.
* [APPS](https://huggingface.co/datasets/codeparrot/apps), Hendrycks et al.,
  revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`: no challenge statements,
  solutions or tests are redistributed here. Upstream/source terms apply.
* [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) and
  [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):
  model settings, revisions, model-file digests and inference receipts are
  included with each cell. No model weights are distributed.

Do not interpret the repository's code license as relicensing third-party
data. `PROVENANCE.json` distinguishes unchanged inputs, extracted functions
and derived numerical exports. `MANIFEST.json` lists every archive input.
