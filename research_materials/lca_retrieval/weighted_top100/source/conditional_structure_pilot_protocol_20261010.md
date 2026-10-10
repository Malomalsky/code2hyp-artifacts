# Conditional structure pilot: protocol and implementation plan

Fixed before new corpus scores, 10 October 2026. The author approved the
separate CPU prototype. This is exploratory method development on an already
analysed cohort, not an independent replication or a change to Stage B/CPRet.

## Question and fixed candidate

Does the conditionally centered score in
`CONDITIONAL_STRUCTURE_CANDIDATE_20261010.md`, equations (1)--(6), add useful
retrieval signal beyond ordinary weighting and calibrated score fusion?
Use the gallery retained-fragment vocabulary, project query counts first,
one total pseudo-count per masked context, and normalization by the original
inverse-conditional-frequency vector norm. Do not renormalize the residual
to unit length, clip negative similarities, tune smoothing, change paths or
introduce query-dependent gates. For numerical stability compute residual
features explicitly; check their scores against the two-product identity.

## Inputs and roles

Use `followup_results/fresh_fusion/selected_programs.jsonl` and the corresponding
metadata/receipts: 650 Codeforces problems, 8 queries and 8 gallery programs
per problem, 5200 by 5200 full-gallery comparisons. Source SHA-256 is
`f5186a1945cbe65a9a38846ca5a305ddfb7a766f89ea10fc8ec98cdd228ab729`.
Verify every input against existing receipts and the saved dense result
SHA `d25fe73921b328604e9bf317af2a7e9c02c0b67b85ed22b9f7e7679da7f73524`.
Reuse the same frozen label-only checkpoint only for its vocabulary/path
selection configuration; call no neural forward, optimizer or training loop.
Rebuilt raw fragment cosine must reproduce the original matrix numerically.

Sort problem IDs by SHA-256 of UTF-8
`conditional-structure-development-20261010:<problem_id>`, with ID as tie
breaker. The first 325 form calibration; the other 325 form internal
validation. Each query stays with its entire problem group. The full fixed
gallery is used for both parts: gallery-only background/IDF statistics are
label-free index statistics, not a separate training set. No calibration
query labels from the validation half enter weight selection. This partition
does not undo prior observation of aggregate results for the whole cohort.

## Methods and calibration

Dense D is the frozen CodeRankEmbed cosine, recovered as `-D.npy`. Additional
channels, each compared separately in `(1-alpha)*D + alpha*channel`, are:

- N: the existing AST-node count cosine, a strong simple structural control.
- F: the original raw retained-fragment cosine.
- I: raw fragment counts with the existing smoothed gallery-IDF and L2 norm.
- W: inverse-conditional-frequency weighted counts, without centering.
- R: the proposed bounded residual inner product.
- T: the existing token TF-IDF cosine, the lexical control.

Distances saved for N, F and T are converted by `1-distance`. New scores
are float64; the saved dense float32 values are promoted, not recomputed.
No per-query rescaling is used: all channels already have bounded cosine or
projection scores. The difference in their actual scale is handled only by
the same declared weight grid. It is not claimed that the scales represent
the same probabilities or that this grid finds an optimal continuous weight.

The shared ascending grid is
`[0, .005, .01, .02, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
  .55, .60, .65, .70, .75, .80, .85, .90, .95, 1]`.
For each channel choose the weight maximizing problem-macro MAP@8 on the
325 calibration problems; exact ties choose the smallest weight. Save all
calibration means and chosen weights before scoring validation. The primary
criterion stays MAP@8, with 8 relevant documents per query and the existing
UTF-8 identifier tie rule. No new significance tests or bootstrap intervals
are needed for this screening pilot. Report standalone channels as well as
chosen hybrids, and retain all unfavorable results. Evaluate the held-out
half only at the chosen weight and at the already-declared standalone weight
one; do not inspect its full grid or adjust weights afterwards.

Report R-hybrid versus D and each calibrated control, including problem-wise
gain/loss counts. Improvement only over raw F or uncalibrated RRF does not
justify the proposed representation. A promising result requires a later,
separately frozen evaluation on unused problems, not further tuning on this
half or on frozen CPRet. A zero selected weight is a valid outcome. A worse
validation result stops this candidate's escalation in the current turn;
do not add nonlinearities or tune another parameter to rescue it.

## Execution and verification

Native resource probe: Mac15,12, 24 GiB RAM, 8 logical CPUs, about 110 GiB
free disk. Use one CPU thread, sparse count/features, memory-mapped old
scores, and blocks of 128 queries for new matrix products. New score matrices
are I, W and R only (about 0.65 GB total). No paid/GPU resource is used.
Measure AST extraction, transform/scoring/calibration/validation time and
whole-process peak RSS; these are engineering measurements, not latency
claims from a controlled benchmark.

Files affected: this protocol; isolated `conditional_structure_pilot_20261010.py`;
its small `test_conditional_structure_pilot_20261010.py`; new output directory
`followup_results/conditional_structure_20261010`; and the research journal.
No production helpers, exports, frozen configuration, article or old output
will be edited. Existing helpers supply fragment extraction, vectorization,
normalization, deterministic ranks, hashing and exclusive JSON output.

Implementation checklist:

- Add focused tests for the exact repair example, zero/singleton blocks,
  PSD/boundedness and sparse-versus-explicit projector equality; test the
  metric against hand-computable ties and stable sorting. Run before and
  after implementing the new scorer.
- Bind the prototype, protocol, immutable input hashes, helper hashes and
  problem partition in a design receipt before feature extraction/scoring.
- Rebuild only non-neural AST counts, reproduce F, compute I/W/R, check the
  residual identity, preserve per-stage hashes and reuse completed stages
  only when their receipt and frozen code bindings match exactly.
- Calibrate, exclusively save chosen weights, then evaluate validation.
  Reconstruct all selected rankings independently with stable sorting,
  verify inputs/outputs and exact selection from saved calibration means.
- Record actual results and limits in a short research note and journal.
  Do not claim empirical improvement until the scores have been computed.
