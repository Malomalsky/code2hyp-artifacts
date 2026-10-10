# Two-stage structural reranking: local exploratory study

## Question and scope

Can structural reranking retain the observed quality gain when applied only
to a short CodeRankEmbed candidate list, and does it reduce CPU ranking time?
The user approved this follow-up on 2026-10-10. The preceding full-gallery
results are already known: this is an exploratory operational study, not an
independent replication, parameter-selection exercise, or equivalence test.

Use the frozen 650-problem Python cohort, 5200 queries and 5200 gallery items,
eight relevant items per query. Reuse the verified Kaggle embeddings and sparse
features. No training, encoding, feature refitting, new data, Java test access,
Stage B changes, or CPRet changes. No network or GPU is required for this run.

## Methods fixed before this run

- Dense CodeRankEmbed cosine alone, D.
- Candidate: 0.80 D + 0.20 retained root-TF-IDF cosine.
- Simpler comparator: 0.75 D + 0.25 retained root-TF cosine, without IDF.
- For each hybrid, full gallery and dense top K = 50, 100, 500.

Rerank only the selected prefix; leave the remaining dense ranking unchanged.
Use the original UTF-8 source-path tie break in both stages. Weights and
representations do not change. Report every K, not a post-hoc winning K.
K=100 is the primary practical operating point; 50 and 500 describe the
trade-off. The established retrieval-then-reranking pattern and default 100
candidates are documented in the official CoRNStack implementation:
https://github.com/gangiswag/cornstack#-running-reranker
This motivates the experiment, not a claim of algorithmic novelty.

## Quality

Use frozen dense scores to preserve the numerical baseline. Recompute sparse
cosines from features, checking full-hybrid scores against the saved matrices.
Measure problem-macro MAP@8, MRR over the complete prefix-plus-tail ranking,
and Success@10. Report signed differences from D and the corresponding full
hybrid, and the fraction of the full MAP gain retained (undefined if its
denominator is nonpositive). Do not call a small difference equivalence.

Report dense candidate recall@K (relevant candidates / 8), fraction of queries
with any candidate hit, and mean fraction of full-hybrid top-eight items in
the candidate list. These distinguish a first-stage ceiling from reranking
error. The best possible AP@8 within a K>=8 prefix is candidate recall@K.
All results are descriptive; no new significance tests or confidence intervals.

## Runtime

Single CPU thread, one query at a time, same float32 dense vectors and float64
sparse features as before. Do not time reads of score matrices as computation.
Time actual dense dot products, exact candidate sorting, sparse scoring
(including candidate-row gathering), fusion and final sorting. Full-gallery
hybrids need no first-stage sort. Return top ten; full-ranking bookkeeping for
MRR belongs to quality evaluation, not serving latency.

Select 512 query IDs by ascending SHA-256 of their UTF-8 source paths, before
timing. Warm all methods on the first 16 selected queries. Run five repetitions
with PCG64 seed 20261010 rotating/shuffling method order at each query to reduce
order effects. Save every component latency and total wall time. Report median
of five per-pass mean latencies, individual-query median and p95. Compare
runtime dense scores and top-ten rankings with the frozen-score path and report
any floating-point ranking differences, rather than assume bitwise equivalence.

This is warm-index CPU ranking latency on this Mac at gallery size 5200, not
end-to-end source-code latency. Exclude neural encoding, AST extraction,
feature construction, index loading, I/O, and network. No ANN index or claim
about million-item scaling. Record resource/library/thread information;
background desktop activity remains a timing limitation.

## Implementation and checks

1. Add only `two_stage_20261010.py` and its small unit-test file. Reuse the
   existing verified input loader, metadata and SHA helpers; do not edit them.
2. Test deterministic ties, prefix containment, full-K equivalence, zero-weight
   dense equivalence, relevant documents outside the prefix, and AP/MRR values.
3. Freeze code, protocol and input hashes before the new run. Store compact
   top-ten IDs, first-relevant ranks and query metrics, not new dense matrices.
4. Run CPU quality and timing. Independently reconstruct every ranking metric
   from the earlier saved full scores; verify compact artifacts and hashes.
5. Write the result report and append the research journal. Do not modify the
   manuscript or upload unpublished material.

Run with the artifacts project's `.venv/bin/python` from this directory:
`-m unittest test_two_stage_20261010`, then `two_stage_20261010.py --prepare`,
`two_stage_20261010.py --run`, and `two_stage_20261010.py --check`.
