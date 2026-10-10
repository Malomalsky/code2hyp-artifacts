# Prospective top-100 validation, 10 October 2026

## Question

Does the fixed root-TF fragment reranker improve solution-to-solution retrieval
over the development-calibrated CodeRankEmbed/BM25 hybrid on unused problems?
Root TF without IDF is the primary structural variant: it is simpler and had
the higher mean in the completed development and later exploratory comparisons.
Root TF-IDF is retained as a fixed secondary variant. No parameters are fitted
to this new cohort.

## Data and selection

Use deepmind/code_contests revision 802411c3010cb00d1b05bad57ca77365a3c699d6,
training shards 33--38 only. Their official LFS hashes and sizes are fixed in
`top100_fresh_20261010.py` before downloading their contents. Target **400**
Codeforces problems with 16 eligible accepted Python3 solutions per problem:
eight queries, eight gallery programs, 3200 queries against the same full
3200-item gallery. The smaller target reflects the six remaining input shards;
it is not a power guarantee. Absolute metrics must not be compared directly
with earlier 5200-item galleries. If fewer than 400 problems survive eligibility,
report the shortfall and stop before encoding or retrieval scores.

Reuse source/AST/terminal eligibility, d0/d1/d2 duplicate rules and deterministic
SHA ordering from the previous selector. Its ordering salt remains
`root-tfidf-external-20261010:`. Exclude all previous Codeforces problem IDs
through shard 32, all archived evaluated metadata and sealed CPRet membership.
Exclude preserved evaluated-program fingerprints and the selected CodeNet
source audit. The previously deleted full CodeNet inventory and unselected
old-program fingerprints cannot be checked; this limitation remains explicit.
Seal membership before extracting similarities. Source parsing does not execute
submitted programs. Use no additional shards after observing eligibility or
performance.

## Fixed pipelines

Use the same CodeRankEmbed model revision, files and inference settings as the
completed Kaggle run: revision 3c4b60807d71f79b43f3c4363786d9493691f8b1,
float32, CLS pooling, normalised embeddings, batch 16, max 512 tokens, empty
query prefix. Encode only on free Kaggle Tesla T4, with explicit
`machine_shape=NvidiaTeslaT4` and `--accelerator NvidiaTeslaT4` after quota and
duplicate checks. The existing encoder saves checkpoints every 128 programs
and enforces a one-hour encoding bound. A source-binding adapter changes only
cohort size and hashes; model/inference code remains the previous implementation.

AST features reuse the same frozen 64-path sampler, category-only terminals,
reversal-invariant fragments of length two and three, and gallery-only feature
vocabulary. Root TF means sqrt(count) followed by L2 normalisation. TF-IDF
applies gallery IDF after the square root. Node counts use ordinary cosine.
BM25 retains the previously fixed tokenizer, positive IDF, k1=1.2, b=.75,
raw query counts and gallery fitting. Its dense hybrid uses per-query full
gallery min-max scaling and fixed alpha=.10 from the completed development fit.

Evaluate exactly these nine rankings:

| Ranking | Rule |
| --- | --- |
| Dense | Full CodeRankEmbed ranking |
| BM25 | Full lexical ranking |
| Dense/BM25 | .90 minmax(D) + .10 minmax(B), full gallery |
| Nodes full | .50 D + .50 node cosine, full gallery |
| Nodes top100 | The same score within dense top100 |
| Root full | .75 D + .25 root-TF cosine, full gallery |
| Root top100 | The same score within dense top100; primary method |
| IDF full | .80 D + .20 root-TF-IDF cosine, full gallery |
| IDF top100 | The same score within dense top100 |

For top100 rankings, append the unchanged dense tail after reranking the first
100 candidates. Ties use UTF-8 source IDs. No new cutoff, weight grid,
normalisation variant, subgroup search or three-channel fusion is introduced.

## Analysis fixed before new outcomes

Primary endpoint: problem-macro MAP@8, denominator eight relevant gallery items
per query. **One primary contrast: Root top100 minus Dense/BM25.** Report its
signed mean and ordinary paired problem-bootstrap percentile 95% interval,
20000 resamples, PCG64 seed 20261010, all eight queries of a problem together.
A positive lower bound supports improvement for the sampled problems conditional
on this fixed gallery. This does not quantify variation across domains, model
training runs or pretraining exposure.

Report all nine MAP@8, full-ranking MRR and Success@10 means; signed descriptive
comparisons with dense and Nodes top100; full-versus-top100 losses; dense
candidate recall@100 and fraction of queries with no relevant candidate.
Do not promote secondary outcomes to primary or add inferential tests after
seeing results. Save compact per-query metrics, first relevant ranks and top10
IDs. Independently check rankings and metric arithmetic before interpreting them.

Existing CPU serving measurements cover a 5200-item gallery. This study records
feature extraction, encoding and evaluation time, but makes no new latency
claim without a separately measured serving run. Novelty remains the empirical
usefulness of the structural channel; BM25, histogram transforms, convex fusion
and candidate reranking are established methods.

Stage B and CPRet retain their existing protocols. Java test remains closed.
Update the manuscript only after verification of the new outcomes.
