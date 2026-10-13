# Crossed corpus/retriever diagnostic, fixed method

Requested after observing the two diagonal cells on 2026-10-11. This is a
diagnostic extension, not a new untouched confirmatory experiment. Freeze
this plan before encoding the missing cells; freeze analysis code before
examining their retrieval scores. Preserve every signed outcome.

## Four cells, no new samples or tuning

| Corpus | CodeRankEmbed | Qwen3-Embedding-0.6B |
|---|---|---|
| CodeContests, sealed 400-problem cohort | Reuse completed embeddings | Encode missing cell |
| CPRet, sealed Python subset | Encode missing cell | Reuse completed embeddings |

CodeContests: 400 problems, 8 queries and 8 gallery programs per problem,
3200 queries and a shared 3200-document gallery. Source SHA-256
ae91ef4ac8b7d7703deb3bb1e90738362b04d9d0bd3ee1cc77de95fdda10a0ea.
CPRet: 276 problems/queries, 3060 gallery programs, 927 binary positive qrels.
Source SHA-256
375b1335f45ee50f84ac944341c490a2528ddf7063213da70c29be4593be3971.
Reuse the sealed membership, source order, relevance, source normalization,
AST features, BM25 counts, and vocabulary in each corpus. No exclusions,
sampling, extra models, updated weights or refitting.

CodeRankEmbed retains revision 3c4b60807d71f79b43f3c4363786d9493691f8b1,
CLS pooling, float32, 768 dimensions, batch 16, max_length 512, empty prefix,
sentence-transformers 3.1.1 and transformers 4.45.1.
Qwen retains revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3,
last-token pooling, float16 weights, normalized float32 1024-dimensional
vectors, batch 8, max_length 2048, left padding, right truncation, SDPA,
transformers 4.57.1 and the exact previously fixed query instruction.
Do not equalize context lengths or precision after seeing the diagonal results:
the factor is the entire frozen retriever configuration, not architecture alone.
Pin model/tokenizer files to the previous verified SHA-256 values.

## Same ranking rules

Reuse six methods: dense D; BM25; D_BM25; nodes_100; root_full; root_100.
root_100: 64 paths, retained ancestor category, 2/3-gram fragments, gallery
vocabulary, sqrt counts, L2/cosine, no IDF, 0.75 dense + 0.25 structure,
reranking only the dense top 100. Preserve the remaining dense order.
nodes_100 uses raw node counts and weight 0.50; root_full uses weight 0.25
over the full gallery. D_BM25 is 0.90 per-query min-max dense plus 0.10
per-query min-max BM25, with k1=1.2, b=0.75 and full-gallery normalization.
Ties use the original UTF-8 gallery identifiers in each corpus.

## Comparable measurement and interpretation

Compute nDCG@10, MRR@10 and Recall@100 in every cell using binary relevance.
Average queries within each problem before averaging problems. Also retain
CodeContests MAP@8/MRR/Success@10 to reproduce the original conclusions;
do not compare CodeContests MAP@8 numerically with CPRet nDCG@10.

The focal diagnostic is the *incremental benefit over pure dense retrieval*,
delta(corpus, retriever) = mean_problem[nDCG10(root_100) - nDCG10(D)].
Report the same deltas over D_BM25 as the previously designated comparator,
without replacing either earlier primary analysis.

Use paired problem bootstrap with 20,000 resamples, PCG64 seed 20261012.
Use identical problem resamples for the two retrievers within a corpus and
independent resamples across corpora. Report two-sided percentile 95% intervals
for each delta, the model difference in delta within each corpus, and the
corpus difference in delta within each retriever. Report the interaction:

    [delta(CPRet,Qwen) - delta(CPRet,CodeRank)]
      - [delta(CodeContests,Qwen) - delta(CodeContests,CodeRank)].

These are exploratory diagnostic intervals, not multiple confirmatory wins
or a familywise significance claim. Intervals condition on the fixed galleries.
Do not select a new method, model or weight based on any of these intervals.
Always report dense quality and candidate Recall@100 alongside increments;
top-100 reranking cannot repair missing relevant candidates.

The crossed design distinguishes empirical corpus and retriever patterns
within these fixed cells. It does not identify a mechanistic cause, isolate
architecture from context/precision/prompt, rule out pretraining exposure,
or generalize to industrial code. The corpus factor includes cohort selection,
gallery composition and relevance density, not just programming style.

## Execution

Only free private Kaggle NvidiaTeslaT4, with explicit machine_shape and
--accelerator NvidiaTeslaT4. Verify quota and duplicate inventory first.
Encode only the two missing cells, sequentially, using separate subprocesses
with the previously fixed package versions. Reuse the completed diagonal
embeddings; do not re-run neural inference or AST extraction for them.
Bound each inference to one hour following a quality-blind resource pilot;
retain checkpoints every 128 programs. Download receipts, model/source hashes,
embeddings and token lengths, and independently verify rankings and metrics.
No training, paid resources, local neural inference, Java test access, Stage B
modification, manuscript upload or public release in this run.
