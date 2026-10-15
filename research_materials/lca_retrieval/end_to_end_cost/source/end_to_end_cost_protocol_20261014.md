# Same-host retrieval quality and computational cost

Status: all six comparators approved by the author on 2026-10-14.
No method or previous result is changed. This is a cost study on an already
inspected cohort, not an independent confirmation of retrieval novelty.

## Question

Does the frozen structural reranker occupy a useful quality/cost position
relative to lexical retrieval, the calibrated lexical-neural hybrid, and
the frozen Qwen models? A lower latency is not by itself evidence of a
useful method if a cheaper alternative achieves better retrieval quality.

## Fixed Comparators

1. CodeRankEmbed alone (reference).
2. BM25 alone: the existing tokenizer, raw query term counts, k1=1.2, b=0.75.
3. CodeRankEmbed+BM25: the existing full-gallery min-max normalization and
   development-selected lexical weight 0.10.
4. CodeRankEmbed+AST: the existing sampled paths, adjacent category pairs
   and triples, square-root frequency weighting, gallery-only vocabulary,
   weight 0.25, and dense top-100 reranking.
5. CodeRankEmbed+Qwen3-Reranker-0.6B: unchanged revision, instruction,
   official template, FP16/SDPA, batch limit eight, 8192-token limit,
   yes-minus-no logits, and top-100 candidates.
6. Qwen3-Embedding-0.6B alone: the previously fixed revision, query
   instruction, last-nonpadding pooling, normalization, FP16/SDPA,
   batch limit eight and 2048-token limit. No structural addition.

The final set is all six methods above. Do not
add an optimized variant, change a model, weight or candidate count,
or search for favourable subgroups after inspecting costs.

## Data And Quality

Reuse the original 400-problem cohort: 3200 query programs and the same
3200-program gallery, eight relevant candidates per query. Original
sources, source membership, relevance mapping, model file hashes and
tie-breaking are immutable. Do not access Java test or modify Stage B.

Report MAP@8, MRR and Success@10 over all 3200 queries, including BM25
alone. Recompute metrics from rankings on the benchmark host. Existing
complete score/embedding outputs may be reused for the quality calculation;
clearly distinguish numerical replay from new inference. Qwen embeddings
for this gallery require new frozen-model inference.

Timing must not use cached query embeddings, lexical query counts,
AST query features or cached reranker scores. Validate the live serving
path against the saved methods on the timing queries. Report any
batch-dependent numerical or ranking differences, rather than silently
selecting the more favourable result or changing precision.

## Host And Measurement

Use one private Kaggle session with machine_shape NvidiaTeslaT4 and
an explicit --accelerator NvidiaTeslaT4 request. Use that session's CPU
for the existing sparse/exact-ranking implementations and T4 for neural
inference. Record CPU model, RAM, visible cores, GPU, software versions
and numerical thread limits. Limit numerical CPU work to one thread for
every comparator. Run methods sequentially without other experiment jobs.
Do not emulate BM25 on a GPU merely to claim identical hardware usage.

Select exactly one query per problem by the smallest SHA-256 of
"end-to-end-cost-20261014:" followed by its existing query ID; break
hash ties by query ID. This uses no query outcomes or measured latencies.
Record the resulting 400 IDs before the benchmark. All methods process
the same IDs in three complete warmed passes. Record and fix method/pass
order before launch; do not retime only slow or disappointing methods.
Methods run in separate fresh processes in a fixed PCG64(20261014)
permutation; each uses the same three seeded query permutations. This
isolates memory footprints, but does not eliminate temporal host-load drift.
The single-query serving call has one raw query; maximum neural batch sizes
remain 16 for gallery CodeRankEmbed, eight for Qwen gallery embeddings and
eight query-candidate pairs for the reranker. Context, precision and pooling
are unchanged. Serving singleton/batched roundoff differences are reported.

The online measurement starts with an in-memory raw query string and
ends with ranked result IDs. It includes tokenization, CPU/GPU transfer,
query encoding, exact dense retrieval, lexical or AST extraction and
vectorization as applicable, candidate gathering, pair tokenization and
Qwen inference as applicable, score combination and final sorting.
Synchronize CUDA at the measurement boundaries. Exclude network request
transport, evaluator computation and checkpoint writing explicitly.

Do not include gallery preparation inside every request. Instead record
separate one-time costs for gallery parsing/tokenization, encoding,
feature extraction, vocabulary/index construction and serialization.
Also record model download and loading separately. Include all models
needed by a complete pipeline in its memory footprint; a reranker does
not replace the first-stage encoder. Use fresh processes or separately
attributable measurements, not a cumulative process peak from all methods.
Report host RAM, peak allocated/reserved GPU memory and index bytes.

Report per-method mean, median and p95 full-request latency, the three
pass means, and measured preparation time. No energy or monetary-cost
claim follows from elapsed time or use of a free GPU account. Report
latency, not concurrent-service throughput. Describe this 3200-item exact
gallery; do not extrapolate to a million-item approximate index.

For workload accounting, show separately measured preparation cost and
Q times measured mean request cost for Q=1, 100, 1000 and 10000. These
are workload projections, not additional measured serving experiments.
Interpret quality and cost jointly, including possible dominance by
BM25, the lexical hybrid or Qwen embeddings. No favourable outcome is
presumed.

## Free Resources And Stop Rules

Read-only Kaggle CLI quota check on 2026-10-14 reported 22.59/30.00 GPU
hours used and 7.41 hours remaining, reset 2026-10-17 00:00 UTC.
The previous reranker kernel is COMPLETE. No kernel or dataset matching
"end-to-end" was found in the account inventory. These checks are not
a launch receipt: repeat quota, active-job and duplicate checks before
uploading or launching the agreed benchmark.

The earlier full reranker averaged 9.93 seconds per query. Thus 400
queries times three passes suggests about 3.3 hours for its pair stage,
not a guaranteed completion time. Freeze a conservative session time
budget and retain at least one hour of free-quota reserve at launch.
The inference/indexing/timing controller has a five-hour wall-clock budget;
the kernel hard limit is six hours, leaving at least one hour of quota.
Stop with resumable checkpoints if the budget is reached; do not
reduce the agreed sample, repetitions or method settings to finish.
Exclude checkpoint writing from request clocks, but retain it in session
resource accounting. Completed quality outputs are not evidence that
all timing passes finished. A continuation on a different VM must retain
session-specific hardware and paired method measurements; do not pool
unmatched hardware timings.

Only free Kaggle compute is authorized. No paid resources, local neural
training, model fine-tuning, new feature design, manuscript submission,
public upload or changes to previous experiments are part of this test.

## Output And Interpretation

Save the fixed design, input/script/model hashes, host receipt, index
preparation measurements, all query/pass timings and result IDs, memory
records, checkpoints and completion receipt. Independently recompute
quality and timing summaries after download. Use ordinary metrics,
paired comparisons and descriptive timing statistics; do not introduce
additional diagnostic tests merely to make the study appear elaborate.

The conclusion must distinguish: a measured low-cost alternative at a
particular quality level; a method dominated by another tested pipeline;
and an incomplete or hardware-incomparable benchmark. None of these
outcomes establishes worldwide novelty or industrial benefit by itself.

## Existing Implementations And Measurement References

- Local frozen methods: coderank_kaggle_20261010.py,
  qwen_external_encode_20261011.py, qwen_reranker_20261013.py,
  bm25_20261010.py, top100_fresh_evaluate_20261010.py and the original
  AST extraction dependencies. Reuse them; do not alter bound originals.
- CUDA measurement boundaries and memory accounting:
  https://docs.pytorch.org/docs/2.11/notes/cuda.html
- Official Qwen model implementations:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
  https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
