# Frozen Qwen weight transfer to APPS (2026-10-11)

## Rationale and sequence

The user chose a different corpus after the planned unused CodeContests
validation/test selection yielded zero eligible problems. That failed selection
is retained unchanged. Qwen's weight was selected on the original 325 development
problems before any APPS program contents or retrieval scores were inspected:
alpha = 0.10, weight-seal SHA-256
72fb97afa09bbd166a9cfca33a6db8eab05867a842ed1cbb7f2acb8f2536822f.
This study tests transfer of that fixed setting, not a new parameter search.

## Corpus and eligibility fixed before contents

Use codeparrot/apps revision 21e74ddf8de1a21436da12e3e653065c5213e9d1,
both upstream train.jsonl and test.jsonl. Both are evaluation inputs here;
no APPS training is performed. Their upstream SHA-256 and sizes are bound in
the adapter and design. Only Codewars kata and Kattis problem URLs are eligible;
Codeforces/AtCoder and unrecognised URLs are excluded, not remapped to new IDs.
Canonical URL identifies a problem across the two files. References sharing a
URL are pooled, with source-level and alpha-normalised-AST deduplication.
Exclude an entire problem if any eligible reference matches any of the three
saved prior-study fingerprints. Use all prior selections covered by the existing
qwen_fresh exclusion inventory, including CPRet, CodeContests and Python Stage A.
The historical fingerprint inventory is incomplete; URL/domain separation and
the available fingerprints do not prove semantic independence or absence from
Qwen's pretraining. Do not claim either.

Apply the unchanged Python eligibility function (64 KiB source, 4096 AST nodes,
parseable Python with at least two terminals), and the unchanged cross-problem
duplicate removal. Require at least 16 distinct surviving solutions per problem.
Choose min(200, eligible) problems, with the existing deterministic hash order
and salt; split 16 references into 8 queries and 8 gallery programs using the
same ordering as the previous external study. Stop without scores if fewer than
30 problems survive. No alternative split, corpus, eligibility relaxation or
second sample is chosen in response to outcomes. Record complete attrition and
selected membership, then seal before features and encoding. The >=16 rule
favours well-represented tasks and bounds generalisation. Dataset-provided
solutions define relevance; their correctness is not independently certified.
Never execute downloaded solutions or inspect Java test data.

## Fixed methods, endpoint and interpretation

Keep the original, uncleaned root-TF AST fragments, frozen path sampler and node
vocabulary. Build the sparse feature vocabulary from the evaluation gallery as
in the existing method; do not use APPS relevance labels to fit representations.
Qwen revision, prompt, max length 2048, batch 8, float16 inference, normalised
1024-dimensional vectors and cosine scoring remain unchanged. Run inference
only on a private free-quota Kaggle Tesla T4 with checkpoints.

Evaluate exactly three methods: dense Qwen, top-100 structural reranking with
old alpha 0.25, and top-100 reranking with sealed alpha 0.10. Mix cosine scores
as (1-alpha)*dense + alpha*structure; preserve the dense tail and UTF-8 ID ties.
Primary endpoint: problem-mean MAP@8 difference, alpha 0.10 minus dense.
Secondary comparison: alpha 0.10 minus alpha 0.25. Also report nDCG@10, MRR@10
and Recall@100 descriptively. Bootstrap whole problems in pairs, 20,000 draws,
PCG64 seed 20261014, percentile 95% interval. A positive primary interval supports
transfer on this selected population, an interval containing zero is inconclusive,
and a negative interval is evidence of harm. Report every sign, not just wins.
Unchanged Recall@100 is expected by construction, not an independent discovery.
Use the existing scorer and independent ranking/metric verifier. No weight or
feature changes after selection. No CPRet rescoring is part of this study.

## Provenance and resources

Primary sources: https://github.com/hendrycks/apps,
https://arxiv.org/abs/2105.09938, and the pinned Hugging Face dataset card/loader.
The upstream Hugging Face card and loader identify an MIT license; originating
site rights are not independently adjudicated here. Keep inference inputs private,
retain URL provenance, and do not publish raw sources or a manuscript in this task.
Use existing code and only a corpus adapter, its tests, receipts and a report.
Do not modify main Java Stage B, existing experiments or the article based on a
result that has not yet been independently verified.
