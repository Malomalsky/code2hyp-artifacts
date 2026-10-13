# Qwen weight transfer: development calibration and fresh evaluation

11 October 2026. The author approved the preceding report's bounded proposal:
calibrate only the Qwen structural weight on the old development problems,
freeze it, then evaluate new problems. No CPRet-based selection and no feature
change. This protocol and its implementation plan precede new Qwen scores.

## Development

Use exactly the 325 problems, 2,600 queries, full 5,200-program gallery and
five 65-problem folds in frequency_factorial_20261010/design_receipt.json.
The other old 325 problems' queries are not encoded or scored. Their gallery
programs remain distractors, as before. Reuse the sealed ret_root.npy scores:
original uncleaned categorical AST, 64 paths, retained LCA, reversal-invariant
2/3-grams, gallery vocabulary, sqrt(TF), L2 normalization, cosine, no IDF.
No cleaned features, variable-binding features, new prompts or normalization.

Qwen3-Embedding-0.6B retains revision
97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 and all settings/model hashes from
qwen_external_encode_20261011.py and the completed Qwen encoding receipt:
float16 inference, float32 normalized output, SDPA, left padding, right
truncation, last nonpadding token, batch 8, max 2048, original query instruction.
Only source membership changes. Free private Kaggle Tesla T4, explicit
machine_shape=NvidiaTeslaT4 and --accelerator NvidiaTeslaT4, no training.
Verify quota and duplicates before each launch. Save progress every 128 rows;
keep the existing one-hour inference bound and resume rather than repeat.

For each query, rerank only its original dense top 100 using
(1-alpha) D + alpha S; retain the dense tail. UTF-8 document IDs break ties.
Use the existing ascending 24-value grid, including zero and one. Unlike the
old full-gallery development grid, this comparison matches the deployed
top-100 pipeline. AP@8 has denominator eight. Average the eight queries per
problem, then problems equally. Each fold selects alpha by training-problem
mean AP@8; exact ties choose the smallest alpha. Report all fold choices and
out-of-fold MAP@8 versus dense and fixed .25. Recompute the fixed .25
CodeRankEmbed control with the same top-100 rule from its existing scores;
do not encode that model again or use it to choose Qwen's weight.

The deployment alpha maximizes mean AP@8 on all 325 development problems,
using the same smallest-alpha tie rule. Save a separate immutable weight seal
binding protocol, design, inputs, model, complete grid and results. Weight zero
is permitted and must not trigger another search. This reused development
set is not independent evidence even with cross-validation.

## New evaluation after the weight seal

Only after the weight is sealed, read CodeContests valid/test source contents
at revision 802411c3010cb00d1b05bad57ca77365a3c699d6. Metadata-only repository
listing was checked before this protocol; no new contents or outcomes read.

| File | Bytes | SHA-256 |
|---|---:|---|
| valid-00000-of-00001-5e672c5751f060d3.parquet | 51829044 | 02e8c1ccedae716f1e43cc813fcb7823c3db666ea92638820aba80e8cef451ab |
| test-00000-of-00001-9c49eeff30aacaa8.parquet | 63077400 | aa426cbdb202bf8703b658bcb31fd1878ca7cfd33ca07d3b703dc94ca6a2b651 |

Use all eligible Codeforces problems from these two fixed shards, with at
least 16 accepted Python3 solutions after the previous source/AST/terminal
eligibility and d0/d1/d2 duplicate rules. Reuse the frozen deterministic
ordering salt from root_tfidf_external_20261010.choose. Take eight queries
and eight gallery programs per eligible problem; do not select by difficulty
or retrieval quality. Pool their galleries. If fewer than 30 problems remain,
report the shortfall and stop without adding another data source. Thirty is
an operational minimum, not a statistical power guarantee.

Exclude all prior problem IDs through all 39 training shards, previously
evaluated metadata and sealed CPRet membership, plus all preserved prior
d0/d1/d2 fingerprints. Also bind and exclude the previous 400-problem cohort.
Full deleted historical fingerprints and model pretraining inventories are
unavailable: novelty to this experiment is not proof of pretraining exclusion.
Do not execute corpus programs. Seal membership before extraction/encoding.
The term test here means this separate Python CodeContests split, NOT Java
CodeNet test; the latter remains closed.

Apply the identical frozen feature algorithm; as in preceding external
evaluations, fit its label-free vocabulary only on the new gallery. No model
or weight fitting on this cohort. Encode Qwen with the same pinned settings.
Evaluate only dense, fixed .25 hybrid and development-selected hybrid.
Primary contrast: selected hybrid minus dense in problem-macro MAP@8.
Report an ordinary paired problem-bootstrap percentile 95% interval, 20,000
resamples, PCG64 seed 20261014. Selected-minus-.25 is secondary/descriptive.
Also report nDCG@10, MRR@10, recall@100, counts of positive/negative/zero
problem differences, encoding/scoring time and truncation. Keep all signs.
If alpha=0, selected equals dense by construction; do not claim a gain.
No subgroup search, confidence gate, additional significance tests or tuning.

## Minimal implementation plan and checks

Reuse existing cohort hashes, root-TF matrix, fold selector, encoder, source
selector, feature extractor, ranking and metric helpers. Add only a Qwen
source adapter, development runner, external runner and focused synthetic
tests. Existing completed scripts/results remain unchanged. No manuscript or
public-package edits. Receipts, rather than changes to the frozen scientific
commit, bind this separate execution.

1. Test top-100 tail preservation, alpha-zero identity, deterministic ties,
   development-only selection and unchanged Qwen input formatting.
2. Verify old development receipts and prepare exactly 7,800 inference rows.
   Bind code/protocol/source/model files before the private Kaggle launch.
3. Download and verify embeddings; calculate the full development grid,
   fold choices and global alpha. Independently verify selected rankings and
   metrics. Freeze alpha before reading new source contents.
4. Select and seal the new corpus using the rules above; reuse AST extraction
   and Qwen encoding. Compare the three frozen rankings once.
5. Independently recompute metrics and paired contrasts from saved rankings,
   save SHA-256, report unfavorable/null results as well, and update the journal.
