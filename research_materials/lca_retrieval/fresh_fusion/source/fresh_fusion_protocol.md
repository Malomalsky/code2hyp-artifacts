# Prospective fixed-fusion evaluation, 2026-10-02

Question: does the previously explored node-count/fragment RRF improve retrieval
on previously unused Codeforces problems? This local prospective follow-up is
fixed before downloading or inspecting the new shard contents or scores. It is
not an external preregistration, independent-team replication, new domain, or
amendment to frozen CodeNet Stage A/B.

## Fixed sample

Use deepmind/code_contests revision
802411c3010cb00d1b05bad57ca77365a3c699d6, exactly train shards 15--23 inclusive.
Only filenames, official sizes and LFS SHA-256 metadata have been consulted.
The runner binds those nine files before downloading their contents.

Reuse correct Codeforces Python3 solutions, 65,536-byte and 4,096-stdlib-AST-node
bounds and at least two augmented terminals. Exclude all Codeforces problem
IDs and eligible normalized-source/token/alpha-AST (D0/D1/D2) hashes from
previously inspected test, validation and train shards 0--14, including
unselected programs/problems. Exclude the complete Python800 inventory too.
Remove new cross-problem D0/D1/D2 matches; within each problem retain one
program per D2 class, smallest D0 first. These are the existing filters,
not guarantees against all near clones, shared authors or related tasks.

Take exactly 650 eligible problems by ascending
SHA-256('fresh-fusion-problems-20261002:' + problem_id). Within each, take the
first 16 distinct programs by
SHA-256('fresh-fusion-sources-20261002:' + problem_id + ':' + D0), assigning
the first eight queries and the next eight gallery. Save this manifest before
scoring. If fewer than 650 remain, stop without scores or retrospective frame
expansion. No replacement after an unfavorable result. This size preserves
the previous 650-problem/5,200-program gallery task; it is not a new power claim.

## Fixed methods and analysis

Reuse raw reversal-canonical path 2+3-grams, their ancestor-category-masked
version, and AST node-type counts. The saved seed-20260711 category vocabulary
and deterministic 64-path sampler are unchanged; checkpoint SHA-256 is
e402d017eb5ea0ddc7d9501a1d690fc713a7e7c93289accccd56ecd92eb204e7.
No learned weights are used for scoring and no training occurs. Use gallery-only
feature vocabularies, float64 L2-normalized cosine vectors, and ascending UTF-8
gallery identifiers for exact ties. Ancestor positions/markers are preserved.

Use exactly the earlier RRF rule: 1/(60 + node_rank) + 1/(60 + fragment_rank),
equal weights and one-based full-gallery ranks. Report five methods: nodes,
retained fragments, masked fragments, nodes+retained, nodes+masked. No transition
model, learned baseline, tuning, new ablation, subgroup or alternative fusion.

The comparison endpoint is problem-macro MAP@8; each problem averages its eight
query AP@8 values. MRR and Success@10 are descriptive. Exactly three contrasts:
retained fusion minus nodes, retained fusion minus retained fragments, and
retained fusion minus masked fusion. Use the existing 20,000 paired problem
percentile-bootstrap draws, PCG64 seed 20261002, shared within this cohort.
Report all differences, 95% intervals and 98.333333% marginal intervals
(Bonferroni adjustment for three comparisons; approximate family coverage).

Support for improvement over both components requires both corresponding
adjusted lower bounds above zero. Support for ancestor utility within fusion
requires the third lower bound above zero. No minimum practical-effect cutoff
is introduced. Zero-crossing intervals are inconclusive, not equivalence.
All signs are reported. No pooling with development cohorts or selection of
further tests based on this result. Intervals condition on the selected gallery.
Success would support this fixed combination within the sampled contest-code
setting, not SOTA, semantic equivalence, new-algorithm novelty or industrial use.

## Execution and integrity

One CPU thread; no GPU, local training, paid resource or Java test access.
Frozen commit c419f6418056b618ce373ebd6fafe6601ff51566 and Stage A/B files remain
unchanged. Bind this protocol, runner, reused helpers, old inputs and checkpoint
before new download. Keep attrition, selected identities, five score matrices,
bootstrap indices, metrics and hashes. Verify ranking and fusion ties with the
existing toy self-check; independently reconstruct scores/metrics/intervals
using the existing NumPy reviewer helper. These are implementation checks, not
additional scientific hypotheses. Preserve every earlier result and public v1.

Integrate the outcome into the existing article by replacing the provisional
fusion discussion and condensing its table, not adding a separate long section.

