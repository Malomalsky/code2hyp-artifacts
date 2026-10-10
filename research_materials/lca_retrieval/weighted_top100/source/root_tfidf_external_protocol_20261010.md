# Frozen new-problem validation of root-TF-IDF, 2026-10-10

## Question and scope

Does the root-TF-IDF fragment hybrid selected in the October 10 exploratory
factorial improve retrieval on previously unused problems? This is a prospective
new-problem replication within Codeforces/Python, not an independent-domain
benchmark, not proof of program equivalence, and not a claim that TF-IDF,
square-root weighting, cosine similarity or linear fusion is novel.

No Stage B or CPRet settings, outputs or evaluation are changed. The manuscript
is not updated until this study produces verified results. Java test stays closed.

## Data fixed before new contents or scores

Use deepmind/code_contests revision
802411c3010cb00d1b05bad57ca77365a3c699d6, training shards 24--32 only.
Official file sizes and LFS SHA-256 identifiers are recorded in the preparation
script and checked on download. Shards 33--38 remain unused. Target 650
Codeforces problems, 16 accepted Python3 solutions per problem, eight queries
and eight gallery solutions, with the full 5,200-program gallery for every query.
The target preserves the previous evaluation size; no power guarantee is claimed.
If fewer than 650 problems survive eligibility, stop without scores and report
the shortfall, rather than extend the sample after seeing performance.

Reuse the existing source-size, AST-size, parser and terminal eligibility rules.
Exclude all previous Codeforces problem IDs: test/validation and training
shards 0--14 from the saved fresh-fusion exclusion receipt, shards 15--23 from
their verified original files, all archived evaluated cohort metadata, and
CPRet's already sealed selection metadata. Only CPRet IDs/fingerprints are read,
not its retrieval results or new source data. Exclude d0/d1/d2 matches against
all archived evaluated Python program metadata and the sealed CPRet selection;
also exclude canonical source hashes in the preserved CodeNet selected-source
audit. The full 240,000-record CodeNet Python inventory has been deleted, as
have original-source fingerprints of previously *unselected* CodeContests
candidates; do not claim these stronger exclusions. This limitation was found
before downloading new data. All old Codeforces problem IDs are excluded anyway.
Within new candidates remove cross-problem d0/d1/d2 collisions and within-problem
d2 duplicates using the established rules. Select problems and solutions by
SHA-256 order with the fixed prefix root-tfidf-external-20261010, then seal
membership before computing similarities. Parsing never executes submitted code.

## Frozen methods

Use the unchanged 64-path AST sampler, vocabulary and category-only terminal
policy from the previous experiment; extract reversal-invariant length-two and
length-three path fragments. Fit the retained fragment vocabulary on the new
gallery only. Project queries onto it *before* grouping masked fragments.
Masking removes only the marked ancestor category and aggregates matching counts.
This preserves the matched-ablation convention used during development.
Fit each representation's IDF on gallery documents, not queries or relevance:
idf(f) = 1 + log((G+1)/(df(f)+1)). No supervised training is performed.

For the candidate, phi(x) = normalize(sqrt(x) * idf), with square root before
IDF, L2 normalization, and zero mapped to zero. Similarities are cosine scores.
D is the same CodeRankEmbed revision 3c4b60807d71f79b43f3c4363786d9493691f8b1,
float32, CLS pooling, normalized output, batch 16, 512 tokens, no query prefix.
Reuse the bounded/checkpointed Colab encoder, changing only cohort binding.
No local neural inference or training; only free remote GPU, no purchases.
Do not interrupt CPRet or allocate a duplicate runtime. Checkpoints go into a
separate Code2Hyp-Colab/root_tfidf_external_20261010 directory.

Report every fixed method, with no new weight grid:

| Method | Score |
|---|---|
| D | Dense cosine |
| Candidate | 0.80 D + 0.20 retained root-TF-IDF cosine |
| Previous TF-IDF | 0.85 D + 0.15 retained raw-TF-IDF cosine |
| Matched mask | 0.80 D + 0.20 masked root-TF-IDF cosine |
| Simpler root | 0.75 D + 0.25 retained root-TF cosine, no IDF |
| Node counts | 0.50 D + 0.50 AST-node-count cosine |
| Fixed-weight control | 0.85 D + 0.15 retained root-TF-IDF cosine |
| Lexical-dense reference | equal-weight RRF(D, lexical TF-IDF), k=60 |

Weights come from the completed development studies, not this cohort. The
last reference is the previously implemented, untuned standard RRF comparator;
the lexical linear-fusion weight selected previously was zero. Do not introduce
a new fitted lexical weight. Report standalone structural and lexical scores
as descriptive context, not additional independent hypotheses.

## Analysis and decision

Primary endpoint: problem-macro MAP@8, eight relevant gallery items per query,
eight queries per problem. Ties use UTF-8 source paths. Primary comparison:
Candidate minus Previous TF-IDF. Two secondary comparisons: Candidate minus D
and Candidate minus Matched mask. Report all three signed differences with
20,000 paired problem-bootstrap resamples, PCG64 seed 20261010. Preserve the
existing percentile 95% intervals and three-comparison Bonferroni-adjusted
98.333% intervals. This is uncertainty over these sampled problems conditional
on the fixed gallery, not across independent datasets, GPU runs or languages.

Only a positive lower adjusted interval for the primary contrast supports a
replicated gain over the previous method. The ancestor-specific claim separately
requires the matched-mask contrast. Report all comparator means, problem-level
differences and negative outcomes. The simpler-root and lexical-dense methods
can limit the practical claim even if the primary contrast is positive.
Fixed-weight control separates the effect of the feature transform from the
changed fusion coefficient. Do not stop early or retune after observing results.

Save source, membership, feature, model, embedding and result hashes. Independently
recompute rankings using the existing reviewer implementation. Measure feature
extraction and scoring time, encoding time and GPU memory; do not count downloads
as model inference. Exclusion and hash checks protect the experiment, rather
than constitute scientific contributions or extra article sections.

## Established-method sources

- Arandjelovic and Zisserman (CVPR 2012), root features/Hellinger:
  https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- Bruch, Gai and Ingber, analysis of hybrid retrieval fusion:
  https://arxiv.org/html/2210.11934v2
- CodeRankEmbed author implementation, frozen revision used previously:
  https://github.com/gangiswag/cornstack/tree/c4f3e1289df917275860affc03e640ddf99d466f

These references motivate standard components; none proves the proposed
combination improves this task. That is the purpose of the new evaluation.
