# Frequency and ancestor-category development study

10 October 2026. The author approved a bounded CPU investigation of three
hypotheses, before computing this study's scores. This protocol includes the
minimal implementation plan. It is not a confirmatory preregistration.

## Scope and competing explanations

The previous pilot favored ordinary TF-IDF fragments over conditional
centering. Distinguish three explanations, without modifying that candidate:

- H1: weighting rare fragment categories improves ranking beyond raw counts.
- H2: repeated fragment occurrences dominate similarity; square-root TF helps.
- H3: retaining the ancestor category adds value under matched frequency rules.

Use the complete 2x2x2 factorial: retained versus masked ancestor category,
raw TF versus square-root TF, IDF absent versus present. No other exponents,
smoothing choices, learned gates, residual normalization or BM25 parameter
search will be introduced after results. A null or unfavorable result stops
escalation of that factor here. All eight configurations and all twelve
single-factor contrasts will be reported, not only the best combination.

## Data and evaluation boundary

Reuse the verified counts, group assignment and gallery from
`followup_results/conditional_structure_20261010/`. The previous design's 325
calibration problems (2,600 queries) are the entire development set for this
study. Do not score queries from its other 325 problems. The fixed full gallery
still contains 5,200 programs, including the other problems as distractors;
label-free gallery vocabulary and IDF are index statistics, not query labels.
No new source programs, neural inference, training, Java test or CPRet test
will be accessed. Existing studies and their files remain unchanged.

All 325 problems have previously been examined, and their earlier aggregate
results informed this design. Five-fold cross-validation is only an internal
development comparison, not a restoration of independent confirmation. Sort
these IDs by SHA-256 of `frequency-factorial-20261010:<id>`; use round-robin
assignment into five folds of 65 problems. Eight queries stay with their problem.

Mask counts by summing the retained gallery vocabulary using the already
frozen group map. Project queries before masking. This intentionally excludes
the additional OOV recovery of the earlier full masked representation. Thus
both branches use the same underlying feature occurrences; it is a controlled
masking comparison, not a replacement for the previous masked baseline.

## Exact transforms and measurements

For each representation, use `idf_f=1+log((G+1)/(df_f+1))` computed only from its
gallery. Feature value is `x_f**beta * idf_f**a`, with beta in {1, 1/2} and a in
{0, 1}, followed by L2 normalization. Zero vectors stay zero. IDF is recomputed
after masking because grouped document frequency is a union, not a sum.
Root TF is applied before IDF; this is not the square root of TF-IDF.

Combine each structural cosine S with unchanged dense cosine D using
`(1-alpha)*D+alpha*S`. Reuse the exact 24-value grid from the previous pilot.
Within each fold, choose alpha by mean problem AP@8 on the other 260 problems;
exact ties choose the smallest alpha. Record the selected weight, training
mean, held-fold mean and pooled out-of-fold problem AP@8 for every method.
Fold assignments are fixed before scoring. Metrics for all weights are
development artifacts, never external-test results.

Also report alpha=0.15 for every configuration, fixed from the preceding pilot
before this study, and alpha=1 (standalone). The common 0.15 comparison isolates
the feature change from different fitted fusion weights. Different score
distributions can still affect fusion; do not claim causal semantic isolation.
The twelve matched single-factor contrasts report mean differences and counts
of higher/lower/unchanged problem AP@8, both fixed-weight and out-of-fold.
Use the existing MAP@8 denominator 8, eight positives/query, full gallery,
and UTF-8 document-ID tie order. Do not add significance tests or bootstrap
intervals to this screening study.

## Mathematical rationale and limits

For a fixed positive diagonal weight w, `phi_beta(x)=normalize(w*x**beta)` is
invariant to uniform rescaling x -> c*x for c>0, both at beta=1 and beta=1/2.
The change therefore is not merely removal of document length. If one count
is multiplied by m in both programs, its unnormalized matching contribution
scales by m**(2*beta): quadratic for raw TF, linear for root TF. The denominator
also changes, so no ranking improvement follows from this algebra alone.

Without IDF, root-TF cosine equals the Bhattacharyya coefficient of normalized
count distributions p and q. Under the frozen grouping h, it decomposes as
`B(p,q)=sum_h sqrt(p_h*q_h)*B(p(.|h),q(.|h))`. Terms with a zero marginal have
zero weight. Consequently `B(mask(p),mask(q))-B(p,q)` is a nonnegative weighted
conditional-disagreement penalty. Equality holds if conditional distributions
agree for every group with positive mass in both programs. Proof: expand the
inner sum within each group and apply Cauchy-Schwarz to two unit square-root
conditional vectors. A larger similarity after masking does not imply better
rankings: candidates can have different penalties.

With root TF and IDF, the effective distribution is proportional to `idf**2*x`,
not x. When the masked branch refits IDF it is not the marginal of that same
distribution, so the monotonicity statement must not be applied to this pair.
These identities are standard kernel/probability reasoning, not a new theorem
or evidence of semantic correctness. We test their practical relevance here.

## Literature checked before computation

- Arandjelovic and Zisserman, CVPR 2012, [RootSIFT, Section 2](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf): square-root histograms and Hellinger similarity are established; their image result is not evidence for code.
- Perronnin, Sanchez and Mensink, ECCV 2010, [Sections 3.1--3.2](https://www.micc.unifi.it/bagdanov/pdfs/peronnin_etal_ECCV10.pdf): power normalization is established; we do not resume tuning the failed residual candidate.
- Bruch, Gai and Ingber, [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/html/2210.11934v2): calibrated convex fusion is prior work, not our novelty.
- Galimzyanov, Kolomyttseva and Bogomolov, [Practical Code RAG at Scale](https://arxiv.org/html/2510.20609v1): simple retrieval baselines can be competitive, but their code-completion/bug-localization tasks differ from our problem-equivalence retrieval.

## Implementation and checks

Scope: new `frequency_factorial_20261010.py`, its focused unittest file, this
protocol, one new output folder, short results note and journal update. Reuse
previous input checks, sparse counts, normalization, hashes and ranking helper.
Do not refactor frozen callers, alter the paper, create cloud jobs or overwrite
outputs. No package installation. Mac15,12, 24 GiB RAM, 8 logical CPUs and
110 GiB free disk were checked; use one CPU thread and query blocks.

1. Write tests first: root-kernel identity/grouping/zero cases, identical-feature
   scaling invariance, and folds that cannot use their own problem values to
   select alpha. Use the existing independent metric for tie/cutoff checks.
2. Verify the completed pilot and counts/group SHA, bind all new source files,
   protocol and folds before scores. Form all eight feature maps from counts.
3. Score only 2,600 development queries. Reproduce previous raw/IDF retained
   similarities, save each matrix/hash and 325x24 problem-metric table.
4. Select fold weights, report all outcomes and contrasts; independently
   reconstruct selected/fixed/standalone AP@8 using the previous rank helper.
5. Save hashes, costs and a bounded interpretation. Do not promote the highest
   development mean to a confirmed improvement; an unused cohort is required.
