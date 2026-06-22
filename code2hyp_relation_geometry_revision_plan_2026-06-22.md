# Code2Hyp relation-geometry revision plan

Date: 2026-06-22

## Current status

The current artifact is best treated as a controlled research workspace for
relation-dependent geometry of terminal-to-terminal AST-path contexts. It is
not yet a final claim that hyperbolic geometry generally improves code2vec or
code2seq. The strongest defensible direction is narrower and more original:

> AST nodes are tree-like, but AST paths are relation-dependent objects. A path
> space can behave as a prefix-trie tree metric, an edit-distance sequence
> space, a set-Jaccard n-gram space, or an endpoint-product space depending on
> which relation is being preserved.

## Closed implementation issues

1. Lorentz distance no longer uses a positive `acosh` clamp. It is computed
   from the Minkowski norm of `x - y`, so `d(x, x) = 0` exactly, including
   small-curvature controls.
2. Prefix-trie terminology is now explicit. New code should call
   `longest_common_prefix_length` and `prefix_trie_distance`. Legacy
   `ast_sequence_lca_depth` and `ast_sequence_tree_distance` remain as aliases
   only for old artifacts.
3. Multi-label target cardinality now counts unique target subtokens. This
   matches the multi-hot target representation.
4. Encoders can sample path contexts with a `context_sample_seed` instead of
   always taking the first `max_contexts` contexts. Validation and test
   encoding remain deterministic unless sampling is explicitly requested.
5. Jaccard is described as set-Jaccard over directed AST-label n-grams, not as
   a full AST distance.
6. README no longer recommends test-split runs for model selection. New
   exploratory and selection runs use `--eval-split val`; `test` is reserved
   for a frozen final configuration.

## Claims that are not yet allowed

The following statements are too strong until the next experiments are done:

1. "Product-hyperbolic geometry outperforms Euclidean geometry."
2. "B62 demonstrates cross-metric generalization."
3. "The midpoint branch split is a true LCA split."
4. "The current Jaccard relation captures full AST structure."
5. "Downstream method-name F1 proves practical usefulness of structural
   faithfulness."

The current safe statement is:

> Branch-sequence product-hyperbolic models with structural supervision show
> strong in-objective structural fidelity on serialized AST-label path
> relations under local Java-small budgets, but geometry-specific causality and
> held-out structural transfer require additional controls.

## Required experiments before a strong article

### 1. Architecture-matched geometry controls

Use the same two-segment branch-sequence scaffold and change only the geometry:

| Control | Geometry | Purpose |
|---|---|---|
| E x E, product L2 | Euclidean factors | Matched Euclidean baseline |
| E x E, product L1 | Euclidean factors | Matched L1 product baseline |
| H(1e-4) x H(1e-4) | Near-Euclidean hyperbolic factors | Curvature-collapse control |
| H(1) x H(1) | Fixed hyperbolic factors | Fixed-curvature control |
| H(c1) x H(c2) | Trainable product curvature | Adaptive-curvature control |
| single H | Matched total dimension | Product-vs-single-manifold control |

All controls must keep token encoders, branch GRUs, hidden dimensions,
attention, decoder, structural objective, schedule, seeds, and training budget
constant.

### 2. Leave-one-relation-out structural transfer

Train on one relation or relation subset and evaluate on held-out relations:

| Training relation | Held-out evaluation |
|---|---|
| prefix | edit, set-Jaccard |
| edit | prefix, set-Jaccard |
| set-Jaccard | prefix, edit |
| prefix + edit | set-Jaccard |
| prefix + set-Jaccard | edit |
| edit + set-Jaccard | prefix |

B62 can be called "multi-objective" now. It can be called "cross-metric" only
after this matrix shows held-out transfer.

### 3. Exact prefix-trie baseline

For a serialized path sequence `s`, define `P(s)` as the set of prefix-trie
edges from the trie root to `s`. The sparse incidence embedding

```text
phi(s)_e = 1 if e in P(s), otherwise 0
```

satisfies

```text
||phi(s) - phi(t)||_1 = |s| + |t| - 2 * LCP(s, t).
```

This gives an exact high-dimensional L1 realization for prefix-trie distance.
The article should compare learned low-dimensional geometries against this
exact baseline through rate-distortion curves.

### 4. Rate-distortion and resource curves

Evaluate dimensions `4, 8, 16, 32, 64, 128` with:

1. normalized stress;
2. Spearman rank correlation;
3. nearest-neighbor recall;
4. parameter count;
5. memory footprint;
6. inference latency.

This reframes the question from "which model wins at one dimension" to "which
geometry compresses which structural relation most efficiently."

### 5. Enriched AST extraction

The current code2seq files expose serialized AST-label paths, not true AST
node identifiers. A stronger study needs an enriched extractor with:

1. method ID;
2. endpoint node IDs;
3. parent IDs;
4. node depths;
5. child indices;
6. edge directions;
7. source spans;
8. true LCA position;
9. full path before truncation;
10. serialized model input after truncation.

This will separate true AST geometry from sequence-proxy geometry.

### 6. Practical structural retrieval benchmark

Create controlled query pairs:

1. alpha-renaming of local identifiers;
2. formatting-only transformations;
3. semantics-preserving rewrites;
4. controlled structural mutations;
5. lexical hard negatives with different structure.

Primary metric:

```text
P[d(z(x), z(T_safe(x))) + margin < d(z(x), z(T_struct(x)))]
```

Additional metrics:

1. Recall@k;
2. MRR;
3. hard-negative ranking;
4. distance calibration by mutation size;
5. robustness after alpha-renaming.

This is the cleanest route from structural faithfulness to practical utility.

## Manuscript framing

The manuscript should be rewritten around relation-dependent AST-path geometry.
Downstream F1 should be secondary. A suitable title is:

> Relation-Dependent Geometry of Terminal-to-Terminal AST-Path Representations

or, if the Code2Hyp name is retained:

> Code2Hyp: A Controlled Study of Product Geometry in AST-Path Context Representations

The abstract should not lead with a single F1 gain. It should lead with the
geometric finding and then report downstream results as a diagnostic trade-off.

## Next engineering tasks

1. Add architecture-matched branch-sequence geometry variants.
2. Add a frozen experiment registry for leave-one-relation-out runs.
3. Add cached target distance matrices for prefix, edit, and set-Jaccard.
4. Add exact sparse L1 prefix-trie baseline.
5. Add a result manifest with config hashes.
6. Add CI, lockfile, LICENSE, CITATION.cff, and release checklist before public
   article submission.
