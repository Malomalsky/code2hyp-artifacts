# Code2Hyp branch-product method revision

Date: 2026-06-22

This note records a method-level revision motivated by the observation that
terminal-to-terminal AST paths are structured objects rather than isolated tree
nodes.

## Motivation

The original structure-oriented Code2Hyp variants encoded the serialized
AST-label path as one hyperbolic factor. This is defensible for the available
prefix-trie proxy, but it does not fully address the geometry of a
terminal-to-terminal path. A path is naturally determined by two terminal-side
branches meeting at an LCA. If original AST node identities are available, a
path \(P(u,v)\) can be represented through the two branches
\(a\rightsquigarrow u\) and \(a\rightsquigarrow v\), where
\(a=\operatorname{LCA}(u,v)\). This motivates a product representation with
separate structural factors for the two branches.

The current code2seq Java-small preprocessing does not expose original AST node
IDs or LCA positions. The first implemented branch-product revision therefore
uses a midpoint split of the observed serialized AST-label path as an LCA
proxy. The later latent-LCA variants replace this hard split with a learned
pivot distribution over observed path positions. Neither variant claims to
recover the original AST LCA.

## Implemented method variants

### B52: pure branch-product structural path channel

`B52_code2hyp_branch_product_context_transform_frechet`

The serialized path-label sequence is split into left and right branch masks.
The midpoint label is included in both branches, matching the role of a shared
LCA proxy. Each branch is pooled, projected, mapped to a Poincare factor, and
used in a product structural distance:

\[
d_{\mathcal M}(p,q)
 =
\sqrt{
  d_{\mathbb H}(p_L,q_L)^2
  + d_{\mathbb H}(p_R,q_R)^2
 }.
\]

The context representation passed to the decoder uses the concatenated
left/right logarithmic maps. This is the most direct factorized formulation,
but it replaces the original whole-path channel.

### B53: pure branch-product architecture without structural loss

`B53_code2hyp_branch_product_context_transform_no_struct`

This is the no-structural-loss control for B52. It separates the effect of the
branch-product architecture from the auxiliary structural objective.

### B54: hybrid whole-path decoder with branch-product structural metric

`B54_code2hyp_context_transform_branch_product_bias_frechet`

This variant keeps the whole serialized path representation in the
code2vec-style decoder channel, as in B44, but uses the branch-product factors
for structural attention bias, structural regularization, Frechet diagnostics,
radius-utilization diagnostics, and pairwise structural distances. This is the
current preferred method candidate because it does not discard whole-path
sequence information while still answering the geometric critique.

### B55: B54 without structural loss

`B55_code2hyp_context_transform_branch_product_bias_no_struct`

This is the no-structural-loss control for B54.

### B56: latent-LCA branch-product structural metric

`B56_code2hyp_context_transform_latent_lca_branch_product_bias_frechet`

This variant removes the hard midpoint assumption used by B54. Because the
official preprocessed code2seq `.c2s` files expose serialized AST-label paths
but do not expose original AST node IDs, edge directions, or true LCA
positions, the split point is treated as a latent variable. The model learns a
pivot distribution over the observed path positions:

\[
\pi_j = \operatorname{softmax}_j(e_j^\top q_{\mathrm{pivot}}).
\]

The expected left and right branch weights are then cumulative probabilities:

\[
w^L_j = \sum_{k \ge j} \pi_k,
\qquad
w^R_j = \sum_{k \le j} \pi_k.
\]

The pivot is therefore included in both expected branches. This is more
faithful to the available data than claiming access to the true AST LCA.

### B57: B56 without structural loss

`B57_code2hyp_context_transform_latent_lca_branch_product_bias_no_struct`

This control isolates the latent branch-product architecture from the auxiliary
structural objective.

### B58: latent-LCA branch-product metric with weak center prior

`B58_code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet`

This variant adds a trainable center prior to the latent pivot distribution.
The prior is not a hard midpoint split. It encodes the weak inductive bias that
in a terminal-to-terminal AST path the transition through the common ancestor
is often near the central region of the serialized path, while still allowing
the learned pivot distribution to move:

\[
s_j = e_j^\top q_{\mathrm{pivot}}
      - \alpha \frac{|j - (|p|-1)/2|}{|p|},
\qquad
\alpha = \operatorname{softplus}(\rho).
\]

### B59: B58 without structural loss

`B59_code2hyp_context_transform_latent_lca_prior_branch_product_bias_no_struct`

This is the no-structural-loss control for the center-prior latent-LCA
variant.

### B60: order-aware branch-sequence product metric

`B60_code2hyp_context_transform_branch_sequence_product_bias_frechet`

This variant keeps the same whole-path decoder channel as B54 but replaces
mean-pooled branch factors with two recurrent branch encoders. The left branch
is read in the reverse direction, from the midpoint/LCA proxy toward the left
terminal; the right branch is read from the midpoint/LCA proxy toward the right
terminal. Thus the structural product factors preserve AST-label order within
each branch instead of treating branch labels as an unordered bag.

### B61: B60 without structural loss

`B61_code2hyp_context_transform_branch_sequence_product_bias_no_struct`

This is the no-structural-loss control for B60. It isolates the effect of the
order-aware branch-sequence architecture from the distance-oriented structural
regularizer.

## Code changes

- Added `ast_path_midpoint_branch_masks`.
- Added product-manifold structural output fields:
  - `structural_product_points`;
  - `context_structural_product_points`;
  - `structural_product_distance_metric`.
- Added `_poincare_product_distance`.
- Updated structural regularizers and diagnostics to prefer product factors
  when present.
- Extended Frechet residual and Poincare radius-utilization diagnostics to
  product factors.
- Added branch-product attention scores with four trainable component weights:
  start-token, left-branch, right-branch, end-token.
- Added B52/B53/B54/B55 to the real-data experiment registry.
- Added B56/B57/B58/B59 latent-LCA variants to the real-data experiment
  registry.
- Added B60/B61 order-aware branch-sequence product variants to the real-data
  experiment registry.
- Added unit tests for branch masks, product distance, finite gradients, B52,
  B54, B56, B58, and B60.

## Numerical stability fix

The initial product-distance implementation used
\(\sqrt{\sum_i d_i^2}\). During structural-loss backpropagation, coincident
branch factors can produce an unstable gradient at exactly zero. The
implementation now computes

\[
\sqrt{\sum_i d_i^2 + 10^{-12}},
\]

which preserves the distance scale while making the gradient finite. A
regression test covers coincident product factors.

## Pilot evidence

The following pilot is exploratory. It uses 1024 training examples, 256
validation examples, two epochs, and three random seeds. Its role is to check
the direction of the method revision before running the larger validation
benchmark.

Command:

```bash
PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 1024 \
  --val-limit 256 \
  --epochs 2 \
  --batch-size 64 \
  --max-contexts 64 \
  --structural-eval-limit 256 \
  --model-seeds 101,202,303 \
  --variants B39_code2vec_context_transform_baseline,B44_code2hyp_context_transform_product_bias_frechet,B54_code2hyp_context_transform_branch_product_bias_frechet,B55_code2hyp_context_transform_branch_product_bias_no_struct \
  --output outputs/code2hyp_b54_branch_product_hybrid_pilot_1k_3seeds.json
```

| Variant | F1 mean+-sd | Spearman mean+-sd | Stress mean+-sd | Exact@1 mean+-sd |
|---|---:|---:|---:|---:|
| B39 matched Euclidean baseline | 0.1438+-0.0490 | -0.2459+-0.0062 | 0.4639+-0.0021 | 0.5192+-0.0071 |
| B44 whole-path product-bias Code2Hyp | 0.1438+-0.0490 | -0.2121+-0.0055 | 0.4392+-0.0031 | 0.5128+-0.0126 |
| B54 hybrid branch-product Code2Hyp | 0.1259+-0.0442 | 0.0652+-0.0099 | 0.3581+-0.0042 | 0.6934+-0.0278 |
| B55 B54 without structural loss | 0.1259+-0.0442 | -0.1879+-0.0285 | 0.4412+-0.0121 | 0.6566+-0.0177 |

Paired B54 differences:

| Comparison | Delta F1 | Delta Spearman | Delta Stress | Delta Exact@1 |
|---|---:|---:|---:|---:|
| B54 - B39 | -0.0180 | +0.3111 | -0.1058 | +0.1742 |
| B54 - B44 | -0.0180 | +0.2773 | -0.0812 | +0.1806 |
| B54 - B55 | +0.0000 | +0.2531 | -0.0832 | +0.0368 |

## Latent-LCA pilot evidence

B56/B57 and B58/B59 were added after the first branch-product pilot because the
midpoint split is only a proxy. The following run uses the same small pilot
budget as above: 1024 training examples, 256 validation examples, two epochs,
and three random seeds.

Commands:

```bash
PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 1024 \
  --val-limit 256 \
  --epochs 2 \
  --batch-size 64 \
  --max-contexts 64 \
  --structural-eval-limit 256 \
  --model-seeds 101,202,303 \
  --variants B39_code2vec_context_transform_baseline,B44_code2hyp_context_transform_product_bias_frechet,B54_code2hyp_context_transform_branch_product_bias_frechet,B55_code2hyp_context_transform_branch_product_bias_no_struct,B56_code2hyp_context_transform_latent_lca_branch_product_bias_frechet,B57_code2hyp_context_transform_latent_lca_branch_product_bias_no_struct \
  --output outputs/code2hyp_b56_latent_lca_branch_product_pilot_1k_3seeds.json

PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 1024 \
  --val-limit 256 \
  --epochs 2 \
  --batch-size 64 \
  --max-contexts 64 \
  --structural-eval-limit 256 \
  --model-seeds 101,202,303 \
  --variants B58_code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet,B59_code2hyp_context_transform_latent_lca_prior_branch_product_bias_no_struct \
  --output outputs/code2hyp_b58_latent_lca_prior_branch_product_pilot_1k_3seeds.json
```

| Variant | F1 mean+-sd | Spearman mean+-sd | Stress mean+-sd | Exact@1 mean+-sd |
|---|---:|---:|---:|---:|
| B39 matched Euclidean baseline | 0.1438+-0.0600 | -0.2459+-0.0076 | 0.4639+-0.0026 | 0.5192+-0.0087 |
| B44 whole-path product-bias Code2Hyp | 0.1438+-0.0600 | -0.2121+-0.0068 | 0.4392+-0.0038 | 0.5128+-0.0155 |
| B54 midpoint branch-product Code2Hyp | 0.1259+-0.0542 | 0.0652+-0.0122 | 0.3581+-0.0052 | 0.6934+-0.0341 |
| B55 B54 without structural loss | 0.1259+-0.0542 | -0.1879+-0.0350 | 0.4412+-0.0149 | 0.6566+-0.0217 |
| B56 latent-LCA branch-product Code2Hyp | 0.1328+-0.0544 | 0.0299+-0.0357 | 0.3648+-0.0068 | 0.6577+-0.0217 |
| B57 B56 without structural loss | 0.1314+-0.0525 | -0.1641+-0.0618 | 0.4353+-0.0151 | 0.5676+-0.0327 |
| B58 latent-LCA with center prior | 0.1328+-0.0544 | 0.0323+-0.0335 | 0.3637+-0.0064 | 0.6630+-0.0188 |
| B59 B58 without structural loss | 0.1314+-0.0525 | -0.1657+-0.0594 | 0.4348+-0.0142 | 0.5774+-0.0361 |

## Interpretation

B52 confirms that a strict branch-product replacement strongly changes the
structural geometry but harms the downstream channel under the small pilot
budget. B54 is methodologically stronger: it preserves the whole-path decoder
channel while using branch-product geometry for structural distances. B56 and
B58 are more faithful to the data limitation because they do not assert a fixed
LCA position. However, under the current prefix-trie proxy and small pilot
budget, B56/B58 improve over their no-struct controls but do not outperform B54
on the prefix-tree structural diagnostics. Therefore B54 remains the strongest
small-budget structure-fidelity candidate, while B56/B58 should be reported as
methodological variants rather than as the main empirical winners.

This is an important negative result: the more semantically cautious latent-LCA
model is not automatically better under the current proxy metric. A stronger
claim about true endpoint-LCA geometry requires enriched AST data with original
node IDs, edge directions, and true LCA positions.

## Order-aware branch-sequence pilot evidence

B60/B61 were added after B54/B56/B58 because all previous branch-product
variants pooled AST-label embeddings inside each branch. This loses the
ordered-path information that code2vec/code2seq normally treats as central.

Command:

```bash
PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 1024 \
  --val-limit 256 \
  --epochs 2 \
  --batch-size 64 \
  --max-contexts 64 \
  --structural-eval-limit 256 \
  --model-seeds 101,202,303 \
  --variants B54_code2hyp_context_transform_branch_product_bias_frechet,B55_code2hyp_context_transform_branch_product_bias_no_struct,B58_code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet,B59_code2hyp_context_transform_latent_lca_prior_branch_product_bias_no_struct,B60_code2hyp_context_transform_branch_sequence_product_bias_frechet,B61_code2hyp_context_transform_branch_sequence_product_bias_no_struct \
  --output outputs/code2hyp_b60_branch_sequence_pilot_1k_3seeds.json
```

| Variant | F1 mean+-sd | Prefix Spearman mean+-sd | Stress mean+-sd | Exact@1 mean+-sd | Edit Spearman mean+-sd | Jaccard Spearman mean+-sd |
|---|---:|---:|---:|---:|---:|---:|
| B54 midpoint branch-product Code2Hyp | 0.1259+-0.0542 | 0.0652+-0.0122 | 0.3581+-0.0052 | 0.6934+-0.0341 | 0.6302+-0.0113 | 0.7019+-0.0194 |
| B55 B54 without structural loss | 0.1259+-0.0542 | -0.1879+-0.0350 | 0.4412+-0.0149 | 0.6566+-0.0217 | 0.6244+-0.0177 | 0.7746+-0.0207 |
| B58 latent-LCA with center prior | 0.1328+-0.0544 | 0.0323+-0.0335 | 0.3637+-0.0064 | 0.6630+-0.0188 | 0.5616+-0.0236 | 0.6610+-0.0407 |
| B59 B58 without structural loss | 0.1314+-0.0525 | -0.1657+-0.0594 | 0.4348+-0.0142 | 0.5774+-0.0361 | 0.5959+-0.0096 | 0.7546+-0.0235 |
| B60 order-aware branch-sequence Code2Hyp | 0.1508+-0.0624 | 0.6250+-0.0313 | 0.2140+-0.0083 | 0.6874+-0.0207 | 0.3991+-0.0083 | 0.2290+-0.0198 |
| B61 B60 without structural loss | 0.1508+-0.0624 | 0.1416+-0.0350 | 0.3559+-0.0094 | 0.5226+-0.0783 | 0.6413+-0.1006 | 0.4978+-0.0842 |

Paired B60 differences:

| Comparison | Delta F1 | Delta Prefix Spearman | Delta Stress | Delta Exact@1 |
|---|---:|---:|---:|---:|
| B60 - B54 | +0.0249+-0.0396 | +0.5598+-0.0239 | -0.1441+-0.0037 | -0.0060+-0.0544 |
| B60 - B61 | +0.0000+-0.0000 | +0.4834+-0.0651 | -0.1419+-0.0144 | +0.1648+-0.0697 |

The small pilot makes B60 the strongest current candidate for prefix-trie
structural fidelity. It also improves downstream F1 relative to B54 in this
pilot. However, B60 reduces transfer to edit-distance and Jaccard-distance
proxies. The safe interpretation is therefore narrow: B60 preserves the
ordered prefix-trie geometry of serialized AST-label paths more effectively
than the previous mean-pooled branch-product variants. It should not yet be
claimed as a universally better structural metric for all AST-path proxy
distances.

## Medium validation: 5000 training examples, GRU path encoder

After the 1024-example pilot, B60/B61 were evaluated in a medium validation run
with a stronger whole-path encoder and a larger validation subset. This run is
a medium-scale screening step before a larger frozen-set validation.

Command:

```bash
PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 5000 \
  --val-limit 2048 \
  --epochs 3 \
  --batch-size 128 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --structural-eval-limit 512 \
  --model-seeds 101,202,303 \
  --sample-seed 20260621 \
  --variants B39_code2vec_context_transform_baseline,B44_code2hyp_context_transform_product_bias_frechet,B54_code2hyp_context_transform_branch_product_bias_frechet,B60_code2hyp_context_transform_branch_sequence_product_bias_frechet,B61_code2hyp_context_transform_branch_sequence_product_bias_no_struct \
  --output outputs/code2hyp_b60_branch_sequence_validation_5k_3seeds.json
```

| Variant | F1 mean+-sd | Prefix Spearman mean+-sd | Stress mean+-sd | Exact@1 mean+-sd | Edit Spearman mean+-sd | Jaccard Spearman mean+-sd |
|---|---:|---:|---:|---:|---:|---:|
| B39 matched Euclidean baseline | 0.0729+-0.0830 | -0.3511+-0.0279 | 0.8235+-0.0228 | 0.3670+-0.0217 | 0.4761+-0.0298 | 0.6438+-0.0066 |
| B44 whole-path product-bias Code2Hyp | 0.1027+-0.0651 | 0.8438+-0.0471 | 0.1969+-0.0241 | 0.6770+-0.0225 | 0.3994+-0.0434 | 0.1107+-0.0530 |
| B54 midpoint branch-product Code2Hyp | 0.0826+-0.0787 | 0.6288+-0.0300 | 0.2439+-0.0129 | 0.6816+-0.0125 | 0.5677+-0.0166 | 0.4347+-0.0314 |
| B60 order-aware branch-sequence Code2Hyp | 0.0797+-0.0828 | 0.9485+-0.0016 | 0.0986+-0.0027 | 0.6987+-0.0155 | 0.3963+-0.0087 | 0.1295+-0.0049 |
| B61 B60 without structural loss | 0.0987+-0.0754 | -0.0457+-0.2266 | 0.4513+-0.1230 | 0.4219+-0.0805 | 0.6072+-0.0667 | 0.5898+-0.0903 |

Paired B60 differences:

| Comparison | Delta F1 | Delta Prefix Spearman | Delta Stress | Delta Exact@1 |
|---|---:|---:|---:|---:|
| B60 - B39 | +0.0069+-0.0072 | +1.2996+-0.0290 | -0.7249+-0.0235 | +0.3318+-0.0230 |
| B60 - B44 | -0.0230+-0.0412 | +0.1046+-0.0475 | -0.0984+-0.0243 | +0.0217+-0.0346 |
| B60 - B54 | -0.0029+-0.0066 | +0.3197+-0.0306 | -0.1453+-0.0129 | +0.0171+-0.0172 |
| B60 - B61 | -0.0190+-0.0352 | +0.9942+-0.2256 | -0.3527+-0.1204 | +0.2768+-0.0668 |

Medium-run interpretation:

- B60 is the strongest current prefix-trie structural-fidelity model.
- B44 remains the strongest model in this medium run by downstream F1.
- The B60 structural objective is effective: B60 strongly outperforms B61 on
  prefix Spearman, stress, and Exact@1.
- B60 still weakens edit/Jaccard proxy transfer compared with B54 and B61.

The appropriate claim boundary is therefore:

> B60 is a strong order-aware representation for the reported prefix-trie
> target over serialized AST-label paths. It should be evaluated as the main
> structure-fidelity candidate in the final 25k/5-seed benchmark, while B44
> remains the downstream-F1 reference candidate.

## Multi-metric validation pilot: B62-B65

Evaluating only the metric used in the structural loss can overstate
structural adequacy. To test cross-metric transfer, the training code now
includes a `multi_metric_distance` objective that averages scale-invariant
distance-preservation losses for three serialized AST-label path relations:

- `prefix_tree`: the truncated AST-label prefix-trie distance;
- `edit`: Levenshtein distance between AST-label sequences;
- `jaccard_bigrams`: Jaccard distance between AST-label bigram sets.

New controlled variants:

- B62: B60 branch-sequence product manifold with multi-metric loss.
- B63: B44 product-bias manifold with multi-metric loss.
- B64: Euclidean context-transform control with multi-metric loss.
- B65: L1 context-transform control with multi-metric loss.

Command:

```bash
.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 1000 \
  --val-limit 512 \
  --epochs 2 \
  --batch-size 128 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --structural-eval-limit 256 \
  --model-seeds 101,202,303 \
  --sample-seed 20260621 \
  --variants B39_code2vec_context_transform_baseline,B44_code2hyp_context_transform_product_bias_frechet,B60_code2hyp_context_transform_branch_sequence_product_bias_frechet,B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet,B63_code2hyp_context_transform_product_bias_multi_metric_frechet,B64_code2vec_context_transform_multi_metric_control,B65_code2vec_context_transform_l1_multi_metric_control \
  --output outputs/code2hyp_b62_multi_metric_pilot_1k_3seeds.json
```

| Variant | F1 mean+-sd | Prefix Spearman mean+-sd | Edit Spearman mean+-sd | Jaccard Spearman mean+-sd | Prefix stress mean+-sd | Edit stress mean+-sd | Jaccard stress mean+-sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| B39 matched Euclidean baseline | 0.0657+-0.0334 | -0.1710+-0.0196 | 0.6114+-0.0122 | 0.6428+-0.0100 | 0.6899+-0.0032 | 0.4872+-0.0036 | 0.5167+-0.0045 |
| B44 product-bias Code2Hyp | 0.0657+-0.0341 | 0.2062+-0.0228 | 0.5319+-0.0163 | 0.2053+-0.0275 | 0.4204+-0.0065 | 0.3465+-0.0051 | 0.3531+-0.0052 |
| B60 branch-sequence product, prefix loss | 0.0681+-0.0381 | 0.5658+-0.0058 | 0.4986+-0.0475 | 0.4599+-0.0442 | 0.2603+-0.0056 | 0.3523+-0.0106 | 0.2696+-0.0075 |
| B62 branch-sequence product, multi-metric loss | 0.0681+-0.0381 | 0.4168+-0.0354 | 0.7455+-0.0450 | 0.6974+-0.0317 | 0.3055+-0.0120 | 0.2796+-0.0200 | 0.2058+-0.0042 |
| B63 product-bias, multi-metric loss | 0.0657+-0.0341 | 0.1945+-0.0196 | 0.5546+-0.0135 | 0.2352+-0.0219 | 0.4225+-0.0065 | 0.3411+-0.0053 | 0.3492+-0.0058 |
| B64 Euclidean multi-metric control | 0.0657+-0.0334 | -0.1671+-0.0193 | 0.6149+-0.0115 | 0.6447+-0.0093 | 0.6879+-0.0032 | 0.4850+-0.0035 | 0.5147+-0.0044 |
| B65 L1 multi-metric control | 0.0657+-0.0334 | -0.1651+-0.0196 | 0.6081+-0.0121 | 0.6383+-0.0083 | 0.6971+-0.0044 | 0.4984+-0.0049 | 0.5282+-0.0065 |

Paired B62-B60 deltas over the three seeds:

| Metric | Mean delta | Per-seed direction |
|---|---:|---|
| F1 | +0.0000 | 0/3 positive, 0/3 negative, 3/3 zero |
| Prefix Spearman | -0.1490 | 0/3 positive, 3/3 negative |
| Edit Spearman | +0.2470 | 3/3 positive |
| Jaccard Spearman | +0.2375 | 3/3 positive |
| Prefix stress | +0.0452 | 3/3 higher stress |
| Edit stress | -0.0726 | 3/3 lower stress |
| Jaccard stress | -0.0638 | 3/3 lower stress |

Interpretation:

- B60 is the prefix-trie specialist.
- B62 is the cross-metric branch-product candidate.
- B62 does not improve downstream F1 relative to B60 in this pilot.
- The B62 effect is not reproduced by B63, B64, or B65. This suggests that the
  order-aware two-branch product representation is important; the multi-metric
  loss alone is not sufficient.
- The result remains exploratory because it uses a 1000-example local budget and
  three model seeds.

This is currently the cleanest evidence for the revised scientific framing:
terminal-to-terminal AST-label paths should be treated as objects whose
geometry depends on the structural relation being preserved. A single
prefix-trie result is not enough to claim universal AST-path adequacy.

## Medium multi-metric validation: 5000 training examples

The B62-B65 controls were then evaluated under the same 5000-example,
3-seed medium protocol used for B60:

```bash
.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --train-limit 5000 \
  --val-limit 2048 \
  --epochs 3 \
  --batch-size 128 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --structural-eval-limit 512 \
  --model-seeds 101,202,303 \
  --sample-seed 20260621 \
  --variants B60_code2hyp_context_transform_branch_sequence_product_bias_frechet,B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet,B63_code2hyp_context_transform_product_bias_multi_metric_frechet,B64_code2vec_context_transform_multi_metric_control,B65_code2vec_context_transform_l1_multi_metric_control \
  --output outputs/code2hyp_b62_multi_metric_validation_5k_3seeds.json
```

| Variant | F1 mean+-sd | Prefix Spearman mean+-sd | Edit Spearman mean+-sd | Jaccard Spearman mean+-sd | Prefix stress mean+-sd | Edit stress mean+-sd | Jaccard stress mean+-sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| B60 branch-sequence product, prefix loss | 0.0797+-0.0828 | 0.9485+-0.0016 | 0.3963+-0.0087 | 0.1295+-0.0049 | 0.0986+-0.0027 | 0.3719+-0.0024 | 0.3515+-0.0009 |
| B62 branch-sequence product, multi-metric loss | 0.0787+-0.0836 | 0.5566+-0.0138 | 0.8986+-0.0061 | 0.7064+-0.0126 | 0.2437+-0.0024 | 0.2161+-0.0009 | 0.1866+-0.0075 |
| B63 product-bias, multi-metric loss | 0.0993+-0.0693 | 0.3604+-0.0726 | 0.7635+-0.0404 | 0.6095+-0.0082 | 0.2880+-0.0142 | 0.2781+-0.0063 | 0.2117+-0.0058 |
| B64 Euclidean multi-metric control | 0.0809+-0.0772 | -0.1343+-0.0499 | 0.5717+-0.0446 | 0.5666+-0.0203 | 0.5374+-0.0460 | 0.3742+-0.0461 | 0.3551+-0.0508 |
| B65 L1 multi-metric control | 0.0996+-0.0725 | -0.1237+-0.0270 | 0.6018+-0.0139 | 0.6296+-0.0169 | 0.5695+-0.0569 | 0.3942+-0.0503 | 0.3856+-0.0656 |

Paired B62 deltas over the three medium-validation seeds:

| Comparison | Delta F1 | Delta Prefix Spearman | Delta Edit Spearman | Delta Jaccard Spearman |
|---|---:|---:|---:|---:|
| B62 - B60 | -0.0010 | -0.3919 | +0.5022 | +0.5769 |
| B62 - B63 | -0.0206 | +0.1962 | +0.1351 | +0.0969 |
| B62 - B64 | -0.0022 | +0.6909 | +0.3268 | +0.1398 |
| B62 - B65 | -0.0209 | +0.6803 | +0.2967 | +0.0768 |

Medium-validation interpretation:

- B60 remains the cleanest prefix-trie structural-fidelity model.
- B62 is the strongest cross-metric structural model across prefix, edit, and
  Jaccard diagnostics, but it intentionally sacrifices the near-perfect
  prefix-trie fit achieved by B60.
- B63 is a relevant downstream/cross-metric compromise: it has higher mean F1
  than B62 in this medium run, but lower structural scores.
- B64/B65 show that Euclidean/L1 multi-metric controls do not reproduce the
  product-manifold structural profile. B65 has competitive F1, but its
  structural stress remains substantially worse.
- The article should frame B60/B62/B63 as a Pareto set, not as a single
  universally dominant variant.

## Next validation step

Run the larger post-review benchmark with B54/B55/B56/B57/B58/B59/B60/B61/B62/B63/B64/B65 added
to the existing factorial set:

```bash
PYTHONPATH=. \
.venv/bin/python \
scripts/run_code2hyp_resumable_benchmark.py \
  --data-root data/code2seq_java_small \
  --eval-split val \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --token-dim 32 \
  --structural-dim 32 \
  --path-encoder gru \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303,404,505 \
  --sample-seed 20260621 \
  --structural-eval-limit 512 \
  --structural-eval-seed 314159 \
  --variants B39_code2vec_context_transform_baseline,B47_code2vec_context_transform_distance_control,B50_code2vec_context_transform_l1_baseline,B51_code2vec_context_transform_l1_distance_control,B48_code2hyp_context_transform_product_bias_no_struct,B49_code2hyp_context_transform_product_bias_near_euclidean,B44_code2hyp_context_transform_product_bias_frechet,B54_code2hyp_context_transform_branch_product_bias_frechet,B55_code2hyp_context_transform_branch_product_bias_no_struct,B56_code2hyp_context_transform_latent_lca_branch_product_bias_frechet,B57_code2hyp_context_transform_latent_lca_branch_product_bias_no_struct,B58_code2hyp_context_transform_latent_lca_prior_branch_product_bias_frechet,B59_code2hyp_context_transform_latent_lca_prior_branch_product_bias_no_struct,B60_code2hyp_context_transform_branch_sequence_product_bias_frechet,B61_code2hyp_context_transform_branch_sequence_product_bias_no_struct,B62_code2hyp_context_transform_branch_sequence_product_bias_multi_metric_frechet,B63_code2hyp_context_transform_product_bias_multi_metric_frechet,B64_code2vec_context_transform_multi_metric_control,B65_code2vec_context_transform_l1_multi_metric_control \
  --output outputs/code2hyp_branch_product_revision_validation_25k_5seeds.json
```

Only after this larger run should the manuscript decide whether B60 replaces
B44 as the main structure-oriented variant. The medium validation suggests a
three-candidate framing: B44 for downstream F1, B60 for prefix-trie structural
fidelity, and B62 for cross-metric AST-label path geometry.
