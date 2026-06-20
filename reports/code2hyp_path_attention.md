# Code2Hyp path-node attention pilot

Date: 2026-06-16

## Purpose

The schedule sweep showed that B19/B20/B21/B22 mostly test optimization timing,
not a new representation. This pilot tests a representation-level change inside
the AST-path encoder.

The baseline family is:

```text
B17 = hyperbolic AST-path message passing + ordinary path pooling
B23 = B17 + learned attention over AST nodes inside each path
B24 = B23 + linear structural-rank schedule
B25 = B23 + depth-aware root-to-leaf attention bias
B26 = B25 + linear structural-rank schedule
B27 = B23 + bidirectional monotone attention-profile regularization
B28 = B23 + soft tree-distance calibration of path-node attention
B29 = B23 + dual root/detail attention with separation regularization
B30 = B29 + global structural-rank objective
B31 = B29 + soft global structural-rank objective
```

B23/B24/B25/B26/B27/B28/B29/B30/B31 are motivated by a concrete limitation of B17: after message
passing along the AST path, B17 still compresses the path through the same
global sequence pooling mechanism. B23 instead lets the model learn which nodes
of a path are most informative before the full path-context is mapped to the
Poincare ball. B25 adds a minimal structural prior directly inside that node
attention mechanism, and B26 tests whether that prior should be combined with
scheduled structural-rank supervision. B27 tests a stronger non-parametric
constraint on the same attention profile: attention along a root-to-leaf AST
path should be approximately monotone in one of the two directions, but the
direction is not hard-coded. B28 tests a stronger global calibration hypothesis:
the attention distribution over AST nodes should induce pairwise tree distances
that remain consistent with ordinary leaf-to-leaf AST distances. B29 changes
the representation itself: instead of asking one attention distribution to
serve both abstraction and detail, it introduces separate root/abstract and
leaf/detail path-node attention channels. B30 keeps the B29 architecture but
adds a global structural-rank objective, testing whether local root/detail
separation and global AST-distance ordering are compatible under the same
training budget. B31 keeps the same architecture and tests a softer version of
that hypothesis: local separation remains the main structural signal, while the
global rank term is retained only as a fractional regularizer.

## Implementation

For each AST-path context, node embeddings are first updated by the same
hyperbolic message passing as in B17. The updated node logs are then pooled by
masked attention:

```text
score_i = <h_i, q_path>
alpha_i = softmax(score_i over valid AST nodes)
path = sum_i alpha_i h_i
```

where `h_i` is the tangent/log representation of the updated AST node and
`q_path` is a learned structural query vector. The rest of the model is the
same full-context hyperbolic code2vec mechanism used by B17.

Parameter cost:

```text
B23 parameter_count = B17 parameter_count + structural_dim
B24 parameter_count = B23 parameter_count
B25 parameter_count = B23 parameter_count + 1
B26 parameter_count = B25 parameter_count
B27 parameter_count = B23 parameter_count
B28 parameter_count = B23 parameter_count
B29 parameter_count > B23 parameter_count
B30 parameter_count = B29 parameter_count
B31 parameter_count = B29 parameter_count
```

So the comparison is not parameter-free, but the extra capacity is small and
mechanistically localized. B27 and B28 add no parameters; they change only the
training objective.

B25 modifies only the attention score:

```text
score_i = <h_i, q_path> + a * depth_i
depth_i in [-1, 1]
```

where `depth_i` is a centered root-to-leaf position inside the extracted AST
path and `a` is one learned scalar. The purpose is not to add capacity, but to
test whether node attention should be biased by the structural role of an AST
node along the path.

B27 adds a bidirectional monotonicity penalty over adjacent attention weights:

```text
L_mono =
  mean_path min(
    mean_i ReLU(alpha_i - alpha_{i+1}),
    mean_i ReLU(alpha_{i+1} - alpha_i)
  )
```

The `min` makes the prior direction-free: a path may put more attention on
root-near nodes or on leaf-near nodes, but a spiky nonmonotone profile is
penalized. This tests whether a low-complexity, interpretable attention profile
is enough to recover structural behavior without adding model capacity.

B28 adds a soft tree-distance calibration loss. For two AST paths `p` and `q`
with node-attention distributions `alpha` and `beta`, the attention-induced
distance is:

```text
D_att(p, q) = sum_i sum_j alpha_i beta_j d(prefix_i(p), prefix_j(q))
```

where `d(prefix_i(p), prefix_j(q))` is the ordinary tree distance between the
two selected prefix nodes. The penalty is the same scale-invariant distance
loss used elsewhere, but applied between `D_att(p, q)` and the ordinary
leaf-to-leaf AST distance. This tests whether node attention can be made
globally metric-consistent without adding parameters.

B29 uses two path-node attention heads:

```text
score_root_i = <h_i, q_root> - depth_i
score_leaf_i = <h_i, q_leaf> + depth_i
path_root = sum_i alpha_root_i h_i
path_leaf = sum_i alpha_leaf_i h_i
path = tanh(W [path_root; path_leaf])
```

The auxiliary separation loss encourages the expected leaf-channel depth to be
larger than the expected root-channel depth:

```text
L_sep = mean_path ReLU(margin - (E_leaf[depth] - E_root[depth]))
```

This is the first path-attention variant in this sequence that explicitly
models the dissertation-level intuition: root-near AST nodes carry
abstraction-level information, while leaf-near nodes carry detail-level
information.

B30 keeps the B29 dual-head encoder and changes only the training objective:

```text
L_struct = L_sep + L_rank
```

where `L_sep` is the local root/detail separation loss above and `L_rank` is
the batch structural-rank loss used in earlier B18/B19/B24/B26 controls. This
tests a stricter hypothesis than B29: the model should not only separate
root-near and leaf-near AST evidence inside a path, but also preserve global
AST-distance order between path-context representations.

B31 keeps the B29 dual-head encoder and uses a softer local/global objective:

```text
L_struct = L_sep + 0.25 * L_rank
```

The coefficient is a predefined fractional ablation, not post-hoc tuning. It
tests whether the B30 failure is caused by the presence of global rank pressure
itself or by applying that pressure too strongly under the small-budget protocol.

## Protocol

Real data only:

- source corpus: code2seq Java-small preprocessed split;
- train-limit: 512;
- validation-limit: 128;
- max contexts: 20;
- path encoder: GRU;
- epochs: 3;
- model seeds: 101, 202, 303;
- regimes: original and structural-only;
- structural regularizer for B24: rank;
- schedule for B24: linear.
- structural regularizer for B27: bidirectional monotone attention profile;
- structural regularizer for B28: soft tree-distance calibration of path-node attention;
- structural regularizer for B29: dual root/detail attention separation;
- structural regularizer for B30: dual separation plus global structural rank;
- structural regularizer for B31: dual separation plus soft global structural rank;
- schedule for B27/B28/B29/B30/B31: linear.

Commands:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 20 \
  --path-encoder gru \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --structural-regularizer rank \
  --variants B17_hyperbolic_path_mp_code2vec,B23_hyperbolic_path_attention_mp_code2vec,B24_hyperbolic_path_attention_mp_rank_annealed,B25_hyperbolic_path_depth_attention_mp_code2vec,B26_hyperbolic_path_depth_attention_mp_rank_annealed,B27_hyperbolic_path_attention_mp_monotone,B28_hyperbolic_path_attention_mp_tree_distance,B29_hyperbolic_path_dual_attention_mp_separated,B30_hyperbolic_path_dual_attention_mp_rank_separated,B31_hyperbolic_path_dual_attention_mp_soft_rank \
  --output outputs/code2hyp_path_attention_original_512_3epochs_3seeds.json

./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --train-limit 512 \
  --val-limit 128 \
  --max-contexts 20 \
  --path-encoder gru \
  --epochs 3 \
  --batch-size 64 \
  --model-seeds 101,202,303 \
  --structural-regularizer rank \
  --lexical-ablation structural_only \
  --variants B17_hyperbolic_path_mp_code2vec,B23_hyperbolic_path_attention_mp_code2vec,B24_hyperbolic_path_attention_mp_rank_annealed,B25_hyperbolic_path_depth_attention_mp_code2vec,B26_hyperbolic_path_depth_attention_mp_rank_annealed,B27_hyperbolic_path_attention_mp_monotone,B28_hyperbolic_path_attention_mp_tree_distance,B29_hyperbolic_path_dual_attention_mp_separated,B30_hyperbolic_path_dual_attention_mp_rank_separated,B31_hyperbolic_path_dual_attention_mp_soft_rank \
  --output outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds.json
```

For computational efficiency after B17-B26 were already computed, B27, B28, B29, B30 and B31
were also run as focused one-variant jobs under the same protocol and then
merged into the two aggregate JSON files:

- `outputs/code2hyp_path_attention_original_512_3epochs_3seeds_b27_only.json`
- `outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds_b27_only.json`
- `outputs/code2hyp_path_attention_original_512_3epochs_3seeds_b28_only.json`
- `outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds_b28_only.json`
- `outputs/code2hyp_path_attention_original_512_3epochs_3seeds_b29_only.json`
- `outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds_b29_only.json`
- `outputs/code2hyp_path_attention_original_512_3epochs_3seeds_b30_only.json`
- `outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds_b30_only.json`
- `outputs/code2hyp_path_attention_original_512_3epochs_3seeds_b31_only.json`
- `outputs/code2hyp_path_attention_structural_only_512_3epochs_3seeds_b31_only.json`

Figure:

```bash
./.venv/bin/python scripts/plot_code2hyp_path_attention.py
```

## Results

### Original

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B17 path message passing | 0.1818 | +0.4678 | 0.1195 | 0.3776 |
| B23 path-node attention | 0.1780 | -0.0166 | 0.1440 | 0.2758 |
| B24 path-node attention + linear rank | 0.1326 | +0.0762 | 0.1352 | 0.2815 |
| B25 depth-aware path-node attention | 0.1780 | -0.0127 | 0.1435 | 0.2728 |
| B26 depth-aware path-node attention + linear rank | 0.1326 | +0.0803 | 0.1350 | 0.2833 |
| B27 path-node attention + monotone profile | 0.1742 | -0.0252 | 0.1450 | 0.2857 |
| B28 path-node attention + soft tree-distance calibration | 0.1780 | -0.0164 | 0.1440 | 0.2799 |
| B29 dual root/detail path-node attention | 0.2008 | +0.1030 | 0.1284 | 0.2870 |
| B30 dual attention + global rank | 0.1174 | +0.1608 | 0.1262 | 0.2892 |
| B31 dual attention + soft global rank | 0.2008 | +0.1629 | 0.1261 | 0.3122 |

### Structural only

| Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---:|---:|---:|---:|
| B17 path message passing | 0.1402 | -0.0488 | 0.1654 | 0.4250 |
| B23 path-node attention | 0.1212 | -0.1215 | 0.1750 | 0.3095 |
| B24 path-node attention + linear rank | 0.1553 | -0.0889 | 0.2040 | 0.2907 |
| B25 depth-aware path-node attention | 0.1212 | -0.1052 | 0.1719 | 0.3226 |
| B26 depth-aware path-node attention + linear rank | 0.1553 | -0.0731 | 0.2004 | 0.2936 |
| B27 path-node attention + monotone profile | 0.1212 | -0.1240 | 0.1755 | 0.3124 |
| B28 path-node attention + soft tree-distance calibration | 0.1212 | -0.1217 | 0.1749 | 0.3113 |
| B29 dual root/detail path-node attention | 0.1856 | +0.1171 | 0.1363 | 0.3455 |
| B30 dual attention + global rank | 0.1705 | +0.0910 | 0.1288 | 0.2596 |
| B31 dual attention + soft global rank | 0.1591 | +0.0918 | 0.1381 | 0.4832 |

Paired diagnostics:

- `reports/code2hyp_path_attention_512_original_f1_vs_b17.md`
- `reports/code2hyp_path_attention_512_original_spearman_vs_b17.md`
- `reports/code2hyp_path_attention_512_structural_only_f1_vs_b17.md`
- `reports/code2hyp_path_attention_512_structural_only_spearman_vs_b17.md`

Figure:

- `figures/code2hyp_path_attention_f1_spearman.png`
- `figures/code2hyp_path_attention_f1_spearman.pdf`

## Interpretation

B23/B24/B25/B26/B27/B28/B29/B30/B31 are scientifically useful, but only B29
currently looks like the most robust next candidate. B31 is a useful
original-regime compromise, not a structural-only replacement for B29.

Main observations:

1. In the original regime, B23 almost preserves B17 F1, but loses structural
   Spearman sharply. This suggests that node-attention pooling can support the
   prediction objective while weakening global AST-distance order.
2. B24 partially repairs structural alignment relative to B23 in the original
   regime, but over-regularizes F1.
3. B25 keeps B23's F1 while slightly improving B23's structural diagnostics in
   the original regime: Spearman `-0.0127` versus `-0.0166`, structural loss
   `0.1435` versus `0.1440`, and rank loss `0.2728` versus `0.2758`.
4. In structural-only stress, B24 improves F1 over B17 (`0.1553` versus
   `0.1402`) and lowers rank loss, but Spearman remains worse than B17.
5. In structural-only stress, B25 improves B23 Spearman (`-0.1052` versus
   `-0.1215`) and structural loss (`0.1719` versus `0.1750`), but does not
   recover F1 and remains weaker than B24 by F1.
6. B26 combines the B25 depth prior with the B24 linear rank schedule. It
   matches B24 F1 in both regimes, while slightly improving Spearman:
   `+0.0803` versus `+0.0762` in original and `-0.0731` versus `-0.0889` in
   structural-only.
7. The result is therefore a useful negative/partial result: naive learned
   node attention is not enough, and a single scalar depth prior is only a weak
   refinement. Scheduled depth-aware attention is a better diagnostic variant
   than B24/B25 alone, but it still does not recover B17's global Spearman.
8. B27 is an important negative control for the new idea: monotone
   attention-profile regularization adds no parameters and is locally
   interpretable, but under the matched `max-contexts=20` protocol it does not
   improve B23. In original mode it slightly lowers F1 (`0.1742` versus
   `0.1780`) and Spearman (`-0.0252` versus `-0.0166`). In structural-only mode
   it essentially reproduces B23 F1 (`0.1212`) and slightly worsens Spearman
   (`-0.1240` versus `-0.1215`).
9. Therefore the failure mode is clearer: local monotonicity of node-level
   attention is not equivalent to global preservation of AST-path distances.
10. B28 strengthens the negative conclusion. It calibrates node attention
    against soft pairwise tree distances, but behaves almost exactly like B23:
    original F1 `0.1780` and Spearman `-0.0164`; structural-only F1 `0.1212`
    and Spearman `-0.1217`. Therefore making attention metrically consistent
    at the path-node level still does not repair global context-level
    hyperbolic geometry.
11. The model needs either a joint local/global structural objective, an
    architecture that separates abstraction-level and detail-level channels, or
    multi-objective checkpoint selection.
12. B29 provides the first positive signal for that direction. It explicitly
    separates root/abstract and leaf/detail attention channels and improves both
    task and structural diagnostics in this small pilot: original F1 `0.2008`
    and Spearman `+0.1030`; structural-only F1 `0.1856` and Spearman `+0.1171`.
    This does not prove superiority, but it is a stronger follow-up candidate
    than B23-B28.
13. B30 is an important diagnostic negative result. It adds the global
    structural-rank objective to B29 and improves some structural diagnostics:
    original Spearman rises to `+0.1608`, and structural-only rank loss drops
    from `0.3455` to `0.2596`. However, this comes at a clear task cost:
    original F1 falls from B29's `0.2008` to `0.1174`, and structural-only F1
    falls from `0.1856` to `0.1705`. Therefore the local/global objective is
    too rigid in this small-budget setting.
14. B31 confirms that the B30 failure is partly a weighting problem. With
    `0.25 * L_rank`, original F1 returns to B29's level (`0.2008`) while
    original Spearman improves from B29's `+0.1030` to `+0.1629` and structural
    loss improves from `0.1284` to `0.1261`. However, this does not generalize to
    the structural-only stress test: F1 drops from B29's `0.1856` to `0.1591`,
    Spearman drops from `+0.1171` to `+0.0918`, and rank loss worsens. Therefore
    soft global rank is promising only as an original-regime balancing device.

Defensible claim:

```text
Adding learned node-level attention inside hyperbolic AST-path message passing
changes the F1/structure trade-off. It can improve structural-only F1 under
rank supervision, and a depth-aware attention bias slightly improves some B23
structural diagnostics. Combining depth bias with a linear rank schedule gives
the best B23-B28 Spearman among the single-attention variants. A parameter-free monotone
attention-profile prior was tested as B27, and a parameter-free soft
tree-distance attention calibration was tested as B28; neither improved B23
under the matched protocol. B29, which separates root/abstract and leaf/detail
attention channels, is the first B23-B31 variant that improves F1 over B17 and
keeps positive Spearman in both original and structural-only regimes. B30 shows
that adding global rank pressure on top of B29 can improve structural losses,
but may over-regularize the task objective. B31 softens this pressure and
recovers B29-level original F1 with higher original Spearman, but remains weaker
than B29 under structural-only stress. This is a promising exploratory result
for B29 and an informative local/global regularization ablation for B30/B31, not
yet a confirmatory claim.
```

Not defensible:

```text
B23/B24/B25/B26/B27/B28 are better final models.
B29 is statistically proven superior.
B30 proves that global rank regularization should always be added to B29.
Soft global rank regularization solves structural-only alignment.
Path-node attention solves the structural alignment problem.
B27 proves that monotone attention is sufficient.
B28 proves that soft tree-distance calibration is sufficient.
The pilot proves statistical superiority.
```

## Next model implication

The next nontrivial model should build on B29, not B23-B28. A single learned
scalar depth bias plus rank schedule is still too weak, and a parameter-free
monotone profile penalty or soft attention-distance calibration is also too
weak. B30 shows that simply summing local separation and global rank losses is
too rigid for the current small-budget setting. B31 shows that fractional global
rank pressure can be useful in the original regime, but still does not give a
robust structural-only solution. More serious candidates are:

```text
B32 = B29/B31 + multi-objective checkpoint selection
B33 = adaptive or uncertainty-weighted local/global structural balancing
B34 = larger split and preregistered confirmatory statistics
```

The intended constraint should not only make attention locally interpretable
along the AST path. It must also preserve global structural order between path
contexts. B27 shows that local monotonicity alone is insufficient; B28 shows
that attention-level soft tree-distance calibration alone is also insufficient;
B30 shows that a hard summed local/global structural penalty can over-regularize
the task objective; B31 shows that softening this penalty helps original-regime
F1/Spearman but still fails as a robust structural-only solution.
