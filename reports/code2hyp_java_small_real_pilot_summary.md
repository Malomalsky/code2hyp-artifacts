# Code2Hyp Java-small Real-data Pilot Summary

Date: 2026-06-14

## Dataset

Real corpus only. Synthetic data is not used as research evidence.

- Source repository: https://github.com/tech-srl/code2seq
- Archive: https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz
- Archive bytes: 479663374
- Archive size check: passed
- Local root in this artifact package: `data/code2seq_java_small`
- Splits:
  - train: 691974 records
  - val: 23844 records
  - test: 57088 records

## Model correction

The first trainable product prototype used `expmap0 -> logmap0` around the origin and then a tangent-space weighted mean. That made the product variant effectively equivalent to the Euclidean variant for the supervised forward pass.

The corrected product variant now aggregates structural path points with a Poincare-ball weighted Einstein midpoint:

1. Map Poincare points to Klein coordinates.
2. Compute the weighted Einstein midpoint with Lorentz gamma weights.
3. Map the midpoint back to the Poincare ball.
4. Use `logmap0` of that midpoint as the structural part of the decoder representation.

This makes B3 geometrically nontrivial while preserving matched capacity: B3 adds one trainable curvature parameter over B1.

## Loss correction

The target-subtoken formulation is highly imbalanced: each method name activates only a few positive subtokens out of a much larger target vocabulary. Plain BCE therefore has a strong trivial-negative bias.

The current pilot supports positive-class weighting:

- `pos_weight = negatives / positives` per target subtoken;
- weights are clamped to `max_positive_weight = 7.0`;
- target subtokens unseen in training are filtered out of validation before scoring.

This is still a pilot-level approximation, but it is a more defensible objective than unweighted BCE for sparse target-subtoken prediction.

## Real pilot protocol

This is a pipeline pilot, not a final article result.

- Train records: 1000 and 4000 pilot settings
- Validation records loaded: 256 and 1024 pilot settings
- Validation records after known-target-subtoken filter: 129 and 639
- Max contexts per method: 30
- Max AST path length: 8
- Epochs: 2
- Batch size: 64 for 1k, 128 for 4k
- Seeds: 101, 202, 303
- Metric: target-subtoken micro precision/recall/F1, with top-k equal to the true number of target subtokens.

## Results

The current defensible architecture is:

- GRU AST-path encoder;
- controlled comparison: B1 is Euclidean, B2 is fixed-curvature product geometry, B3 adds trainable curvature, B4 maps the full code2vec context to the Poincare ball, B4T adds trainable curvature to B4, B8 replaces B4's closed-form Poincare/Klein midpoint with an unrolled Fréchet/Karcher mean refinement, B9 replaces the Poincare ball with the Lorentz hyperboloid and a projected ambient centroid, B10 uses a factorized mixed-product metric with Euclidean lexical channels and a Poincare AST-path channel, B11 adds structural rank regularization to B10, B12 adds two learned positive mixed-product metric weights to B11, B13 adds a low-rank nonlinear residual channel mixer after B11-style product aggregation, B16 splits the product metric into start/path/end weights, B17 adds hyperbolic AST-path message passing, B18 adds structural-rank supervision to B17, B19 linearly anneals the structural-rank supervision for B17, B20 applies delayed-linear structural-rank supervision after an unregularized warmup epoch, B21 applies cosine structural-rank scheduling, B22 applies warmup-decay structural-rank scheduling, B23 adds learned attention over updated AST-path nodes, B24 adds a B19-style linear rank schedule to B23, B25 adds one learned root-to-leaf depth bias to B23 attention, B26 adds the same linear rank schedule to B25, B7 keeps hyperbolic distance attention but removes Poincare/Klein midpoint aggregation, B6 is a Euclidean metric-code2vec control with the same full context and distance-based attention as B4, B14 is the same metric-code2vec control constrained to a bounded Euclidean ball, B_tree is a Euclidean metric-code2vec control with explicit tree-distance/LCA attention bias, B5 is Euclidean with the structural auxiliary loss;
- product representation: Euclidean lexical channel and hyperbolic structural channel;
- Poincare/Klein weighted Einstein midpoint for structural aggregation;
- sparse target-subtoken BCE with positive-class weighting;
- identity representation transform by default.

### Main identity-transform pilots

1000 train records, 256 validation records loaded, 129 validation records after known-target filtering:

| Variant | F1 mean±sd | Structural distance loss mean±sd | Rank loss mean±sd | Spearman mean±sd |
|---|---:|---:|---:|---:|
| B1 Euclidean | 0.1378±0.0073 | 0.7999±0.0104 | 0.1593±0.0130 | -0.3274±0.0132 |
| B2 Product fixed curvature | 0.1393±0.0021 | 0.1434±0.0123 | 0.6648±0.0231 | +0.1244±0.0439 |
| B3 Product | 0.1393±0.0021 | 0.1443±0.0120 | 0.7018±0.0322 | +0.1281±0.0486 |
| B4 Hyperbolic code2vec | 0.1778±0.0189 | 0.1169±0.0015 | 0.3082±0.0280 | +0.4176±0.0168 |
| B4T Hyperbolic code2vec trainable curvature | 0.1778±0.0189 | 0.1172±0.0016 | 0.3063±0.0305 | +0.4149±0.0177 |
| B8 Hyperbolic Frechet code2vec | 0.1778±0.0189 | 0.1163±0.0015 | 0.3333±0.0203 | +0.4288±0.0166 |
| B9 Lorentz hyperboloid code2vec | 0.1556±0.0166 | 0.1351±0.0034 | 0.2441±0.0049 | +0.3035±0.0167 |
| B10 Factorized mixed-product code2vec | 0.1778±0.0189 | 0.2210±0.0158 | 0.8213±0.0505 | -0.2780±0.0206 |
| B11 Factorized mixed-product + structural rank | 0.1600±0.0227 | 0.1188±0.0009 | 0.6334±0.0841 | +0.4203±0.0199 |
| B12 Factorized mixed-product learned metric + structural rank | 0.1600±0.0227 | 0.1190±0.0009 | 0.6300±0.0835 | +0.4193±0.0193 |
| B13 Factorized mixed-product channel mixer + structural rank | 0.1407±0.0042 | 0.1704±0.0207 | 0.2709±0.0322 | +0.1865±0.1003 |
| B7 Hyperbolic attention only | 0.1733±0.0251 | 0.1186±0.0018 | 0.3442±0.0290 | +0.4102±0.0266 |
| B6 Euclidean metric-code2vec | 0.1511±0.0109 | 0.1958±0.0076 | 0.1050±0.0043 | +0.1281±0.0266 |
| B14 Bounded Euclidean metric-code2vec | 0.1585±0.0171 | 0.1818±0.0012 | 0.0770±0.0053 | +0.1598±0.0182 |
| B_tree Euclidean LCA/tree bias | 0.1733±0.0251 | 0.1911±0.0056 | 0.0986±0.0028 | +0.1273±0.0249 |
| B5 Euclidean + structural loss | 0.1556±0.0251 | 0.7626±0.0084 | 0.2706±0.0209 | -0.3082±0.0068 |

4000 train records, 1024 validation records loaded, 639 validation records after known-target filtering:

| Variant | F1 mean±sd | Structural distance loss mean±sd | Rank loss mean±sd | Spearman mean±sd |
|---|---:|---:|---:|---:|
| B1 Euclidean | 0.1302±0.0079 | 0.8807±0.0032 | 0.0488±0.0008 | -0.3346±0.0280 |
| B2 Product fixed curvature | 0.1386±0.0030 | 0.1362±0.0040 | 0.3720±0.0404 | +0.0392±0.0448 |
| B3 Product | 0.1383±0.0030 | 0.1583±0.0028 | 0.3873±0.1014 | -0.0054±0.0340 |
| B4 Hyperbolic code2vec | 0.1453±0.0028 | 0.1074±0.0003 | 1.3574±0.1096 | +0.4086±0.0079 |
| B4T Hyperbolic code2vec trainable curvature | 0.1456±0.0031 | 0.1086±0.0004 | 1.4476±0.0673 | +0.3867±0.0066 |
| B8 Hyperbolic Frechet code2vec | 0.1432±0.0013 | 0.1166±0.0027 | 1.3382±0.1156 | +0.2237±0.0243 |
| B9 Lorentz hyperboloid code2vec | 0.1320±0.0084 | 0.1382±0.0001 | 0.7899±0.0088 | +0.1833±0.0018 |
| B10 Factorized mixed-product code2vec | 0.1420±0.0004 | 0.4071±0.0112 | 1.1454±0.1355 | -0.3879±0.0032 |
| B11 Factorized mixed-product + structural rank | 0.1447±0.0041 | 0.1065±0.0016 | 0.2221±0.0925 | +0.3033±0.0435 |
| B12 Factorized mixed-product learned metric + structural rank | 0.1396±0.0039 | 0.1067±0.0010 | 0.2402±0.0602 | +0.3093±0.0342 |
| B13 Factorized mixed-product channel mixer + structural rank | 0.1275±0.0067 | 0.2050±0.0152 | 0.0563±0.0406 | -0.0208±0.0272 |
| B7 Hyperbolic attention only | 0.1341±0.0090 | 0.1278±0.0005 | 1.4828±0.0209 | +0.1675±0.0042 |
| B6 Euclidean metric-code2vec | 0.1323±0.0085 | 0.1853±0.0009 | 0.1993±0.0021 | +0.1920±0.0015 |
| B14 Bounded Euclidean metric-code2vec | 0.1386±0.0000 | 0.2418±0.0048 | 0.2028±0.0064 | +0.1630±0.0027 |
| B_tree Euclidean LCA/tree bias | 0.1329±0.0083 | 0.1802±0.0019 | 0.1988±0.0083 | +0.1955±0.0040 |
| B5 Euclidean + structural loss | 0.1323±0.0090 | 0.8413±0.0734 | 0.0383±0.0009 | -0.2238±0.0380 |

Mean paired F1 deltas:

- 1k: B2 minus B1 = +0.0015; B3 minus B1 = +0.0015; B4 minus B1 = +0.0400; B8 minus B1 = +0.0400; B9 minus B1 = +0.0178; B10 minus B1 = +0.0400; B11 minus B1 = +0.0222; B12 minus B1 = +0.0222; B13 minus B1 = +0.0030; B7 minus B1 = +0.0356; B6 minus B1 = +0.0133; B14 minus B1 = +0.0207; B_tree minus B1 = +0.0356; B4 minus B8 = +0.0000; B4 minus B9 = +0.0222; B4 minus B10 = +0.0000; B4 minus B11 = +0.0178; B4 minus B12 = +0.0178; B4 minus B13 = +0.0370; B4 minus B7 = +0.0044; B4 minus B6 = +0.0267; B4 minus B14 = +0.0193; B4 minus B_tree = +0.0044; B4 minus B3 = +0.0385; B4T minus B4 = +0.0000.
- 4k: B2 minus B1 = +0.0085; B3 minus B1 = +0.0082; B4 minus B1 = +0.0151; B8 minus B1 = +0.0130; B9 minus B1 = +0.0018; B10 minus B1 = +0.0118; B11 minus B1 = +0.0145; B12 minus B1 = +0.0094; B13 minus B1 = -0.0027; B7 minus B1 = +0.0039; B6 minus B1 = +0.0021; B14 minus B1 = +0.0085; B_tree minus B1 = +0.0027; B4 minus B8 = +0.0021; B4 minus B9 = +0.0133; B4 minus B10 = +0.0033; B4 minus B11 = +0.0006; B4 minus B12 = +0.0057; B4 minus B13 = +0.0178; B4 minus B7 = +0.0112; B4 minus B6 = +0.0130; B4 minus B14 = +0.0066; B4 minus B_tree = +0.0124; B4 minus B3 = +0.0069; B4T minus B4 = +0.0003.

Paired seed-level checks:

- On the 4k pilot, B4 exceeds B6 by `+0.0130` F1 and B_tree by `+0.0124` F1.
- On the 4k pilot, B4 exceeds B8 by `+0.0021` F1; on 1k, B4 and B8 are tied by F1.
- On the 4k pilot, B4 exceeds B10 by `+0.0033` F1 and by `+0.7965` Spearman; B10 is therefore a negative structural control despite competitive F1.
- On the 4k pilot, B4 and B11 are effectively tied by F1 (`+0.0006` for B4), while B11 slightly improves structural distance loss (`0.1065` versus `0.1074`) and B4 remains higher by Spearman (`+0.1053`).
- On the 4k pilot, B12 does not improve B11: B4 exceeds B12 by `+0.0057` F1 and `+0.0992` Spearman. B12 learned weights remain close to the initial equal metric: lexical/path weights are `[1.0963, 1.0190]`, `[1.0380, 0.9960]`, `[1.0997, 0.9764]` for seeds 101, 202, 303.
- On the 4k pilot, B13 does not improve B11/B12: B4 exceeds B13 by `+0.0178` F1 and `+0.4294` Spearman on all 3 matched seeds. B13 reduces local structural rank loss (`0.0563`) but loses global Spearman (`-0.0208`), which means the local rank objective can be satisfied without preserving the global AST-distance order.
- On the 4k pilot, B14 improves B6 by `+0.0063` F1, but B4 still exceeds B14 by `+0.0066` F1 and `+0.2456` Spearman on all 3 matched seeds. Therefore the B4 effect is not explained by bounded Euclidean norm plus distance attention alone.
- On the 4k pilot, B4 exceeds B6 by `+0.2166` Spearman and B_tree by `+0.2131` Spearman on all 3 matched seeds.
- Exact two-sided sign-test p-value is `0.25` for 3/3 directional wins. Therefore this is directionally consistent pilot evidence, not a statistically significant confirmatory result.
- Full paired tables are stored in `reports/code2hyp_paired_effects_1k_f1.md`, `reports/code2hyp_paired_effects_4k_f1.md`, `reports/code2hyp_paired_effects_1k_spearman.md`, and `reports/code2hyp_paired_effects_4k_spearman.md`.

The predictive gain of product hyperbolic geometry over B1/B5 is positive but small and currently pilot-level. The B2 control shows that fixed-curvature product geometry explains most of the B3 gain. Trainable curvature does not provide a stable independent advantage over fixed curvature. The stronger result remains B4: full-context hyperbolic code2vec is the best or tied-best F1 variant and gives the strongest Spearman structural alignment on 4k. B8 is the intrinsic-aggregation robustness control: it replaces the closed-form Poincare/Klein midpoint by an unrolled Fréchet/Karcher refinement. On 1k, B8 is tied with B4 by F1 and slightly improves structural diagnostics; on 4k, it stays close by F1 but loses Spearman. B10 tests the more theoretically conservative factorized mixed-product model, with Euclidean lexical channels and a Poincare AST-path channel. It is competitive by F1 but structurally fails without explicit rank alignment: Spearman is negative on both 1k and 4k. B11 adds the same structural rank regularizer used for B5 to B10. This repairs the structural distance loss and makes B11 nearly tied with B4 by 4k F1, but B4 still has higher Spearman. B12 adds two learned positive product-metric weights to B11, but does not improve it; the weights remain near one and slightly favor the lexical block. Therefore the B11 limitation is not just a fixed-distance-scale issue. B13 adds a low-rank nonlinear residual channel mixer after B11-style product aggregation, but also does not improve B11: it sharply lowers local rank loss while degrading F1 and global Spearman. B16 shows that splitting the product metric into start-token, AST-path and end-token scales also does not solve this limitation. B17 is the first hyperbolic message-passing AST-path encoder in the pilot. It improves rank/structural diagnostics in the structural-only stress test, but it loses F1 in the main regime. B18 confirms that adding a fixed structural-rank penalty to B17 further improves structural losses while over-regularizing the task objective. B19 shows that linear annealing recovers much of the F1 lost by B18. B20 shows that a delayed warmup can push structural-only Spearman closest to zero, but gives up F1 relative to B19. B7 is the hyperbolic-attention-only ablation: it keeps Poincare distance attention but replaces Poincare/Klein midpoint aggregation by a tangent/log-space weighted sum. B6 is the metric-attention anti-confounding control, B14 is the bounded-Euclidean anti-confounding control, and B_tree is the explicit tree-distance/LCA control. B14 improves over B6 in F1, showing that a bounded latent space can help, but it does not close the F1 gap to B4 and remains far below B4 by structural Spearman. B4T does not materially improve B4, so the current gain should not be attributed to the extra curvature parameter.

### New B4: hyperbolic code2vec context space

After the product-channel controls, a closer analogue of code2vec was added:

1. Each path-context `(start token, AST path, end token)` is first represented as the usual code2vec-style context vector.
2. The full context vector is mapped to the Poincare ball with `exp_0`.
3. Attention is computed by negative squared hyperbolic distance to a learned query point.
4. Contexts are aggregated with the Poincare/Klein weighted midpoint.
5. The resulting code vector is mapped back with `log_0` for the target-subtoken decoder.
6. B8 replaces step 4 by an unrolled Fréchet/Karcher mean refinement initialized from the Poincare/Klein midpoint.
7. B7 removes step 4 and instead aggregates `log_0(z_i)` in tangent/log space with the same hyperbolic attention weights.
8. B9 replaces the Poincare-ball coordinate model by a Lorentz hyperboloid model and aggregates by an ambient weighted centroid projected back to the hyperboloid.
9. B10 factorizes the context space into Euclidean lexical channels and a Poincare AST-path channel, then computes attention by the corresponding mixed-product distance.
10. B11 adds structural rank regularization to B10.
11. B12 adds two learned positive product-metric weights to B11: lexical-block weight and AST-path hyperbolic-distance weight.
12. B13 adds a low-rank nonlinear residual channel mixer after B11-style factorized product aggregation.
13. B14 projects the same full code2vec context vectors to a bounded Euclidean ball and uses Euclidean distance attention and Euclidean weighted averaging, without negative curvature.
14. B16 separates the product metric into start-token, AST-path and end-token distances.
15. B17 performs message passing over nodes along each AST path before the code2vec-style context readout.
16. B18 adds structural-rank supervision to B17.
17. B19/B20 test whether that structural-rank supervision should be scheduled rather than applied as a constant penalty.
18. B21/B22 test alternative structural-rank schedules: cosine ramp-up and warmup-decay.
19. B23/B24/B25/B26 test whether the AST-path message-passing encoder should use node-level attention, whether that attention needs an explicit root-to-leaf structural prior, and whether the prior should be combined with scheduled rank supervision.

This is the first direct `hyperbolic code2vec` ablation. It differs from B2/B3 because hyperbolic geometry is applied to the whole path-context representation, not only to the AST-path channel. B4 uses fixed curvature `c=1`; B4T uses the same architecture with one trainable curvature parameter. B8 tests whether replacing the practical Einstein midpoint by a more canonical iterative Fréchet/Karcher mean improves the mechanism. B9 tests whether the B4 result is robust to a different standard model of the same hyperbolic geometry, namely the Lorentz hyperboloid. B10 tests a mixed-curvature product-space alternative in which only the AST-path channel is hyperbolic; B11 tests whether this factorized formulation needs explicit structural rank alignment; B12 tests whether a learned product-metric scale is sufficient to improve B11; B13 tests whether a shallow nonlinear channel-mixing layer after factorized aggregation is sufficient to improve B11; B16 tests whether separate start/path/end metric scales are sufficient to improve the factorized-product family. B17 changes the representation itself by adding hyperbolic message passing over AST-path nodes. B18 tests whether the B17 representation benefits from explicit structural-rank supervision. B19/B20/B21/B22 test whether the same structural-rank objective is schedule-sensitive. B23/B24/B25/B26 test whether AST-path message passing needs learned and structurally biased node-level attention. B7 keeps Poincare distance attention but removes the Poincare/Klein midpoint, so it tests whether the gain comes from the full hyperbolic aggregation or only from a hyperbolic attention score. B6 is the matched Euclidean metric-attention control for B4: it keeps the full context vector and negative squared distance attention, but removes the Poincare exp/log maps, geodesic distance and Poincare/Klein midpoint. B14 is the bounded Euclidean metric-attention control: it keeps the ball constraint but removes negative curvature. B_tree tests a stronger non-hyperbolic alternative by adding explicit tree-distance/LCA features to the same Euclidean metric-attention mechanism.

Visualization:

- `figures/code2hyp_b4_java_small_1k_metrics.png`
- `figures/code2hyp_b4_java_small_1k_metrics.pdf`

Current 1k and 4k tables are reported above in `Main identity-transform pilots`.
The older B4-only duplicate table was removed to avoid mixing the B9 and B11
artifact generations.

### Control: no positive target-subtoken weighting

Note: the control tables below were produced before the corrected minibatch
`target_sizes` alignment. They remain useful for qualitative/structural
diagnostics, but their F1 values are legacy diagnostic numbers and should be
rerun before being cited as corrected predictive results.

1024 train records, identity transform, no positive weighting:

| Variant | F1 mean±sd | Structural distance loss mean±sd | Rank loss mean±sd | Spearman mean±sd |
|---|---:|---:|---:|---:|
| B1 Euclidean | 0.1311±0.0052 | 0.7827±0.0220 | 0.1837±0.0058 | -0.3109±0.0111 |
| B2 Product fixed curvature | 0.1297±0.0059 | 0.1355±0.0050 | 0.3679±0.0785 | 0.1394±0.0546 |
| B3 Product | 0.1297±0.0059 | 0.1394±0.0060 | 0.3675±0.0819 | 0.1248±0.0471 |
| B5 Euclidean + structural loss | 0.1325±0.0052 | 0.7413±0.0107 | 0.3140±0.0133 | -0.2923±0.0101 |

Without positive weighting, B3 does not improve downstream F1. This supports the methodological decision to use class weighting for sparse multi-label target-subtoken prediction, but it also warns that F1 differences are sensitive to the loss formulation.

### Control: rank regularization for B5

1024 train records, identity transform, positive weighting, B5 trained with adjacent-rank structural regularizer:

| Variant | F1 mean±sd | Structural distance loss mean±sd | Rank loss mean±sd | Spearman mean±sd |
|---|---:|---:|---:|---:|
| B1 Euclidean | 0.1158±0.0342 | 0.7757±0.0272 | 0.1966±0.0052 | -0.3119±0.0141 |
| B2 Product fixed curvature | 0.1381±0.0357 | 0.1361±0.0082 | 0.4415±0.1583 | 0.1501±0.0631 |
| B3 Product | 0.1478±0.0306 | 0.1404±0.0096 | 0.4282±0.1524 | 0.1307±0.0704 |
| B5 Euclidean + structural loss | 0.1102±0.0259 | 0.7444±0.0092 | 0.3832±0.0057 | -0.2896±0.0080 |

The current adjacent-rank regularizer does not improve B5. It should be treated as a failed/negative ablation, not as a main method.

### Control: tanh representation transform

1024 train records, positive weighting, tanh representation transform:

| Setting | Variant | F1 mean±sd | Structural distance loss mean±sd | Spearman mean±sd |
|---|---|---:|---:|---:|
| 2 epochs | B1 Euclidean | 0.0739±0.0154 | 0.5818±0.0232 | -0.2204±0.0196 |
| 2 epochs | B2 Product fixed curvature | 0.0907±0.0110 | 0.1280±0.0031 | 0.2233±0.0608 |
| 2 epochs | B3 Product | 0.0907±0.0110 | 0.1290±0.0038 | 0.2146±0.0629 |
| 2 epochs | B5 Euclidean + structural loss | 0.0795±0.0239 | 0.4821±0.0247 | -0.1394±0.0242 |
| 5 epochs | B1 Euclidean | 0.1144±0.0342 | 0.5289±0.0324 | -0.1906±0.0231 |
| 5 epochs | B2 Product fixed curvature | 0.1227±0.0317 | 0.1289±0.0006 | 0.2023±0.0189 |
| 5 epochs | B3 Product | 0.1213±0.0205 | 0.1294±0.0016 | 0.1993±0.0200 |
| 5 epochs | B5 Euclidean + structural loss | 0.1032±0.0479 | 0.2117±0.0082 | 0.0552±0.0202 |

The tanh transform improves some structural diagnostics but lowers downstream F1 relative to the identity-transform product pilots. It remains an ablation, not the default architecture. B2 again matches or slightly exceeds B3, which reinforces the current conclusion that trainable curvature is not yet independently useful in this pilot.

### Exploratory curvature sweep

Because B2 fixed-curvature product geometry explains most of the B3 gain, curvature must be treated as an experimental factor rather than a fixed convention. A first one-seed sweep was run on the 1024/256 pilot setting:

| c | Variant | F1 | Structural distance loss | Spearman |
|---:|---|---:|---:|---:|
| 0.1 | B2 Product fixed curvature | 0.1632 | 0.7416 | -0.3242 |
| 0.1 | B3 Product trainable curvature | 0.1632 | 0.7472 | -0.3247 |
| 0.3 | B2 Product fixed curvature | 0.1674 | 0.5371 | -0.2743 |
| 0.3 | B3 Product trainable curvature | 0.1632 | 0.5587 | -0.2819 |
| 1.0 | B2 Product fixed curvature | 0.1674 | 0.1410 | 0.1487 |
| 1.0 | B3 Product trainable curvature | 0.1674 | 0.1415 | 0.1369 |
| 3.0 | B2 Product fixed curvature | 0.1255 | 0.1206 | 0.2595 |
| 3.0 | B3 Product trainable curvature | 0.1255 | 0.1202 | 0.2691 |

Interpretation:

- `c=1.0` is currently the best compromise between F1 and structural diagnostics.
- `c=3.0` improves structural alignment but hurts downstream F1.
- `c=0.1/0.3` preserves F1 on this seed but weakens structural alignment.
- B3 does not yet learn a substantially different curvature in two epochs; trainable curvature remains a hypothesis for longer runs or a dedicated curvature schedule.

### Control: endpoint-token obfuscation

The endpoint-token obfuscation control removes the original lexical surface of
`start_token` and `end_token` while preserving token equality:

```text
foo -> LEX_<stable_hash(foo)>
bar -> LEX_<stable_hash(bar)>
foo -> LEX_<stable_hash(foo)>
```

This tests whether the B4 signal survives when the model cannot directly use
identifier spelling in path-context endpoints.

1024 train records, 256 validation records, GRU path encoder, positive
weighting, 3 seeds:

| Variant | F1 mean | Structural distance loss mean | Spearman mean |
|---|---:|---:|---:|
| B1 Euclidean | 0.1328 | 0.7757 | -0.3119 |
| B4 Hyperbolic code2vec | 0.1715 | 0.1181 | 0.3980 |
| B6 Euclidean metric-code2vec | 0.1521 | 0.1921 | 0.1336 |
| B14 Bounded Euclidean metric-code2vec | 0.1397 | 0.1910 | 0.1337 |
| B_tree Euclidean LCA/tree bias | 0.1425 | 0.1983 | 0.0983 |
| B11 Factorized product + structural rank | 0.1660 | 0.1661 | -0.1299 |

Paired seed-level deltas for B4 under obfuscation:

| Comparison | F1 delta | Spearman delta | Direction |
|---|---:|---:|---|
| B4 - B1 | +0.0387 | +0.7099 | F1 +/2 -/0 0/1; Spearman +/3 -/0 0/0 |
| B4 - B6 | +0.0194 | +0.2644 | F1 +/3 -/0 0/0; Spearman +/3 -/0 0/0 |
| B4 - B14 | +0.0318 | +0.2644 | F1 +/3 -/0 0/0; Spearman +/3 -/0 0/0 |
| B4 - B_tree | +0.0290 | +0.2997 | F1 +/3 -/0 0/0; Spearman +/3 -/0 0/0 |
| B4 - B11 | +0.0055 | +0.5280 | F1 +/2 -/1 0/0; Spearman +/3 -/0 0/0 |

Interpretation:

- This is the strongest current anti-leakage pilot result: B4 remains ahead of
  the Euclidean metric, bounded Euclidean, and explicit tree-bias controls
  after lexical endpoint surfaces are removed.
- The result is still exploratory because `n = 3` seeds and the train/validation
  limits are small. It supports running the larger confirmatory O1 protocol; it
  is not yet final statistical proof.
- B11 remains an important mixed-product competitor: it is close to B4 by F1
  but still much weaker by global structural Spearman in this obfuscated pilot.

Detailed paired tables:

- `reports/code2hyp_paired_effects_1k_obfuscated_f1.md`
- `reports/code2hyp_paired_effects_1k_obfuscated_spearman.md`

### Stress-test: structural-only endpoint masking

The structural-only control replaces every `start_token` and `end_token` with
`<LEXICAL_MASK>`. This removes both lexical surface and endpoint-token equality.
It should be interpreted as a stress-test of AST-path structure, not as the
primary downstream setting.

1024 train records, 256 validation records, GRU path encoder, positive
weighting, 3 seeds:

| Variant | F1 mean | Structural distance loss mean | Spearman mean |
|---|---:|---:|---:|
| B1 Euclidean | 0.1245 | 0.7855 | -0.2998 |
| B2 Product fixed curvature | 0.1245 | 0.1348 | 0.1353 |
| B3 Product | 0.1411 | 0.1392 | 0.1185 |
| B4 Hyperbolic code2vec | 0.1314 | 0.2008 | -0.2225 |
| B6 Euclidean metric-code2vec | 0.1328 | 0.7863 | -0.2738 |
| B8 Hyperbolic Frechet code2vec | 0.1452 | 0.2008 | -0.2430 |
| B10 Factorized product code2vec | 0.1425 | 0.2147 | -0.2520 |
| B13 Factorized product channel mixer | 0.1355 | 0.1210 | 0.1747 |
| B14 Bounded Euclidean metric-code2vec | 0.1176 | 0.2261 | 0.0370 |
| B_tree Euclidean LCA/tree bias | 0.1245 | 0.7844 | -0.2690 |

Interpretation:

- When endpoint-token information is completely removed, B4 no longer dominates
  the task metric. B8, B10 and B3 become stronger by F1 in this small pilot.
- B4 still improves structural Spearman over B1, B6 and B_tree, but not over
  B2/B3/B13/B14. This means the strongest scientific claim should use O1
  obfuscation, not O2 structural-only masking, as the primary lexical-weakening
  condition.
- The result is useful because it prevents an overclaim: full-context
  hyperbolic B4 benefits from having endpoint context available, while product
  AST-path geometry may be better suited when only path structure remains.

### Control: Lorentz-product AST-path channel

B15 `Lorentz-product code2vec` was added to test whether the better
hierarchical behavior attributed to the Lorentz model transfers to the
path-context product setup:

```text
start token: Euclidean
AST path: Lorentz hyperboloid
end token: Euclidean
attention: Euclidean start/end distances + Lorentz AST-path distance
aggregation: Euclidean weighted mean for tokens + Lorentz centroid for AST path
```

This is parameter-matched to B4 and B10.

1k/256, GRU path encoder, positive weighting, 3 seeds:

| Regime | B15 F1 | B15 Spearman | B15 structural loss | Interpretation |
|---|---:|---:|---:|---|
| Original | 0.1521 | -0.2481 | 0.5628 | worse than B4/B8 and not better than B6 by F1 |
| Obfuscated | 0.1521 | -0.2481 | 0.5628 | identical to original because token equality is preserved |
| Structural only | 0.1328 | -0.2513 | 0.5721 | not competitive with B3/B8/B10 |

Conclusion:

- B15 is a useful negative control, not a better main model.
- The Lorentz coordinate model alone does not explain the B4/B8 effect.
- The next model-search step should target the failure of product models to
  preserve global AST-distance order, not merely switch Poincare to Lorentz.

Detailed paired tables:

- `reports/code2hyp_paired_effects_1k_structural_only_f1.md`
- `reports/code2hyp_paired_effects_1k_structural_only_spearman.md`
- `reports/code2hyp_paired_effects_1k_with_b15_f1.md`
- `reports/code2hyp_paired_effects_1k_with_b15_spearman.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b15_f1.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b15_spearman.md`

Lexical-ablation summary figure:

- `figures/code2hyp_lexical_ablation_metrics.png`
- `figures/code2hyp_lexical_ablation_metrics.pdf`

### Control: factorized three-metric product geometry

B16 `factorized mixed-product three-metric + structural rank` was added after
B15 to test a more precise product-geometry hypothesis. Instead of a broad
lexical/path scale as in B12, B16 learns three positive metric weights:

```text
start token: tau_s ||s_i - q_s||^2
AST path: tau_h d_H(p_i, q_h)^2
end token: tau_e ||e_i - q_e||^2
```

It keeps the same Poincare AST-path channel and structural-rank supervision
line as B11/B12, so the control asks whether the remaining B4-to-B11 gap is
caused by endpoint/path metric-scale imbalance.

1k/256, GRU path encoder, positive weighting, 3 seeds:

| Regime | B16 F1 | B16 Spearman | B16 structural loss | B4 - B16 F1 | B4 - B16 Spearman |
|---|---:|---:|---:|---:|---:|
| Original | 0.1674 | -0.1305 | 0.1662 | +0.0041 | +0.5286 |
| Obfuscated | 0.1674 | -0.1305 | 0.1662 | +0.0041 | +0.5286 |
| Structural only | 0.1328 | -0.1375 | 0.1709 | -0.0014 | -0.0851 |

Learned metric weights remain close to one:

```text
Original seed 101: [1.0543, 0.9761, 1.0102]
Original seed 202: [1.0471, 0.9935, 0.9782]
Original seed 303: [1.0549, 0.9853, 1.0496]
Structural-only seed 101: [1.0001, 0.9749, 1.0001]
Structural-only seed 202: [1.0000, 0.9695, 1.0000]
Structural-only seed 303: [1.0001, 0.9646, 1.0001]
```

Conclusion:

- B16 is a stronger diagnostic product control than B12, because it separates
  start-token, AST-path and end-token metric scales.
- B16 does not improve B4/B8 on the main original/obfuscated regime.
- In structural-only, B16 improves Spearman relative to B4, but both remain
  negative and the task F1 is essentially tied.
- Therefore the weakness of B11/B12 is not explained by a simple product-scale
  imbalance. A stronger next model should change the representation or rank
  objective itself, not merely reweight product-distance components.

### Control: hyperbolic AST-path message passing

B17 `hyperbolic AST-path message passing` was added after the negative
B15/B16 controls. The goal is to test a stronger structural hypothesis:
the AST-path channel may need a geometry-aware encoder before code2vec readout,
not merely a different metric scale after a path representation has already
been collapsed.

Mechanism:

```text
1. Embed AST nodes.
2. Map node embeddings to the Poincare ball.
3. Exchange messages along neighboring nodes in each extracted AST path.
4. Map updated node states back to the tangent space at the origin.
5. Feed the updated path sequence to the same GRU path encoder.
6. Use the normal code2vec-style attention/readout afterwards.
```

B18 `hyperbolic AST-path message passing + structural rank` keeps the same
architecture but adds the structural-rank/alignment loss. This tests whether
the improved structural encoder benefits from explicit rank supervision, or
whether the rank objective over-regularizes the downstream prediction task.

B19 keeps the B17/B18 architecture but replaces the constant rank-loss weight
with linear annealing. This tests whether the structural signal is useful when
introduced gradually instead of being imposed at full strength from the first
epoch.

B20 keeps the same architecture but uses delayed-linear annealing. The first
epoch is unregularized by the structural-rank objective, and the rank loss is
introduced only after warmup. This tests whether the model first needs to learn
the target-subtoken task before structural alignment is imposed.

B21/B22 extend the same schedule question: B21 uses cosine ramp-up of the
structural-rank weight, and B22 uses warmup-decay. They are schedule controls,
not new architectures.

B23/B24 move from schedule control to representation control. B23 keeps the
B17 hyperbolic AST-path message passing core but pools updated AST-path nodes
with learned node-level attention before the full path-context is mapped to the
Poincare ball. B24 adds the B19-style linear structural-rank schedule to B23.
B25 keeps B23 but adds one learned scalar depth bias to the node-attention
score, so that attention can use the root-to-leaf role of a node inside the AST
path. B26 adds the B19/B24-style linear structural-rank schedule to B25.

1k/256, GRU path encoder, positive weighting, 3 seeds:

| Regime | Variant | F1 | Spearman | Structural loss | Rank loss |
|---|---|---:|---:|---:|---:|
| Original | B4 | 0.1715 | +0.3980 | 0.1181 | 0.5536 |
| Original | B8 | 0.1715 | +0.4077 | 0.1176 | 0.5308 |
| Original | B17 | 0.1425 | +0.3531 | 0.1180 | 0.4647 |
| Original | B18 | 0.1245 | +0.3553 | 0.1177 | 0.4698 |
| Original | B19 | 0.1577 | +0.3429 | 0.1183 | 0.4557 |
| Original | B20 | 0.1577 | +0.3459 | 0.1181 | 0.5269 |
| Obfuscated | B4 | 0.1715 | +0.3980 | 0.1181 | 0.5536 |
| Obfuscated | B17 | 0.1425 | +0.3531 | 0.1180 | 0.4647 |
| Obfuscated | B18 | 0.1245 | +0.3553 | 0.1177 | 0.4698 |
| Obfuscated | B19 | 0.1577 | +0.3429 | 0.1183 | 0.4557 |
| Obfuscated | B20 | 0.1577 | +0.3459 | 0.1181 | 0.5269 |
| Structural only | B4 | 0.1314 | -0.2225 | 0.2008 | 0.9306 |
| Structural only | B8 | 0.1452 | -0.2430 | 0.2008 | 0.9086 |
| Structural only | B17 | 0.1231 | -0.0341 | 0.1544 | 0.6636 |
| Structural only | B18 | 0.1010 | -0.0356 | 0.1472 | 0.5261 |
| Structural only | B19 | 0.1577 | -0.0263 | 0.1452 | 0.5515 |
| Structural only | B20 | 0.1383 | -0.0016 | 0.1433 | 0.6361 |

Interpretation:

- In the main original/obfuscated regime, B17/B18/B19/B20 do not beat B4/B8 by task
  F1 or global Spearman.
- B17 substantially improves structural-only Spearman relative to B4/B8:
  it moves the correlation from strongly negative to near zero.
- B18 further improves structural loss and rank loss in structural-only mode,
  but drops F1. This is evidence for over-regularization by a fixed structural
  rank penalty, not evidence for a better final model.
- B19 validates the annealing hypothesis directionally: relative to B18, it
  recovers much of the lost F1 while keeping structural-only Spearman near zero
  and structural loss far below B4/B8.
- B20 clarifies the schedule trade-off: delayed warmup gives the best
  structural-only Spearman among B4/B8/B17/B18/B19/B20, but loses F1 relative
  to B19 in the structural-only stress test.
- B21/B22 clarify that cosine and warmup-decay schedules do not dominate the
  simpler linear B19 schedule in the compact schedule sweep.
- B23/B24 show that unconstrained learned node attention is not enough: B23
  nearly preserves original-regime F1 but weakens global AST-distance Spearman,
  while B24 improves structural-only F1 relative to B17 but still does not
  repair Spearman.
- B25 shows that a weak depth-aware prior is a useful refinement over B23 by
  some structural diagnostics, but not enough to recover B17's global
  AST-distance Spearman or to dominate B24 in structural-only F1.
- B26 shows that depth-aware attention and scheduled rank pressure compose:
  it matches B24 F1 and slightly improves B24 Spearman in both regimes, but it
  still remains far below B17 by original-regime global Spearman.
- B27 shows that bidirectional monotone attention-profile regularization is
  not sufficient by itself. It adds no parameters and makes the local
  root-to-leaf attention profile more constrained, but under the matched
  max-contexts=20 protocol it does not improve B23 by F1 or global Spearman.
- B28 shows that soft tree-distance calibration of path-node attention is also
  not sufficient by itself. It adds no parameters and makes node attention
  metrically interpretable through expected AST-prefix distances, but it still
  behaves almost exactly like B23 under the matched protocol.
- B29 is the first positive path-attention follow-up in this sequence. It
  separates root/abstract and leaf/detail AST-node attention channels and gives
  a higher exploratory F1/Spearman trade-off in both original and structural-only
  regimes under the same 512/128, 3-seed matched protocol.
- B30 keeps the B29 dual-head architecture and adds a global structural-rank
  objective. It improves some structural diagnostics, but over-regularizes the
  task objective in the original regime, so it is a diagnostic negative result
  rather than a replacement for B29.
- B31 keeps the B29 dual-head architecture and softens the B30 local/global
  objective by using `L_sep + 0.25 * L_rank`. It preserves B29-level F1 and
  improves Spearman in the original regime, but remains weaker than B29 in the
  structural-only stress test.
- The next serious model direction is therefore structurally constrained
  path-node attention built on B29/B31 with adaptive local/global balancing and
  multi-objective checkpoint selection, not another constant rank-loss coefficient, scalar
  product-distance reweighting, unconstrained attention layer, one-scalar depth
  bias, local monotone-profile penalty, or attention-only soft tree-distance
  calibration alone.

Detailed paired tables:

- `reports/code2hyp_paired_effects_1k_with_b16_f1.md`
- `reports/code2hyp_paired_effects_1k_with_b16_spearman.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b16_f1.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b16_spearman.md`
- `reports/code2hyp_paired_effects_1k_with_b18_f1.md`
- `reports/code2hyp_paired_effects_1k_with_b18_spearman.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b18_f1.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b18_spearman.md`
- `reports/code2hyp_paired_effects_1k_with_b19_f1.md`
- `reports/code2hyp_paired_effects_1k_with_b19_spearman.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b19_f1.md`
- `reports/code2hyp_paired_effects_1k_structural_only_with_b19_spearman.md`
- `reports/code2hyp_b20_schedule_ablation.md`
- `reports/code2hyp_schedule_sweep.md`
- `reports/code2hyp_path_attention.md`
- `reports/code2hyp_paired_effects_focused_b20_original_f1.md`
- `reports/code2hyp_paired_effects_focused_b20_original_spearman.md`
- `reports/code2hyp_paired_effects_focused_b20_structural_only_f1.md`
- `reports/code2hyp_paired_effects_focused_b20_structural_only_spearman.md`
- `reports/code2hyp_path_attention_512_original_f1_vs_b17.md`
- `reports/code2hyp_path_attention_512_original_spearman_vs_b17.md`
- `reports/code2hyp_path_attention_512_structural_only_f1_vs_b17.md`
- `reports/code2hyp_path_attention_512_structural_only_spearman_vs_b17.md`
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

### Generated figure

The main identity-transform pilot is visualized in:

- `figures/code2hyp_java_small_pilot_metrics.png`
- `figures/code2hyp_java_small_pilot_metrics.pdf`

The B4/B4T/B8/B9/B17/B18/B19/B20/B7/B6/B14/B_tree hyperbolic-code2vec and Euclidean controls are visualized in:

- `figures/code2hyp_b4_java_small_1k_with_b19_metrics.png`
- `figures/code2hyp_b4_java_small_1k_with_b19_metrics.pdf`
- `figures/code2hyp_focused_b20_original_1k_metrics.png`
- `figures/code2hyp_focused_b20_original_1k_metrics.pdf`
- `figures/code2hyp_focused_b20_lexical_ablation_metrics.png`
- `figures/code2hyp_focused_b20_lexical_ablation_metrics.pdf`
- `figures/code2hyp_b20_f1_spearman_tradeoff.png`
- `figures/code2hyp_b20_f1_spearman_tradeoff.pdf`
- `figures/code2hyp_schedule_sweep_f1_spearman.png`
- `figures/code2hyp_schedule_sweep_f1_spearman.pdf`
- `figures/code2hyp_path_attention_f1_spearman.png`
- `figures/code2hyp_path_attention_f1_spearman.pdf`

## Interpretation

The pilot establishes thirty-nine points:

1. The pipeline now runs on a real code2seq Java-small split, not on synthetic data.
2. The mean AST-node encoder is too weak for a serious code2vec/code2seq comparison.
3. The GRU path encoder materially improves the real-data pilot.
4. Positive target-subtoken weighting is necessary for this sparse multi-label objective; without it, B3 does not show a stable advantage.
5. With positive weighting and identity transform, product hyperbolic variants B2/B3 show a weak positive pilot signal over B1 and B5, but the evidence is underpowered and not yet suitable for a final article claim.
6. B2 is now a critical control: fixed-curvature hyperbolic product geometry explains most of the current pilot gain; trainable curvature is not yet independently validated.
7. The strongest current signal is structural rather than predictive: B2/B3 substantially lower validation structural-distance loss relative to B1 and B5.
8. The rank/Spearman diagnostics are mixed. Product geometry gives positive Spearman on 1k and substantially less negative Spearman than B1 on 4k, but global rank preservation of AST distances is not yet proven.
9. B4 `hyperbolic code2vec` is now the strongest 1k and 4k pilot variant: unlike B2/B3, it applies hyperbolic geometry to the whole path-context representation and currently improves both F1 and structural diagnostics.
10. B8 `hyperbolic Frechet code2vec` is the intrinsic aggregation robustness control. It shows that Fréchet/Karcher refinement is compatible with B4 on 1k, but does not improve the 4k pilot over the simpler Einstein midpoint.
11. B7 `hyperbolic attention only` is the attention-only aggregation control for B4. It shows that Poincare distance attention alone is not enough on the 4k pilot; hyperbolic aggregation is part of the current mechanism.
12. B6 `Euclidean metric-code2vec` is the main anti-confounding control for B4. It shows that full-context metric attention is useful, but it does not close the gap to hyperbolic B4.
13. B10 `factorized mixed-product code2vec` is a negative structural control: it is competitive by F1 but fails by global Spearman without explicit structural rank alignment.
14. B11 `factorized mixed-product + structural rank` is the strongest factorized product candidate: it nearly closes the 4k F1 gap to B4, but remains lower by global Spearman.
15. B12/B13 show that neither simple learned product-metric rescaling nor shallow nonlinear post-aggregation channel mixing explains the remaining B11-to-B4 gap.
16. B14 `bounded Euclidean metric-code2vec` is the bounded-latent-space control. It improves B6 by F1, but does not close the F1 or Spearman gap to hyperbolic B4.
17. B_tree `Euclidean LCA/tree bias` is the explicit tree-distance/LCA control. It improves structural diagnostics relative to weak Euclidean baselines, but it also does not close the gap to hyperbolic B4.
18. B4T shows that adding trainable curvature to B4 does not improve this pilot.
19. Endpoint-token obfuscation is now implemented and logged as an anti-leakage control. It removes identifier spelling from path-context endpoints while preserving endpoint-token equality.
20. In the 1k obfuscated pilot, B4 remains ahead of B6, B14 and B_tree by F1 and structural Spearman on all three matched seeds. This strengthens the case for a confirmatory lexical-weakening experiment, but remains exploratory.
21. Structural-only endpoint masking is a much harsher stress-test. Under this condition, B4 is no longer the best F1 variant; B8, B10 and B3 are stronger in the 1k pilot.
22. Therefore the most defensible scientific claim is not "B4 is universally best", but "negative-curvature path-context geometry gives a robust signal under endpoint-token obfuscation, while pure AST-path-only structure may require a product/factorized formulation."
23. B15 `Lorentz-product code2vec` is implemented as a parameter-matched Lorentz AST-path product control. It does not improve over B4/B8 and is weak by structural diagnostics.
24. B16 `factorized mixed-product three-metric + structural rank` is implemented as the stronger start/path/end product-metric control. It does not improve B4/B8 in the main regime and does not make product geometry structurally reliable.
25. B17 `hyperbolic AST-path message passing` is implemented as the first geometry-aware path encoder control. It improves structural-only rank diagnostics, but it is not yet competitive with B4/B8 by F1.
26. B18 `hyperbolic AST-path message passing + structural rank` confirms the structural/task trade-off: structural losses improve, while F1 falls.
27. B19 `hyperbolic AST-path message passing + annealed structural rank` validates the schedule hypothesis directionally: it recovers F1 relative to B18 and gives the best structural-only F1 among B4/B8/B17/B18/B19.
28. B20 `hyperbolic AST-path message passing + delayed structural rank` gives the best structural-only Spearman among B4/B8/B17/B18/B19/B20, but loses F1 relative to B19.
29. B21/B22 show that cosine and warmup-decay schedules are useful controls but not current winners; B19 remains the most stable scheduled AST-path MP compromise in the compact sweep.
30. B23/B24 show that learned node-level attention inside the AST path is a meaningful representation probe, but unconstrained attention can preserve/improve F1 while weakening global AST-distance order.
31. B25 shows that root-to-leaf depth bias should be modeled, but a single learned scalar bias is too weak as a final method.
32. B26 shows that adding a linear rank schedule to B25 slightly improves the B24/B25 structural trade-off, but still does not recover B17's original-regime global Spearman.
33. B27 shows that local monotone attention-profile regularization is not enough: it adds no parameters, but under the matched protocol it behaves close to B23 and does not repair global AST-distance alignment.
34. B28 shows that attention-level soft tree-distance calibration is also not enough: it adds no parameters and is geometrically interpretable, but it remains close to B23 by F1 and global Spearman.
35. B29 is the first positive path-attention follow-up: dual root/detail attention with separation regularization improves the exploratory F1/Spearman trade-off in both original and structural-only regimes.
36. B30 tests the direct local/global objective B29 plus structural rank. It improves structural losses in places, but sharply reduces original-regime F1, which suggests that a hard summed rank penalty is too rigid under the current small-budget protocol.
37. B31 tests a softer local/global objective B29 plus `0.25 * structural rank`. It preserves B29's original-regime F1 and improves original-regime Spearman, but remains weaker than B29 under structural-only stress.
38. Therefore the next mathematically justified candidate should build on B29/B31 with adaptive local/global balancing and multi-objective selection, not merely add a constant structural penalty, another schedule, a local attention-shape penalty, attention-only tree-distance calibration, or a hard summed rank penalty.
39. The strongest current article claim remains comparative and bounded: full-context negative-curvature code2vec is the best balanced pilot family so far, while B29/B31 identify the most promising next structural encoder direction for a larger confirmatory experiment.

Therefore the honest next research step is not to claim a proven hyperbolic advantage yet. The next step is a larger controlled experiment with:

- GRU path encoder as the minimum viable path encoder;
- multiple seeds;
- larger train/validation limits or full split;
- reporting paired deltas and confidence intervals;
- learning-rate and curvature initialization sweep;
- geometry diagnostics: curvature trajectory, AST-distance stress, Spearman correlation between AST path distance and embedding distance;
- sampled pair diagnostics for large validation sets, because full pairwise structural diagnostics become expensive;
- a better rank-preservation objective or a scheduled multi-objective criterion if rank preservation is elevated from diagnostic to training criterion;
- comparison against B1, B2, B3, B4, B4T, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31 and B_tree under matched data, seeds and training budget.
