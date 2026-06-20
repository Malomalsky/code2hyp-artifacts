# Code2Hyp benchmark protocol

Дата фиксации: 2026-06-16.

## 1. Serious task

Основная задача для подтверждения инструмента:

```text
Java method-name subtoken prediction on the official code2seq Java-small
preprocessed corpus.
```

Почему это подходящая задача:

```text
1. Это стандартная задача линии code2vec/code2seq.
2. Вход уже представлен как AST path contexts: start token, AST path, end token.
3. Метрики из литературы совпадают с нашими: subtoken precision, recall, F1.
4. Корпус не синтетический и не собран нами post hoc.
```

Локальный корпус:

```text
data/code2seq_java_small/java-small/java-small.train.c2s  691974 lines
data/code2seq_java_small/java-small/java-small.val.c2s     23844 lines
data/code2seq_java_small/java-small/java-small.test.c2s    57088 lines
```

## 2. Model roles

После 4k pilot зафиксированы три роли:

```text
B39_code2vec_context_transform_baseline
  Matched code2vec-style Euclidean baseline:
  h_i = tanh(W [start; path; end] + b), attention, multilabel subtoken decoder.

B36_code2hyp_product_frechet_neighbor
  Main performance candidate:
  product space R_start x H_AST_path x R_end, trainable curvature,
  Frechet AST-path aggregation, local AST-neighborhood regularizer.

B44_code2hyp_context_transform_product_bias_frechet
  Main structural-faithfulness candidate:
  preserves Euclidean code2vec context transform and adds trainable
  hyperbolic product-metric structural attention bias.
```

Optional conservative replacement:

```text
B40_code2hyp_context_transform_frechet
  Keeps the code2vec context-transform interface while replacing AST-path
  representation/aggregation by a hyperbolic Frechet channel.
```

## 3. External literature numbers

Primary external comparison is `code2seq` Table 1 on Java-small because it uses
the same Java-small benchmark family and the same subtoken precision/recall/F1
definition.

Primary sources:

```text
code2vec:
  Alon, Zilberstein, Levy, Yahav. code2vec: Learning Distributed
  Representations of Code. POPL 2019.
  https://arxiv.org/abs/1803.09473

code2seq:
  Alon, Brody, Levy, Yahav. code2seq: Generating Sequences from Structured
  Representations of Code. ICLR 2019.
  https://arxiv.org/abs/1808.01400

official datasets:
  https://github.com/tech-srl/code2seq
```

| Model | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---|
| ConvAttention | 50.25 | 24.62 | 33.05 | Allamanis et al. 2016; code2seq Table 1 |
| Paths+CRFs | 8.39 | 5.63 | 6.74 | Alon et al. 2018; code2seq Table 1 |
| code2vec | 18.51 | 18.74 | 18.62 | Alon et al. 2019; code2seq Table 1 |
| 2-layer BiLSTM, no token splitting | 32.40 | 20.40 | 25.03 | code2seq Table 1 |
| 2-layer BiLSTM | 42.63 | 29.97 | 35.20 | code2seq Table 1 |
| TreeLSTM | 40.02 | 31.84 | 35.46 | Tai et al. 2015; code2seq Table 1 |
| Transformer | 38.13 | 26.70 | 31.41 | Vaswani et al. 2017; code2seq Table 1 |
| code2seq | 50.64 | 37.40 | 43.02 | Alon et al. 2019; code2seq Table 1 |

Important boundary:

```text
These are full-budget literature results. They must not be mixed with a
Code2Hyp subset run as if both had identical compute/training budgets.
```

## 4. Confirmatory comparison inside our tool

Primary controlled comparison:

```text
B36 - B39 on official test split subtoken F1.
```

Secondary controlled comparisons:

```text
B40 - B39 on subtoken F1 and structural Spearman.
B44 - B39 on structural Spearman and Overlap@3.
B36 - B39 under record_obfuscated lexical control.
B36/B40/B44 - B39 under structural_only stress test.
```

This gives two publishable axes:

```text
1. Downstream performance axis:
   Does product hyperbolic AST-path modeling improve method-name prediction?

2. Structural-faithfulness axis:
   Does the learned representation preserve AST structural order better than
   the matched Euclidean code2vec-style baseline?
```

Lexical controls:

```text
original:
  keep endpoint token identity.

obfuscated:
  stable one-to-one token renaming. This is a sanity check only: it preserves
  global lexical identity and is therefore isomorphic to original for learned
  embeddings. It must not be used as evidence that lexical information was
  removed.

record_obfuscated:
  local per-record endpoint renaming. This preserves within-context equality
  patterns but breaks global cross-record lexical identity. This is the proper
  lexical-control condition for testing how much the downstream F1 result
  relies on shared token names.

structural_only:
  replace endpoint tokens by a small constant vocabulary while keeping AST
  paths. This is a severe structural stress test, not a realistic production
  condition.
```

## 5. Final benchmark command

Validation is for model selection:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --eval-split val \
  --train-limit 10000 \
  --val-limit 4096 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_val_benchmark_10k_5epochs_3seeds.json
```

Test is for fixed final comparison only:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --eval-split test \
  --train-limit 10000 \
  --val-limit 4096 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_10k_5epochs_3seeds.json
```

Paper-ready summary:

```bash
./.venv/bin/python scripts/summarize_code2hyp_paper_benchmark.py \
  outputs/code2hyp_test_benchmark_10k_5epochs_3seeds.json \
  --output reports/code2hyp_test_benchmark_10k_5epochs_3seeds.md
```

Final lexical-control runs:

```bash
./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --eval-split test \
  --train-limit 10000 \
  --val-limit 4096 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --lexical-ablation record_obfuscated \
  --variants B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_10k_5epochs_3seeds_record_obfuscated_with_stress.json

./.venv/bin/python scripts/run_code2hyp_java_small_pilot.py \
  --eval-split test \
  --train-limit 10000 \
  --val-limit 4096 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --lexical-ablation structural_only \
  --variants B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_10k_5epochs_3seeds_structural_only_with_stress.json
```

For scale-up runs beyond the fixed 10k benchmark, use the resumable runner
instead of the monolithic runner:

```bash
./.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --model-seeds 101,202,303 \
  --max-positive-weight 7.0 \
  --variants B36_code2hyp_product_frechet_neighbor,B39_code2vec_context_transform_baseline,B40_code2hyp_context_transform_frechet,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_test_benchmark_25k_5epochs_3seeds_resumable_with_stress.json
```

The scale-up run must be interpreted as external validation of the frozen 10k
protocol, not as a new model-selection stage.

## 6. Statistical reporting

Required reporting:

```text
mean +/- sample sd over seeds
paired seed deltas against B39
test split size before and after known-target filtering
precision, recall, F1 in percent
structural normalized stress, Spearman and Overlap@3 as diagnostic metrics
```

Structural diagnostic interpretation:

```text
normalized stress:
  scale-invariant metric distortion between learned structural distances and
  AST tree distances after optimal scalar alignment. Lower is better.

structural Spearman:
  rank-order agreement between learned structural distances and AST tree
  distances. Higher is better.

Overlap@3:
  local-neighborhood preservation: whether nearest learned structural
  neighbors are also close in AST tree distance. Higher is better.
```

Do not report:

```text
p-values for n=3 seeds as if they were strong confirmatory evidence.
SOTA claims against code2seq unless training budget and model scale are made
comparable.
```

## 7. Current claim boundary

If the 10k/5-epoch test run reproduces the 4k pilot pattern, the defensible
claim is:

```text
Code2Hyp is a working geometry-aware extension of code2vec-style AST path
models. On the official Java-small method-name prediction task, it improves
the matched local Euclidean baseline under the same subset training budget and
adds structural-faithfulness diagnostics absent from standard code2vec/code2seq
reports.
```

The stronger claim:

```text
Code2Hyp outperforms full-budget published code2vec/code2seq baselines.
```

requires a full-budget run and is not established by the current pilots.

Final 10k/5-epoch conclusion after controls:

```text
The local downstream F1 gain is strongest in the original lexical condition:
B36 improves the matched B39 baseline by +2.89 F1 points. Under
record_obfuscated, the B36 F1 advantage disappears, so the downstream gain
should be described as a joint lexical-structural effect rather than as a pure
geometry-only effect.

The structural-faithfulness result is robust: B36/B40/B44 substantially reduce
normalized AST-distance stress and improve structural Spearman/Overlap@3 across
original, record_obfuscated and structural_only conditions. B44 is the clearest
structural-faithfulness model; B36 is the clearest original-condition
downstream-performance model.
```
