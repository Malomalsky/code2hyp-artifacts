# CodeNet Python800 Stage A validation, RunPod execution v1

Date: 2026-07-29

Role: frozen validation-only curvature selection before any test opening.

Local artifact directory:

`outputs/codenet_python800_stage_a_validation_runpod_v1/`

RunPod pod:

- Validation pod id: `3rqo93j9xgjiy6`
- Pod accelerator available: NVIDIA GeForce RTX 3090
- Computation device used by runner v4: CPU
- Runner commit: `cca28ffaccc1ea33256bfa8824fc6589716e3356`
- Runner tag: `codenet-stage-a-validation-runner-v4`

The device description was corrected after a source-level audit on 2026-08-20;
see `reports/codenet_python800_stage_a_compute_device_correction_2026-08-20.md`.

## Protocol Boundary

The run completed the registered validation stage only. The progress and seal
checks record:

- `test_program_ids_materialized = false`
- `test_relevance_labels_opened = false`
- `test_retrieval_metrics_computed = false`
- `validation_only = true`

Therefore the test set was not opened during curvature selection.

## Completion Status

- Registered seeds: 10
- Complete seed results: 10
- Planned cells per seed: 7
- Distance matrices: 70
- Encoder checkpoints: 10
- Validation seed seals: 10
- Selection seal: present
- Tracebacks or runtime errors in RunPod logs: 0

## Frozen Validation Selection

The frozen selection rule chose:

- Selected cell: `HEE_c3_true_LCA`
- Selected active curvature: `3.0`

## Validation Metrics

Problem-level MAP is averaged across the 10 registered seeds. The baseline for
the delta column is `EEE_true_LCA`.

| Cell | Problem MAP mean | Problem MAP sd | Delta vs `EEE_true_LCA` | MRR mean | Recall@1 mean | Recall@5 mean | Recall@10 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `HEE_c3_true_LCA` | 0.043404 | 0.001101 | +0.001081 | 0.249056 | 0.137113 | 0.347938 | 0.476160 |
| `HEE_c1_true_LCA` | 0.042630 | 0.001310 | +0.000307 | 0.245805 | 0.134021 | 0.343428 | 0.472036 |
| `HEE_c0p3_true_LCA` | 0.042398 | 0.001265 | +0.000075 | 0.245031 | 0.132990 | 0.343041 | 0.471649 |
| `HEE_c0p1_true_LCA` | 0.042336 | 0.001244 | +0.000013 | 0.244820 | 0.132861 | 0.343170 | 0.471392 |
| `EEE_true_LCA` | 0.042322 | 0.001250 | +0.000000 | 0.244809 | 0.132861 | 0.343428 | 0.471134 |
| `HEE_near_zero_true_LCA` | 0.042322 | 0.001250 | +0.000000 | 0.244809 | 0.132861 | 0.343428 | 0.471134 |
| `EEE_zero_anchor` | 0.019647 | 0.001303 | -0.022675 | 0.151127 | 0.065077 | 0.205928 | 0.329510 |

## Seal Checks

Each seed seal independently verified:

- frozen protocol and execution config match;
- exact seven-cell validation design;
- checkpoint hash and distance-matrix hashes;
- `float64` distance tensor shape `[776, 776]`;
- validation cardinalities: 776 queries, 776 gallery items, 97 problem clusters;
- MAP recomputation from stored distance matrices and frozen cluster IDs;
- rounding-aware Gate 0;
- validation-only boundary flags.

The selection seal verified:

- all registered seed results are present;
- all seed results match their seals;
- the selected curvature is recomputed from the frozen rule;
- the complete selection remains validation-only.

## Next Registered Step

The next step in the frozen protocol is the single confirmatory test opening
from immutable tag `codenet-stage-a-test-runner-v4`, using the validation
selection `HEE_c3_true_LCA` and curvature `3.0`. This step materializes the test
program IDs and computes test retrieval metrics once.
