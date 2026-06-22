#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python scripts/run_code2hyp_resumable_benchmark.py \
  --data-root data/code2seq_java_small \
  --eval-split test \
  --train-limit 25000 \
  --val-limit 8192 \
  --max-contexts 30 \
  --max-path-length 8 \
  --token-dim 32 \
  --structural-dim 32 \
  --curvature 1.0 \
  --path-encoder gru \
  --representation-transform identity \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 0.003 \
  --structural-loss-weight 0.05 \
  --structural-regularizer distance \
  --max-positive-weight 7.0 \
  --model-seeds 101,202,303,404,505 \
  --structural-eval-limit 512 \
  --structural-eval-seed 314159 \
  --variants B39_code2vec_context_transform_baseline,B47_code2vec_context_transform_distance_control,B50_code2vec_context_transform_l1_baseline,B51_code2vec_context_transform_l1_distance_control,B48_code2hyp_context_transform_product_bias_no_struct,B49_code2hyp_context_transform_product_bias_near_euclidean,B36_code2hyp_product_frechet_neighbor,B44_code2hyp_context_transform_product_bias_frechet \
  --output outputs/code2hyp_postreview_benchmark_25k_5epochs_5seeds_with_b49_l1_and_geometry_diagnostics.json
