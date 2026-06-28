#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MAX_FILES="${MAX_FILES:-250}"
MAX_METHODS="${MAX_METHODS:-32}"
SAMPLE_SEED="${SAMPLE_SEED:-20260623}"
ALPHA="${ALPHA:-0.75}"
EPSILON="${EPSILON:-0.05}"
GW_ITERATIONS="${GW_ITERATIONS:-8}"
SINKHORN_ITERATIONS="${SINKHORN_ITERATIONS:-80}"

mkdir -p outputs reports logs

projects=(
  "training/cassandra"
  "training/elasticsearch"
  "training/gradle"
  "training/hibernate-orm"
  "training/presto"
  "training/spring-framework"
  "training/wildfly"
  "training/liferay-portal"
  "training/intellij-community"
  "validation/libgdx"
  "test/hadoop"
)

relations=(
  "endpoint"
  "lca_depth"
  "lca_anchored_product"
  "edge_jaccard"
  "path_length"
  "multi_endpoint_lca_edge"
)

for project in "${projects[@]}"; do
  tag="${project//\//_}"
  source="data/code2seq_java_small_raw/extracted/java-small/${project}"
  for relation in "${relations[@]}"; do
    output="outputs/raw_ast_fgw_suite_${tag}_${MAX_METHODS}methods_seed${SAMPLE_SEED}_${relation}_alpha0p75.json"
    report="reports/raw_ast_fgw_suite_${tag}_${MAX_METHODS}methods_seed${SAMPLE_SEED}_${relation}_alpha0p75.md"
    if [[ -s "$output" ]]; then
      printf 'SKIP project=%s relation=%s output=%s\n' "$project" "$relation" "$output"
      continue
    fi
    printf 'RUN project=%s relation=%s output=%s\n' "$project" "$relation" "$output"
    nice -n 10 "$PYTHON_BIN" scripts/run_raw_ast_fgw_benchmark.py \
      --source "$source" \
      --max-files "$MAX_FILES" \
      --max-methods "$MAX_METHODS" \
      --sample-seed "$SAMPLE_SEED" \
      --max-paths-per-method 16 \
      --min-paths-per-method 4 \
      --structural-relation "$relation" \
      --alpha "$ALPHA" \
      --epsilon "$EPSILON" \
      --gw-iterations "$GW_ITERATIONS" \
      --sinkhorn-iterations "$SINKHORN_ITERATIONS" \
      --output "$output" \
      --report "$report"
  done
done
