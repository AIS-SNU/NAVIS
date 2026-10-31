#!/usr/bin/env bash
# Insertion only: 32 insertion threads, no concurrent search.
#
#   DATA_DIR=/data/fineweb bash scripts/fineweb_insert_only.sh
set -euo pipefail
source "$(dirname "$0")/fineweb_env.sh"
need_tool navis_bench

run_logged fineweb_insert_only "$BUILD_DIR/navis_bench" -i "$INDEX" -q "$QUERY" \
    --insert-data "$DATA" "${BENCH_COMMON[@]}" \
    --search-threads 0 --insert-threads "${INSERT_THREADS:-32}" \
    --batch-size "$BATCH_SIZE" --num-batches "$NUM_BATCHES"
