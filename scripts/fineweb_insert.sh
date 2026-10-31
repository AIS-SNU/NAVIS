#!/usr/bin/env bash
# Concurrent search + insertion: 22 search and 10 insertion threads (paper
# default), NUM_BATCHES x BATCH_SIZE insertions.
#
#   DATA_DIR=/data/fineweb bash scripts/fineweb_insert.sh
set -euo pipefail
source "$(dirname "$0")/fineweb_env.sh"
need_tool navis_bench

run_logged fineweb_insert "$BUILD_DIR/navis_bench" -i "$INDEX" -q "$QUERY" -g "$GT_DIR" \
    --insert-data "$DATA" "${BENCH_COMMON[@]}" \
    --search-threads "${SEARCH_THREADS:-22}" --insert-threads "${INSERT_THREADS:-10}" \
    --batch-size "$BATCH_SIZE" --num-batches "$NUM_BATCHES"
