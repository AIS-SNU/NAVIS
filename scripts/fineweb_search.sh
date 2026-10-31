#!/usr/bin/env bash
# Search only, 32 threads: warm the cache with 10 passes over the queries, then
# measure one pass per search frontier L (default 40).
#
#   DATA_DIR=/data/fineweb bash scripts/fineweb_search.sh [L ...]
set -euo pipefail
source "$(dirname "$0")/fineweb_env.sh"
need_tool navis_bench

L_VALUES=("$@")
[[ ${#L_VALUES[@]} -gt 0 ]] || L_VALUES=(40)

run_logged fineweb_search "$BUILD_DIR/navis_bench" -i "$INDEX" -q "$QUERY" -g "$GT_DIR" \
    "${BENCH_COMMON[@]}" --num-batches 0 --search-threads "${SEARCH_THREADS:-32}" \
    --warmup-passes "${WARMUP_PASSES:-10}" "${L_VALUES[@]}"
