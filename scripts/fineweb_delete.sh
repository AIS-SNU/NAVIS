#!/usr/bin/env bash
# Deletion workload (paper §9.9): each batch inserts BATCH_SIZE new vectors and
# deletes the BATCH_SIZE oldest, so the live set stays at BASE_N. 22 search,
# 10 insertion, and 1 deletion thread; recall against the sliding-window
# ground truth from fineweb_prepare.sh.
#
#   DATA_DIR=/data/fineweb bash scripts/fineweb_delete.sh
set -euo pipefail
source "$(dirname "$0")/fineweb_env.sh"
need_tool navis_bench

run_logged fineweb_delete "$BUILD_DIR/navis_bench" -i "$INDEX" -q "$QUERY" -g "$GT_DELETE_DIR" \
    --insert-data "$DATA" "${BENCH_COMMON[@]}" --workload-change \
    --search-threads "${SEARCH_THREADS:-22}" --insert-threads "${INSERT_THREADS:-10}" \
    --delete-threads "${DELETE_THREADS:-1}" \
    --batch-size "$BATCH_SIZE" --num-batches "$NUM_BATCHES"
