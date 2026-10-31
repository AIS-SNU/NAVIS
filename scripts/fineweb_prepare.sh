#!/usr/bin/env bash
# Prepare FineWeb: base subset, ground truth (insert and delete workloads), and
# the NAVIS index. Outputs that already exist are skipped.
#
#   DATA_DIR=/data/fineweb bash scripts/fineweb_prepare.sh
set -euo pipefail
source "$(dirname "$0")/fineweb_env.sh"
need_tool navis_convert
need_tool navis_build

echo "FineWeb prepare: base $BASE_N, $NUM_BATCHES x $BATCH_SIZE insertions (total $TOTAL_N)"

# 1. Base set: the first BASE_N vectors.
if [[ -f "$BASE" ]]; then
    echo "[1/3] base exists: $BASE"
else
    echo "[1/3] base subset -> $BASE"
    run_logged prepare_subset "$BUILD_DIR/navis_convert" subset "$DATA" "$BASE" --count "$BASE_N"
fi

# 2. Exact top-10 ground truth for every batch boundary, for the insert
#    workload (GT_DIR) and the delete workload (GT_DELETE_DIR), in one pass.
LAST_GT=gt_$(( TOTAL_N - BASE_N )).bin
GT_OUT=()
[[ -f "$GT_DIR/$LAST_GT" ]]        || GT_OUT+=(-o "$GT_DIR")
[[ -f "$GT_DELETE_DIR/$LAST_GT" ]] || GT_OUT+=(--delete-o "$GT_DELETE_DIR")
if [[ ${#GT_OUT[@]} -eq 0 ]]; then
    echo "[2/3] ground truth exists: $GT_DIR, $GT_DELETE_DIR"
else
    echo "[2/3] ground truth -> ${GT_OUT[*]}"
    run_logged prepare_gt "$BUILD_DIR/navis_convert" batch-gt "$DATA" "$QUERY" \
        --index-npts "$BASE_N" --total-npts "$TOTAL_N" --batch "$BATCH_SIZE" \
        -K 10 --threads "$GT_THREADS" --mem-budget "$MEM_BUDGET_GB" "${GT_OUT[@]}"
fi

# 3. NAVIS index (R=96, L=100, 128-byte PQ).
if [[ -f "${INDEX}_disk.edge" && -f "${INDEX}_mem.index" ]]; then
    echo "[3/3] index exists: $INDEX"
else
    echo "[3/3] index -> $INDEX"
    run_logged prepare_build "$BUILD_DIR/navis_build" -d "$BASE" -o "$INDEX" \
        -t float -R 96 -L 100 --pq-bytes 128 --mem-budget "$MEM_BUDGET_GB"
fi
echo "done"
