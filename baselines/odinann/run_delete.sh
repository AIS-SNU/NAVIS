#!/usr/bin/env bash
# OdinANN deletion workload, as scripts/fineweb_delete.sh: each batch inserts
# BATCH_SIZE new vectors and deletes the BATCH_SIZE oldest. OdinANN deletes
# lazily and merges the deletes into the graph every 20 batches.
# With "cache", runs OdinANN+$: OdinANN with a CACHE_GB NAVIS-Cache.
#
#   DATA_DIR=/data/fineweb bash baselines/odinann/run_delete.sh [cache]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline odinann/tests/overall_performance

name=odinann_delete
export ODIN_PACKED_CACHE_GB=0
if [[ ${1:-} == cache ]]; then
    name=odinann_cache_delete
    ODIN_PACKED_CACHE_GB=$CACHE_GB
fi
export ODIN_SEARCH_THREADS=$SEARCH_THREADS ODIN_INSERT_THREADS=$INSERT_THREADS
export ODIN_DELETE_THREADS=${DELETE_THREADS:-1} ODIN_DELETE_BATCHES=$NUM_BATCHES
export ODIN_MEM_L=10   # searches start from the entrance graph's 10 nearest vertices

rm -f "${ODINANN_INDEX}"_{shadow,merge}_* "${ODINANN_INDEX}_packed_cache_stats.json"

# Arguments: type, data, |E_pos|, index, queries, ground truth, K, beam width,
# base size / batch size, |E_search|.
run_logged "$name" "$BASELINES_BUILD/odinann/tests/overall_performance" \
    float "$DATA" 100 "$ODINANN_INDEX" "$QUERY" "$GT_DELETE_DIR" 10 4 \
    $(( BASE_N / BATCH_SIZE )) 40
