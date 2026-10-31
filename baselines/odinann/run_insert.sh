#!/usr/bin/env bash
# OdinANN concurrent search + insertion, the workload of scripts/fineweb_insert.sh.
# With "cache", runs OdinANN+$: OdinANN with a CACHE_GB NAVIS-Cache.
#
#   DATA_DIR=/data/fineweb bash baselines/odinann/run_insert.sh [cache]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline odinann/tests/test_insert_search

name=odinann_insert
export ODIN_PACKED_CACHE_GB=0
if [[ ${1:-} == cache ]]; then
    name=odinann_cache_insert
    ODIN_PACKED_CACHE_GB=$CACHE_GB
fi

# Every run starts from the converted index: OdinANN works on a copy
# (_shadow_*) and merges into _merge_*.
rm -f "${ODINANN_INDEX}"_{shadow,merge}_* "${ODINANN_INDEX}_packed_cache_stats.json"

# Arguments: type, data, |E_pos|, batch size, batches, insert threads, search
# threads, search mode (0 = beam), index, queries, ground truth, ground-truth
# offset, K, insert beam width, search beam width, entrance-graph L (searches
# start from the entrance graph's 10 nearest vertices), |E_search|.
run_logged "$name" "$BASELINES_BUILD/odinann/tests/test_insert_search" \
    float "$DATA" 100 "$BATCH_SIZE" "$NUM_BATCHES" "$INSERT_THREADS" "$SEARCH_THREADS" 0 \
    "$ODINANN_INDEX" "$QUERY" "$GT_DIR" 0 10 4 4 10 40
