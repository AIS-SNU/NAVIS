#!/usr/bin/env bash
# IP-DiskANN concurrent search + insertion, the workload of
# scripts/fineweb_insert.sh, with insertion matched to OdinANN and NAVIS
# (exact reranking of the top-L candidates, neighbor pruning on full
# vectors). Host memory: CACHE_GB for the 1-bit codes plus 4 GiB of BfTree
# page cache.
#
#   DATA_DIR=/data/fineweb bash baselines/ipdiskann/run_insert.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline ipdiskann/release/bench_diskann3

# Every run starts from a fresh copy of the prepared index.
RUN=$IPDISKANN_DIR/run
python3 "$BASELINES/ipdiskann/prepare_shadow.py" --force \
    --source-prefix "$IPDISKANN_DIR/quantized/index" --dest-prefix "$RUN/index"

mkdir -p "$LOG_DIR"
csv=$LOG_DIR/ipdiskann_insert_$(date +%Y%m%d_%H%M%S).csv
run_logged ipdiskann_insert "$BASELINES_BUILD/ipdiskann/release/bench_diskann3" \
    --dataset fineweb --index-prefix "$RUN/index" --storage-dir "$RUN" \
    --query "$QUERY" --insert-data "$DATA" --groundtruth "$GT_DIR" --csv "$csv" \
    --base-n "$BASE_N" --batch-size "$BATCH_SIZE" --num-batches "$NUM_BATCHES" \
    --search-threads "$SEARCH_THREADS" --insert-threads "$INSERT_THREADS" \
    --cache-gb 4 --quant-cache-gb "$CACHE_GB" --K 10 --search-l 40 --insert-l 100 \
    --search-beam-width 4 --insert-beam-width 4 --insert-pipeline matched \
    --pruned-degree 96 --backedge-ratio 1.0 --tick-seconds 5
echo "csv: $csv"
