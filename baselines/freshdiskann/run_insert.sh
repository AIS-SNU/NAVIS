#!/usr/bin/env bash
# FreshDiskANN concurrent search + insertion, the workload of
# scripts/fineweb_insert.sh. Insertions go to an in-memory index (built with
# L = 128, FreshDiskANN's default) that is merged into the disk index every
# MERGE_SIZE points (3M = 30 batches). As in
# OdinANN's evaluation, FreshDiskANN is restarted after every merge to avoid
# its memory corruption; each restart resumes from the merged index.
# Searches start from the 10 nearest vertices of NAVIS's entrance graph
# (static, read from the NAVIS index), as for OdinANN.
#
#   DATA_DIR=/data/fineweb bash baselines/freshdiskann/run_insert.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline tools/bench_freshdiskann
[[ -f ${INDEX}_mem.index ]] || { echo "error: missing NAVIS entrance graph ${INDEX}_mem.index" >&2; exit 1; }

MERGE_SIZE=${MERGE_SIZE:-3000000}
export FRESH_MERGE_THREADS=${FRESH_MERGE_THREADS:-$INSERT_THREADS}

# Every run starts from the converted index (_shadow1_* is the working copy;
# merges alternate between it and _merge_*).
P=$FRESHDISKANN_INDEX
for f in _disk.index _disk.index.tags _pq_pivots.bin _pq_compressed.bin; do
    cp "$P$f" "${P}_shadow1$f"
    cp "$P$f" "${P}_merge$f"
done

mkdir -p "$LOG_DIR"
run=$LOG_DIR/freshdiskann_insert_$(date +%Y%m%d_%H%M%S)
start=0 elapsed=0 chunk=0
while (( start < NUM_BATCHES )); do
    out=${run}_chunk$(printf %02d $chunk)
    echo "chunk $chunk: batches $start.. ($(date))"
    rc=0
    "$BASELINES_BUILD/tools/bench_freshdiskann" --dataset fineweb \
        --index "$P" --query "$QUERY" --insert-data "$DATA" --groundtruth "$GT_DIR" \
        --csv "$out.csv" --base-n "$BASE_N" --batch-size "$BATCH_SIZE" \
        --num-batches "$NUM_BATCHES" --merge-size "$MERGE_SIZE" \
        --start-batch "$start" --elapsed-offset "$elapsed" --ckpt "$chunk" \
        --search-threads "$SEARCH_THREADS" --insert-threads "$INSERT_THREADS" \
        -K 10 --search-L 40 --insert-L 100 --mem-L 128 \
        --entrance-graph "${INDEX}_mem.index" --entry-L 10 > "$out.log" 2>&1 || rc=$?
    # After a merge the driver exits and leaves <csv>.next_start to resume from.
    if [[ -f $out.csv.next_start ]]; then
        start=$(sed -n 's/^next_start_batch=//p' "$out.csv.next_start")
        elapsed=$(sed -n 's/^elapsed_s=//p' "$out.csv.next_start")
        chunk=$((chunk + 1))
    elif (( rc != 0 )); then
        echo "error: chunk $chunk failed (exit $rc), see $out.log" >&2
        exit "$rc"
    else
        break
    fi
done

# One CSV for the run, with a single run-wide throughput row.
python3 "$BASELINES/freshdiskann/aggregate_chunks.py" "$run.csv"
echo "csv: $run.csv"
