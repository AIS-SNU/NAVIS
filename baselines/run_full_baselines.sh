#!/usr/bin/env bash
# Run every baseline on FineWeb, one system at a time: convert the NAVIS index,
# run its workloads, then delete the converted index and working copies to
# free disk for the next system. The NAVIS index and the data are only read.
# Finishes with summarize.py over all runs.
#
#   DATA_DIR=/data/fineweb bash baselines/run_full_baselines.sh [system ...]
#
# Systems (default: all, in this order): odinann ipdiskann freshdiskann
#   KEEP_INDEX=1       keep each converted index after its runs
#   MEM_LIMIT_GB=<n>   cap every step's memory (systemd-run --user scope, no swap)
#   MEM_ALLOW_SWAP=1   with MEM_LIMIT_GB, let a step swap out cold pages instead of
#                      being OOM-killed (FreshDiskANN's merge at FineWeb 60M)
set -uo pipefail
source "$(dirname "$0")/env.sh"

declare -A STEPS=(
    [odinann]="prepare.sh;run_insert.sh;run_insert.sh cache;run_delete.sh;run_delete.sh cache"
    [ipdiskann]="prepare.sh;run_insert.sh"
    [freshdiskann]="prepare.sh;run_insert.sh"
)
declare -A INDEX_DIR=(
    [odinann]=$(dirname "$ODINANN_INDEX")
    [ipdiskann]=$IPDISKANN_DIR
    [freshdiskann]=$(dirname "$FRESHDISKANN_INDEX")
)
systems=("$@")
[[ ${#systems[@]} -gt 0 ]] || systems=(odinann ipdiskann freshdiskann)
for sys in "${systems[@]}"; do
    [[ -n ${STEPS[$sys]:-} ]] || { echo "unknown system: $sys" >&2; exit 1; }
done

mkdir -p "$LOG_DIR"
status=$LOG_DIR/baselines_$(date +%Y%m%d_%H%M%S).status
started=$status.start
touch "$started"
note() { echo "$(date '+%F %T') $*" | tee -a "$status"; }

# run_step <script> [arg]: one step, with its peak memory in the status file.
run_step() {
    local cmd=(bash "$BASELINES/$sys/$1" ${2:+"$2"}) rc
    if [[ -n ${MEM_LIMIT_GB:-} ]]; then
        local swap=(-p MemorySwapMax=0)
        [[ ${MEM_ALLOW_SWAP:-0} == 1 ]] && swap=()
        cmd=(systemd-run --user --scope -q -p "MemoryMax=${MEM_LIMIT_GB}G" "${swap[@]}" "${cmd[@]}")
    fi
    /usr/bin/time -f "peak RSS %M KB, %e s" -o "$status.time" "${cmd[@]}"
    rc=$?
    note "done  $sys/$1${2:+ $2} (exit $rc; $(tail -1 "$status.time"))"
    return "$rc"
}

for sys in "${systems[@]}"; do
    ok=1
    IFS=';' read -ra steps <<< "${STEPS[$sys]}"
    for step in "${steps[@]}"; do
        read -r script arg <<< "$step"
        note "start $sys/$step"
        if ! run_step "$script" "${arg:-}"; then
            ok=0
            [[ $script == prepare.sh ]] && break   # nothing to run without the index
        fi
    done
    dir=${INDEX_DIR[$sys]}
    if (( ok )) && [[ ${KEEP_INDEX:-0} != 1 ]]; then
        note "deleting $dir ($(du -sh "$dir" | cut -f1))"
        rm -rf -- "$dir"
    elif (( !ok )); then
        note "keeping $dir: a $sys step failed"
    fi
done
rm -f "$status.time"

# Summary of every run this script started, with the metrics of the README table.
mapfile -t runs < <(find "$LOG_DIR" -maxdepth 1 -newer "$started" \( \
    -name 'odinann_*insert_*.log' -o -name 'odinann_*delete_*.log' \
    -o -name 'ipdiskann_insert_*.csv' -o \( -name 'freshdiskann_insert_*.csv' ! -name '*_chunk*' \) \) | sort)
rm -f "$started"
if [[ ${#runs[@]} -gt 0 ]]; then
    python3 "$BASELINES/summarize.py" "${runs[@]}" | tee -a "$status"
fi
note "status: $status"
