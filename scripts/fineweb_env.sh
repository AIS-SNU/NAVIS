# Common settings for the FineWeb scripts (sourced, not executed).
# Every variable can be overridden from the environment, e.g.
#   DATA_DIR=/data/fineweb NUM_BATCHES=10 bash scripts/fineweb_insert.sh

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$REPO/build}
LOG_DIR=${LOG_DIR:-$REPO/logs}

# Inputs: FineWeb-Edu x EmbeddingGemma, 768-d float32 (see README).
DATA_DIR=${DATA_DIR:?set DATA_DIR to the directory with the FineWeb .bin files (on the NVMe SSD)}
DATA=${DATA:-$DATA_DIR/80M_float.bin}      # base rows first, then the insert stream
QUERY=${QUERY:-$DATA_DIR/80M_query.bin}    # 10,000 queries

# Outputs of fineweb_prepare.sh.
BASE=${BASE:-$DATA_DIR/base_50M.bin}
INDEX=${INDEX:-$DATA_DIR/navis_fineweb}
GT_DIR=${GT_DIR:-$DATA_DIR/gt}                    # insert workload ground truth
GT_DELETE_DIR=${GT_DELETE_DIR:-$DATA_DIR/gt_delete}  # sliding-window (delete) ground truth

# Workload (paper defaults: 50M base, 100 batches x 100K = 10M insertions).
BASE_N=${BASE_N:-50000000}
BATCH_SIZE=${BATCH_SIZE:-100000}
NUM_BATCHES=${NUM_BATCHES:-100}
TOTAL_N=$(( BASE_N + BATCH_SIZE * NUM_BATCHES ))

CACHE_GB=${CACHE_GB:-16}          # NAVIS-Cache
MEM_BUDGET_GB=${MEM_BUDGET_GB:-48} # ground truth + index build (keep well below RAM)
GT_THREADS=${GT_THREADS:-$(nproc)}

# Search / insertion parameters shared by every run (paper configuration).
BENCH_COMMON=(-t float -K 10 --search-L 40 --insert-L 100 --mem-L 10 --beam-width 4
              --cache-gb "$CACHE_GB")

need_tool() {
    if [[ ! -x "$BUILD_DIR/$1" ]]; then
        echo "error: $BUILD_DIR/$1 not found; build first: cmake -B build && cmake --build build -j" >&2
        exit 1
    fi
}

# run_logged <name> <command...>: run a command, echoing it and teeing all
# output to $LOG_DIR/<name>_<timestamp>.log.
run_logged() {
    local name=$1; shift
    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "+ $*" | tee "$log"
    local rc=0
    "$@" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
    echo "log: $log (exit $rc)"
    return "$rc"
}
