# Common settings for the baseline scripts (sourced, not executed). Builds on
# scripts/fineweb_env.sh, so the data, ground truth, workload, and cache size
# are the same as for NAVIS:
#   DATA_DIR=/data/fineweb bash baselines/odinann/run_insert.sh

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/fineweb_env.sh"

BASELINES=$REPO/baselines
BASELINES_BUILD=${BASELINES_BUILD:-$REPO/build/baselines}

# Converted indexes (each about the size of the NAVIS index).
ODINANN_INDEX=${ODINANN_INDEX:-$DATA_DIR/odinann/odinann}
FRESHDISKANN_INDEX=${FRESHDISKANN_INDEX:-$DATA_DIR/freshdiskann/freshdiskann}
IPDISKANN_DIR=${IPDISKANN_DIR:-$DATA_DIR/ipdiskann}

SEARCH_THREADS=${SEARCH_THREADS:-22}
INSERT_THREADS=${INSERT_THREADS:-10}

# need_baseline <path under build/baselines>: fail early if a binary is missing.
need_baseline() {
    if [[ ! -x "$BASELINES_BUILD/$1" ]]; then
        echo "error: $BASELINES_BUILD/$1 not found; build first: bash baselines/build.sh" >&2
        exit 1
    fi
}
