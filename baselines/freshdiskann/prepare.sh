#!/usr/bin/env bash
# Convert the NAVIS index (scripts/fineweb_prepare.sh) into a FreshDiskANN
# index: the same graph in DiskANN's packed layout, PQ retrained on the base,
# and empty in-memory indexes.
#
#   DATA_DIR=/data/fineweb bash baselines/freshdiskann/prepare.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline tools/navis_to_freshdiskann

if [[ -f $FRESHDISKANN_INDEX.ready ]]; then
    echo "FreshDiskANN index exists: $FRESHDISKANN_INDEX"
    exit 0
fi
mkdir -p "$(dirname "$FRESHDISKANN_INDEX")"
run_logged freshdiskann_prepare "$BASELINES_BUILD/tools/navis_to_freshdiskann" \
    --navis-prefix "$INDEX" --out-prefix "$FRESHDISKANN_INDEX" --base "$BASE" \
    --R 96 --pq-bytes 128
touch "$FRESHDISKANN_INDEX.ready"
