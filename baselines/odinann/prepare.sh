#!/usr/bin/env bash
# Convert the NAVIS index (scripts/fineweb_prepare.sh) into an OdinANN index:
# the same graph in DiskANN's packed layout, with PQ retrained on the base,
# and NAVIS's entrance graph (vertex ids are unchanged by the conversion).
#
#   DATA_DIR=/data/fineweb bash baselines/odinann/prepare.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline tools/navis_to_odinann

if [[ -f $ODINANN_INDEX.ready ]]; then
    echo "OdinANN index exists: $ODINANN_INDEX"
    exit 0
fi
mkdir -p "$(dirname "$ODINANN_INDEX")"
run_logged odinann_prepare "$BASELINES_BUILD/tools/navis_to_odinann" \
    --navis-prefix "$INDEX" --out-prefix "$ODINANN_INDEX" --base "$BASE" \
    --R 96 --pq-bytes 128
for f in _mem.index _mem.index.data _mem.index.tags; do
    cp "$INDEX$f" "$ODINANN_INDEX$f"
done
touch "$ODINANN_INDEX.ready"
