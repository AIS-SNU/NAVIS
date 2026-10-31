#!/usr/bin/env bash
# Convert the NAVIS graph into an IP-DiskANN (DiskANN3) index: import the
# graph and full-precision vectors into BfTree stores, then add the 1-bit
# spherical codes that IP-DiskANN keeps in memory. Reads the graph from
# ${INDEX}_mem_graph, which navis_build leaves next to the index.
#
#   DATA_DIR=/data/fineweb bash baselines/ipdiskann/prepare.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
need_baseline ipdiskann/release/navis_to_diskann3
need_baseline ipdiskann/release/add_spherical_quantization

FULL=$IPDISKANN_DIR/full/index        # graph + full-precision vectors
QUANT=$IPDISKANN_DIR/quantized/index  # + 1-bit codes: the index the runs use
THREADS=${GT_THREADS:-$(nproc)}
mkdir -p "$(dirname "$FULL")" "$(dirname "$QUANT")"

if [[ -f ${FULL}_neighbors.bftree ]]; then
    echo "graph import exists: $FULL"
else
    run_logged ipdiskann_import "$BASELINES_BUILD/ipdiskann/release/navis_to_diskann3" \
        --graph "${INDEX}_mem_graph" --data "$BASE" --base-n "$BASE_N" \
        --output-prefix "$FULL" --capacity "$TOTAL_N" --metric l2 \
        --pruned-degree 96 --l-build 100 --alpha 1.2 --backedge-ratio 1.0 \
        --vector-cache-mib 2048 --neighbor-cache-mib 2048 --threads "$THREADS"
fi

if [[ -f ${QUANT}_quant_data.bin ]]; then
    echo "IP-DiskANN index exists: $QUANT"
else
    run_logged ipdiskann_quantize "$BASELINES_BUILD/ipdiskann/release/add_spherical_quantization" \
        --source-prefix "$FULL" --output-prefix "$QUANT" --data "$BASE" \
        --graph "${INDEX}_mem_graph" --base-n "$BASE_N" --train-points 100000 \
        --quant-cache-gib "$CACHE_GB" --quant-bits 1 --threads "$THREADS" \
        --l-build 100 --alpha 1.2 --backedge-ratio 1.0
fi
