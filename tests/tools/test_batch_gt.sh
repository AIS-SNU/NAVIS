#!/usr/bin/env bash
# tests/tools/test_batch_gt.sh — navis_convert batch-gt against a brute-force
# oracle, for both the insert and the delete (sliding-window) workload.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
NC="${1:-$REPO/build/navis_convert}"

[ -x "$NC" ] || { echo "FAIL: navis_convert not found at $NC" >&2; exit 1; }

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# 3-d data with duplicates (ties are broken by the smaller id) and 3 queries.
python3 - "$WORK" <<'EOF'
import random, struct, sys
work = sys.argv[1]
rng = random.Random(7)
n, dim = 30, 3
data = [[rng.randint(0, 9) for _ in range(dim)] for _ in range(n)]
data[17] = list(data[3]); data[25] = list(data[3])          # duplicates
queries = [[rng.randint(0, 9) for _ in range(dim)] for _ in range(3)]
def write(path, rows):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(rows), dim))
        for r in rows: f.write(struct.pack(f"<{dim}f", *r))
write(f"{work}/data.bin", data)
write(f"{work}/query.bin", queries)
index, total, batch, K = 12, 24, 4, 3     # rows 24..29 are never live
def topk(q, lo, hi):
    d = sorted((sum((a - b) ** 2 for a, b in zip(data[i], q)), i) for i in range(lo, hi))
    return [i for _, i in d[:K]]
with open(f"{work}/expected.txt", "w") as f:
    for off in range(0, total - index + 1, batch):
        for mode, lo in (("insert", 0), ("delete", off)):
            ids = [topk(q, lo, index + off) for q in queries]
            f.write(f"{mode} {off} " + " ".join(str(i) for r in ids for i in r) + "\n")
EOF

# A tiny budget forces several partitions (and segment-aligned boundaries).
"$NC" batch-gt "$WORK/data.bin" "$WORK/query.bin" \
    --index-npts 12 --total-npts 24 --batch 4 -K 3 --mem-budget 0.00004 \
    -o "$WORK/insert" --delete-o "$WORK/delete" > "$WORK/run.out"
grep -q "partitions" "$WORK/run.out"

python3 - "$WORK" <<'EOF'
import struct, sys
work = sys.argv[1]
for line in open(f"{work}/expected.txt"):
    mode, off, *ids = line.split()
    with open(f"{work}/{mode}/gt_{off}.bin", "rb") as f:
        nq, k = struct.unpack("<II", f.read(8))
        got = list(struct.unpack(f"<{nq * k}I", f.read(nq * k * 4)))
    assert (nq, k) == (3, 3), (mode, off, nq, k)
    assert got == [int(i) for i in ids], (mode, off, got, ids)
EOF

# Error cases: non-divisible stream, and a data file shorter than total-npts.
for args in "--index-npts 12 --total-npts 23 --batch 4" \
            "--index-npts 12 --total-npts 40 --batch 4"; do
    set +e
    "$NC" batch-gt "$WORK/data.bin" "$WORK/query.bin" $args -K 3 -o "$WORK/bad" \
        > /dev/null 2> "$WORK/err"
    rc=$?
    set -e
    [ "$rc" -ne 0 ] || { echo "FAIL: batch-gt $args unexpectedly succeeded" >&2; exit 1; }
    grep -q -E "divisible|fewer than" "$WORK/err" || { cat "$WORK/err" >&2; exit 1; }
done

echo "PASS"
