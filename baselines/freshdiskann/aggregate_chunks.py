"""Combine the per-chunk CSVs of a FreshDiskANN run into one CSV.

run_insert.sh runs bench_freshdiskann in chunks (one per merge) that write
<run>_chunkNN.{csv,log}. This writes <run>.csv with:
- each merge-boundary batch row (tick_idx=-1) annotated with merge_wait_s,
  insert_wall_s, and pure_insert_throughput_vps (the batch also waited for
  the merge, from the "trigger_merge inline after batch N (Xs)" log line);
- one run-wide tick_idx=-4 row in place of the per-chunk ones: inserts over
  the total insertion and merge time.

Usage: python3 aggregate_chunks.py <run>.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# `[bench_freshdiskann] trigger_merge inline after batch 29 (2984.51s)`
TRIGGER_MERGE_RE = re.compile(
    r"trigger_merge inline after batch (\d+) \(([\d.]+)s\)"
)

CSV_COLS = [
    "system", "dataset", "base_n", "batch_idx", "batch_size", "tick_idx",
    "t_elapsed_s", "inserted_so_far", "qps", "mean_lat_ms",
    "p50_ms", "p90_ms", "p95_ms", "p99_ms", "p999_ms",
    "recall_at_10", "mean_disk_ios", "rss_mib", "cache_hit_rate",
    "disk_index_bytes", "search_threads", "insert_threads",
    "cache_gb", "search_L", "insert_L", "K", "notes",
]


def parse_notes(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in (s or "").split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def notes_to_str(d: dict[str, str]) -> str:
    return ";".join(f"{k}={v}" for k, v in d.items())


def find_chunks(run_csv: Path) -> list[Path]:
    """Discover chunk CSVs by stripping .csv and matching _chunkNN.csv."""
    base = run_csv.with_suffix("").name
    parent = run_csv.parent
    rx = re.compile(re.escape(base) + r"_chunk(\d+)\.csv$")
    found = []
    for p in parent.iterdir():
        m = rx.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    return [p for _, p in found]


def extract_merge_durations(chunk_logs: list[Path]) -> dict[int, float]:
    """batch_idx -> merge_wait_s from 'trigger_merge inline after batch N (Xs)' log lines."""
    out: dict[int, float] = {}
    for lp in chunk_logs:
        if not lp.exists():
            continue
        try:
            txt = lp.read_text(errors="ignore")
        except Exception:
            continue
        for m in TRIGGER_MERGE_RE.finditer(txt):
            batch = int(m.group(1))
            wait_s = float(m.group(2))
            # Last match wins if duplicates; normally there's just one per chunk.
            out[batch] = wait_s
    return out


def rebuild(run_csv: Path, out_csv: Path) -> tuple[int, int, int]:
    chunk_csvs = find_chunks(run_csv)
    if not chunk_csvs:
        raise SystemExit(f"ERROR: no chunk CSVs found for {run_csv}")
    chunk_logs = [c.with_suffix(".log") for c in chunk_csvs]
    merge_waits = extract_merge_durations(chunk_logs)

    out_rows: list[dict] = []
    patched_count = 0
    grand_inserts = 0
    grand_insert_wall_s = 0.0
    grand_drain_s = 0.0
    grand_total_time_s = 0.0
    per_chunk_summaries: list[tuple[int, dict]] = []  # (chunk_idx, summary_row)

    for chunk_idx, cp in enumerate(chunk_csvs):
        with open(cp, newline="") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                try:
                    tick = int(r["tick_idx"])
                    batch = int(r["batch_idx"])
                except (TypeError, ValueError):
                    out_rows.append(r)
                    continue

                if tick == -1 and batch in merge_waits:
                    # Patch merge-boundary batch summary.
                    notes = parse_notes(r.get("notes", ""))
                    merge_wait = merge_waits[batch]
                    batch_wall = float(notes.get("batch_wall_s", "0") or 0)
                    insert_wall = max(batch_wall - merge_wait, 0.0)
                    bs = int(r.get("batch_size", 0) or 0)
                    pure_vps = (bs / insert_wall) if insert_wall > 0 else 0.0
                    notes["merge_wait_s"] = f"{merge_wait:.3f}"
                    notes["insert_wall_s"] = f"{insert_wall:.3f}"
                    notes["pure_insert_throughput_vps"] = f"{pure_vps:.2f}"
                    r = dict(r)
                    r["notes"] = notes_to_str(notes)
                    patched_count += 1

                if tick == -4:
                    # Stash per-chunk summary, drop from concat; we'll emit
                    # one grand-total -4 row at the end.
                    per_chunk_summaries.append((chunk_idx, dict(r)))
                    continue

                out_rows.append(r)

    # Compute grand total from per-chunk summaries.
    for idx, s in per_chunk_summaries:
        n = parse_notes(s.get("notes", ""))
        try:
            grand_inserts = max(grand_inserts, int(s.get("inserted_so_far", 0) or 0))
        except (TypeError, ValueError):
            pass
        try:
            grand_insert_wall_s += float(n.get("total_insert_time_s", 0) or 0)
            grand_drain_s += float(n.get("total_merge_drain_s", 0) or 0)
            grand_total_time_s += float(n.get("total_time_s", 0) or 0)
        except (TypeError, ValueError):
            pass

    # Emit one grand-total tick=-4 row at the end.
    if per_chunk_summaries:
        last = per_chunk_summaries[-1][1]
        grand_fair = (grand_inserts / grand_total_time_s) if grand_total_time_s > 0 else 0.0
        notes = {
            "fair_throughput_vps":   f"{grand_fair:.2f}",
            "total_insert_time_s":   f"{grand_insert_wall_s:.3f}",
            "total_merge_drain_s":   f"{grand_drain_s:.3f}",
            "total_time_s":          f"{grand_total_time_s:.3f}",
            "chunks":                str(len(per_chunk_summaries)),
        }
        grand = dict(last)
        grand["t_elapsed_s"] = last.get("t_elapsed_s", "")
        grand["batch_idx"] = "-1"
        grand["tick_idx"] = "-4"
        grand["inserted_so_far"] = str(grand_inserts)
        grand["notes"] = notes_to_str(notes)
        out_rows.append(grand)

    # Write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLS})

    return len(out_rows), patched_count, len(chunk_csvs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_csv", type=Path,
                    help="<run>.csv; reads <run>_chunkNN.{csv,log}")
    args = ap.parse_args()

    n_rows, n_patched, n_chunks = rebuild(args.run_csv, args.run_csv)
    print(f"wrote {n_rows} rows from {n_chunks} chunks "
          f"({n_patched} merge-boundary batches patched) -> {args.run_csv}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
