#!/usr/bin/env python3
"""One README table row per run, from a navis_bench log, an OdinANN log, or a
baseline CSV (FreshDiskANN, IP-DiskANN):

- Throughput: vectors inserted / total batch insertion time (FreshDiskANN:
  including its merges, from the run-wide summary row).
- QPS, P50 / P99 latency: averages over the search passes that run
  concurrently with the batches.
- Recall@10: average over the search passes after each batch.

usage: summarize.py <log or csv> ...
"""
import csv
import re
import statistics as st
import sys


def notes(row):
    return dict(kv.split('=', 1) for kv in row['notes'].split(';') if '=' in kv)


def from_csv(path):
    rows = list(csv.DictReader(open(path)))
    conc = [r for r in rows if int(r['tick_idx']) >= 0]
    post = [r for r in rows if r['tick_idx'] == '-2' and int(r['batch_idx']) >= 0]
    summary = [notes(r) for r in rows if r['tick_idx'] == '-4']
    if summary and 'fair_throughput_vps' in summary[-1]:
        rate = float(summary[-1]['fair_throughput_vps'])
    else:
        batches = [r for r in rows if r['tick_idx'] == '-1']
        # bench harness writes batch_wall_s; bench_diskann3 writes insert_wall_s
        wall = sum(float(notes(r).get('batch_wall_s') or notes(r)['insert_wall_s'])
                   for r in batches)
        rate = sum(int(r['batch_size']) for r in batches) / wall
    return (rate, [float(r['qps']) for r in conc], [float(r['p50_ms']) for r in conc],
            [float(r['p99_ms']) for r in conc], [100 * float(r['recall_at_10']) for r in post])


def from_navis_log(text):
    tick = re.compile(r'^\s*\[\s*\d+s\] inserted .*QPS\s+([\d.]+)\s+mean\s+[\d.]+ us\s+'
                      r'p50\s+([\d.]+)\s+p99\s+([\d.]+)', re.M)
    ticks = [tuple(map(float, m.groups())) for m in tick.finditer(text)]
    batches = re.findall(r'^\s*Batch \d+ done in ([\d.]+)s: (\d+) inserts', text, re.M)
    recall = re.findall(r'^\s*after batch \d+:.*Recall@K\s+([\d.]+)%', text, re.M)
    rate = sum(int(n) for _, n in batches) / sum(float(t) for t, _ in batches)
    return (rate, [t[0] for t in ticks], [t[1] / 1000 for t in ticks],
            [t[2] / 1000 for t in ticks], [float(r) for r in recall])


def from_odinann_log(text):
    # Search rows: L QPS mean P50 P90 P95 P99 P99.9 Recall ...; recall 0 while
    # inserting, > 0 for the other passes. Those count only right after a
    # batch: the first one precedes the first batch, and the deletion workload
    # adds one at each merge.
    conc, post, after_batch = [], [], False
    for line in text.splitlines():
        if re.search(r'Inserted \d+ points in', line):
            after_batch = True
        elif re.match(r'^\s*40\s+[\d.]+\s', line):
            row = line.split()
            if float(row[8]) == 0:
                conc.append(row)
            else:
                if after_batch:
                    post.append(row)
                after_batch = False
    batches = re.findall(r'Inserted (\d+) points in ([\d.]+)s', text)
    rate = sum(int(n) for n, _ in batches) / sum(float(t) for _, t in batches)
    return (rate, [float(r[1]) for r in conc], [float(r[3]) for r in conc],
            [float(r[6]) for r in conc], [float(r[8]) for r in post])


def summarize(path):
    if path.endswith('.csv'):
        return from_csv(path)
    text = open(path, errors='replace').read().replace('\r', '\n')
    return from_navis_log(text) if 'navis_bench' in text else from_odinann_log(text)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(f"{'run':<44} {'vectors/s':>10} {'QPS':>9} {'P50/P99 (ms)':>14} {'Recall@10':>10}")
    for path in sys.argv[1:]:
        rate, qps, p50, p99, recall = summarize(path)
        name = path.rsplit('/', 1)[-1]
        print(f"{name:<44} {rate:>10,.0f} {st.mean(qps):>9,.0f} "
              f"{st.mean(p50):>6.2f} / {st.mean(p99):<5.2f} {st.mean(recall):>9.2f}%")


if __name__ == '__main__':
    main()
