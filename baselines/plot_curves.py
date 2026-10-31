#!/usr/bin/env python3
"""Curves of the README comparison, in the layout of the paper's figures, from
navis_bench logs, OdinANN logs, and baseline CSVs (FreshDiskANN, IP-DiskANN):

insert (Fig. 10), over elapsed time:
  (a) insertion throughput: vectors inserted so far / elapsed time; FreshDiskANN
      at its time-averaged rate, merges included, as in the paper;
  (b) search QPS and (c) mean search latency of the concurrent search passes,
      averaged over 1-minute windows;
  (d) Recall@10 after each batch, over the vectors inserted.
delete (Fig. 19), over the update progress (vectors replaced, % of the base):
  (a) replace throughput of each batch, (b) mean search QPS during each batch,
  (c) Recall@10 after each batch.

usage: plot_curves.py insert|delete <out.png|out.pdf> <log or csv> ...
"""
import csv
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_N, BATCH, QUERIES = 50_000_000, 100_000, 10_000
SMOOTH_S = 60    # insert: window for (b) and (c)
XMAX_MIN = 380   # insert: time axis; longer runs continue past the edge, as in the paper
STYLE = {  # the paper's line styles and colors (blue and orange darkened for contrast)
    'FreshDiskANN': dict(color='#237a3b', ls='-.', lw=0.8),
    'IP-DiskANN':   dict(color='#6a3d9a', ls=':', lw=1.6),
    'OdinANN':      dict(color='#b22222', ls=(0, (5, 2)), lw=1.4),
    'OdinANN+$':    dict(color='#c47a08', ls=(0, (3, 1.5)), lw=1.4),
    'NAVIS':        dict(color='#255a9e', ls='-', lw=1.6),
}

# Every parser returns one run:
#   ticks    [(t_s, batch, qps, mean_lat_ms)]  concurrent search passes
#   batches  [(t_end_s, batch_wall_s)]         one per batch, in order
#   recall   [recall_pct]                      after each batch, in order
#   fair     (vectors/s, total_s) when merges are averaged over time (FreshDiskANN)


def from_csv(path):
    rows = list(csv.DictReader(open(path)))
    name = {'freshdiskann': 'FreshDiskANN', 'ipdiskann': 'IP-DiskANN'}[rows[0]['system']]
    ticks = [(float(r['t_elapsed_s']), int(r['batch_idx']), float(r['qps']), float(r['mean_lat_ms']))
             for r in rows if int(r['tick_idx']) >= 0]
    batches = [(float(r['t_elapsed_s']),
                float(re.search(r'(?:batch|insert)_wall_s=([\d.]+)', r['notes']).group(1)))
               for r in rows if r['tick_idx'] == '-1']
    recall = [100 * float(r['recall_at_10'])
              for r in rows if r['tick_idx'] == '-2' and int(r['batch_idx']) >= 0]
    notes = [r['notes'] for r in rows if r['tick_idx'] == '-4'][-1]
    fair = None
    if name == 'FreshDiskANN':
        fair = (float(re.search(r'fair_throughput_vps=([\d.]+)', notes).group(1)),
                float(re.search(r'total_time_s=([\d.]+)', notes).group(1)))
    return name, dict(ticks=ticks, batches=batches, recall=recall, fair=fair)


def from_navis(text):
    # Tick times count from the batch start; a batch's run time is in its
    # 'done' line, and the pass after it takes QUERIES / QPS.
    tick = re.compile(r'^\s*\[\s*(\d+)s\] inserted \d+/\d+.*QPS\s+([\d.]+)\s+mean\s+([\d.]+) us')
    done = re.compile(r'^\s*Batch (\d+) done in ([\d.]+)s')
    after = re.compile(r'^\s*after batch \d+:.*QPS\s+([\d.]+).*Recall@K\s+([\d.]+)%')
    ticks, batches, recall = [], [], []
    t_base = 0.0
    for line in text.splitlines():
        if m := tick.match(line):
            ticks.append((t_base + int(m.group(1)), len(batches),
                          float(m.group(2)), float(m.group(3)) / 1000))
        elif m := done.match(line):
            t_base += float(m.group(2))
            batches.append((t_base, float(m.group(2))))
        elif m := after.match(line):
            t_base += QUERIES / float(m.group(1))
            recall.append(float(m.group(2)))
    return 'NAVIS', dict(ticks=ticks, batches=batches, recall=recall, fair=None)


def from_odinann(text):
    # Search rows: L QPS mean P50 P90 P95 P99 P99.9 Recall ...; Recall@10 is 0
    # while a batch runs. A row is timed by the last 'current time: N' before it.
    # Recall passes count only right after a batch: the first one precedes the
    # first batch, and the deletion workload adds one at each merge.
    name = 'OdinANN+$' if 'Enabling OdinPackedCache' in text else 'OdinANN'
    ticks, batches, recall = [], [], []
    t, after_batch = 0, False
    for line in text.splitlines():
        if m := re.search(r'current time: (\d+)', line):
            t = int(m.group(1))
        elif m := re.search(r'Inserted \d+ points in ([\d.]+)s', line):
            batches.append((t, float(m.group(1))))
            after_batch = True
        elif re.match(r'^\s*40\s+[\d.]+\s', line):
            row = line.split()
            if float(row[8]) == 0:
                ticks.append((t, len(batches), float(row[1]), float(row[2])))
            else:
                if after_batch:
                    recall.append(float(row[8]))
                after_batch = False
    t0 = ticks[0][0]
    return name, dict(ticks=[(t - t0, *rest) for t, *rest in ticks],
                      batches=[(t - t0, w) for t, w in batches], recall=recall, fair=None)


def load(path):
    if path.endswith('.csv'):
        return from_csv(path)
    text = open(path, errors='replace').read().replace('\r', '\n')
    return from_navis(text) if 'navis_bench' in text else from_odinann(text)


def by_window(points, window):
    bins = {}
    for t, *vals in points:
        bins.setdefault(int(t // window), []).append(vals)
    return [((b + 0.5) * window, *map(st.mean, zip(*bins[b]))) for b in sorted(bins)]


def style_axis(ax, ylabel, xlabel, title, ylim):
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f'{xlabel}\n{title}')
    ax.set_ylim(*ylim)
    ax.grid(True, color='#dddddd', ls='--', lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))


def plot_insert(runs):
    fig, (ax_tp, ax_qps, ax_lat, ax_rec) = plt.subplots(1, 4, figsize=(13, 2.7))
    for name, d in runs.items():
        s = STYLE[name]
        end = d['fair'][1] if d['fair'] else max(t for t, _ in d['batches'])
        if d['fair']:
            ax_tp.plot([0, end / 60], [d['fair'][0]] * 2, **s)
        else:
            ax_tp.plot([t / 60 for t, _ in d['batches']],
                       [(i + 1) * BATCH / t for i, (t, _) in enumerate(d['batches'])], **s)
        sm = by_window([(t, q, l) for t, _, q, l in d['ticks']], SMOOTH_S)
        ax_qps.plot([t / 60 for t, *_ in sm], [q for _, q, _ in sm], **s)
        ax_lat.plot([t / 60 for t, *_ in sm], [l for *_, l in sm], **s)
        for ax in (ax_tp, ax_qps, ax_lat):
            ax.axvline(end / 60, color=s['color'], ls=':', lw=1.4)
        ax_rec.plot([(i + 1) * BATCH / 1e6 for i in range(len(d['recall']))], d['recall'],
                    label=name, **s)
    t_axis = 'Elapsed Time [min]'
    style_axis(ax_tp, 'Throughput [VPS]', t_axis, '(a) Insertion Throughput over Time', (0, 2000))
    style_axis(ax_qps, 'Throughput [QPS]', t_axis, '(b) Search Throughput over Time', (0, 14000))
    style_axis(ax_lat, 'Avg. Lat. [ms]', t_axis, '(c) Search Latency over Time', (0, 20))
    style_axis(ax_rec, 'Recall@10 [%]', 'Inserted Vectors [M]', '(d) Recall over Batches', (50, 100))
    for ax in (ax_tp, ax_qps, ax_lat):
        ax.set_xlim(0, XMAX_MIN)
    ax_rec.set_xlim(0, 10.2)
    ax_rec.axhline(95, color='#888888', ls='--', lw=1.0)
    ax_rec.text(9.8, 93, 'Target Recall=95', ha='right', va='top', fontsize=8, color='#555555')
    return fig, ax_rec


def plot_delete(runs):
    fig, (ax_tp, ax_qps, ax_rec) = plt.subplots(1, 3, figsize=(10, 2.7))
    pct = lambda b: (b + 1) * BATCH / BASE_N * 100
    for name, d in runs.items():
        s = STYLE[name]
        ax_tp.plot([pct(b) for b in range(len(d['batches']))],
                   [BATCH / w for _, w in d['batches']], **s)
        per_batch = {}
        for _, b, q, _ in d['ticks']:
            per_batch.setdefault(b, []).append(q)
        ax_qps.plot([pct(b - 0.5) for b in sorted(per_batch)],
                    [st.mean(per_batch[b]) for b in sorted(per_batch)], **s)
        ax_rec.plot([pct(b) for b in range(len(d['recall']))], d['recall'], label=name, **s)
    x_axis = 'Update Progress [% of Base]'
    style_axis(ax_tp, 'Throughput [VPS]', x_axis, '(a) Replace Throughput', (0, 2500))
    style_axis(ax_qps, 'Throughput [QPS]', x_axis, '(b) Search Throughput', (0, 14000))
    style_axis(ax_rec, 'Recall@10 [%]', x_axis, '(c) Recall', (75, 100))
    for ax in (ax_tp, ax_qps, ax_rec):
        ax.set_xlim(0, 20.4)
    return fig, ax_rec


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ('insert', 'delete'):
        sys.exit(__doc__)
    loaded = dict(load(p) for p in sys.argv[3:])
    runs = {name: loaded[name] for name in STYLE if name in loaded}
    for name, d in runs.items():
        print(f"{name:<13} {len(d['batches'])} batches, {len(d['recall'])} recall passes, "
              f"{len(d['ticks'])} concurrent passes")

    plt.rcParams.update({'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.edgecolor': '#222222', 'axes.labelcolor': '#222222',
                         'xtick.color': '#222222', 'ytick.color': '#222222'})
    fig, ax_legend = (plot_insert if sys.argv[1] == 'insert' else plot_delete)(runs)
    handles, labels = ax_legend.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=True,
               edgecolor='#222222', fancybox=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(sys.argv[2], dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
