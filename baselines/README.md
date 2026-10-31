# Baselines

The baselines compared with NAVIS on FineWeb, as in the [Performance](../README.md#performance) section and the paper's evaluation: OdinANN, OdinANN+$, IP-DiskANN, and FreshDiskANN.

Every baseline starts from the NAVIS index of the [walkthrough](../README.md#walkthrough-fineweb). Its graph is converted into the baseline's own format, so all systems begin from the same graph over the same 50M vectors. Each baseline then runs the same workload as NAVIS: 100 batches of 100K insertions with 22 search and 10 insertion threads, beam width 4, |E_search| = 40, |E_pos| = 100, and K = 10, scored against the same ground truth.

| System | Directory | What it is |
|---|---|---|
| OdinANN | [`odinann/`](odinann) | In-place updates on DiskANN's packed layout (FAST'26) |
| OdinANN+$ | [`odinann/`](odinann) | OdinANN with a 16 GiB NAVIS-Cache over its packed pages |
| IP-DiskANN | [`ipdiskann/`](ipdiskann) | The in-place update path of DiskANN3 (Bf-tree storage, 1-bit spherical codes in memory) |
| FreshDiskANN | [`freshdiskann/`](freshdiskann) | Buffered insertion: an in-memory index merged into the disk index every 3M points |

## Requirements

On top of the [NAVIS requirements](../README.md#requirements):

```bash
sudo apt install libgoogle-glog-dev libaio-dev libmkl-full-dev
```

- OdinANN uses glog, jemalloc, and BLAS. FreshDiskANN uses glog, libaio, and Intel MKL.
- IP-DiskANN needs Rust 1.97.1: `rustup toolchain install 1.97.1`.
- Disk: each converted index takes about 200 GB for FineWeb (one 4 KiB page per vertex for OdinANN and FreshDiskANN), and every run works on a fresh copy of it (two for FreshDiskANN). `run_full_baselines.sh` deletes a system's files once its runs are done.
- Run one system at a time, and not next to a NAVIS run: the runs compete for memory, CPU, and SSD bandwidth.

## Build

```bash
bash baselines/build.sh                                   # all three
bash baselines/build.sh odinann freshdiskann              # or a subset
CARGO="cargo +1.97.1" bash baselines/build.sh ipdiskann   # if 1.97.1 is not the default toolchain
```

Everything goes to `build/baselines/`. `build.sh` downloads liburing 2.10, the same pinned release as the NAVIS build, and checks its hash.

## Run on FineWeb

First, run the [walkthrough](../README.md#walkthrough-fineweb) up to the NAVIS index (`scripts/fineweb_prepare.sh`). Keep `navis_fineweb_mem_graph`: IP-DiskANN imports the graph from it. The baseline scripts read the same settings as the NAVIS scripts (`scripts/fineweb_env.sh`), so set `DATA_DIR` the same way.

To run everything, one system after another:

```bash
DATA_DIR=/data/fineweb bash baselines/run_full_baselines.sh            # or a subset: odinann ipdiskann freshdiskann
DATA_DIR=/data/fineweb MEM_LIMIT_GB=60 bash baselines/run_full_baselines.sh   # cap each step's memory
DATA_DIR=/data/fineweb MEM_LIMIT_GB=60 MEM_ALLOW_SWAP=1 bash baselines/run_full_baselines.sh freshdiskann
```

For each system it converts the NAVIS index, runs the workloads, and then deletes the converted index and its working copies, so only one system's files are on disk at a time (`KEEP_INDEX=1` keeps them). The NAVIS index and the data are only read. At the end it prints the metrics of every run with `summarize.py`, and its status file in `logs/` records each step's exit code and peak memory.

`MEM_LIMIT_GB` runs each step in a systemd scope with that memory limit and no swap, so a step that goes over it is killed. FreshDiskANN's merges go over 60 GB on FineWeb: on top of the steady 45 GB (two in-memory indexes, the PQ codes, and the entrance graph), a merge loads a second copy of the PQ codes, its graph deltas, and the 3M-point in-memory index from disk. `MEM_ALLOW_SWAP=1` lets the capped steps swap instead. Our FreshDiskANN run stayed at the 60 GB limit with up to 30 GB swapped out. This needs that much free swap, best on a different disk than the benchmark SSD.

The same steps one at a time:

```bash
export DATA_DIR=/data/fineweb

# OdinANN and OdinANN+$
bash baselines/odinann/prepare.sh
bash baselines/odinann/run_insert.sh          # OdinANN
bash baselines/odinann/run_insert.sh cache    # OdinANN+$
bash baselines/odinann/run_delete.sh          # deletion workload; add "cache" for OdinANN+$

# IP-DiskANN
bash baselines/ipdiskann/prepare.sh
bash baselines/ipdiskann/run_insert.sh

# FreshDiskANN
bash baselines/freshdiskann/prepare.sh
bash baselines/freshdiskann/run_insert.sh
```

`prepare.sh` converts the NAVIS index once, into `$DATA_DIR/{odinann,ipdiskann,freshdiskann}/`. Every run then starts from that converted index on a fresh working copy. Logs and CSVs go to `logs/`, next to the NAVIS logs.

`summarize.py` computes the metrics of the README table from the logs (NAVIS, OdinANN) and CSVs (IP-DiskANN, FreshDiskANN):

```bash
python3 baselines/summarize.py logs/fineweb_insert_*.log logs/odinann_insert_*.log \
    logs/ipdiskann_insert_*.csv logs/freshdiskann_insert_*.csv
```

```
run                                           vectors/s       QPS   P50/P99 (ms)  Recall@10
fineweb_insert_20261025_191151.log                1,485    11,499   1.73 / 4.04      98.02%
...
```

`plot_curves.py` (needs matplotlib) draws the README figures from the same files, in the layout of the paper's Fig. 10 (insertion) and Fig. 19 (deletion). It names each run by its contents, so pass one run per system:

```bash
python3 baselines/plot_curves.py insert assets/fineweb_insert.png logs/fineweb_insert_*.log \
    logs/odinann_insert_*.log logs/odinann_cache_insert_*.log logs/ipdiskann_insert_*.csv logs/freshdiskann_insert_*.csv
python3 baselines/plot_curves.py delete assets/fineweb_delete.png logs/fineweb_delete_*.log \
    logs/odinann_delete_*.log logs/odinann_cache_delete_*.log
```

The insertion throughput in the README table sets the run time. Our FineWeb runs took about 4 hours each for OdinANN and OdinANN+$, 6 hours for FreshDiskANN, and 12 hours for IP-DiskANN on the insertion workload, and about 13 hours each for OdinANN and OdinANN+$ on the deletion workload.

## Notes per system

**OdinANN** (`odinann/OdinANN`). The OdinANN artifact (FAST'26), which builds on PipeANN (OSDI'25) and DiskANN, reduced to what these experiments run:
- `tests/test_insert_search` runs the insertion workload. `tests/overall_performance` runs the deletion workload, where deletes are lazy and merged into the graph every 20 batches.
- Both binaries are built with OdinANN's optimizations (`-DBG_IO_THREAD -DDELTA_PRUNING`).
- Searches start from the 10 nearest vertices of NAVIS's entrance graph (static; `prepare.sh` copies it), as in the paper. Insertions start from the medoid, as in the artifact.
- OdinANN+$ is the same binary with `ODIN_PACKED_CACHE_GB` set. Because OdinANN keeps the packed layout, the cache holds whole pages (a vector together with its edge list).
- The conversion (`navis_to_odinann`) writes each vertex's vector and edge list into one packed node and retrains 128-byte PQ on the base vectors.
- Our changes to the artifact:
  - the NAVIS-Cache port for OdinANN+$;
  - lock-table and tag-map changes that made the deletion workload run reliably;
  - the entrance graph in the deletion benchmark (`ODIN_MEM_L`), with its vertex ids remapped at every merge, since merges renumber vertices;
  - no per-query sleep in the search threads;
  - removal of the search modes and prototype code these experiments do not use.

**IP-DiskANN** (`ipdiskann/DiskANN3`). DiskANN3 0.56.0 ([microsoft/DiskANN](https://github.com/microsoft/DiskANN)), reduced to the crates the benchmark needs. We added APIs to attach and snapshot the quantized store, and insertion with exact reranking. The benchmark crate (`ipdiskann/src`) runs IP-DiskANN with insertion matched to OdinANN and NAVIS: exact reranking of the top-L candidates and neighbor pruning on full vectors. It has three binaries:
- `navis_to_diskann3` imports the NAVIS graph and vectors into BfTree stores;
- `add_spherical_quantization` adds the in-memory 1-bit codes;
- `bench_diskann3` runs the workload.

DiskANN3 starts every search from a single start point; it has no entrance graph.

**FreshDiskANN** (`freshdiskann/FreshDiskANN`). The FreshDiskANN version that OdinANN's authors evaluated against, reduced to the library:
- Insertions go to an in-memory index, built with L = 128 (FreshDiskANN's default; the paper's runs used 0) and merged into the disk index every 3M points (6% of the base). The merge runs with the insertion threads, and its time counts toward throughput, as in the paper.
- Searches start from the 10 nearest vertices of NAVIS's entrance graph, as for OdinANN. `run_insert.sh` reads it from the NAVIS index; it stays static, and its ids stay valid because merges only number the new points. Insertions and the merge's own searches start from the medoid.
- As in OdinANN's evaluation, `run_insert.sh` restarts FreshDiskANN after every merge to avoid its memory corruption. `aggregate_chunks.py` joins the parts into one CSV.
- The merge runs inside the insertion phase, so the 5 s search passes continue during merges.
- Merges need more than 60 GB on FineWeb; see `MEM_ALLOW_SWAP` above.
- Our changes:
  - entrance-graph start points for searches (`--entrance-graph`, `--entry-L`), which FreshDiskANN does not have;
  - the idle in-memory index is freed before it is reallocated, so switching indexes does not briefly hold a third one (about 18 GB);
  - the number of merge threads (`FRESH_MERGE_THREADS`);
  - the disk degree is set after the index loads;
  - no deletion bookkeeping, since the workload only inserts;
  - I/O context registration on the merge thread.

## Licenses

OdinANN, FreshDiskANN, and DiskANN3 are MIT-licensed. Their license and notice files are kept in their directories. FreshDiskANN also bundles a few third-party headers under their own licenses, stated in each file: `ctpl_stl.h` (Apache-2.0), `tcmalloc/malloc_extension.h` (BSD-3-Clause), `threadpool.h` (zlib), and `tsl/` (MIT).
