use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
        mpsc::{self, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use diskann::{
    graph::{DiskANNIndex, config, search::Knn, search_output_buffer::IdDistance},
    provider::DefaultContext,
};
use diskann_bftree::{
    BfTreeProvider,
    provider::{BFTREE_IO_THREADS, ParallelRerank, SavedParams, SeededParallelRerank},
    quant::QuantVectorProvider,
};
use diskann_providers::{
    model::graph::provider::async_::common::Quantized,
    storage::{FileStorageProvider, LoadWith},
};
use navis_diskann3_bench::{CSV_HEADER, FloatBinReader, parse_metric, read_bin_header};

type Provider = BfTreeProvider<f32, QuantVectorProvider>;
type Index = DiskANNIndex<Provider>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InsertPipeline {
    /// Odin/NAVIS parity path: exact top-L rerank and exact forward prune.
    Matched,
    /// Native DiskANN3 insertion with only the graph-search beam overridden.
    NativeBeamOnly,
}

impl InsertPipeline {
    fn label(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::NativeBeamOnly => "native_beam_only",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "NAVIS-compatible streaming benchmark for DiskANN3's SSD Bf-tree provider")]
struct Args {
    #[arg(long)]
    dataset: String,
    #[arg(long)]
    index_prefix: PathBuf,
    #[arg(long)]
    storage_dir: PathBuf,
    #[arg(long)]
    query: PathBuf,
    #[arg(long)]
    insert_data: PathBuf,
    #[arg(long)]
    groundtruth: PathBuf,
    #[arg(long)]
    csv: PathBuf,
    #[arg(long)]
    base_n: usize,
    #[arg(long)]
    batch_size: usize,
    #[arg(long)]
    num_batches: usize,
    #[arg(long, default_value_t = 22)]
    search_threads: usize,
    #[arg(long, default_value_t = 10)]
    insert_threads: usize,
    #[arg(long, default_value_t = 4.0)]
    cache_gb: f64,
    #[arg(long, default_value_t = 16.0)]
    quant_cache_gb: f64,
    #[arg(long = "K", default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 40)]
    search_l: usize,
    #[arg(long, default_value_t = 100)]
    insert_l: usize,
    #[arg(long, default_value_t = 4)]
    search_beam_width: usize,
    #[arg(long, default_value_t = 4)]
    insert_beam_width: usize,
    #[arg(long, value_enum, default_value_t = InsertPipeline::Matched)]
    insert_pipeline: InsertPipeline,
    #[arg(long, default_value_t = 96)]
    pruned_degree: usize,
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,
    #[arg(long, default_value_t = 1.0)]
    backedge_ratio: f32,
    #[arg(long, default_value_t = 5.0)]
    tick_seconds: f64,
    #[arg(long)]
    query_limit: Option<usize>,
}

#[derive(Debug, Clone)]
struct SearchMetrics {
    qps: f64,
    mean_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    p999_ms: f64,
    recall: f64,
    mean_cmps: f64,
    mean_hops: f64,
}

#[derive(Debug, Clone, Copy)]
struct InsertMeasurement {
    latency_us: u64,
    candidates: usize,
    out_degree: usize,
    reused_full_vectors: usize,
}

fn parallel_map<T, F>(threads: usize, len: usize, f: &F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync,
{
    let next = AtomicUsize::new(0);
    let mut indexed = thread::scope(|scope| -> Result<Vec<(usize, T)>> {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let index = next.fetch_add(1, AtomicOrdering::Relaxed);
                        if index >= len {
                            break;
                        }
                        local.push((index, f(index)?));
                    }
                    Ok::<_, anyhow::Error>(local)
                })
            })
            .collect();

        let mut output = Vec::with_capacity(len);
        for handle in handles {
            let mut local = handle
                .join()
                .map_err(|_| anyhow::anyhow!("parallel worker panicked"))??;
            output.append(&mut local);
        }
        Ok(output)
    })?;
    indexed.sort_unstable_by_key(|(index, _)| *index);
    Ok(indexed.into_iter().map(|(_, value)| value).collect())
}

#[derive(Debug, Clone, Copy, Default)]
struct Usage {
    rss_current_kib: u64,
    rss_peak_kib: u64,
    storage_current_bytes: u64,
    storage_allocated_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct UsagePeaks {
    rss_kib: u64,
    storage_bytes: u64,
    storage_allocated_bytes: u64,
}

struct UsageMonitor {
    stop: Arc<AtomicBool>,
    peaks: Arc<Mutex<UsagePeaks>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl UsageMonitor {
    fn start(storage_dir: PathBuf) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peaks = Arc::new(Mutex::new(UsagePeaks::default()));
        let stop_worker = stop.clone();
        let peaks_worker = peaks.clone();
        let handle = thread::spawn(move || {
            while !stop_worker.load(AtomicOrdering::Relaxed) {
                if let Ok(usage) = read_usage(&storage_dir) {
                    update_peaks(&peaks_worker, usage);
                }
                thread::sleep(Duration::from_millis(250));
            }
            if let Ok(usage) = read_usage(&storage_dir) {
                update_peaks(&peaks_worker, usage);
            }
        });
        Self {
            stop,
            peaks,
            thread: Some(handle),
        }
    }

    fn snapshot(&self, storage_dir: &Path) -> Result<(Usage, UsagePeaks)> {
        let usage = read_usage(storage_dir)?;
        update_peaks(&self.peaks, usage);
        let peaks = *self.peaks.lock().expect("usage peak mutex poisoned");
        Ok((usage, peaks))
    }
}

impl Drop for UsageMonitor {
    fn drop(&mut self) {
        self.stop.store(true, AtomicOrdering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

struct CsvOutput {
    writer: BufWriter<File>,
}

struct Row<'a> {
    batch_idx: i64,
    batch_size: usize,
    tick_idx: i64,
    elapsed_s: f64,
    inserted: usize,
    search: Option<&'a SearchMetrics>,
    notes: String,
}

impl CsvOutput {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("create CSV {}", path.display()))?,
        );
        writeln!(writer, "{CSV_HEADER}")?;
        Ok(Self { writer })
    }

    fn write(&mut self, args: &Args, usage: Usage, row: Row<'_>) -> Result<()> {
        ensure!(
            !row.notes.contains(','),
            "CSV notes must not contain commas: {}",
            row.notes
        );
        let s = row.search;
        writeln!(
            self.writer,
            "ipdiskann,{},{},{},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},0,{:.3},0,{},{},{},{:.3},{},{},{},{}",
            args.dataset,
            args.base_n,
            row.batch_idx,
            row.batch_size,
            row.tick_idx,
            row.elapsed_s,
            row.inserted,
            s.map_or(0.0, |m| m.qps),
            s.map_or(0.0, |m| m.mean_ms),
            s.map_or(0.0, |m| m.p50_ms),
            s.map_or(0.0, |m| m.p90_ms),
            s.map_or(0.0, |m| m.p95_ms),
            s.map_or(0.0, |m| m.p99_ms),
            s.map_or(0.0, |m| m.p999_ms),
            s.map_or(0.0, |m| m.recall),
            usage.rss_current_kib as f64 / 1024.0,
            usage.storage_current_bytes,
            args.search_threads,
            args.insert_threads,
            args.cache_gb,
            args.search_l,
            args.insert_l,
            args.k,
            row.notes
        )?;
        self.writer.flush()?;
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let mut csv = CsvOutput::create(&args.csv)?;
    let monitor = UsageMonitor::start(args.storage_dir.clone());

    let query_header = read_bin_header(&args.query)?;
    let mut query_reader = FloatBinReader::open(&args.query)?;
    let query_count = args
        .query_limit
        .unwrap_or(query_header.rows)
        .min(query_header.rows);
    let queries = Arc::new(query_reader.read_rows(query_count)?);

    let insert_header = read_bin_header(&args.insert_data)?;
    ensure!(
        insert_header.dim == query_header.dim,
        "query/insert dimension mismatch"
    );
    let total_inserts = args
        .batch_size
        .checked_mul(args.num_batches)
        .context("insert count overflow")?;
    ensure!(
        args.base_n + total_inserts <= insert_header.rows,
        "insert file has {} rows but run needs rows through {}",
        insert_header.rows,
        args.base_n + total_inserts
    );
    let mut insert_reader = FloatBinReader::open(&args.insert_data)?;
    insert_reader.seek_row(args.base_n)?;

    let prefix = args
        .index_prefix
        .to_str()
        .context("index prefix is not valid UTF-8")?
        .to_owned();
    let saved: SavedParams =
        serde_json::from_reader(BufReader::new(File::open(format!("{prefix}_params.json"))?))?;
    ensure!(
        saved.dim == query_header.dim,
        "index/query dimension mismatch"
    );
    ensure!(
        !saved.is_memory,
        "benchmark requires an SSD-backed Bf-tree snapshot"
    );
    ensure!(
        saved.max_points >= args.base_n + total_inserts,
        "index capacity {} is smaller than final point count {}",
        saved.max_points,
        args.base_n + total_inserts
    );
    ensure!(
        saved.max_degree as usize >= args.pruned_degree,
        "physical max degree is smaller than pruned degree"
    );
    let quant_params = saved
        .quant_params
        .as_ref()
        .context("benchmark requires a spherical-quantized snapshot")?;
    let configured_cache_gb =
        (saved.params_vector.bytes + saved.params_neighbor.bytes) as f64 / (1u64 << 30) as f64;
    ensure!(
        (configured_cache_gb - args.cache_gb).abs() < 0.001,
        "--cache-gb={} does not match snapshot Bf-tree caches ({configured_cache_gb:.3} GiB)",
        args.cache_gb
    );
    let configured_quant_cache_gb = quant_params.params_quant.bytes as f64 / (1u64 << 30) as f64;
    ensure!(
        (configured_quant_cache_gb - args.quant_cache_gb).abs() < 0.001,
        "--quant-cache-gb={} does not match snapshot quantized-code cache ({configured_quant_cache_gb:.3} GiB)",
        args.quant_cache_gb
    );
    if let Some(graph_params) = &saved.graph_params {
        ensure!(
            graph_params.l_build == args.insert_l,
            "--insert-l does not match snapshot graph parameters"
        );
        ensure!(
            (graph_params.alpha - args.alpha).abs() < 1e-6,
            "--alpha does not match snapshot graph parameters"
        );
        ensure!(
            (graph_params.backedge_ratio - args.backedge_ratio).abs() < 1e-6,
            "--backedge-ratio does not match snapshot graph parameters"
        );
    }
    let metric = parse_metric(&saved.metric)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads((args.search_threads + args.insert_threads).max(1))
        .enable_all()
        .build()?;
    let provider = runtime
        .block_on(Provider::load_with(&FileStorageProvider, &prefix))
        .map_err(|e| anyhow::anyhow!("load DiskANN3 Bf-tree snapshot: {e:?}"))?;
    let mut builder = config::Builder::new(
        args.pruned_degree,
        config::MaxDegree::new(saved.max_degree as usize),
        args.insert_l,
        metric.into(),
    );
    builder
        .alpha(args.alpha)
        .backedge_ratio(args.backedge_ratio);
    if args.insert_pipeline == InsertPipeline::Matched {
        // Odin and NAVIS fill the remaining R-P positions from the sorted
        // explored set. Apply the same policy to reverse-edge overflow prunes;
        // the parity insertion API forces it independently for forward prune.
        builder.saturate_after_prune(true);
    }
    let graph_config = builder.build()?;
    let thread_hint = NonZeroUsize::new(args.search_threads + args.insert_threads);
    let index = Arc::new(DiskANNIndex::new(graph_config, provider, thread_hint));

    let mut last_post: Option<SearchMetrics> = None;
    let inserted_done = Arc::new(AtomicUsize::new(0));
    let insert_candidates_sum = Arc::new(AtomicU64::new(0));
    let insert_out_degree_sum = Arc::new(AtomicU64::new(0));
    let mut total_insert_wall_s = 0.0f64;
    // Snapshot recovery is setup, not update work. Start the shared benchmark
    // clock only after the mutable index is fully loaded so elapsed time and
    // fair throughput match the other baseline harnesses.
    let workload_started = Instant::now();

    for batch_idx in 0..args.num_batches {
        let batch_start_offset = batch_idx * args.batch_size;
        let vectors = Arc::new(insert_reader.read_rows(args.batch_size)?);
        let gt_before = Arc::new(read_groundtruth(
            &args
                .groundtruth
                .join(format!("gt_{batch_start_offset}.bin")),
            query_count,
            args.k,
        )?);
        let index_insert = index.clone();
        let inserted_counter = inserted_done.clone();
        let candidates_counter = insert_candidates_sum.clone();
        let out_degree_counter = insert_out_degree_sum.clone();
        let dim = insert_header.dim;
        let base_n = args.base_n;
        let batch_size = args.batch_size;
        let insert_threads = args.insert_threads;
        let insert_beam_width = args.insert_beam_width;
        let insert_pipeline = args.insert_pipeline;
        let expected_insert_candidates = args.insert_l;
        let expected_out_degree = args.pruned_degree;
        let (insert_tx, insert_rx) = mpsc::sync_channel(1);
        let insert_job = thread::spawn(move || {
            let result = parallel_map(insert_threads, batch_size, &|local| {
                let id = u32::try_from(base_n + batch_start_offset + local)
                    .context("insert id exceeds u32")?;
                let row = &vectors[local * dim..(local + 1) * dim];
                let started = Instant::now();
                let mut measurement = match insert_pipeline {
                    InsertPipeline::Matched => {
                        let exact = SeededParallelRerank::<f32, u32>::default();
                        let stats = futures_executor::block_on(index_insert.insert_with_rerank(
                            &Quantized,
                            &exact,
                            &exact,
                            &Quantized,
                            &DefaultContext,
                            &id,
                            row,
                            insert_beam_width,
                        ))
                        .map_err(|e| anyhow::anyhow!("insert id {id}: {e:?}"))?;
                        ensure!(
                            stats.candidates == expected_insert_candidates,
                            "insert id {id}: reranked {} candidates, expected top-L size {}",
                            stats.candidates,
                            expected_insert_candidates,
                        );
                        ensure!(
                            stats.out_degree == expected_out_degree,
                            "insert id {id}: installed {} outgoing edges, expected R={}",
                            stats.out_degree,
                            expected_out_degree,
                        );
                        ensure!(
                            exact.reused_vectors() == stats.candidates,
                            "insert id {id}: reused {} of {} reranked full vectors during forward prune",
                            exact.reused_vectors(),
                            stats.candidates,
                        );
                        InsertMeasurement {
                            latency_us: 0,
                            candidates: stats.candidates,
                            out_degree: stats.out_degree,
                            reused_full_vectors: exact.reused_vectors(),
                        }
                    }
                    InsertPipeline::NativeBeamOnly => {
                        futures_executor::block_on(index_insert.insert_with_beam(
                            &Quantized,
                            &DefaultContext,
                            &id,
                            row,
                            insert_beam_width,
                        ))
                        .map_err(|e| anyhow::anyhow!("insert id {id}: {e:?}"))?;
                        InsertMeasurement {
                            latency_us: 0,
                            candidates: 0,
                            out_degree: 0,
                            reused_full_vectors: 0,
                        }
                    }
                };
                candidates_counter
                    .fetch_add(measurement.candidates as u64, AtomicOrdering::Relaxed);
                out_degree_counter
                    .fetch_add(measurement.out_degree as u64, AtomicOrdering::Relaxed);
                inserted_counter.fetch_add(1, AtomicOrdering::Relaxed);
                measurement.latency_us =
                    started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                Ok(measurement)
            });
            let _ = insert_tx.send(result);
        });

        let batch_started = Instant::now();
        let mut tick_idx = 0i64;
        let tick_duration = Duration::from_secs_f64(args.tick_seconds);
        let insert_measurements = loop {
            match insert_rx.recv_timeout(tick_duration) {
                Ok(result) => break result?,
                Err(RecvTimeoutError::Timeout) => {
                    let current = inserted_done.load(AtomicOrdering::Relaxed);
                    let completed = current.max(1) as f64;
                    let insert_progress = format!(
                        "insert_pipeline={};mean_insert_candidates_so_far={:.3};mean_insert_out_degree_so_far={:.3};search_beam_width={};insert_beam_width={}",
                        args.insert_pipeline.label(),
                        insert_candidates_sum.load(AtomicOrdering::Relaxed) as f64 / completed,
                        insert_out_degree_sum.load(AtomicOrdering::Relaxed) as f64 / completed,
                        args.search_beam_width,
                        args.insert_beam_width,
                    );
                    let metrics = run_search(
                        &index,
                        args.search_threads,
                        &queries,
                        query_header.dim,
                        &gt_before,
                        query_count,
                        args.k,
                        args.search_l,
                        args.search_beam_width,
                    )?;
                    let (usage, peaks) = monitor.snapshot(&args.storage_dir)?;
                    let notes = usage_notes(
                        "concurrent",
                        usage,
                        peaks,
                        Some(&metrics),
                        Some(batch_start_offset),
                        Some(&insert_progress),
                    );
                    csv.write(
                        &args,
                        usage,
                        Row {
                            batch_idx: batch_idx as i64,
                            batch_size: args.batch_size,
                            tick_idx,
                            elapsed_s: workload_started.elapsed().as_secs_f64(),
                            inserted: current,
                            search: Some(&metrics),
                            notes,
                        },
                    )?;
                    tick_idx += 1;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = insert_job.join();
                    bail!("insert worker exited without returning a result");
                }
            }
        };
        insert_job
            .join()
            .map_err(|_| anyhow::anyhow!("insert worker panicked"))?;
        let insert_wall_s = batch_started.elapsed().as_secs_f64();
        total_insert_wall_s += insert_wall_s;
        let mut insert_latencies: Vec<u64> = insert_measurements
            .iter()
            .map(|sample| sample.latency_us)
            .collect();
        insert_latencies.sort_unstable();
        let mean_insert_candidates = insert_measurements
            .iter()
            .map(|sample| sample.candidates as f64)
            .sum::<f64>()
            / insert_measurements.len().max(1) as f64;
        let mean_insert_out_degree = insert_measurements
            .iter()
            .map(|sample| sample.out_degree as f64)
            .sum::<f64>()
            / insert_measurements.len().max(1) as f64;
        let mean_reused_full_vectors = insert_measurements
            .iter()
            .map(|sample| sample.reused_full_vectors as f64)
            .sum::<f64>()
            / insert_measurements.len().max(1) as f64;
        let inserted = inserted_done.load(AtomicOrdering::Relaxed);
        let (usage, peaks) = monitor.snapshot(&args.storage_dir)?;
        let insert_notes = format!(
            "{};insert_pipeline={};insert_throughput_vps={:.6};insert_10p_us={};insert_50p_us={};insert_90p_us={};insert_99p_us={};insert_99_9p_us={};insert_wall_s={:.6};pure_insert_throughput_vps={:.6};mean_insert_candidates={mean_insert_candidates:.3};mean_insert_out_degree={mean_insert_out_degree:.3};mean_insert_reused_full_vectors={mean_reused_full_vectors:.3};insert_beam_width={};merge_wait_s=0;merge_policy=none",
            usage_notes("insert", usage, peaks, None, None, None),
            args.insert_pipeline.label(),
            args.batch_size as f64 / insert_wall_s.max(1e-9),
            percentile_u64(&insert_latencies, 0.10),
            percentile_u64(&insert_latencies, 0.50),
            percentile_u64(&insert_latencies, 0.90),
            percentile_u64(&insert_latencies, 0.99),
            percentile_u64(&insert_latencies, 0.999),
            insert_wall_s,
            args.batch_size as f64 / insert_wall_s.max(1e-9),
            args.insert_beam_width,
        );
        csv.write(
            &args,
            usage,
            Row {
                batch_idx: batch_idx as i64,
                batch_size: args.batch_size,
                tick_idx: -1,
                elapsed_s: workload_started.elapsed().as_secs_f64(),
                inserted,
                search: None,
                notes: insert_notes,
            },
        )?;

        let batch_end_offset = (batch_idx + 1) * args.batch_size;
        let gt_after = read_groundtruth(
            &args.groundtruth.join(format!("gt_{batch_end_offset}.bin")),
            query_count,
            args.k,
        )?;
        let post = run_search(
            &index,
            args.search_threads,
            &queries,
            query_header.dim,
            &gt_after,
            query_count,
            args.k,
            args.search_l,
            args.search_beam_width,
        )?;
        let (usage, peaks) = monitor.snapshot(&args.storage_dir)?;
        let notes = usage_notes(
            "post_batch",
            usage,
            peaks,
            Some(&post),
            Some(batch_end_offset),
            Some("merge_policy=none;merge_wait_s=0"),
        );
        csv.write(
            &args,
            usage,
            Row {
                batch_idx: batch_idx as i64,
                batch_size: args.batch_size,
                tick_idx: -2,
                elapsed_s: workload_started.elapsed().as_secs_f64(),
                inserted,
                search: Some(&post),
                notes,
            },
        )?;
        last_post = Some(post);
    }

    let (usage, peaks) = monitor.snapshot(&args.storage_dir)?;
    // Match the shared C++ harness: include each concurrent insert/search
    // window, but exclude the post-batch pure-search pass. DiskANN3 has no
    // asynchronous merge drain to add.
    let fair_throughput = total_inserts as f64 / total_insert_wall_s.max(1e-9);
    let insert_pipeline_notes = match args.insert_pipeline {
        InsertPipeline::Matched => {
            "insert_exact_rerank=1;insert_rerank_prune_vector_reuse=1;insert_exact_forward_prune=1;insert_quant_forward_prune=0;insert_quant_reverse_prune=1;insert_candidate_pool=top_l;prune_saturate_to_r=1"
        }
        InsertPipeline::NativeBeamOnly => {
            "insert_exact_rerank=0;insert_rerank_prune_vector_reuse=0;insert_exact_forward_prune=0;insert_quant_forward_prune=1;insert_quant_reverse_prune=1;insert_candidate_pool=native_visited;prune_saturate_to_r=0"
        }
    };
    let summary_extra = format!(
        "provider=bftree_spherical_cache_only;quantized_traversal=1;ssd_full_vector_rerank=1;parallel_ssd_rerank=1;parallel_beam_reads=1;bftree_io_threads={};insert_pipeline={};{};search_beam_width={};insert_beam_width={};quant_cache_gib={:.3};ssd_backed=1;merge_policy=none;fair_throughput_vps={fair_throughput:.6};total_insert_time_s={total_insert_wall_s:.6};total_merge_drain_s=0;total_time_s={total_insert_wall_s:.6}",
        BFTREE_IO_THREADS,
        args.insert_pipeline.label(),
        insert_pipeline_notes,
        args.search_beam_width,
        args.insert_beam_width,
        args.quant_cache_gb,
    );
    let notes = usage_notes(
        "run_summary",
        usage,
        peaks,
        last_post.as_ref(),
        Some(args.batch_size * args.num_batches),
        Some(&summary_extra),
    );
    csv.write(
        &args,
        usage,
        Row {
            batch_idx: -1,
            batch_size: args.batch_size,
            tick_idx: -4,
            elapsed_s: workload_started.elapsed().as_secs_f64(),
            inserted: inserted_done.load(AtomicOrdering::Relaxed),
            search: last_post.as_ref(),
            notes,
        },
    )?;

    // BfTree's Drop implementation breadth-first scans every node before
    // deallocating its process-local state.  For an SSD benchmark shadow this
    // adds a full-index read after all metrics have been flushed (roughly
    // 302 GiB for FineWeb) without improving the disposable run artifact.
    // Let process teardown reclaim the index instead.
    std::mem::forget(index);
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        matches!(args.dataset.as_str(), "fineweb" | "msmarco" | "deep"),
        "dataset must be fineweb msmarco or deep"
    );
    ensure!(args.batch_size > 0, "batch size must be positive");
    ensure!(args.num_batches > 0, "number of batches must be positive");
    ensure!(args.search_threads > 0, "search threads must be positive");
    ensure!(args.insert_threads > 0, "insert threads must be positive");
    ensure!(
        args.search_beam_width > 0,
        "search beam width must be positive"
    );
    ensure!(
        args.insert_beam_width > 0,
        "insert beam width must be positive"
    );
    ensure!(args.k > 0, "K must be positive");
    ensure!(args.search_l >= args.k, "search L must be at least K");
    ensure!(args.insert_l > 0, "insert L must be positive");
    ensure!(args.tick_seconds > 0.0, "tick interval must be positive");
    ensure!(
        args.storage_dir.is_dir(),
        "storage directory does not exist"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_search(
    index: &Arc<Index>,
    search_threads: usize,
    queries: &[f32],
    dim: usize,
    groundtruth: &[u32],
    query_count: usize,
    k: usize,
    search_l: usize,
    search_beam_width: usize,
) -> Result<SearchMetrics> {
    let wall_started = Instant::now();
    let rows: Vec<(u64, Vec<u32>, u32, u32)> =
        parallel_map(search_threads, query_count, &|query_id| {
            let query = &queries[query_id * dim..(query_id + 1) * dim];
            let mut ids = vec![u32::MAX; k];
            let mut distances = vec![f32::INFINITY; k];
            let mut output = IdDistance::new(&mut ids, &mut distances);
            let started = Instant::now();
            let stats = futures_executor::block_on(index.search_with(
                Knn::new(search_l, Some(search_beam_width))?,
                &Quantized,
                ParallelRerank,
                &DefaultContext,
                query,
                &mut output,
            ))
            .map_err(|e| anyhow::anyhow!("search query {query_id}: {e:?}"))?;
            Ok((
                started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                ids,
                stats.cmps,
                stats.hops,
            ))
        })?;
    let wall_s = wall_started.elapsed().as_secs_f64();
    let mut latencies: Vec<u64> = rows.iter().map(|r| r.0).collect();
    latencies.sort_unstable();
    let total_us: u128 = latencies.iter().map(|&v| u128::from(v)).sum();
    let hits: usize = rows
        .iter()
        .enumerate()
        .map(|(query_id, row)| {
            let truth = &groundtruth[query_id * k..(query_id + 1) * k];
            row.1.iter().filter(|id| truth.contains(id)).count()
        })
        .sum();
    Ok(SearchMetrics {
        qps: query_count as f64 / wall_s.max(1e-9),
        mean_ms: total_us as f64 / query_count as f64 / 1000.0,
        p50_ms: percentile_u64(&latencies, 0.50) as f64 / 1000.0,
        p90_ms: percentile_u64(&latencies, 0.90) as f64 / 1000.0,
        p95_ms: percentile_u64(&latencies, 0.95) as f64 / 1000.0,
        p99_ms: percentile_u64(&latencies, 0.99) as f64 / 1000.0,
        p999_ms: percentile_u64(&latencies, 0.999) as f64 / 1000.0,
        recall: hits as f64 / (query_count * k) as f64,
        mean_cmps: rows.iter().map(|r| u64::from(r.2)).sum::<u64>() as f64 / query_count as f64,
        mean_hops: rows.iter().map(|r| u64::from(r.3)).sum::<u64>() as f64 / query_count as f64,
    })
}

fn read_groundtruth(path: &Path, query_count: usize, k: usize) -> Result<Vec<u32>> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open ground truth {}", path.display()))?,
    );
    let rows = read_u32(&mut reader)? as usize;
    let width = read_u32(&mut reader)? as usize;
    ensure!(
        rows >= query_count,
        "{} has too few queries",
        path.display()
    );
    ensure!(
        width >= k,
        "{} ground-truth width is below K",
        path.display()
    );
    let mut all = vec![0u32; rows * width];
    reader.read_exact(bytemuck::cast_slice_mut(&mut all))?;
    let mut selected = Vec::with_capacity(query_count * k);
    for row in all.chunks_exact(width).take(query_count) {
        selected.extend_from_slice(&row[..k]);
    }
    Ok(selected)
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn percentile_u64(sorted: &[u64], quantile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (quantile * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn read_usage(storage_dir: &Path) -> Result<Usage> {
    let status = fs::read_to_string("/proc/self/status")?;
    let mut rss_current_kib = 0;
    let mut rss_peak_kib = 0;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            rss_current_kib = parse_status_kib(value)?;
        } else if let Some(value) = line.strip_prefix("VmHWM:") {
            rss_peak_kib = parse_status_kib(value)?;
        }
    }
    let (storage_current_bytes, storage_allocated_bytes) = directory_usage(storage_dir)?;
    Ok(Usage {
        rss_current_kib,
        rss_peak_kib,
        storage_current_bytes,
        storage_allocated_bytes,
    })
}

fn parse_status_kib(value: &str) -> Result<u64> {
    value
        .split_whitespace()
        .next()
        .context("missing /proc status value")?
        .parse()
        .context("invalid /proc status value")
}

fn directory_usage(path: &Path) -> Result<(u64, u64)> {
    let mut logical = 0u64;
    let mut allocated = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current)
            .with_context(|| format!("scan storage directory {}", current.display()))?
        {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                stack.push(entry.path());
            } else if metadata.is_file() {
                logical = logical.saturating_add(metadata.len());
                allocated = allocated.saturating_add(metadata.blocks().saturating_mul(512));
            }
        }
    }
    Ok((logical, allocated))
}

fn update_peaks(peaks: &Mutex<UsagePeaks>, usage: Usage) {
    let mut peaks = peaks.lock().expect("usage peak mutex poisoned");
    peaks.rss_kib = peaks
        .rss_kib
        .max(usage.rss_peak_kib.max(usage.rss_current_kib));
    peaks.storage_bytes = peaks.storage_bytes.max(usage.storage_current_bytes);
    peaks.storage_allocated_bytes = peaks
        .storage_allocated_bytes
        .max(usage.storage_allocated_bytes);
}

fn usage_notes(
    phase: &str,
    usage: Usage,
    peaks: UsagePeaks,
    search: Option<&SearchMetrics>,
    gt_offset: Option<usize>,
    extra: Option<&str>,
) -> String {
    let mut notes = format!(
        "phase={phase};rss_current_mib={:.3};rss_peak_mib={:.3};storage_current_bytes={};storage_peak_bytes={};storage_allocated_current_bytes={};storage_allocated_peak_bytes={};mean_disk_ios_available=0;cache_hit_rate_available=0",
        usage.rss_current_kib as f64 / 1024.0,
        peaks.rss_kib as f64 / 1024.0,
        usage.storage_current_bytes,
        peaks.storage_bytes,
        usage.storage_allocated_bytes,
        peaks.storage_allocated_bytes,
    );
    if let Some(search) = search {
        notes.push_str(&format!(
            ";mean_distance_computations={:.3};mean_graph_hops={:.3}",
            search.mean_cmps, search.mean_hops
        ));
    }
    if let Some(offset) = gt_offset {
        notes.push_str(&format!(";gt_offset={offset}"));
    }
    if let Some(extra) = extra {
        notes.push(';');
        notes.push_str(extra);
    }
    notes
}
