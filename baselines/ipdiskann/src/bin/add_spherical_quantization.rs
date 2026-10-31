use std::{
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use diskann_bftree::{
    BfTreePaths, GraphParams, VectorDtype,
    provider::{BfTreeParams, QuantParams, SavedParams},
    quant::QuantVectorProvider,
};
use diskann_quantization::{
    algorithms::transforms::{TargetDim, TransformKind},
    alloc::{GlobalAllocator, Poly},
    spherical::{
        PreScale, SphericalQuantizer, SupportedMetric,
        iface::{self as spherical_iface, Quantizer},
    },
};
use diskann_utils::views::Matrix;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::Serialize;

use navis_diskann3_bench::{
    FloatBinReader, VamanaReader, checked_u32, read_bin_header, require_absent,
};

#[derive(Debug, Parser)]
#[command(about = "Attach memory-resident spherical codes to an existing DiskANN3 SSD snapshot")]
struct Args {
    #[arg(long)]
    source_prefix: PathBuf,
    #[arg(long)]
    output_prefix: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    base_n: usize,
    #[arg(long, default_value_t = 100_000)]
    train_points: usize,
    #[arg(long, default_value_t = 16)]
    quant_cache_gib: usize,
    #[arg(long, default_value_t = 1)]
    quant_bits: usize,
    #[arg(long, default_value_t = 16_384)]
    import_batch: usize,
    #[arg(long, default_value_t = 32)]
    threads: usize,
    #[arg(long, default_value_t = 100)]
    l_build: usize,
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,
    #[arg(long, default_value_t = 1.0)]
    backedge_ratio: f32,
    #[arg(long)]
    force: bool,
}

#[derive(Serialize)]
struct Manifest<'a> {
    format: &'static str,
    source_prefix: &'a Path,
    source_data: &'a Path,
    source_graph: &'a Path,
    base_points: usize,
    capacity: usize,
    dim: usize,
    max_degree: u32,
    medoid: usize,
    metric: &'a str,
    l_build: usize,
    alpha: f32,
    backedge_ratio: f32,
    quantizer: &'static str,
    quant_bits: usize,
    transform: &'static str,
    pre_scale: &'static str,
    train_points: usize,
    quantized_vector_bytes: usize,
    quant_cache_gib: usize,
    quant_record_bytes: usize,
    quant_leaf_page_bytes: usize,
    quant_cache_only: bool,
}

fn output_paths(prefix: &Path) -> Result<Vec<PathBuf>> {
    let prefix = prefix
        .to_str()
        .context("output prefix is not valid UTF-8")?;
    Ok([
        "_params.json",
        "_vectors.bftree",
        "_neighbors.bftree",
        "_quant.bftree",
        "_quant_data.bin",
        "_manifest.json",
    ]
    .into_iter()
    .map(|suffix| PathBuf::from(format!("{prefix}{suffix}")))
    .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.base_n > 0, "--base-n must be positive");
    ensure!(args.train_points > 0, "--train-points must be positive");
    ensure!(
        args.train_points <= args.base_n,
        "training set exceeds base"
    );
    ensure!(args.import_batch > 0, "--import-batch must be positive");
    ensure!(args.threads > 0, "--threads must be positive");
    ensure!(
        args.quant_cache_gib > 0,
        "--quant-cache-gib must be positive"
    );
    ensure!(
        matches!(args.quant_bits, 1 | 2 | 4),
        "--quant-bits must be one of 1 2 or 4"
    );

    let header = read_bin_header(&args.data)?;
    ensure!(args.base_n <= header.rows, "base exceeds source data rows");
    let graph = VamanaReader::open(&args.graph, args.base_n)?;
    let medoid = graph.header().medoid as usize;
    ensure!(
        medoid < args.base_n,
        "graph medoid is outside the base graph"
    );

    let source = args
        .source_prefix
        .to_str()
        .context("source prefix is not valid UTF-8")?
        .to_owned();
    let mut source_params: SavedParams = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(format!("{source}_params.json"))?,
    ))?;
    ensure!(
        source_params.quant_params.is_none(),
        "source snapshot is already quantized"
    );
    ensure!(
        source_params.dim == header.dim,
        "snapshot/data dim mismatch"
    );
    ensure!(
        source_params.max_points >= args.base_n,
        "snapshot capacity is smaller than the base"
    );
    for path in [
        BfTreePaths::vectors_bftree(&source),
        BfTreePaths::neighbors_bftree(&source),
    ] {
        ensure!(
            path.is_file(),
            "source snapshot file is missing: {}",
            path.display()
        );
    }

    let outputs = output_paths(&args.output_prefix)?;
    require_absent(&outputs, args.force)?;
    if let Some(parent) = args.output_prefix.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let started = Instant::now();
    let mut training_reader = FloatBinReader::open(&args.data)?;
    let training = training_reader
        .read_rows(args.train_points)?
        .into_boxed_slice();
    let training = Matrix::try_from(training, args.train_points, header.dim)
        .map_err(|e| anyhow::anyhow!("construct training matrix: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let quantizer = SphericalQuantizer::train(
        training.as_view(),
        TransformKind::DoubleHadamard {
            target_dim: TargetDim::Same,
        },
        SupportedMetric::SquaredL2,
        PreScale::ReciprocalMeanNorm,
        &mut rng,
        GlobalAllocator,
    )?;
    let quantizer: Poly<dyn Quantizer> = match args.quant_bits {
        1 => {
            let implementation = spherical_iface::Impl::<1>::new(quantizer)?;
            diskann_quantization::poly!(Quantizer, implementation, GlobalAllocator)?
        }
        2 => {
            let implementation = spherical_iface::Impl::<2>::new(quantizer)?;
            diskann_quantization::poly!(Quantizer, implementation, GlobalAllocator)?
        }
        4 => {
            let implementation = spherical_iface::Impl::<4>::new(quantizer)?;
            diskann_quantization::poly!(Quantizer, implementation, GlobalAllocator)?
        }
        _ => unreachable!("quantization bit width was validated above"),
    };
    let quantized_vector_bytes = quantizer.bytes();
    let serialized_quantizer = quantizer
        .serialize(GlobalAllocator)
        .map_err(|e| anyhow::anyhow!("serialize quantizer: {e}"))?
        .to_vec();

    let cache_bytes = args
        .quant_cache_gib
        .checked_mul(1usize << 30)
        .context("quant cache byte size overflow")?;
    ensure!(
        cache_bytes.is_power_of_two(),
        "quant cache size must be a power of two"
    );
    let output = args
        .output_prefix
        .to_str()
        .context("output prefix is not valid UTF-8")?
        .to_owned();
    let live_quant = PathBuf::from(format!(
        "{output}.building.{}_quant.live",
        std::process::id()
    ));
    require_absent(std::slice::from_ref(&live_quant), true)?;
    let quant_record_bytes = std::mem::size_of::<usize>() + quantized_vector_bytes;
    let quant_leaf_page_bytes = 4096;
    let mut quant_config = bf_tree::Config::new(&live_quant, cache_bytes);
    quant_config
        .cache_only(true)
        .cb_max_record_size(quant_record_bytes)
        .leaf_page_size(quant_leaf_page_bytes)
        .use_snapshot(true);
    quant_config
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid quant BfTree config: {e:?}"))?;

    eprintln!("[add_spherical_quantization] creating independent in-memory code tree");
    let provider = Arc::new(
        QuantVectorProvider::new_with_config(quantizer, quant_config)
            .map_err(|e| anyhow::anyhow!("create quantized store: {e:?}"))?,
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    // Keep disk reads one batch ahead of the CPU encoder. The bounded channel
    // prevents a large RAM spike while all requested Rayon workers quantize the
    // current batch.
    let producer_path = args.data.clone();
    let producer_base_n = args.base_n;
    let producer_batch = args.import_batch;
    let (batch_tx, batch_rx) = mpsc::sync_channel::<(usize, Vec<f32>)>(2);
    let producer = thread::spawn(move || -> Result<()> {
        let mut data = FloatBinReader::open(&producer_path)?;
        let mut offset = 0usize;
        while offset < producer_base_n {
            let count = producer_batch.min(producer_base_n - offset);
            let vectors = data.read_rows(count)?;
            if batch_tx.send((offset, vectors)).is_err() {
                break;
            }
            offset += count;
        }
        Ok(())
    });
    let encode_result = (|| -> Result<()> {
        for (imported, vectors) in batch_rx {
            let count = vectors.len() / header.dim;
            ensure!(count * header.dim == vectors.len(), "partial vector batch");
            let provider_ref = &provider;
            pool.install(|| {
                (0..count)
                    .into_par_iter()
                    .try_for_each(|local| -> Result<()> {
                        let id = checked_u32(imported + local, "point id")?;
                        let vector = &vectors[local * header.dim..(local + 1) * header.dim];
                        provider_ref
                            .set_vector_sync(id as usize, vector)
                            .map_err(|e| anyhow::anyhow!("quantize vector {id}: {e:?}"))
                    })
            })?;
            let completed = imported + count;
            if completed == args.base_n || imported / 1_000_000 != completed / 1_000_000 {
                eprintln!(
                    "[add_spherical_quantization] encoded {completed}/{} ({:.1} K vectors/s)",
                    args.base_n,
                    completed as f64 / started.elapsed().as_secs_f64().max(1e-9) / 1000.0,
                );
            }
        }
        Ok(())
    })();
    let producer_result = producer
        .join()
        .map_err(|_| anyhow::anyhow!("vector reader thread panicked"))?;
    encode_result?;
    producer_result?;

    let mut medoid_reader = FloatBinReader::open(&args.data)?;
    medoid_reader.seek_row(medoid)?;
    let medoid_vector = medoid_reader.read_rows(1)?;
    provider
        .set_vector_sync(source_params.max_points, &medoid_vector)
        .map_err(|e| anyhow::anyhow!("quantize start point: {e:?}"))?;

    eprintln!("[add_spherical_quantization] writing code snapshot and linking SSD trees {output}");
    provider.save_snapshot(&BfTreePaths::quant_bftree(&output));
    std::fs::write(BfTreePaths::quant_data_bin(&output), serialized_quantizer)?;
    for (source_path, output_path) in [
        (
            BfTreePaths::vectors_bftree(&source),
            BfTreePaths::vectors_bftree(&output),
        ),
        (
            BfTreePaths::neighbors_bftree(&source),
            BfTreePaths::neighbors_bftree(&output),
        ),
    ] {
        std::fs::hard_link(&source_path, &output_path).with_context(|| {
            format!(
                "hard-link immutable SSD tree {} -> {}",
                source_path.display(),
                output_path.display()
            )
        })?;
    }
    source_params.prefix = output.clone();
    source_params.quant_params = Some(QuantParams {
        params_quant: BfTreeParams {
            bytes: cache_bytes,
            max_record_size: quant_record_bytes,
            leaf_page_size: quant_leaf_page_bytes,
        },
    });
    source_params.graph_params = Some(GraphParams {
        l_build: args.l_build,
        alpha: args.alpha,
        backedge_ratio: args.backedge_ratio,
        vector_dtype: VectorDtype::F32,
    });
    std::fs::write(
        BfTreePaths::params_json(&output),
        serde_json::to_vec(&source_params)?,
    )?;
    if live_quant.exists() {
        std::fs::remove_file(&live_quant)
            .with_context(|| format!("remove temporary quant tree {}", live_quant.display()))?;
    }

    let manifest_path = PathBuf::from(format!("{output}_manifest.json"));
    let manifest = Manifest {
        format: "diskann3-bftree-spherical-cache-only-v1",
        source_prefix: &args.source_prefix,
        source_data: &args.data,
        source_graph: &args.graph,
        base_points: args.base_n,
        capacity: source_params.max_points,
        dim: header.dim,
        max_degree: source_params.max_degree,
        medoid,
        metric: &source_params.metric,
        l_build: args.l_build,
        alpha: args.alpha,
        backedge_ratio: args.backedge_ratio,
        quantizer: "spherical",
        quant_bits: args.quant_bits,
        transform: "double_hadamard_same",
        pre_scale: "reciprocal_mean_norm",
        train_points: args.train_points,
        quantized_vector_bytes,
        quant_cache_gib: args.quant_cache_gib,
        quant_record_bytes,
        quant_leaf_page_bytes,
        quant_cache_only: true,
    };
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    std::mem::forget(provider);
    eprintln!(
        "[add_spherical_quantization] complete in {:.1}s; bits={}; code={} bytes; cache={} GiB",
        started.elapsed().as_secs_f64(),
        args.quant_bits,
        quantized_vector_bytes,
        args.quant_cache_gib,
    );
    Ok(())
}
