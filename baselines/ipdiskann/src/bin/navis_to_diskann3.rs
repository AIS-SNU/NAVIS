use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use diskann::provider::{DefaultContext, Guard, SetElement};
use diskann_bftree::{BfTreeProvider, BfTreeProviderParameters, GraphParams, NoStore, VectorDtype};
use diskann_providers::storage::{FileStorageProvider, SaveWith};
use diskann_utils::views::Matrix;
use rayon::prelude::*;
use serde::Serialize;

use navis_diskann3_bench::{
    FloatBinReader, VamanaReader, checked_u32, neighbor_store_config, parse_metric, require_absent,
    snapshot_paths, vector_store_config,
};

#[derive(Debug, Parser)]
#[command(
    about = "Import a NAVIS/Vamana memory graph into DiskANN3's mutable SSD Bf-tree provider"
)]
struct Args {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    data: PathBuf,
    /// Number of leading data rows represented by the NAVIS graph.
    #[arg(long)]
    base_n: Option<usize>,
    #[arg(long)]
    output_prefix: PathBuf,
    #[arg(long)]
    capacity: usize,
    #[arg(long, default_value = "l2")]
    metric: String,
    #[arg(long, default_value_t = 96)]
    pruned_degree: usize,
    #[arg(long, default_value_t = 100)]
    l_build: usize,
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,
    #[arg(long, default_value_t = 1.0)]
    backedge_ratio: f32,
    #[arg(long, default_value_t = 2048)]
    vector_cache_mib: usize,
    #[arg(long, default_value_t = 2048)]
    neighbor_cache_mib: usize,
    #[arg(long, default_value_t = 16_384)]
    import_batch: usize,
    #[arg(long, default_value_t = 32)]
    threads: usize,
    #[arg(long)]
    force: bool,
}

#[derive(Serialize)]
struct Manifest<'a> {
    format: &'static str,
    source_graph: &'a Path,
    source_data: &'a Path,
    base_points: usize,
    capacity: usize,
    dim: usize,
    max_degree: u32,
    medoid: u32,
    metric: &'a str,
    pruned_degree: usize,
    l_build: usize,
    alpha: f32,
    backedge_ratio: f32,
    vector_cache_mib: usize,
    neighbor_cache_mib: usize,
    vector_record_bytes: usize,
    vector_leaf_page_bytes: usize,
    neighbor_record_bytes: usize,
    neighbor_leaf_page_bytes: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.import_batch > 0, "--import-batch must be positive");
    ensure!(args.threads > 0, "--threads must be positive");
    let metric = parse_metric(&args.metric)?;
    let mut data = FloatBinReader::open(&args.data)?;
    let data_header = data.header();
    let base_points = args.base_n.unwrap_or(data_header.rows);
    ensure!(
        base_points <= data_header.rows,
        "base point count {} exceeds data rows {}",
        base_points,
        data_header.rows
    );
    ensure!(
        args.capacity >= base_points,
        "capacity {} is smaller than base point count {}",
        args.capacity,
        base_points
    );
    checked_u32(args.capacity, "capacity")?;

    let mut graph = VamanaReader::open(&args.graph, base_points)?;
    let graph_header = graph.header();
    ensure!(
        args.pruned_degree <= graph_header.max_degree as usize,
        "pruned degree exceeds physical graph degree"
    );

    let mut outputs = snapshot_paths(&args.output_prefix)?;
    let manifest_path = PathBuf::from(format!("{}_manifest.json", args.output_prefix.display()));
    outputs.push(manifest_path.clone());
    require_absent(&outputs, args.force)?;
    if let Some(parent) = args.output_prefix.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let output = args
        .output_prefix
        .to_str()
        .context("output prefix is not valid UTF-8")?;
    let build_prefix = format!("{output}.building.{}", std::process::id());
    let live_vector = PathBuf::from(format!("{build_prefix}_vectors.live"));
    let live_neighbors = PathBuf::from(format!("{build_prefix}_neighbors.live"));
    require_absent(&[live_vector.clone(), live_neighbors.clone()], true)?;

    let vector_config =
        vector_store_config(&live_vector, args.vector_cache_mib << 20, data_header.dim)?;
    let neighbor_config = neighbor_store_config(
        &live_neighbors,
        args.neighbor_cache_mib << 20,
        graph_header.max_degree,
    )?;
    let vector_record_bytes = vector_config.get_cb_max_record_size();
    let vector_leaf_page_bytes = vector_config.get_leaf_page_size();
    let neighbor_record_bytes = neighbor_config.get_cb_max_record_size();
    let neighbor_leaf_page_bytes = neighbor_config.get_leaf_page_size();

    let mut medoid_reader = FloatBinReader::open(&args.data)?;
    medoid_reader.seek_row(graph_header.medoid as usize)?;
    let start_data = medoid_reader.read_rows(1)?.into_boxed_slice();
    let start_points = Matrix::try_from(start_data, 1, data_header.dim)
        .map_err(|e| anyhow::anyhow!("construct start-point matrix: {e}"))?;

    let params = BfTreeProviderParameters {
        max_points: args.capacity,
        num_start_points: NonZeroUsize::new(1).expect("one is nonzero"),
        dim: data_header.dim,
        metric,
        max_degree: graph_header.max_degree,
        vector_provider_config: vector_config,
        quant_vector_provider_config: Default::default(),
        neighbor_list_provider_config: neighbor_config,
        graph_params: Some(GraphParams {
            l_build: args.l_build,
            alpha: args.alpha,
            backedge_ratio: args.backedge_ratio,
            vector_dtype: VectorDtype::F32,
        }),
        use_snapshot: true,
    };
    let provider = Arc::new(
        BfTreeProvider::<f32, NoStore>::new(params, start_points.as_view(), NoStore)
            .map_err(|e| anyhow::anyhow!("create Bf-tree provider: {e:?}"))?,
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let started = Instant::now();
    let mut imported = 0usize;
    let mut medoid_neighbors: Option<Vec<u32>> = None;
    while imported < base_points {
        let count = args.import_batch.min(base_points - imported);
        let vectors = data.read_rows(count)?;
        let adjacency = graph.read_batch(count)?;
        if (imported..imported + count).contains(&(graph_header.medoid as usize)) {
            medoid_neighbors = Some(adjacency[graph_header.medoid as usize - imported].clone());
        }
        let provider_ref = &provider;
        pool.install(|| {
            (0..count).into_par_iter().try_for_each_init(
                || vec![0u32; graph_header.max_degree as usize + 1],
                |neighbor_buf, local| -> Result<()> {
                    let id_usize = imported + local;
                    let id = checked_u32(id_usize, "point id")?;
                    let row = &vectors[local * data_header.dim..(local + 1) * data_header.dim];
                    let guard = futures_executor::block_on(provider_ref.set_element(
                        &DefaultContext,
                        &id,
                        row,
                    ))
                    .map_err(|e| anyhow::anyhow!("set vector {id}: {e:?}"))?;
                    futures_executor::block_on(guard.complete());
                    provider_ref
                        .neighbors()
                        .set_neighbors(id, &adjacency[local], neighbor_buf)
                        .map_err(|e| anyhow::anyhow!("set neighbors for {id}: {e:?}"))?;
                    Ok(())
                },
            )
        })?;
        imported += count;
        eprintln!(
            "[navis_to_diskann3] imported {imported}/{} ({:.1} K vectors/s)",
            base_points,
            imported as f64 / started.elapsed().as_secs_f64().max(1e-9) / 1000.0
        );
    }
    graph.finish()?;

    let frozen_id = checked_u32(args.capacity, "frozen start id")?;
    let mut start_neighbors = Vec::with_capacity(graph_header.max_degree as usize);
    start_neighbors.push(graph_header.medoid);
    for id in medoid_neighbors.context("medoid adjacency was not observed")? {
        if id != graph_header.medoid && start_neighbors.len() < graph_header.max_degree as usize {
            start_neighbors.push(id);
        }
    }
    let mut start_buf = vec![0u32; graph_header.max_degree as usize + 1];
    provider
        .neighbors()
        .set_neighbors(frozen_id, &start_neighbors, &mut start_buf)
        .map_err(|e| anyhow::anyhow!("set frozen start neighbors: {e:?}"))?;

    eprintln!("[navis_to_diskann3] writing CPR snapshot {output}");
    futures_executor::block_on(provider.save_with(&FileStorageProvider, &output.to_owned()))
        .map_err(|e| anyhow::anyhow!("save snapshot: {e:?}"))?;
    // BfTree::drop traverses every page to deallocate its in-process mapping.
    // For this one-shot converter that becomes a redundant O(index-size) read
    // after save_with has already closed the durable snapshot. On Unix it is
    // safe to unlink the temporary live stores while the provider still has
    // them open; process exit then releases the descriptors and all memory.
    for live in [&live_vector, &live_neighbors] {
        if live.exists() {
            std::fs::remove_file(live)
                .with_context(|| format!("remove temporary live tree {}", live.display()))?;
        }
    }

    let manifest = Manifest {
        format: "diskann3-bftree-full-precision-v1",
        source_graph: &args.graph,
        source_data: &args.data,
        base_points,
        capacity: args.capacity,
        dim: data_header.dim,
        max_degree: graph_header.max_degree,
        medoid: graph_header.medoid,
        metric: metric.as_str(),
        pruned_degree: args.pruned_degree,
        l_build: args.l_build,
        alpha: args.alpha,
        backedge_ratio: args.backedge_ratio,
        vector_cache_mib: args.vector_cache_mib,
        neighbor_cache_mib: args.neighbor_cache_mib,
        vector_record_bytes,
        vector_leaf_page_bytes,
        neighbor_record_bytes,
        neighbor_leaf_page_bytes,
    };
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    std::mem::forget(provider);
    eprintln!(
        "[navis_to_diskann3] complete in {:.1}s; vector_record={} vector_leaf={} neighbor_record={} neighbor_leaf={}",
        started.elapsed().as_secs_f64(),
        vector_record_bytes,
        vector_leaf_page_bytes,
        neighbor_record_bytes,
        neighbor_leaf_page_bytes
    );
    Ok(())
}
