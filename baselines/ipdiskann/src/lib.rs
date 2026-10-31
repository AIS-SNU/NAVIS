use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use diskann_bftree::Config as BfTreeConfig;

pub const BIN_HEADER_BYTES: u64 = 8;
pub const VAMANA_HEADER_BYTES: u64 = 24;
pub const CSV_HEADER: &str = "system,dataset,base_n,batch_idx,batch_size,tick_idx,t_elapsed_s,inserted_so_far,qps,mean_lat_ms,p50_ms,p90_ms,p95_ms,p99_ms,p999_ms,recall_at_10,mean_disk_ios,rss_mib,cache_hit_rate,disk_index_bytes,search_threads,insert_threads,cache_gb,search_L,insert_L,K,notes";

#[derive(Debug, Clone, Copy)]
pub struct BinHeader {
    pub rows: usize,
    pub dim: usize,
}

pub fn read_bin_header(path: &Path) -> Result<BinHeader> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open vector file {}", path.display()))?,
    );
    let rows = read_u32(&mut reader)? as usize;
    let dim = read_u32(&mut reader)? as usize;
    ensure!(
        rows > 0 && dim > 0,
        "{} has an empty bin header",
        path.display()
    );
    let expected = BIN_HEADER_BYTES
        .checked_add(
            (rows as u64)
                .checked_mul(dim as u64)
                .and_then(|v| v.checked_mul(4))
                .context("vector file size overflow")?,
        )
        .context("vector file size overflow")?;
    let actual = reader.get_ref().metadata()?.len();
    ensure!(
        actual == expected,
        "{} size mismatch: header expects {expected} bytes but file has {actual}",
        path.display()
    );
    Ok(BinHeader { rows, dim })
}

pub struct FloatBinReader {
    reader: BufReader<File>,
    header: BinHeader,
    next_row: usize,
}

impl FloatBinReader {
    pub fn open(path: &Path) -> Result<Self> {
        let header = read_bin_header(path)?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
        reader.seek(SeekFrom::Start(BIN_HEADER_BYTES))?;
        Ok(Self {
            reader,
            header,
            next_row: 0,
        })
    }

    pub fn header(&self) -> BinHeader {
        self.header
    }

    pub fn read_rows(&mut self, count: usize) -> Result<Vec<f32>> {
        ensure!(
            self.next_row + count <= self.header.rows,
            "requested rows {}..{} past vector count {}",
            self.next_row,
            self.next_row + count,
            self.header.rows
        );
        let mut values = vec![0.0f32; count * self.header.dim];
        self.reader
            .read_exact(bytemuck::cast_slice_mut(&mut values))?;
        self.next_row += count;
        Ok(values)
    }

    pub fn seek_row(&mut self, row: usize) -> Result<()> {
        ensure!(row <= self.header.rows, "row {row} is out of range");
        let offset = BIN_HEADER_BYTES + (row * self.header.dim * 4) as u64;
        self.reader.seek(SeekFrom::Start(offset))?;
        self.next_row = row;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VamanaHeader {
    pub declared_size: u64,
    pub max_degree: u32,
    pub medoid: u32,
    pub frozen_points: u64,
}

pub struct VamanaReader {
    reader: BufReader<File>,
    header: VamanaHeader,
    nodes: usize,
    next_node: usize,
}

impl VamanaReader {
    pub fn open(path: &Path, nodes: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open graph {}", path.display()))?;
        let actual_size = file.metadata()?.len();
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let header = VamanaHeader {
            declared_size: read_u64(&mut reader)?,
            max_degree: read_u32(&mut reader)?,
            medoid: read_u32(&mut reader)?,
            frozen_points: read_u64(&mut reader)?,
        };
        ensure!(
            header.declared_size == actual_size,
            "{} graph header declares {} bytes but file has {}",
            path.display(),
            header.declared_size,
            actual_size
        );
        ensure!(header.max_degree > 0, "graph max degree is zero");
        ensure!(
            (header.medoid as usize) < nodes,
            "graph medoid is out of range"
        );
        ensure!(
            header.frozen_points == 0,
            "converter expects a static NAVIS graph (frozen_points=0), got {}",
            header.frozen_points
        );
        Ok(Self {
            reader,
            header,
            nodes,
            next_node: 0,
        })
    }

    pub fn header(&self) -> VamanaHeader {
        self.header
    }

    pub fn read_batch(&mut self, count: usize) -> Result<Vec<Vec<u32>>> {
        ensure!(
            self.next_node + count <= self.nodes,
            "graph batch exceeds node count"
        );
        let mut rows = Vec::with_capacity(count);
        for node in self.next_node..self.next_node + count {
            let degree = read_u32(&mut self.reader)? as usize;
            ensure!(
                degree <= self.header.max_degree as usize,
                "node {node} degree {degree} exceeds graph max degree {}",
                self.header.max_degree
            );
            let mut neighbors = Vec::with_capacity(degree);
            for _ in 0..degree {
                let id = read_u32(&mut self.reader)?;
                ensure!(
                    (id as usize) < self.nodes,
                    "node {node} has out-of-range neighbor {id}"
                );
                neighbors.push(id);
            }
            rows.push(neighbors);
        }
        self.next_node += count;
        Ok(rows)
    }

    pub fn finish(mut self) -> Result<()> {
        ensure!(self.next_node == self.nodes, "graph import stopped early");
        let mut trailing = [0u8; 1];
        ensure!(
            self.reader.read(&mut trailing)? == 0,
            "graph contains trailing bytes"
        );
        Ok(())
    }
}

pub fn vector_store_config(path: &Path, cache_bytes: usize, dim: usize) -> Result<BfTreeConfig> {
    let record_size = std::mem::size_of::<usize>()
        .checked_add(dim.checked_mul(4).context("vector record size overflow")?)
        .context("vector record size overflow")?;
    disk_store_config(path, cache_bytes, record_size)
}

pub fn neighbor_store_config(
    path: &Path,
    cache_bytes: usize,
    max_degree: u32,
) -> Result<BfTreeConfig> {
    let record_size =
        std::mem::size_of::<u32>() + (max_degree as usize + 1) * std::mem::size_of::<u32>();
    disk_store_config(path, cache_bytes, record_size)
}

fn disk_store_config(path: &Path, cache_bytes: usize, record_size: usize) -> Result<BfTreeConfig> {
    ensure!(cache_bytes > 0, "Bf-tree cache must be nonzero");
    let cb_size = cache_bytes.next_power_of_two();
    let mut leaf_size = 4096usize;
    while leaf_size < record_size.saturating_mul(2).saturating_add(256) {
        leaf_size = leaf_size
            .checked_mul(2)
            .context("leaf page size overflow")?;
    }
    let mut config = BfTreeConfig::new(path, cb_size);
    config
        .cb_max_record_size(record_size)
        .leaf_page_size(leaf_size)
        // CPR recovery can materialize cache-tagged records as disk base pages.
        // Full-page promotion attempts to tag them as Cache again and panics in
        // bf-tree 0.5.x. Mini-page reads/writes are the intended SSD path here
        // and avoid that broken recovery/promotion combination.
        .read_record_cache(false)
        .write_load_full_page(false)
        .use_snapshot(true);
    config
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid Bf-tree config: {e:?}"))?;
    Ok(config)
}

pub fn snapshot_paths(prefix: &Path) -> Result<Vec<PathBuf>> {
    let prefix = prefix
        .to_str()
        .context("DiskANN3 snapshot prefix is not valid UTF-8")?;
    Ok(["_params.json", "_vectors.bftree", "_neighbors.bftree"]
        .into_iter()
        .map(|suffix| PathBuf::from(format!("{prefix}{suffix}")))
        .collect())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub fn parse_metric(value: &str) -> Result<diskann_vector::distance::Metric> {
    use std::str::FromStr;
    diskann_vector::distance::Metric::from_str(value)
        .map_err(|e| anyhow::anyhow!("invalid metric {value:?}: {e}"))
}

pub fn checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label}={value} exceeds DiskANN3 u32 ids"))
}

pub fn require_absent(paths: &[PathBuf], force: bool) -> Result<()> {
    let existing: Vec<_> = paths.iter().filter(|p| p.exists()).collect();
    if !existing.is_empty() && !force {
        bail!(
            "refusing to overwrite existing output {}; pass --force",
            existing[0].display()
        );
    }
    for path in existing {
        std::fs::remove_file(path)
            .with_context(|| format!("remove prior output {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_dimensional_vector_config_is_valid() {
        let config = vector_store_config(Path::new("/tmp/vector.bftree"), 64 << 20, 768)
            .expect("768D config should validate");
        assert_eq!(config.get_cb_max_record_size(), 3080);
        assert_eq!(config.get_leaf_page_size(), 8192);
        assert!(!config.is_memory_backend());
    }

    #[test]
    fn neighbor_config_is_disk_backed() {
        let config = neighbor_store_config(Path::new("/tmp/neighbors.bftree"), 64 << 20, 96)
            .expect("R=96 config should validate");
        assert_eq!(config.get_cb_max_record_size(), 392);
        assert_eq!(config.get_leaf_page_size(), 4096);
        assert!(!config.is_memory_backend());
    }
}
