#include "navis/build/build.h"
#include "navis/separated_layout.h"
#include "navis/util/file_utils.h"
#include "navis/io/uring_io.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace navis {
namespace build {

// ===========================================================================
// create_starling_partition — Starling-style graph partitioner.
//
// Output: {prefix}_partition.bin.aligned, format consumed by the separated
// layout builder:
//   [u64 C][u64 partition_nums][u64 nd]
//   for p in 0..partition_nums:
//     [u32 count][u32 * C node_ids]   (padding slots = UINT32_MAX)
//
// Algorithm: BFS-based greedy growth. Seed from the medoid; grow each
// partition by popping from a BFS queue until C ids are placed; then
// start a new partition. When the queue runs dry mid-partition we pad
// with the next unassigned id in sweep order. Each node is enqueued at
// most once (via an `enqueued` bitmap), so memory stays O(npts).
//
// Matches the Starling paper's core GP idea (co-traversed nodes land on
// the same page) without the METIS / k-way-balanced objective. Quality
// is approximate but close, and runtime is O(|V| * avg_degree).
// ===========================================================================

static constexpr uint32_t kInvalidID = UINT32_MAX;

namespace {

// Parse a Vamana graph file, returning both the adjacency list and the
// medoid id (from the file's header).
template <typename T>
void load_vamana(const std::string& graph_path, uint32_t npts,
                 std::vector<std::vector<uint32_t>>& adj,
                 uint32_t& medoid_out, uint32_t& max_degree_out) {
    UringReader graph_in(graph_path);
    uint64_t index_size;
    uint32_t max_degree, medoid;
    uint64_t frozen_num;
    graph_in.read(&index_size, sizeof(uint64_t));
    graph_in.read(&max_degree, sizeof(uint32_t));
    graph_in.read(&medoid,     sizeof(uint32_t));
    graph_in.read(&frozen_num, sizeof(uint64_t));

    medoid_out     = medoid;
    max_degree_out = max_degree;
    adj.assign(npts, {});

    for (uint32_t i = 0; i < npts; i++) {
        uint32_t deg;
        graph_in.read(&deg, sizeof(uint32_t));
        uint32_t kept = std::min(deg, max_degree);
        adj[i].resize(kept);
        graph_in.read(adj[i].data(), kept * sizeof(uint32_t));
        if (deg > max_degree) {
            graph_in.skip((deg - max_degree) * sizeof(uint32_t));
        }
    }
}

} // anonymous

template <typename T>
void create_starling_partition(const std::string& graph_path,
                               uint32_t npts,
                               uint64_t edges_per_page,
                               const std::string& output_prefix) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<uint32_t>> adj;
    uint32_t medoid, max_degree;
    load_vamana<T>(graph_path, npts, adj, medoid, max_degree);

    const uint64_t C = edges_per_page;
    const uint64_t num_pages = div_round_up(npts, C);

    std::printf("[partition] npts=%u, C=%lu, num_pages=%lu, seed medoid=%u\n",
                npts, C, num_pages, medoid);

    // Page assignment + "already in queue" tracking (each node enqueued ≤ 1×).
    std::vector<uint32_t> page_of(npts, kInvalidID);
    std::vector<uint8_t>  enqueued(npts, 0);

    std::deque<uint32_t> Q;
    Q.push_back(medoid);
    enqueued[medoid] = 1;

    // Rolling cursor for picking a fresh seed when BFS runs out.
    uint32_t next_seed = 0;

    // Page layout: [num_pages][C] slots, UINT32_MAX = empty.
    std::vector<uint32_t> page_ids(num_pages * C, kInvalidID);
    std::vector<uint32_t> page_counts(num_pages, 0);

    uint64_t placed = 0;
    for (uint64_t p = 0; p < num_pages; p++) {
        uint32_t slot = 0;
        while (slot < C) {
            // Drop stale queue entries (nodes assigned via other paths —
            // can't happen with enqueued[] gate, but keep belt-and-suspenders).
            while (!Q.empty() && page_of[Q.front()] != kInvalidID) Q.pop_front();

            if (Q.empty()) {
                // Find the next unassigned id.
                while (next_seed < npts && page_of[next_seed] != kInvalidID)
                    next_seed++;
                if (next_seed >= npts) break;  // all nodes placed, tail partition
                Q.push_back(next_seed);
                enqueued[next_seed] = 1;
                next_seed++;
            }

            uint32_t v = Q.front();
            Q.pop_front();
            if (page_of[v] != kInvalidID) continue;

            page_of[v] = static_cast<uint32_t>(p);
            page_ids[p * C + slot] = v;
            slot++;
            placed++;

            // Expand BFS frontier with unassigned, un-enqueued neighbors.
            for (uint32_t nbr : adj[v]) {
                if (nbr < npts && page_of[nbr] == kInvalidID && !enqueued[nbr]) {
                    Q.push_back(nbr);
                    enqueued[nbr] = 1;
                }
            }
        }
        page_counts[p] = slot;

        if (p % 1024 == 0 || p + 1 == num_pages) {
            int pct = static_cast<int>(100.0 * (p + 1) / num_pages);
            std::printf("\r[partition] pages: %3d%% (placed %lu/%u)", pct, placed, npts);
            std::fflush(stdout);
        }
    }
    std::printf("\n");

    // Write {prefix}_partition.bin.aligned
    std::string out_path = output_prefix + "_partition.bin.aligned";
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("create_starling_partition: cannot write " + out_path);
    }
    uint64_t C64  = C;
    uint64_t pn64 = num_pages;
    uint64_t nd64 = npts;
    out.write(reinterpret_cast<const char*>(&C64),  sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&pn64), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&nd64), sizeof(uint64_t));
    for (uint64_t p = 0; p < num_pages; p++) {
        uint32_t s = page_counts[p];
        out.write(reinterpret_cast<const char*>(&s), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(page_ids.data() + p * C),
                  C * sizeof(uint32_t));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Sanity pass: every id should appear exactly once.
    uint64_t missing = 0;
    for (uint32_t id = 0; id < npts; id++) {
        if (page_of[id] == kInvalidID) missing++;
    }
    if (missing > 0) {
        std::fprintf(stderr,
            "[partition] WARNING: %lu ids unassigned (expected 0)\n", missing);
    }

    std::printf("[partition] Wrote %s (%.2f MB) in %.2fs, avg fill = %.1f/%lu\n",
                out_path.c_str(),
                file_size(out_path) / (1024.0 * 1024.0),
                secs,
                static_cast<double>(placed) / num_pages, C);
}

// Explicit instantiations
template void create_starling_partition<float>(const std::string&, uint32_t,
                                               uint64_t, const std::string&);
template void create_starling_partition<int8_t>(const std::string&, uint32_t,
                                                uint64_t, const std::string&);
template void create_starling_partition<uint8_t>(const std::string&, uint32_t,
                                                 uint64_t, const std::string&);

} // namespace build
} // namespace navis
