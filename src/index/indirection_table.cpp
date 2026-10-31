#include "navis/index/indirection_table.h"
#include "navis/cache/navis_cache.h"
#include "navis/util/file_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace navis {

IndirectionTable::~IndirectionTable() = default;

void IndirectionTable::init_identity(uint64_t npts) {
    // No backing array; id2edge_loc(id) returns id inline.
    is_identity_.store(true, std::memory_order_release);
    cur_edge_loc_.store(npts, std::memory_order_relaxed);
    // capacity stays 0 — no buffer allocated in identity mode.
}

bool IndirectionTable::load(const std::string& prefix, uint64_t capacity_hint) {
    std::string path = prefix + "_indirection.bin";
    if (!file_exists(path)) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("IndirectionTable: failed to open " + path);
    }

    // Determine file size to detect v1 vs v2 format.
    in.seekg(0, std::ios::end);
    uint64_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    uint64_t npts;
    in.read(reinterpret_cast<char*>(&npts), sizeof(uint64_t));
    if (!in) {
        throw std::runtime_error("IndirectionTable: failed to read header in " + path);
    }

    // v1 format: [u64 npts][u32 edge_loc[npts]]
    // v2 format: [u64 npts][u64 cur_edge][u64 cur_vec][u32 edge_loc[npts]]
    //            [u32 vec_loc[npts]][u64 loc2id_n][u32 loc2id[loc2id_n]]
    //            [u64 num_empty][u64 empty_pages[num_empty]]
    uint64_t v1_expected = 8 + npts * 4;
    bool is_v2 = (file_size > v1_expected + 16);

    uint64_t cap = std::max<uint64_t>({npts, capacity_hint, uint64_t{1024}});

    uint64_t cur_edge_val = 0, cur_vec_val = 0;

    if (is_v2) {
        in.read(reinterpret_cast<char*>(&cur_edge_val), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&cur_vec_val), sizeof(uint64_t));
    }

    // --- Forward edge map ---
    auto edge_buf = std::make_unique<uint32_t[]>(cap);
    in.read(reinterpret_cast<char*>(edge_buf.get()), npts * sizeof(uint32_t));
    if (!in) throw std::runtime_error("IndirectionTable: truncated edge map in " + path);
    if (cap > npts) std::memset(edge_buf.get() + npts, 0, (cap - npts) * sizeof(uint32_t));

    bool edge_identity = true;
    for (uint64_t i = 0; i < npts; ++i) {
        if (edge_buf[i] != static_cast<uint32_t>(i)) { edge_identity = false; break; }
    }

    if (!is_v2 && edge_identity) {
        init_identity(npts);
        return true;
    }

    if (!is_v2) {
        for (uint64_t i = 0; i < npts; ++i) {
            if (edge_buf[i] + 1ULL > cur_edge_val) cur_edge_val = edge_buf[i] + 1ULL;
        }
    }

    // Install edge forward map.
    if (edge_identity) {
        is_identity_.store(true, std::memory_order_release);
        cur_edge_loc_.store(cur_edge_val ? cur_edge_val : npts, std::memory_order_relaxed);
    } else {
        edge_base_.store(edge_buf.get(), std::memory_order_release);
        capacity_.store(cap, std::memory_order_release);
        cur_edge_loc_.store(cur_edge_val, std::memory_order_relaxed);
        is_identity_.store(false, std::memory_order_release);
        retired_edge_.push_back(std::move(edge_buf));
    }

    // --- Forward vec map (v2 only) ---
    if (is_v2) {
        auto vec_buf = std::make_unique<uint32_t[]>(cap);
        in.read(reinterpret_cast<char*>(vec_buf.get()), npts * sizeof(uint32_t));
        if (!in) throw std::runtime_error("IndirectionTable: truncated vec map in " + path);
        if (cap > npts) std::memset(vec_buf.get() + npts, 0, (cap - npts) * sizeof(uint32_t));

        bool vec_identity = true;
        for (uint64_t i = 0; i < npts; ++i) {
            if (vec_buf[i] != static_cast<uint32_t>(i)) { vec_identity = false; break; }
        }
        if (vec_identity) {
            vec_is_identity_.store(true, std::memory_order_release);
        } else {
            vec_base_.store(vec_buf.get(), std::memory_order_release);
            vec_capacity_.store(cap, std::memory_order_release);
            vec_is_identity_.store(false, std::memory_order_release);
            retired_vec_.push_back(std::move(vec_buf));
        }
        cur_vec_loc_.store(cur_vec_val ? cur_vec_val : npts, std::memory_order_relaxed);
    } else {
        vec_is_identity_.store(true, std::memory_order_release);
        cur_vec_loc_.store(npts, std::memory_order_relaxed);
    }

    // --- Reverse edge map (v2 only) ---
    if (is_v2) {
        uint64_t loc2id_n;
        in.read(reinterpret_cast<char*>(&loc2id_n), sizeof(uint64_t));
        if (!in) throw std::runtime_error("IndirectionTable: truncated loc2id header in " + path);

        uint64_t loc_cap = std::max<uint64_t>({loc2id_n, capacity_hint, uint64_t{1024}});
        auto loc_buf = std::make_unique<uint32_t[]>(loc_cap);
        in.read(reinterpret_cast<char*>(loc_buf.get()), loc2id_n * sizeof(uint32_t));
        if (!in) throw std::runtime_error("IndirectionTable: truncated loc2id in " + path);
        std::fill(loc_buf.get() + loc2id_n, loc_buf.get() + loc_cap, kInvalidID);

        loc2id_base_.store(loc_buf.get(), std::memory_order_release);
        loc_capacity_.store(loc_cap, std::memory_order_release);
        retired_loc_.push_back(std::move(loc_buf));

        // --- Empty pages (v2 only) ---
        uint64_t num_empty;
        in.read(reinterpret_cast<char*>(&num_empty), sizeof(uint64_t));
        if (in && num_empty > 0) {
            std::vector<uint64_t> empties(num_empty);
            in.read(reinterpret_cast<char*>(empties.data()), num_empty * sizeof(uint64_t));
            for (auto p : empties) {
                bool val = false;
                if (empty_pages_index_.insert(p, val)) {
                    empty_pages_queue_.push(p);
                }
            }
        }
    } else if (!edge_identity) {
        // v1 fallback: build reverse map from forward edge map.
        uint64_t loc_cap = std::max<uint64_t>({cur_edge_val, uint64_t{1024}});
        auto loc_buf = std::make_unique<uint32_t[]>(loc_cap);
        std::fill(loc_buf.get(), loc_buf.get() + loc_cap, kInvalidID);
        uint32_t* eb = edge_base_.load(std::memory_order_acquire);
        for (uint32_t id = 0; id < npts; ++id) {
            if (eb[id] < loc_cap) loc_buf[eb[id]] = id;
        }
        loc2id_base_.store(loc_buf.get(), std::memory_order_release);
        loc_capacity_.store(loc_cap, std::memory_order_release);
        retired_loc_.push_back(std::move(loc_buf));
    }

    return true;
}

void IndirectionTable::save(const std::string& prefix, uint64_t npts) const {
    std::string path = prefix + "_indirection.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("IndirectionTable: cannot write " + path);
    }

    auto write_u64 = [&](uint64_t v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(uint64_t));
    };
    auto dump_identity_or_array = [&](bool identity, uint32_t* base, uint64_t count) {
        if (identity) {
            constexpr uint64_t CHUNK = 1ULL << 20;
            std::vector<uint32_t> buf(std::min(count, CHUNK));
            for (uint64_t off = 0; off < count; off += CHUNK) {
                uint64_t n = std::min(CHUNK, count - off);
                for (uint64_t i = 0; i < n; ++i) buf[i] = static_cast<uint32_t>(off + i);
                out.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(uint32_t));
            }
        } else {
            out.write(reinterpret_cast<const char*>(base), count * sizeof(uint32_t));
        }
    };

    uint64_t cur_edge = cur_edge_loc_.load(std::memory_order_acquire);
    uint64_t cur_vec  = cur_vec_loc_ .load(std::memory_order_acquire);

    // v2 header
    write_u64(npts);
    write_u64(cur_edge);
    write_u64(cur_vec);

    // Forward edge map
    dump_identity_or_array(is_identity_.load(std::memory_order_acquire),
                           edge_base_.load(std::memory_order_acquire), npts);
    // Forward vec map
    dump_identity_or_array(vec_is_identity_.load(std::memory_order_acquire),
                           vec_base_.load(std::memory_order_acquire), npts);

    // Reverse edge map
    uint64_t loc_n = std::max<uint64_t>(cur_edge, npts);
    write_u64(loc_n);
    uint32_t* loc2id = loc2id_base_.load(std::memory_order_acquire);
    if (loc2id) {
        out.write(reinterpret_cast<const char*>(loc2id), loc_n * sizeof(uint32_t));
    } else {
        // Synthesize identity reverse map.
        constexpr uint64_t CHUNK = 1ULL << 20;
        std::vector<uint32_t> buf(std::min(loc_n, CHUNK));
        for (uint64_t off = 0; off < loc_n; off += CHUNK) {
            uint64_t n = std::min(CHUNK, loc_n - off);
            for (uint64_t i = 0; i < n; ++i) buf[i] = static_cast<uint32_t>(off + i);
            out.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(uint32_t));
        }
    }

    // Empty pages snapshot
    std::vector<uint64_t> empty_snap;
    {
        auto* self = const_cast<IndirectionTable*>(this);
        std::lock_guard<std::mutex> g(self->alloc_lock_);
        auto tmp = self->empty_pages_queue_;
        while (!tmp.empty()) {
            empty_snap.push_back(tmp.front());
            tmp.pop();
        }
    }
    write_u64(empty_snap.size());
    if (!empty_snap.empty()) {
        out.write(reinterpret_cast<const char*>(empty_snap.data()),
                  empty_snap.size() * sizeof(uint64_t));
    }
}

void IndirectionTable::ensure_edge_capacity(uint64_t new_cap) {
    if (is_identity_.load(std::memory_order_acquire)) return;  // no array to grow
    if (capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    std::lock_guard<std::mutex> lock(grow_mu_);
    if (capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    grow_edge_to_locked(new_cap);
}

void IndirectionTable::grow_edge_to_locked(uint64_t new_cap) {
    uint64_t cur_cap = capacity_.load(std::memory_order_relaxed);
    uint64_t target = std::max<uint64_t>(cur_cap * 2, new_cap);
    if (target < 1024) target = 1024;

    auto new_edge = std::make_unique<uint32_t[]>(target);
    uint32_t* old_edge = edge_base_.load(std::memory_order_acquire);
    if (old_edge && cur_cap > 0) {
        std::memcpy(new_edge.get(), old_edge, cur_cap * sizeof(uint32_t));
    }
    if (target > cur_cap) {
        std::memset(new_edge.get() + cur_cap, 0, (target - cur_cap) * sizeof(uint32_t));
    }

    edge_base_.store(new_edge.get(), std::memory_order_release);
    capacity_.store(target, std::memory_order_release);
    retired_edge_.push_back(std::move(new_edge));
}

void IndirectionTable::set_edge_loc(uint32_t id, uint32_t edge_loc) {
    if (is_identity_.load(std::memory_order_acquire)) {
        // Under identity, edge_loc must equal id. If a caller wants to set
        // a non-identity mapping, they must first drop the identity flag —
        // out of scope today.
        return;
    }
    if (id >= capacity_.load(std::memory_order_relaxed)) {
        ensure_edge_capacity(static_cast<uint64_t>(id) + 1);
    }
    edge_base_.load(std::memory_order_acquire)[id] = edge_loc;
}

// ---------------------------------------------------------------------------
// Geometry setter
// ---------------------------------------------------------------------------
void IndirectionTable::set_geometry(uint32_t edges_per_page) {
    edges_per_page_ = edges_per_page;
}

// ---------------------------------------------------------------------------
// Build-pipeline initializer
// ---------------------------------------------------------------------------
void IndirectionTable::init_from_perm(const uint32_t* edge_perm,
                                      const uint32_t* vec_perm,
                                      uint64_t npts) {
    // --- Edge side ---
    {
        std::lock_guard<std::mutex> lock(grow_mu_);
        uint64_t cap = std::max<uint64_t>(npts, uint64_t{1024});
        auto buf = std::make_unique<uint32_t[]>(cap);
        std::memcpy(buf.get(), edge_perm, npts * sizeof(uint32_t));
        if (cap > npts)
            std::memset(buf.get() + npts, 0, (cap - npts) * sizeof(uint32_t));

        uint64_t cur = 0;
        for (uint64_t i = 0; i < npts; i++)
            if (uint64_t(edge_perm[i]) + 1 > cur) cur = uint64_t(edge_perm[i]) + 1;

        edge_base_.store(buf.get(), std::memory_order_release);
        capacity_.store(cap, std::memory_order_release);
        cur_edge_loc_.store(cur, std::memory_order_relaxed);
        is_identity_.store(false, std::memory_order_release);
        retired_edge_.push_back(std::move(buf));
    }

    // --- Vector side ---
    if (vec_perm) {
        std::lock_guard<std::mutex> lock(grow_mu_);
        uint64_t cap = std::max<uint64_t>(npts, uint64_t{1024});
        auto buf = std::make_unique<uint32_t[]>(cap);
        std::memcpy(buf.get(), vec_perm, npts * sizeof(uint32_t));
        if (cap > npts)
            std::memset(buf.get() + npts, 0, (cap - npts) * sizeof(uint32_t));

        uint64_t cur = 0;
        for (uint64_t i = 0; i < npts; i++)
            if (uint64_t(vec_perm[i]) + 1 > cur) cur = uint64_t(vec_perm[i]) + 1;

        vec_base_.store(buf.get(), std::memory_order_release);
        vec_capacity_.store(cap, std::memory_order_release);
        cur_vec_loc_.store(cur, std::memory_order_relaxed);
        vec_is_identity_.store(false, std::memory_order_release);
        retired_vec_.push_back(std::move(buf));
    }
    // else: vec side stays identity

    // --- Reverse edge map: invert edge_perm ---
    {
        uint64_t max_loc = 0;
        for (uint64_t i = 0; i < npts; ++i) {
            if (uint64_t(edge_perm[i]) + 1 > max_loc)
                max_loc = uint64_t(edge_perm[i]) + 1;
        }
        uint64_t loc_cap = std::max<uint64_t>(max_loc, uint64_t{1024});
        auto buf = std::make_unique<uint32_t[]>(loc_cap);
        std::fill(buf.get(), buf.get() + loc_cap, kInvalidID);
        for (uint32_t id = 0; id < npts; ++id) {
            buf[edge_perm[id]] = id;
        }
        loc2id_base_.store(buf.get(), std::memory_order_release);
        loc_capacity_.store(loc_cap, std::memory_order_release);
        retired_loc_.push_back(std::move(buf));
    }
}

// ---------------------------------------------------------------------------
// Vector forward map
// ---------------------------------------------------------------------------
void IndirectionTable::set_vec_loc(uint32_t id, uint32_t vec_loc) {
    if (vec_is_identity_.load(std::memory_order_acquire)) return;
    if (id >= vec_capacity_.load(std::memory_order_relaxed))
        ensure_vec_capacity(static_cast<uint64_t>(id) + 1);
    vec_base_.load(std::memory_order_acquire)[id] = vec_loc;
}

void IndirectionTable::ensure_vec_capacity(uint64_t new_cap) {
    if (vec_is_identity_.load(std::memory_order_acquire)) return;
    if (vec_capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    std::lock_guard<std::mutex> lock(grow_mu_);
    if (vec_capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    grow_vec_to_locked(new_cap);
}

void IndirectionTable::grow_vec_to_locked(uint64_t new_cap) {
    uint64_t cur_cap = vec_capacity_.load(std::memory_order_relaxed);
    uint64_t target  = std::max<uint64_t>(cur_cap * 2, new_cap);
    if (target < 1024) target = 1024;

    auto buf = std::make_unique<uint32_t[]>(target);
    uint32_t* old = vec_base_.load(std::memory_order_acquire);
    if (old && cur_cap > 0)
        std::memcpy(buf.get(), old, cur_cap * sizeof(uint32_t));
    if (target > cur_cap)
        std::memset(buf.get() + cur_cap, 0, (target - cur_cap) * sizeof(uint32_t));

    vec_base_.store(buf.get(), std::memory_order_release);
    vec_capacity_.store(target, std::memory_order_release);
    retired_vec_.push_back(std::move(buf));
}

// ---------------------------------------------------------------------------
// Reverse edge map
// ---------------------------------------------------------------------------
void IndirectionTable::ensure_loc_capacity(uint64_t new_cap) {
    const uint64_t max_cap = static_cast<uint64_t>(kAllocatedID);
    if (new_cap > max_cap) {
        throw std::overflow_error(
            "IndirectionTable::ensure_loc_capacity: edge-location space exhausted");
    }
    if (loc_capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    std::lock_guard<std::mutex> lock(grow_mu_);
    if (loc_capacity_.load(std::memory_order_relaxed) >= new_cap) return;
    grow_loc_to_locked(new_cap);
}

void IndirectionTable::grow_loc_to_locked(uint64_t new_cap) {
    uint64_t cur_cap = loc_capacity_.load(std::memory_order_relaxed);
    const uint64_t max_cap = static_cast<uint64_t>(kAllocatedID);
    if (cur_cap > max_cap || new_cap > max_cap) {
        throw std::overflow_error(
            "IndirectionTable::grow_loc_to_locked: edge-location space exhausted");
    }

    const uint64_t doubled =
        cur_cap > max_cap / 2 ? max_cap : cur_cap * 2;
    uint64_t target = std::max<uint64_t>(doubled, new_cap);
    if (target < 1024) target = 1024;

    auto buf = std::make_unique<uint32_t[]>(target);
    uint32_t* old = loc2id_base_.load(std::memory_order_acquire);
    if (old && cur_cap > 0)
        std::memcpy(buf.get(), old, cur_cap * sizeof(uint32_t));
    // New slots are invalid (free).
    std::fill(buf.get() + cur_cap, buf.get() + target, kInvalidID);

    loc2id_base_.store(buf.get(), std::memory_order_release);
    loc_capacity_.store(target, std::memory_order_release);
    retired_loc_.push_back(std::move(buf));
}

// ---------------------------------------------------------------------------
// 3-tier edge-slot allocator
// ---------------------------------------------------------------------------
std::vector<uint32_t> IndirectionTable::alloc_edge_locs(
        int n,
        const std::set<uint64_t>& hint_pages,
        std::set<uint64_t>& pages_need_to_read) {
    std::lock_guard<std::mutex> lock(alloc_lock_);
    if (n < 0) {
        throw std::invalid_argument(
            "IndirectionTable::alloc_edge_locs: n must be non-negative");
    }

    if (edges_per_page_ == 0) {
        std::fprintf(stderr, "[indirection] alloc_edge_locs called before set_geometry\n");
        std::abort();
    }

    constexpr int INDEX_SIZE_FACTOR = 4;
    uint32_t threshold = static_cast<uint32_t>(
        (edges_per_page_ + INDEX_SIZE_FACTOR - 1) / INDEX_SIZE_FACTOR);

    uint32_t epp = edges_per_page_;
    if (n == 0) return {};

    const uint64_t tail = cur_edge_loc_.load(std::memory_order_relaxed);
    auto tier3_bounds = [tail, epp](uint64_t count,
                                    uint64_t& allocation_end,
                                    uint64_t& padded_end) {
        if (count > std::numeric_limits<uint64_t>::max() - tail) return false;
        allocation_end = tail + count;

        const uint64_t page_remainder = allocation_end % uint64_t{epp};
        const uint64_t padding =
            page_remainder == 0 ? 0 : uint64_t{epp} - page_remainder;
        if (padding > std::numeric_limits<uint64_t>::max() - allocation_end)
            return false;
        padded_end = allocation_end + padding;

        // Both kAllocatedID and kInvalidID are sentinels. kAllocatedID is the
        // exclusive slot count, so valid edge locations are [0, kAllocatedID).
        return padded_end <= static_cast<uint64_t>(kAllocatedID);
    };

    uint64_t unused_allocation_end = 0;
    uint64_t unused_padded_end = 0;
    const bool rollback_enabled = !tier3_bounds(
        static_cast<uint64_t>(n), unused_allocation_end, unused_padded_end);

    // A near-exhaustion request may still succeed entirely from Tiers 1/2.
    // Snapshot only that rare path so a later Tier-3 rejection can restore
    // every allocator-visible mutation without penalizing normal workloads.
    std::queue<uint64_t> empty_pages_snapshot;
    std::set<uint64_t> pages_need_to_read_snapshot;
    std::vector<uint64_t> popped_empty_pages;
    if (rollback_enabled) {
        empty_pages_snapshot = empty_pages_queue_;
        pages_need_to_read_snapshot = pages_need_to_read;
        const uint64_t pages_for_request =
            static_cast<uint64_t>(n) / epp +
            (static_cast<uint64_t>(n) % epp != 0);
        popped_empty_pages.reserve(static_cast<size_t>(std::min<uint64_t>(
            empty_pages_queue_.size(), pages_for_request)));
    }

    std::vector<uint32_t> ret;
    ret.reserve(static_cast<size_t>(n));
    int cur = 0;

    auto commit_empty_page_pops = [&] {
        if (!rollback_enabled) return;
        for (uint64_t page_no : popped_empty_pages) {
            empty_pages_index_.erase(page_no);
        }
    };

    auto rollback_reuse = [&] {
        uint32_t* loc2id = loc2id_base_.load(std::memory_order_acquire);
        for (uint32_t loc : ret) {
            loc2id[loc] = kInvalidID;
        }
        empty_pages_queue_.swap(empty_pages_snapshot);
        pages_need_to_read.swap(pages_need_to_read_snapshot);
    };

    auto claim_slot = [&](uint32_t loc) {
        ensure_loc_capacity(static_cast<uint64_t>(loc) + 1);
        loc2id_base_.load(std::memory_order_acquire)[loc] = kAllocatedID;
        ret.push_back(loc);
        ++cur;
    };

    // -------------- Tier 1: empty pages (no read needed) ----------------
    while (cur < n && !empty_pages_queue_.empty()) {
        uint64_t page_no = empty_pages_queue_.front();
        if (rollback_enabled) popped_empty_pages.push_back(page_no);
        empty_pages_queue_.pop();
        if (!rollback_enabled) empty_pages_index_.erase(page_no);

        uint32_t st = static_cast<uint32_t>((page_no - 1) * epp);
        uint32_t ed = st + epp;
        ensure_loc_capacity(ed);
        uint32_t* loc2id = loc2id_base_.load(std::memory_order_acquire);
        for (uint32_t i = st; i < ed && cur < n; ++i) {
            if (loc2id[i] != kInvalidID) {
                std::fprintf(stderr,
                    "[indirection] empty_pages invariant violated: page %lu slot %u = %u\n",
                    (unsigned long)page_no, i, loc2id[i]);
                std::abort();
            }
            claim_slot(i);
        }
    }
    if (cur == n) {
        commit_empty_page_pops();
        return ret;
    }

    // -------------- Tier 2: hint pages with holes -----------------------
    for (auto p : hint_pages) {
        if (cur == n) break;
        uint32_t st = static_cast<uint32_t>((p - 1) * epp);
        uint32_t ed = st + epp;
        if (ed > loc_capacity_.load(std::memory_order_relaxed)) continue;

        uint32_t* loc2id = loc2id_base_.load(std::memory_order_acquire);
        uint32_t cnt = 0;
        for (uint32_t i = st; i < ed; ++i) {
            if (loc2id[i] == kInvalidID) ++cnt;
        }
        if (cnt < threshold) continue;
        if (cnt < epp) {
            pages_need_to_read.insert(p);
        }
        for (uint32_t i = st; i < ed && cur < n; ++i) {
            if (loc2id[i] == kInvalidID) {
                claim_slot(i);
            }
        }
    }
    if (cur == n) {
        commit_empty_page_pops();
        return ret;
    }

    // -------------- Tier 3: fresh tail (no read needed) -----------------
    const uint64_t remaining = static_cast<uint64_t>(n - cur);

    {
        uint64_t allocation_end = 0;
        uint64_t padded_end = 0;
        if (!tier3_bounds(remaining, allocation_end, padded_end)) {
            if (rollback_enabled) rollback_reuse();
            throw std::overflow_error(
                "IndirectionTable::alloc_edge_locs: edge-location space exhausted");
        }

        try {
            ensure_loc_capacity(padded_end);
        } catch (...) {
            if (rollback_enabled) rollback_reuse();
            throw;
        }
        uint32_t* loc2id = loc2id_base_.load(std::memory_order_acquire);

        for (uint64_t loc = tail; loc < allocation_end; ++loc) {
            loc2id[loc] = kAllocatedID;
            ret.push_back(static_cast<uint32_t>(loc));
        }

        // Pad tail to page boundary so future tier-1/2 sees clean page state.
        for (uint64_t loc = allocation_end; loc < padded_end; ++loc) {
            loc2id[loc] = kInvalidID;
        }
        cur_edge_loc_.store(padded_end, std::memory_order_release);
        commit_empty_page_pops();
    }
    return ret;
}

// ---------------------------------------------------------------------------
// Bind / Invalidate
// ---------------------------------------------------------------------------
void IndirectionTable::bind_edge_loc(uint32_t id, uint32_t edge_loc) {
    // Forward map.
    set_edge_loc(id, edge_loc);
    // Reverse map.
    ensure_loc_capacity(static_cast<uint64_t>(edge_loc) + 1);
    loc2id_base_.load(std::memory_order_acquire)[edge_loc] = id;
}

void IndirectionTable::invalidate_edge_loc(uint32_t edge_loc) {
    std::lock_guard<std::mutex> lock(alloc_lock_);
    uint32_t* base = loc2id_base_.load(std::memory_order_acquire);
    if (!base || edge_loc >= loc_capacity_.load(std::memory_order_relaxed))
        return;
    base[edge_loc] = kInvalidID;

    if (edges_per_page_ == 0) return;
    uint32_t epp = edges_per_page_;
    uint64_t page_no   = 1 + edge_loc / epp;
    uint32_t first_loc = static_cast<uint32_t>((page_no - 1) * epp);
    uint64_t cap       = loc_capacity_.load(std::memory_order_relaxed);
    bool     all_free  = true;
    for (uint32_t sl = first_loc; sl < first_loc + epp; ++sl) {
        if (sl >= cap) break;
        if (base[sl] != kInvalidID) { all_free = false; break; }
    }
    if (all_free) {
        bool val = false;
        if (empty_pages_index_.insert(page_no, val)) {
            empty_pages_queue_.push(page_no);
        }
        // Eager eviction hint for NAVIS-Cache: a fully-invalidated page
        // is dead weight in the read cache (no id2edge_loc points into
        // it until tier-1/2 reuse). Reclaim the cache slot now instead
        // of waiting for W-TinyLFU cold-evict. page_no equals the
        // physical sector number in the edge file (sector 0 is metadata,
        // edge page k lives at sector k). Non-owning pointer; null when
        // the cache is disabled, so the call is safe in both modes.
        if (navis_cache_) {
            navis_cache_->evict(page_no);
        }
    }
}

} // namespace navis
