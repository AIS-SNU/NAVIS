#pragma once

#include "navis/types.h"
#include <vector>
#include <string>

namespace navis {

// ---------------------------------------------------------------------------
// IOBackend — abstract storage I/O interface
//
// Production implementation: UringBackend (io_uring + O_DIRECT).
// This interface exists so unit tests can substitute a mock backend
// without touching real storage.
// ---------------------------------------------------------------------------
class IOBackend {
public:
    virtual ~IOBackend() = default;

    // Open the underlying storage device
    virtual void open(const std::string& device_name, bool writable) = 0;
    virtual void close() = 0;

    // Synchronous batch I/O: submit all, block until all complete.
    // All req.buf must be 4KB-aligned memory (for O_DIRECT).
    virtual void read_batch(std::vector<IORequest>& reqs) = 0;
    virtual void write_batch(std::vector<IORequest>& reqs) = 0;

    // Per-thread lifecycle: create/destroy thread-local I/O channels.
    // Must be called from the worker thread itself.
    virtual void register_thread() = 0;
    virtual void deregister_thread() = 0;
};

} // namespace navis
