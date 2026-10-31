#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// ===========================================================================
// io_uring-based sequential reader and writer for the build pipeline.
//
// C API implemented in src/io/uring_io.c (compiled as C because liburing's
// header includes C++ <atomic> and can't be used from C++ directly).
//
// Reader: io_uring + O_DIRECT, 64MB aligned buffer, 4MB SQEs.
// Writer: 64MB user-space buffer, flushed via io_uring in 4MB chunks.
// ===========================================================================

extern "C" {
    // --- Reader ---
    struct uring_reader;
    struct uring_reader* uring_reader_open(const char* path, size_t buf_size);
    void uring_reader_close(struct uring_reader* r);
    int uring_reader_read(struct uring_reader* r, void* dst, size_t n);
    void uring_reader_seek(struct uring_reader* r, uint64_t offset);
    void uring_reader_skip(struct uring_reader* r, size_t n);
    uint64_t uring_reader_file_size(struct uring_reader* r);

    // --- Writer ---
    struct uring_writer;
    struct uring_writer* uring_writer_open(const char* path, size_t buf_size);
    void uring_writer_close(struct uring_writer* w);
    int uring_writer_write(struct uring_writer* w, const void* src, size_t n);
    int uring_writer_seek(struct uring_writer* w, uint64_t offset);
    uint64_t uring_writer_tell(struct uring_writer* w);
}

namespace navis {

class UringReader {
public:
    explicit UringReader(const std::string& path, size_t buf_size = 64 * 1024 * 1024)
        : handle_(uring_reader_open(path.c_str(), buf_size)) {
        if (!handle_)
            throw std::runtime_error("UringReader: failed to open " + path);
    }

    ~UringReader() { if (handle_) uring_reader_close(handle_); }

    UringReader(const UringReader&) = delete;
    UringReader& operator=(const UringReader&) = delete;

    void read(void* dst, size_t n) {
        if (uring_reader_read(handle_, dst, n) < 0)
            throw std::runtime_error("UringReader: read failed");
    }

    void seek(uint64_t offset) { uring_reader_seek(handle_, offset); }
    void skip(size_t n) { uring_reader_skip(handle_, n); }

private:
    struct uring_reader* handle_;
};

class UringWriter {
public:
    explicit UringWriter(const std::string& path, size_t buf_size = 64 * 1024 * 1024)
        : handle_(uring_writer_open(path.c_str(), buf_size)) {
        if (!handle_)
            throw std::runtime_error("UringWriter: failed to open " + path);
    }

    ~UringWriter() { if (handle_) uring_writer_close(handle_); }

    UringWriter(const UringWriter&) = delete;
    UringWriter& operator=(const UringWriter&) = delete;

    void write(const void* src, size_t n) {
        if (uring_writer_write(handle_, src, n) < 0)
            throw std::runtime_error("UringWriter: write failed");
    }

    void seek(uint64_t offset) {
        if (uring_writer_seek(handle_, offset) < 0)
            throw std::runtime_error("UringWriter: seek failed");
    }

    uint64_t tell() { return uring_writer_tell(handle_); }

private:
    struct uring_writer* handle_;
};

} // namespace navis
