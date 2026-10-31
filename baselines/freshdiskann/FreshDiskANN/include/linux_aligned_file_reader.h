// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#ifndef USE_AIO
#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  FileHandle   file_desc;
  io_context_t bad_ctx = nullptr;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  IOContext get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);
  void write(std::vector<AlignedRead> &write_reqs, IOContext &ctx,
             bool async = false);
  void read_fd(int fd, std::vector<AlignedRead> &read_reqs, IOContext &ctx);
  void write_fd(int fd, std::vector<AlignedRead> &write_reqs, IOContext &ctx);
};
#else
#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t     file_sz;
  FileHandle   file_desc;
  io_context_t bad_ctx = (io_context_t) -1;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  IOContext &get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false);
  void write(std::vector<AlignedRead> &write_reqs, IOContext &ctx,
             bool async = false);
  void read_fd(int fd, std::vector<AlignedRead> &read_reqs, IOContext &ctx);
  void write_fd(int fd, std::vector<AlignedRead> &write_reqs, IOContext &ctx);

  void sequential_write(AlignedRead &write_req, IOContext &ctx);
};
#endif  // USE_AIO

#endif
