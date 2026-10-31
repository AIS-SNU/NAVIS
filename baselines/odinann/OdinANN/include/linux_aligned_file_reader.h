#pragma once

#include "aligned_file_reader.h"
#include "v2/lock_table.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz;
  FileHandle file_desc;
  void *bad_ctx = nullptr;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *get_ctx(int flag = 0);

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false);
  void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false);
  void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
  void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

  // read and update cache.
  void read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr, bool verbose = false);
  // read but not update cache.
  int send_read_no_alloc(IORequest &req, void *ctx);
  int send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx);

  void send_io(IORequest &reqs, void *ctx, bool write);
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
  int poll(void *ctx);
  void poll_all(void *ctx);
  void poll_wait(void *ctx);

  // register thread-id for a context
  void register_thread(int flag = 0);

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();
};

namespace v2 {
  inline std::vector<uint64_t> lockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<IORequest> &reqs) {
    std::vector<uint64_t> ret;
    for (auto &req : reqs) {
      for (uint64_t i = 0; i < req.len; i += SECTOR_LEN) {
        ret.push_back((req.offset + i) / SECTOR_LEN);
      }
    }
    lock_table.sort_and_deduplicate(ret);
    for (auto &x : ret) {
      lock_table.wrlock(x);
    }
    return ret;
  }

  inline void unlockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<uint64_t> &reqs) {
    for (auto &x : reqs) {
      lock_table.unlock(x);
    }
  }

};  // namespace v2
