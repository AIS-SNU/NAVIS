#ifndef USE_AIO
#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "liburing.h"

#define MAX_EVENTS 1024

namespace {
  constexpr uint64_t kNoUserData = 0;
  void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    while (true) {
      for (uint64_t j = 0; j < reqs.size(); j++) {
        auto sqe = io_uring_get_sqe(ring);
        sqe->user_data = kNoUserData;
        if (write) {
          io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        } else {
          io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      bool fail = false;
      for (uint64_t j = 0; j < reqs.size(); j++) {
        int ret = 0;
        do {
          ret = io_uring_wait_cqe(ring, &cqe);
        } while (ret == -EINTR);

        if (ret < 0 || cqe->res < 0) {
          fail = true;
          LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                     << reqs[j].len << " " << reqs[j].offset;
          break;  // CQE broken.
        }
        io_uring_cqe_seen(ring, cqe);
      }
      if (!fail) {  // repeat until no fails.
        break;
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = new io_uring();
    io_uring_queue_init(MAX_EVENTS, ioctx::ring, flag);
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_uring_queue_exit(ioctx::ring);
  delete ioctx::ring;
  ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  auto sqe = io_uring_get_sqe(ring);
  req.finished = false;
  sqe->user_data = (uint64_t) &req;
  if (write) {
    io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
  }
  io_uring_submit(ring);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    reqs[j].finished = false;
    sqe->user_data = (uint64_t) &reqs[j];
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
  }
  io_uring_submit(ring);
}

int LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = io_uring_peek_cqe(ring, &cqe);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = 0;
  do {
    ret = io_uring_wait_cqe(ring, &cqe);
  } while (ret == -EINTR);
  if (ret < 0 || cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
}

#else
#include "linux_aligned_file_reader.h"

#include <libaio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb iocb_t;

  void execute_io(void *ctx, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS), (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *> cbs(n_ops, nullptr);
      std::vector<io_event_t> evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                        reqs[j + iter * MAX_EVENTS].offset);
        }
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno << "="
                     << ::strerror((int) -ret) << ", try #" << n_tries + 1 << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents((io_context_t) ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            LOG(ERROR) << "io_getevents() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                       << "=" << ::strerror((int) -ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_context_t ctx;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ctx == nullptr)) {
    register_thread(flag);
  }
  return (void *) ioctx::ctx;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ctx == nullptr) {
    int ret = io_setup(MAX_EVENTS, &ioctx::ctx);
    if (ret != 0) {
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_destroy((io_context_t) this->get_ctx());
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  uint64_t n_ops = std::min(reqs.size(), (uint64_t) MAX_EVENTS);
  std::vector<iocb_t *> cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (uint64_t j = 0; j < n_ops; j++) {
    if (write) {
      io_prep_pwrite(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_prep_pread(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    reqs[j].finished = false;  // reset finished flag
  }

  for (uint64_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  // issue reads
  int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
  // if requests didn't get accepted
  if (ret != (int64_t) n_ops) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", " << strerror(errno);
  }
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  iocb_t cb;
  req.finished = false;  // reset finished flag
  if (write) {
    io_prep_pwrite(&cb, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_prep_pread(&cb, this->file_desc, req.buf, req.len, req.offset);
  }
  cb.data = (void *) &req;  // set user data to point to the request

  iocb_t *cbs[1] = {&cb};  // create an array of iocb_t pointers
  int ret = io_submit((io_context_t) ctx, 1, cbs);
  if (ret != 1) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  // Poll a single completed IO request in the io_uring context.
  io_event event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, 1, &event, nullptr);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (ret) {
    IORequest *req = (IORequest *) event.data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  // Poll all completed IO requests in the io_uring context.
  static __thread io_event_t evts[MAX_EVENTS];
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_event_t event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 1, 1, &event, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  IORequest *req = (IORequest *) event.data;
  if (req != nullptr) {
    req->finished = true;
  }
}

#endif

int LinuxAlignedFileReader::send_read_no_alloc(IORequest &req, void *ring) {
#ifndef READ_ONLY_TESTS
  if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
    send_io(req, ring, false);
  } else {
    req.finished = true;  // mark as finished for cache miss
  }
#else
  send_io(req, ring, false);
#endif
  return 1;
}

int LinuxAlignedFileReader::send_read_no_alloc(std::vector<IORequest> &reqs, void *ring) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;
  // fetch from cache.
  for (auto &req : reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len != SECTOR_LEN) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
      disk_read_reqs.push_back(req);
    }
  }
  send_io(disk_read_reqs, ring, false);
  return disk_read_reqs.size();
#else
  send_io(reqs, ring, false);
  return reqs.size();
#endif
}

void LinuxAlignedFileReader::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref,
                                        bool verbose) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;

  // TODO(gh): introduce size_per_io to cache.
  for (auto &req : read_reqs) {
    if (req.offset % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
      crash();
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true)) {
      disk_read_reqs.push_back(req);
    }
  }

  if (verbose) {
    LOG(INFO) << "#disk read reqs: " << disk_read_reqs.size();
  }

  if (disk_read_reqs.size() > 0) {
    read(disk_read_reqs, ctx);
    for (auto &req : disk_read_reqs) {
      v2::cache.put(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true);
    }
  }

  // ref.
  if (page_ref != nullptr) {
    for (auto &req : read_reqs) {
      page_ref->push_back(req.offset / SECTOR_LEN);
    }
  }
#else
  read(read_reqs, ctx);
#endif
}