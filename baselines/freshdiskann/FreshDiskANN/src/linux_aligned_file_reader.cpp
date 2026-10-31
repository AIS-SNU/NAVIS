#ifndef USE_AIO
#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "liburing.h"
#include "tsl/robin_map.h"
#include "utils.h"

#define MAX_EVENTS 256

namespace {
  constexpr uint64_t kNoUserData = 0;
  void execute_io(io_context_t ring, int fd, std::vector<AlignedRead> &reqs,
                  uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    for (uint64_t i = 0; i < reqs.size(); i += MAX_EVENTS) {
      auto ed = std::min(reqs.size(), i + MAX_EVENTS);
      for (uint64_t j = i; j < ed; j++) {
        auto sqe = io_uring_get_sqe(ring);
        sqe->user_data = kNoUserData;
        if (write) {
          io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len,
                              reqs[j].offset);
        } else {
          io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      for (uint64_t j = i; j < ed; j++) {
        int ret = io_uring_wait_cqe(ring, &cqe);
        // LOG(INFO) << "Wait CQE ring " << ring << " ret " << ret;
        io_uring_cqe_seen(ring, cqe);
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
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_context_t ring = nullptr;
};

io_context_t LinuxAlignedFileReader::get_ctx() {
  // if (ioctx::ring == nullptr) {
  //   ioctx::ring = new io_uring();
  //   io_uring_queue_init(MAX_EVENTS, ioctx::ring, 0);
  // }
  auto ring = new io_uring();
  io_uring_queue_init(MAX_EVENTS, ring, 0);
  return ring;
  // return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread() {
  // if (ioctx::ring == nullptr) {
  //   ioctx::ring = new io_uring();
  //   io_uring_queue_init(MAX_EVENTS, ioctx::ring, 0);
  // }
}

void LinuxAlignedFileReader::deregister_thread() {
  // io_uring_queue_exit(ioctx::ring);
  // delete ioctx::ring;
  // ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::open(const std::string &fname,
                                  bool               enable_writes = false,
                                  bool               enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags);
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

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  IOContext &ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<AlignedRead> &write_reqs,
                                   IOContext &ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int                       fd,
                                     std::vector<AlignedRead> &read_reqs,
                                     IOContext                &ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int                       fd,
                                      std::vector<AlignedRead> &write_reqs,
                                      IOContext                &ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}
#else
#include "linux_aligned_file_reader.h"

#include <aio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 256

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &reqs,
                  uint64_t n_retries = 0, bool write = false) {
#ifdef DEBUG
    for (auto &req : reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // diskann::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS),
                                (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *>    cbs(n_ops, nullptr);
      std::vector<io_event_t>  evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf,
                         reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf,
                        reqs[j + iter * MAX_EVENTS].len,
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
        int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          std::cerr << "io_submit() failed; returned " << ret
                    << ", expected=" << n_ops << ", ernno=" << errno << "="
                    << ::strerror((int) -ret) << ", try #" << n_tries + 1
                    << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                             nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            std::cerr << "io_getevents() failed; returned " << ret
                      << ", expected=" << n_ops << ", ernno=" << errno << "="
                      << ::strerror((int) -ret) << ", try #" << n_tries + 1
                      << std::endl;
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
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

io_context_t &LinuxAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void LinuxAlignedFileReader::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()) {
    return;
  }
  io_context_t ctx = 0;
  int          ret = io_setup(MAX_EVENTS, &ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
              << ":" << ::strerror(errno) << std::endl;
  } else {
    ctx_map[my_id] = ctx;
  }
  lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  io_destroy(ctx);
  //  assert(ret == 0);
  lk.lock();
  ctx_map.erase(my_id);
  //  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);

  for (auto &iter : ctx_map) {
    io_context_t ctx = iter.second;
    io_destroy(ctx);
    //  assert(ret == 0);
    //    std::cerr << "returned ctx from thread-id:" << iter.first <<
    //    std::endl;
  }
  ctx_map.clear();
  lk.unlock();
}

void LinuxAlignedFileReader::open(const std::string &fname,
                                  bool               enable_writes = false,
                                  bool               enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags);
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

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                  IOContext &ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<AlignedRead> &write_reqs,
                                   IOContext &ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int                       fd,
                                     std::vector<AlignedRead> &read_reqs,
                                     IOContext                &ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int                       fd,
                                      std::vector<AlignedRead> &write_reqs,
                                      IOContext                &ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::sequential_write(AlignedRead &write_req,
                                              IOContext   &ctx) {
  assert(this->file_desc != -1);
  // check inputs
  assert(IS_ALIGNED(write_req.offset, 4096));
  assert(IS_ALIGNED(write_req.buf, 4096));
  assert(IS_ALIGNED(write_req.len, 4096));

  // create write request
  io_event_t  evt;
  struct iocb cb;
  iocb_t     *cbs = &cb;
  io_prep_pwrite(&cb, this->file_desc, write_req.buf, write_req.len,
                 write_req.offset);

  uint64_t n_tries = 0;
  // issue reads
  int64_t ret = io_submit(ctx, (int64_t) 1, &cbs);
  // if requests didn't get accepted
  if (ret != (int64_t) 1) {
    std::cerr << "io_submit() failed; returned " << ret << ", expected=" << 1
              << ", ernno=" << errno << "=" << ::strerror((int) -ret)
              << ", try #" << n_tries + 1;
    diskann::cout << "ctx: " << ctx << "\n";
    exit(-1);
  } else {
    // wait on io_getevents
    ret = io_getevents(ctx, (int64_t) 1, (int64_t) 1, &evt, nullptr);
    // if requests didn't complete
    if (ret != (int64_t) 1) {
      std::cerr << "io_getevents() failed; returned " << ret
                << ", expected=" << 1 << ", ernno=" << errno << "="
                << ::strerror((int) -ret) << ", try #" << n_tries + 1;
      exit(-1);
    }
  }
}
#endif