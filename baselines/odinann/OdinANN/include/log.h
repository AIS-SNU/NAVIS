#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_

#ifdef USE_GLOG
#include <glog/logging.h>
#else
#include <stdarg.h>

#include <iostream>
#include <sstream>
#include <string>

// Attention: stringstream has multi-thread performance issues,
// but never mind because no one will frequently log in the multi-threading
// case.
class LogStream {
 public:
  static constexpr int kMaxFmtLen = 255;
  LogStream(const char *file, int line, const char *level) {
    std::string filename = std::string(file);
    size_t pos = filename.find_last_of('/');
    if (pos != std::string::npos) {
      filename = filename.substr(pos + 1);
    }
    ss_ << "[" + filename + ":" + std::to_string(line) + ":" + level + "] ";
  }

  ~LogStream() {
    ss_ << std::endl;
    std::cerr << ss_.str();
  }

  template<class T>
  inline LogStream &operator<<(const T &t) {
    ss_ << t;
    return *this;
  }

  inline LogStream &operator()(const char *format, ...) {
    char va_buf[1 + kMaxFmtLen];
    va_list args;
    va_start(args, format);
    sprintf(va_buf, format, args);
    va_end(args);
    ss_ << va_buf;
    return *this;
  }

 private:
  std::stringstream ss_;
};

#define LOG(level) \
  if (true)        \
  LogStream(__FILE__, __LINE__, #level)
#ifdef NDEBUG
class NullStream {
 public:
  template<typename T>
  inline NullStream &operator<<(const T &) {
    return *this;
  }

  inline NullStream &operator()(const char *, ...) {
    return *this;
  }
};
#define DLOG(level) NullStream()
#else
#define DLOG(level) LOG(level)
#endif  // NDEBUG
#endif  // USE_GLOG

#endif  // UTILS_LOG_H_