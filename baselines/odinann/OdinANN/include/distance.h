#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace pipeann {

  template<typename T>
  inline float compute_l2_norm(const T *vector, uint64_t ndims) {
    float norm = 0.0f;
    for (uint64_t i = 0; i < ndims; i++) {
      norm += (float) (vector[i] * vector[i]);
    }
    return std::sqrt(norm);
  }

  //  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
  template<typename T>
  class Distance {
   public:
    virtual float compare(const T *a, const T *b, unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceCosineInt8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
  };

  class DistanceCosineFloat : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t length) const;
  };

  class SlowDistanceCosineUInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
  };

  class DistanceL2Int8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
  };

  class DistanceL2UInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
  };

  class DistanceL2 : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
  };
}  // namespace pipeann
