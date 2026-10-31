#include "navis/distance.h"
#include <immintrin.h>
#include <stdexcept>

namespace navis {

// ===========================================================================
// SIMD helpers — squared difference primitives
// ===========================================================================

#ifdef __AVX2__

static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

#endif // __AVX2__

// ===========================================================================
// L2 squared distance — float
// ===========================================================================

float DistanceL2Float::compare(const float* a, const float* b, uint32_t dim) const {
#ifdef __AVX512F__
    // AVX-512: 16 floats per iter (vs 8 in AVX-2). At 768D this drops
    // the loop from 96 to 48 iters; FMA latency-throughput-bound.
    const float* end16 = a + ((dim >> 4) << 4);
    const float* end   = a + dim;
    __m512 sum = _mm512_setzero_ps();

    while (a < end16) {
        __m512 va = _mm512_loadu_ps(a);
        __m512 vb = _mm512_loadu_ps(b);
        __m512 d  = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(d, d, sum);
        a += 16;
        b += 16;
    }
    float result = _mm512_reduce_add_ps(sum);

    while (a < end) {
        float d = *a++ - *b++;
        result += d * d;
    }
    return result;
#elif defined(__AVX2__)
    const float* end8 = a + ((dim >> 3) << 3);
    const float* end  = a + dim;
    __m256 sum = _mm256_setzero_ps();

    while (a < end8) {
        __m256 va = _mm256_loadu_ps(a);
        __m256 vb = _mm256_loadu_ps(b);
        __m256 d  = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(d, d, sum);
        a += 8;
        b += 8;
    }
    float result = hsum_avx2(sum);

    while (a < end) {
        float d = *a++ - *b++;
        result += d * d;
    }
    return result;
#else
    float result = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
#endif
}

// ===========================================================================
// L2 squared distance — int8
// ===========================================================================

float DistanceL2Int8::compare(const int8_t* a, const int8_t* b, uint32_t dim) const {
    int32_t result = 0;
#pragma omp simd reduction(+ : result)
    for (int32_t i = 0; i < static_cast<int32_t>(dim); i++) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        result += d * d;
    }
    return static_cast<float>(result);
}

// ===========================================================================
// L2 squared distance — uint8
// ===========================================================================

float DistanceL2UInt8::compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const {
    uint32_t result = 0;
#pragma omp simd reduction(+ : result)
    for (int32_t i = 0; i < static_cast<int32_t>(dim); i++) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        result += static_cast<uint32_t>(d * d);
    }
    return static_cast<float>(result);
}

// ===========================================================================
// Cosine distance — float
// ===========================================================================

float DistanceCosineFloat::compare(const float* a, const float* b, uint32_t dim) const {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
#pragma omp simd reduction(+ : dot, norm_a, norm_b)
    for (int32_t i = 0; i < static_cast<int32_t>(dim); i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 0.0f) ? (1.0f - dot / denom) : 1.0f;
}

// ===========================================================================
// Cosine distance — int8
// ===========================================================================

float DistanceCosineInt8::compare(const int8_t* a, const int8_t* b, uint32_t dim) const {
    int64_t dot = 0, norm_a = 0, norm_b = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot    += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        norm_a += static_cast<int32_t>(a[i]) * static_cast<int32_t>(a[i]);
        norm_b += static_cast<int32_t>(b[i]) * static_cast<int32_t>(b[i]);
    }
    float denom = std::sqrt(static_cast<float>(norm_a)) *
                  std::sqrt(static_cast<float>(norm_b));
    return (denom > 0.0f) ? (1.0f - static_cast<float>(dot) / denom) : 1.0f;
}

// ===========================================================================
// Cosine distance — uint8
// ===========================================================================

float DistanceCosineUInt8::compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const {
    uint64_t dot = 0, norm_a = 0, norm_b = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot    += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
        norm_a += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(a[i]);
        norm_b += static_cast<uint32_t>(b[i]) * static_cast<uint32_t>(b[i]);
    }
    float denom = std::sqrt(static_cast<float>(norm_a)) *
                  std::sqrt(static_cast<float>(norm_b));
    return (denom > 0.0f) ? (1.0f - static_cast<float>(dot) / denom) : 1.0f;
}

// ===========================================================================
// Inner product distance — returns negative dot product (smaller = more similar)
// ===========================================================================

float DistanceIPFloat::compare(const float* a, const float* b, uint32_t dim) const {
    float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
    for (int32_t i = 0; i < static_cast<int32_t>(dim); i++) {
        dot += a[i] * b[i];
    }
    return -dot;
}

float DistanceIPInt8::compare(const int8_t* a, const int8_t* b, uint32_t dim) const {
    int64_t dot = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return -static_cast<float>(dot);
}

float DistanceIPUInt8::compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const {
    uint64_t dot = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }
    return -static_cast<float>(dot);
}

// ===========================================================================
// Factory
// ===========================================================================

template <>
std::unique_ptr<Distance<float>> make_distance(Metric m) {
    switch (m) {
        case Metric::L2:             return std::make_unique<DistanceL2Float>();
        case Metric::COSINE:         return std::make_unique<DistanceCosineFloat>();
        case Metric::INNER_PRODUCT:  return std::make_unique<DistanceIPFloat>();
        default: throw std::invalid_argument("Unsupported metric for float");
    }
}

template <>
std::unique_ptr<Distance<int8_t>> make_distance(Metric m) {
    switch (m) {
        case Metric::L2:             return std::make_unique<DistanceL2Int8>();
        case Metric::COSINE:         return std::make_unique<DistanceCosineInt8>();
        case Metric::INNER_PRODUCT:  return std::make_unique<DistanceIPInt8>();
        default: throw std::invalid_argument("Unsupported metric for int8");
    }
}

template <>
std::unique_ptr<Distance<uint8_t>> make_distance(Metric m) {
    switch (m) {
        case Metric::L2:             return std::make_unique<DistanceL2UInt8>();
        case Metric::COSINE:         return std::make_unique<DistanceCosineUInt8>();
        case Metric::INNER_PRODUCT:  return std::make_unique<DistanceIPUInt8>();
        default: throw std::invalid_argument("Unsupported metric for uint8");
    }
}

} // namespace navis
