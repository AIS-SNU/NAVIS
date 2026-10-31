#include <immintrin.h>
#include <iostream>
#include <vector>

#include "utils.h"
#include "distance.h"

namespace pipeann {
  template<typename T>
  float compute_cosine_similarity(const T *left, const T *right, uint64_t ndims) {
    float left_norm = compute_l2_norm<T>(left, ndims);
    float right_norm = compute_l2_norm<T>(right, ndims);
    float dot = 0.0f;
    for (uint64_t i = 0; i < ndims; i++) {
      dot += (float) (left[i] * right[i]);
    }
    float cos_sim = dot / (left_norm * right_norm);
    return cos_sim;
  }

  std::vector<float> compute_cosine_similarity_batch(const float *query, const unsigned *indices, const float *all_data,
                                                     const unsigned ndims, const unsigned npts) {
    std::vector<float> cos_dists;
    cos_dists.reserve(npts);

    for (size_t i = 0; i < npts; i++) {
      const float *point = all_data + (size_t) (indices[i]) * (size_t) (ndims);
      cos_dists.push_back(compute_cosine_similarity<float>(point, query, ndims));
    }
    return cos_dists;
  }

  // Cosine similarity.
  float DistanceCosineInt8::compare(const int8_t *a, const int8_t *b, uint32_t length) const {
    return pipeann::compute_cosine_similarity(a, b, length);
  }

  float DistanceCosineFloat::compare(const float *a, const float *b, uint32_t length) const {
    return pipeann::compute_cosine_similarity(a, b, length);
  }

  float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b, uint32_t length) const {
    int magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += ((uint32_t) a[i]) * ((uint32_t) a[i]);
      magB += ((uint32_t) b[i]) * ((uint32_t) b[i]);
      scalarProduct += ((uint32_t) a[i]) * ((uint32_t) b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
  }

#ifdef USE_AVX512  // AVX512 support.
#define DIFF128 diff128
#define DIFF256 diff256

  inline __m128 _mm_sqdf_epi8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
  }

  inline __m128 _mm_mul_epu8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
  }

  inline __m128 _mm_sqdf_epu8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
  }

  inline __m128 _mm_mul_epi16(__m128i X, __m128i Y) {
    return _mm_cvtepi32_ps(_mm_madd_epi16(X, Y));
  }

  inline __m128 _mm_sqdf_epi16(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi16(X, zero);
    __m128i sign_y = _mm_cmplt_epi16(Y, zero);

    __m128i xlo = _mm_unpacklo_epi16(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi16(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi16(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi16(Y, sign_y);

    __m128 dlo = _mm_cvtepi32_ps(_mm_sub_epi32(xlo, ylo));
    __m128 dhi = _mm_cvtepi32_ps(_mm_sub_epi32(xhi, yhi));

    return _mm_add_ps(_mm_mul_ps(dlo, dlo), _mm_mul_ps(dhi, dhi));
  }

  inline __m128 _mm_sqdf_ps(__m128 X, __m128 Y) {
    __m128 d = _mm_sub_ps(X, Y);
    return _mm_mul_ps(d, d);
  }

  inline __m256 _mm256_sqdf_epi8(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
  }

  inline __m256 _mm256_mul_epu8(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
  }

  inline __m256 _mm256_sqdf_epu8(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
  }

  inline __m256 _mm256_mul_epi16(__m256i X, __m256i Y) {
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(X, Y));
  }

  inline __m256 _mm256_sqdf_epi16(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi16(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi16(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi16(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi16(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi16(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi16(Y, sign_y);

    __m256 dlo = _mm256_cvtepi32_ps(_mm256_sub_epi32(xlo, ylo));
    __m256 dhi = _mm256_cvtepi32_ps(_mm256_sub_epi32(xhi, yhi));

    return _mm256_add_ps(_mm256_mul_ps(dlo, dlo), _mm256_mul_ps(dhi, dhi));
  }

  inline __m256 _mm256_sqdf_ps(__m256 X, __m256 Y) {
    __m256 d = _mm256_sub_ps(X, Y);
    return _mm256_mul_ps(d, d);
  }

  // Do not use intrinsics not supported by old MS compiler version
  inline __m512 _mm512_mul_epi8(__m512i X, __m512i Y) {
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask(zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask(zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
  }

  inline __m512 _mm512_sqdf_epi8(__m512i X, __m512i Y) {
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask(zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask(zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
  }

  inline __m512 _mm512_mul_epu8(__m512i X, __m512i Y) {
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
  }

  inline __m512 _mm512_sqdf_epu8(__m512i X, __m512i Y) {
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
  }

  inline __m512 _mm512_mul_epi16(__m512i X, __m512i Y) {
    return _mm512_cvtepi32_ps(_mm512_madd_epi16(X, Y));
  }

  inline __m512 _mm512_sqdf_epi16(__m512i X, __m512i Y) {
    __m512i zero = _mm512_setzero_si512();

    __mmask32 sign_x_mask = _mm512_cmpgt_epi16_mask(zero, X);
    __mmask32 sign_y_mask = _mm512_cmpgt_epi16_mask(zero, Y);

    __m512i sign_x = _mm512_movm_epi16(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi16(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi16(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi16(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi16(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi16(Y, sign_y);

    __m512 dlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(xlo, ylo));
    __m512 dhi = _mm512_cvtepi32_ps(_mm512_sub_epi32(xhi, yhi));

    return _mm512_add_ps(_mm512_mul_ps(dlo, dlo), _mm512_mul_ps(dhi, dhi));
  }

  inline __m512 _mm512_sqdf_ps(__m512 X, __m512 Y) {
    __m512 d = _mm512_sub_ps(X, Y);
    return _mm512_mul_ps(d, d);
  }

#define REPEAT(type, ctype, delta, load, exec, acc, result) \
  {                                                         \
    type c1 = load((ctype *) (pX));                         \
    type c2 = load((ctype *) (pY));                         \
    pX += delta;                                            \
    pY += delta;                                            \
    result = acc(result, exec(c1, c2));                     \
  }

  // L2 distance functions.
  float DistanceL2Int8::compare(const int8_t *pX, const int8_t *pY, uint32_t length) const {
    const std::int8_t *pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t *pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t *pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t *pEnd1 = pX + length;

    const std::int8_t *pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
      REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));

    while (pX < pEnd32) {
      REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
      REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
      float c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
    }
    while (pX < pEnd1) {
      float c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
    }
    return diff;
  }

  float DistanceL2UInt8::compare(const uint8_t *pX, const uint8_t *pY, uint32_t length) const {
    const std::uint8_t *pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t *pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t *pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t *pEnd1 = pX + length;

    const std::uint8_t *pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
      REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));

    while (pX < pEnd32) {
      REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
      REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
      float c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
      c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
    }
    while (pX < pEnd1) {
      float c1 = ((float) (*pX++) - (float) (*pY++));
      diff += c1 * c1;
    }
    return diff;
  }

  float DistanceL2::compare(const float *pX, const float *pY, uint32_t length) const {
    const float *pEnd8 = pX + ((length >> 3) << 3);
    const float *pEnd4 = pX + ((length >> 2) << 2);
    const float *pEnd1 = pX + length;

    const float *pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
      REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_sqdf_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));

    while (pX < pEnd8) {
      REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4) {
      REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
      float c1 = (*pX++) - (*pY++);
      diff += c1 * c1;
    }
    return diff;
  }
#else  // without AVX512 support.

#ifdef USE_AVX2
  static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
  }
#endif

  // L2 distance functions.
  float DistanceL2Int8::compare(const int8_t *a, const int8_t *b, uint32_t size) const {
    int32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) * ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
  }

  float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b, uint32_t size) const {
    uint32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) * ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
  }

  float DistanceL2::compare(const float *a, const float *b, uint32_t size) const {
    a = (const float *) __builtin_assume_aligned(a, 32);
    b = (const float *) __builtin_assume_aligned(b, 32);

    float result = 0;
#ifdef USE_AVX2
    // assume size is divisible by 8
    uint16_t niters = (uint16_t) (size / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
      // scope is a[8j:8j+7], b[8j:8j+7]
      // load a_vec
      if (j < (niters - 1)) {
        _mm_prefetch((char *) (a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (b + 8 * (j + 1)), _MM_HINT_T0);
      }
      __m256 a_vec = _mm256_load_ps(a + 8 * j);
      // load b_vec
      __m256 b_vec = _mm256_load_ps(b + 8 * j);
      // a_vec - b_vec
      __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
      /*
      // (a_vec - b_vec)**2
          __m256 tmp_vec2 = _mm256_mul_ps(tmp_vec, tmp_vec);
      // accumulate sum
          sum = _mm256_add_ps(sum, tmp_vec2);
      */
      // sum = (tmp_vec**2) + sum
      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }

    // horizontal add sum
    result = _mm256_reduce_add_ps(sum);
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
    for (_s32 i = 0; i < (_s32) size; i++) {
      result += (a[i] - b[i]) * (a[i] - b[i]);
    }
#endif
    return result;
  }
#endif  // USE_AVX512
}  // namespace pipeann