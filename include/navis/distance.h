#pragma once

#include "navis/types.h"
#include <cmath>
#include <cstdint>
#include <memory>

namespace navis {

// ---------------------------------------------------------------------------
// Distance — abstract base for vector distance computation
// ---------------------------------------------------------------------------
template <typename T>
class Distance {
public:
    virtual ~Distance() = default;
    virtual float compare(const T* a, const T* b, uint32_t dim) const = 0;
};

// ---------------------------------------------------------------------------
// L2 squared distance implementations
// ---------------------------------------------------------------------------
class DistanceL2Float : public Distance<float> {
public:
    float compare(const float* a, const float* b, uint32_t dim) const override;
};

class DistanceL2Int8 : public Distance<int8_t> {
public:
    float compare(const int8_t* a, const int8_t* b, uint32_t dim) const override;
};

class DistanceL2UInt8 : public Distance<uint8_t> {
public:
    float compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const override;
};

// ---------------------------------------------------------------------------
// Cosine distance implementations (returns 1 - cosine_similarity)
// ---------------------------------------------------------------------------
class DistanceCosineFloat : public Distance<float> {
public:
    float compare(const float* a, const float* b, uint32_t dim) const override;
};

class DistanceCosineInt8 : public Distance<int8_t> {
public:
    float compare(const int8_t* a, const int8_t* b, uint32_t dim) const override;
};

class DistanceCosineUInt8 : public Distance<uint8_t> {
public:
    float compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const override;
};

// ---------------------------------------------------------------------------
// Inner product distance (returns -dot so smaller = more similar)
// ---------------------------------------------------------------------------
class DistanceIPFloat : public Distance<float> {
public:
    float compare(const float* a, const float* b, uint32_t dim) const override;
};

class DistanceIPInt8 : public Distance<int8_t> {
public:
    float compare(const int8_t* a, const int8_t* b, uint32_t dim) const override;
};

class DistanceIPUInt8 : public Distance<uint8_t> {
public:
    float compare(const uint8_t* a, const uint8_t* b, uint32_t dim) const override;
};

// ---------------------------------------------------------------------------
// Factory: create distance function from metric and data type
// ---------------------------------------------------------------------------
template <typename T>
std::unique_ptr<Distance<T>> make_distance(Metric m);

// ---------------------------------------------------------------------------
// Utility: L2 norm
// ---------------------------------------------------------------------------
template <typename T>
inline float compute_l2_norm(const T* vec, uint64_t dim) {
    float norm = 0.0f;
    for (uint64_t i = 0; i < dim; i++) {
        norm += static_cast<float>(vec[i]) * static_cast<float>(vec[i]);
    }
    return std::sqrt(norm);
}

} // namespace navis
