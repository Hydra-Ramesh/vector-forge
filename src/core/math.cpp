#include <omp.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "vectorforge/core/math.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <limits>
#include <algorithm>

namespace vectorforge {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#ifndef _MSC_VER
__attribute__((target("avx2,fma")))
#endif
inline float compute_distance_avx2(const float* left_vector, const float* right_vector, size_t dimension, Metric metric) {
    float distance = 0.0f;
    if (metric == Metric::L2) {
        __m256 sum_squared_differences = _mm256_setzero_ps();
        size_t dimension_index = 0;
        for (; dimension_index + 7 < dimension; dimension_index += 8) {
            __m256 left_values = _mm256_loadu_ps(left_vector + dimension_index);
            __m256 right_values = _mm256_loadu_ps(right_vector + dimension_index);
            __m256 difference = _mm256_sub_ps(left_values, right_values);
            sum_squared_differences = _mm256_fmadd_ps(difference, difference, sum_squared_differences);
        }
        float lane_sums[8];
        _mm256_storeu_ps(lane_sums, sum_squared_differences);
        for (int lane_index = 0; lane_index < 8; ++lane_index) {
            distance += lane_sums[lane_index];
        }
        for (; dimension_index < dimension; ++dimension_index) {
            float difference = left_vector[dimension_index] - right_vector[dimension_index];
            distance += difference * difference;
        }
    } else if (metric == Metric::Cosine) {
        float dot_product = 0.0f;
        float left_norm_squared = 0.0f;
        float right_norm_squared = 0.0f;
        __m256 dot_product_sum = _mm256_setzero_ps();
        __m256 left_norm_sum = _mm256_setzero_ps();
        __m256 right_norm_sum = _mm256_setzero_ps();
        size_t dimension_index = 0;
        for (; dimension_index + 7 < dimension; dimension_index += 8) {
            __m256 left_values = _mm256_loadu_ps(left_vector + dimension_index);
            __m256 right_values = _mm256_loadu_ps(right_vector + dimension_index);
            dot_product_sum = _mm256_fmadd_ps(left_values, right_values, dot_product_sum);
            left_norm_sum = _mm256_fmadd_ps(left_values, left_values, left_norm_sum);
            right_norm_sum = _mm256_fmadd_ps(right_values, right_values, right_norm_sum);
        }
        float dot_product_lanes[8], left_norm_lanes[8], right_norm_lanes[8];
        _mm256_storeu_ps(dot_product_lanes, dot_product_sum);
        _mm256_storeu_ps(left_norm_lanes, left_norm_sum);
        _mm256_storeu_ps(right_norm_lanes, right_norm_sum);
        for (int lane_index = 0; lane_index < 8; ++lane_index) {
            dot_product += dot_product_lanes[lane_index];
            left_norm_squared += left_norm_lanes[lane_index];
            right_norm_squared += right_norm_lanes[lane_index];
        }
        for (; dimension_index < dimension; ++dimension_index) {
            dot_product += left_vector[dimension_index] * right_vector[dimension_index];
            left_norm_squared += left_vector[dimension_index] * left_vector[dimension_index];
            right_norm_squared += right_vector[dimension_index] * right_vector[dimension_index];
        }
        if (left_norm_squared == 0.0f || right_norm_squared == 0.0f) {
            distance = 1.0f;
        } else {
            distance = 1.0f - (dot_product / (std::sqrt(left_norm_squared) * std::sqrt(right_norm_squared)));
        }
    }
    return distance;
}
#endif

#if defined(__ARM_NEON)
inline float compute_distance_neon(const float* left_vector, const float* right_vector, size_t dimension, Metric metric) {
    float distance = 0.0f;
    if (metric == Metric::L2) {
        float32x4_t sum_squared_differences = vdupq_n_f32(0.0f);
        size_t dimension_index = 0;
        for (; dimension_index + 3 < dimension; dimension_index += 4) {
            float32x4_t left_values = vld1q_f32(left_vector + dimension_index);
            float32x4_t right_values = vld1q_f32(right_vector + dimension_index);
            float32x4_t difference = vsubq_f32(left_values, right_values);
            sum_squared_differences = vmlaq_f32(sum_squared_differences, difference, difference);
        }
        float lane_sums[4];
        vst1q_f32(lane_sums, sum_squared_differences);
        for (int lane_index = 0; lane_index < 4; ++lane_index) {
            distance += lane_sums[lane_index];
        }
        for (; dimension_index < dimension; ++dimension_index) {
            float difference = left_vector[dimension_index] - right_vector[dimension_index];
            distance += difference * difference;
        }
    } else if (metric == Metric::Cosine) {
        float dot_product = 0.0f;
        float left_norm_squared = 0.0f;
        float right_norm_squared = 0.0f;
        float32x4_t dot_product_sum = vdupq_n_f32(0.0f);
        float32x4_t left_norm_sum = vdupq_n_f32(0.0f);
        float32x4_t right_norm_sum = vdupq_n_f32(0.0f);
        size_t dimension_index = 0;
        for (; dimension_index + 3 < dimension; dimension_index += 4) {
            float32x4_t left_values = vld1q_f32(left_vector + dimension_index);
            float32x4_t right_values = vld1q_f32(right_vector + dimension_index);
            dot_product_sum = vmlaq_f32(dot_product_sum, left_values, right_values);
            left_norm_sum = vmlaq_f32(left_norm_sum, left_values, left_values);
            right_norm_sum = vmlaq_f32(right_norm_sum, right_values, right_values);
        }
        float dot_product_lanes[4], left_norm_lanes[4], right_norm_lanes[4];
        vst1q_f32(dot_product_lanes, dot_product_sum);
        vst1q_f32(left_norm_lanes, left_norm_sum);
        vst1q_f32(right_norm_lanes, right_norm_sum);
        for (int lane_index = 0; lane_index < 4; ++lane_index) {
            dot_product += dot_product_lanes[lane_index];
            left_norm_squared += left_norm_lanes[lane_index];
            right_norm_squared += right_norm_lanes[lane_index];
        }
        for (; dimension_index < dimension; ++dimension_index) {
            dot_product += left_vector[dimension_index] * right_vector[dimension_index];
            left_norm_squared += left_vector[dimension_index] * left_vector[dimension_index];
            right_norm_squared += right_vector[dimension_index] * right_vector[dimension_index];
        }
        if (left_norm_squared == 0.0f || right_norm_squared == 0.0f) {
            distance = 1.0f;
        } else {
            distance = 1.0f - (dot_product / (std::sqrt(left_norm_squared) * std::sqrt(right_norm_squared)));
        }
    }
    return distance;
}
#endif

inline float compute_distance_scalar(const float* left_vector, const float* right_vector, size_t dimension, Metric metric) {
    float distance = 0.0f;
    if (metric == Metric::L2) {
        #pragma omp simd reduction(+:distance)
        for (size_t dimension_index = 0; dimension_index < dimension; ++dimension_index) {
            float difference = left_vector[dimension_index] - right_vector[dimension_index];
            distance += difference * difference;
        }
    } else if (metric == Metric::Cosine) {
        float dot_product = 0.0f;
        float left_norm_squared = 0.0f;
        float right_norm_squared = 0.0f;
        #pragma omp simd reduction(+:dot_product, left_norm_squared, right_norm_squared)
        for (size_t dimension_index = 0; dimension_index < dimension; ++dimension_index) {
            dot_product += left_vector[dimension_index] * right_vector[dimension_index];
            left_norm_squared += left_vector[dimension_index] * left_vector[dimension_index];
            right_norm_squared += right_vector[dimension_index] * right_vector[dimension_index];
        }
        if (left_norm_squared == 0.0f || right_norm_squared == 0.0f) {
            distance = 1.0f;
        } else {
            distance = 1.0f - (dot_product / (std::sqrt(left_norm_squared) * std::sqrt(right_norm_squared)));
        }
    }
    return distance;
}

DistanceFunction get_distance_function() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(_MSC_VER)
    return compute_distance_avx2; // On MSVC, assume AVX2 if compiled with /arch:AVX2
#else
    if (__builtin_cpu_supports("avx2")) {
        return compute_distance_avx2;
    }
#endif
#elif defined(__ARM_NEON)
    return compute_distance_neon;
#endif
    return compute_distance_scalar;
}

float compute_distance(const float* a, const float* b, size_t dim, Metric metric) {
    static DistanceFunction func = get_distance_function();
    return func(a, b, dim, metric);
}

float compute_distance(const std::vector<float>& a, const std::vector<float>& b, size_t dim, Metric metric) {
    return compute_distance(a.data(), b.data(), dim, metric);
}

float compute_distance(const float* a, const std::vector<float>& b, size_t dim, Metric metric) {
    return compute_distance(a, b.data(), dim, metric);
}

std::vector<float> train_kmeans(const float* data, size_t vector_count, size_t dimension, size_t cluster_count, Metric metric, int max_iterations) {
    if (vector_count == 0 || cluster_count == 0 || dimension == 0) {
        throw std::invalid_argument("Invalid K-Means parameters");
    }
    if (cluster_count > vector_count) {
        cluster_count = vector_count;
    }

    std::vector<float> centroids(cluster_count * dimension);

    std::mt19937 random_generator(42);
    std::uniform_int_distribution<size_t> random_index(0, vector_count - 1);
    
    std::vector<size_t> chosen;
    for (size_t centroid_index = 0; centroid_index < cluster_count; ++centroid_index) {
        size_t source_index = random_index(random_generator);
        while (std::find(chosen.begin(), chosen.end(), source_index) != chosen.end()) {
            source_index = random_index(random_generator);
        }
        chosen.push_back(source_index);
        for (size_t dimension_index = 0; dimension_index < dimension; ++dimension_index) {
            centroids[centroid_index * dimension + dimension_index] = data[source_index * dimension + dimension_index];
        }
    }

    std::vector<size_t> assignments(vector_count, 0);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        int changed_assignments = 0;

        // Assign points to nearest centroid
        #pragma omp parallel for reduction(+:changed_assignments)
        for (int64_t vector_index = 0; vector_index < static_cast<int64_t>(vector_count); ++vector_index) {
            float nearest_distance = std::numeric_limits<float>::max();
            size_t nearest_centroid = 0;
            for (size_t centroid_index = 0; centroid_index < cluster_count; ++centroid_index) {
                float distance_to_centroid = compute_distance(data + vector_index * dimension, centroids.data() + centroid_index * dimension, dimension, metric);
                if (distance_to_centroid < nearest_distance) {
                    nearest_distance = distance_to_centroid;
                    nearest_centroid = centroid_index;
                }
            }
            if (assignments[vector_index] != nearest_centroid) {
                assignments[vector_index] = nearest_centroid;
                changed_assignments++;
            }
        }

        if (changed_assignments == 0) {
            break;
        }

        // Update centroids
        std::vector<float> new_centroids(cluster_count * dimension, 0.0f);
        std::vector<size_t> counts(cluster_count, 0);

        for (size_t vector_index = 0; vector_index < vector_count; ++vector_index) {
            size_t centroid_index = assignments[vector_index];
            counts[centroid_index]++;
            for (size_t dimension_index = 0; dimension_index < dimension; ++dimension_index) {
                new_centroids[centroid_index * dimension + dimension_index] += data[vector_index * dimension + dimension_index];
            }
        }

        for (size_t centroid_index = 0; centroid_index < cluster_count; ++centroid_index) {
            if (counts[centroid_index] > 0) {
                for (size_t dimension_index = 0; dimension_index < dimension; ++dimension_index) {
                    centroids[centroid_index * dimension + dimension_index] = new_centroids[centroid_index * dimension + dimension_index] / counts[centroid_index];
                }
            }
        }
    }

    return centroids;
}

std::vector<uint8_t> binarize_vector(const float* vec, size_t dim) {
    size_t byte_count = (dim + 7) / 8;
    std::vector<uint8_t> binarized(byte_count, 0);
    for (size_t i = 0; i < dim; ++i) {
        if (vec[i] > 0.0f) {
            binarized[i / 8] |= (1 << (i % 8));
        }
    }
    return binarized;
}

float compute_distance_hamming(const uint8_t* a, const uint8_t* b, size_t original_dim) {
    size_t byte_count = (original_dim + 7) / 8;
    size_t num_uint64 = byte_count / 8;
    size_t remainder = byte_count % 8;
    
    const uint64_t* a_64 = reinterpret_cast<const uint64_t*>(a);
    const uint64_t* b_64 = reinterpret_cast<const uint64_t*>(b);
    
    uint64_t distance = 0;
    
    for (size_t i = 0; i < num_uint64; ++i) {
#if defined(_MSC_VER)
        distance += __popcnt64(a_64[i] ^ b_64[i]);
#else
        distance += __builtin_popcountll(a_64[i] ^ b_64[i]);
#endif
    }
    
    size_t offset = num_uint64 * 8;
    for (size_t i = 0; i < remainder; ++i) {
#if defined(_MSC_VER)
        distance += __popcnt16(a[offset + i] ^ b[offset + i]);
#else
        distance += __builtin_popcount(a[offset + i] ^ b[offset + i]);
#endif
    }
    
    return static_cast<float>(distance);
}

} // namespace vectorforge
