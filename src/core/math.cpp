#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "vectorforge/core/math.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <limits>
#include <algorithm>

namespace vectorforge {

float compute_distance(const float* a, const float* b, size_t dim, Metric metric) {
    float dist = 0.0f;
    if (metric == Metric::L2) {
#if defined(__AVX2__) && defined(__FMA__)
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        for (int j = 0; j < 8; ++j) {
            dist += temp[j];
        }
        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
#else
        #pragma omp simd reduction(+:dist)
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
#endif
    } else if (metric == Metric::Cosine) {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 sum_dot = _mm256_setzero_ps();
        __m256 sum_norm_a = _mm256_setzero_ps();
        __m256 sum_norm_b = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
            sum_norm_a = _mm256_fmadd_ps(va, va, sum_norm_a);
            sum_norm_b = _mm256_fmadd_ps(vb, vb, sum_norm_b);
        }
        float temp_dot[8], temp_norm_a[8], temp_norm_b[8];
        _mm256_storeu_ps(temp_dot, sum_dot);
        _mm256_storeu_ps(temp_norm_a, sum_norm_a);
        _mm256_storeu_ps(temp_norm_b, sum_norm_b);
        for (int j = 0; j < 8; ++j) {
            dot += temp_dot[j];
            norm_a += temp_norm_a[j];
            norm_b += temp_norm_b[j];
        }
        for (; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
#else
        #pragma omp simd reduction(+:dot, norm_a, norm_b)
        for (size_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
#endif
        if (norm_a == 0.0f || norm_b == 0.0f) {
            dist = 1.0f; 
        } else {
            dist = 1.0f - (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
        }
    }
    return dist;
}

std::vector<float> train_kmeans(const float* data, size_t num_vectors, size_t dim, size_t k, Metric metric, int max_iter) {
    if (num_vectors == 0 || k == 0 || dim == 0) {
        throw std::invalid_argument("Invalid K-Means parameters");
    }
    if (k > num_vectors) {
        k = num_vectors; // Cannot have more clusters than points
    }

    std::vector<float> centroids(k * dim);

    // Initialize centroids by picking random data points
    std::mt19937 gen(42); // deterministic
    std::uniform_int_distribution<size_t> dist(0, num_vectors - 1);
    
    std::vector<size_t> chosen;
    for (size_t i = 0; i < k; ++i) {
        size_t idx = dist(gen);
        // Simple distinct choice
        while(std::find(chosen.begin(), chosen.end(), idx) != chosen.end()) {
            idx = dist(gen);
        }
        chosen.push_back(idx);
        for(size_t d = 0; d < dim; ++d) {
            centroids[i * dim + d] = data[idx * dim + d];
        }
    }

    std::vector<size_t> assignments(num_vectors, 0);

    for (int iter = 0; iter < max_iter; ++iter) {
        int changed = 0;

        // Assign points to nearest centroid
        #pragma omp parallel for reduction(+:changed)
        for (int64_t i = 0; i < static_cast<int64_t>(num_vectors); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            size_t best_c = 0;
            for (size_t c = 0; c < k; ++c) {
                float d = compute_distance(data + i * dim, centroids.data() + c * dim, dim, metric);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = c;
                }
            }
            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                changed++;
            }
        }

        if (changed == 0) {
            break; // Converged
        }

        // Update centroids
        std::vector<float> new_centroids(k * dim, 0.0f);
        std::vector<size_t> counts(k, 0);

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t c = assignments[i];
            counts[c]++;
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[c * dim + d] += data[i * dim + d];
            }
        }

        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    centroids[c * dim + d] = new_centroids[c * dim + d] / counts[c];
                }
            }
        }
    }

    return centroids;
}

} // namespace vectorforge
