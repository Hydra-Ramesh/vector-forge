#include <omp.h>
#include "vectorforge/core/pq.hpp"
#include "vectorforge/core/math.hpp"
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace vectorforge {

ProductQuantizer::ProductQuantizer(size_t dim, size_t m, size_t k_sub)
    : dim_(dim), m_(m), k_sub_(k_sub), is_trained_(false) {
    if (dim % m != 0) {
        throw std::invalid_argument("Dimension must be perfectly divisible by m");
    }
    if (k_sub > 256) {
        throw std::invalid_argument("k_sub > 256 is not supported (requires larger than uint8_t)");
    }
    sub_dim_ = dim / m;
}

void ProductQuantizer::train(const std::vector<Vector>& training_data, Metric metric) {
    size_t vector_count = training_data.size();
    std::vector<float> flat_data(vector_count * dim_);
    for (size_t vector_index = 0; vector_index < vector_count; ++vector_index) {
        std::copy(training_data[vector_index].begin(), training_data[vector_index].end(), flat_data.begin() + vector_index * dim_);
    }
    train_flat(flat_data.data(), vector_count, metric);
}

void ProductQuantizer::train_flat(const float* data, size_t num_vectors, Metric metric) {
    centroids_.resize(m_ * k_sub_ * sub_dim_);

    #pragma omp parallel for
    for (int subquantizer_index = 0; subquantizer_index < static_cast<int>(m_); ++subquantizer_index) {
        std::vector<float> subvector_data(num_vectors * sub_dim_);
        for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
            for (size_t dimension_index = 0; dimension_index < sub_dim_; ++dimension_index) {
                subvector_data[vector_index * sub_dim_ + dimension_index] = data[vector_index * dim_ + subquantizer_index * sub_dim_ + dimension_index];
            }
        }

        auto subquantizer_centroids = train_kmeans(subvector_data.data(), num_vectors, sub_dim_, k_sub_, metric);
        
        for (size_t centroid_index = 0; centroid_index < k_sub_; ++centroid_index) {
            for (size_t dimension_index = 0; dimension_index < sub_dim_; ++dimension_index) {
                centroids_[subquantizer_index * k_sub_ * sub_dim_ + centroid_index * sub_dim_ + dimension_index] = subquantizer_centroids[centroid_index * sub_dim_ + dimension_index];
            }
        }
    }
    is_trained_ = true;
}

std::vector<uint8_t> ProductQuantizer::encode(const float* vector) const {
    if (!is_trained_) throw std::runtime_error("PQ not trained");

    std::vector<uint8_t> code(m_);
    for (size_t subquantizer_index = 0; subquantizer_index < m_; ++subquantizer_index) {
        float nearest_distance = std::numeric_limits<float>::max();
        uint8_t nearest_centroid = 0;
        const float* subvector = vector + subquantizer_index * sub_dim_;
        const float* subquantizer_centroids = centroids_.data() + subquantizer_index * k_sub_ * sub_dim_;

        for (size_t centroid_index = 0; centroid_index < k_sub_; ++centroid_index) {
            float distance = compute_distance(subvector, subquantizer_centroids + centroid_index * sub_dim_, sub_dim_, Metric::L2);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_centroid = static_cast<uint8_t>(centroid_index);
            }
        }
        code[subquantizer_index] = nearest_centroid;
    }
    return code;
}

std::vector<float> ProductQuantizer::compute_lut(const float* query, Metric metric) const {
    if (!is_trained_) throw std::runtime_error("PQ not trained");

    std::vector<float> lut(m_ * k_sub_);
    for (size_t subquantizer_index = 0; subquantizer_index < m_; ++subquantizer_index) {
        const float* subquery = query + subquantizer_index * sub_dim_;
        const float* subquantizer_centroids = centroids_.data() + subquantizer_index * k_sub_ * sub_dim_;

        for (size_t centroid_index = 0; centroid_index < k_sub_; ++centroid_index) {
            lut[subquantizer_index * k_sub_ + centroid_index] = compute_distance(subquery, subquantizer_centroids + centroid_index * sub_dim_, sub_dim_, metric);
        }
    }
    return lut;
}

float ProductQuantizer::compute_adc(const uint8_t* code, const float* lut) const {
    float distance = 0.0f;
    if (m_ == 8) {
        for (size_t subquantizer_index = 0; subquantizer_index < 8; ++subquantizer_index) distance += lut[subquantizer_index * k_sub_ + code[subquantizer_index]];
    } else if (m_ == 16) {
        for (size_t subquantizer_index = 0; subquantizer_index < 16; ++subquantizer_index) distance += lut[subquantizer_index * k_sub_ + code[subquantizer_index]];
    } else {
        #pragma omp simd reduction(+:distance)
        for (size_t subquantizer_index = 0; subquantizer_index < m_; ++subquantizer_index) {
            distance += lut[subquantizer_index * k_sub_ + code[subquantizer_index]];
        }
    }
    return distance;
}

} // namespace vectorforge
