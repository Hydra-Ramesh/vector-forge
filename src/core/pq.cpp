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
    size_t num_vectors = training_data.size();
    std::vector<float> flat_data(num_vectors * dim_);
    for (size_t i = 0; i < num_vectors; ++i) {
        std::copy(training_data[i].begin(), training_data[i].end(), flat_data.begin() + i * dim_);
    }
    train_flat(flat_data.data(), num_vectors, metric);
}

void ProductQuantizer::train_flat(const float* data, size_t num_vectors, Metric metric) {
    centroids_.resize(m_ * k_sub_ * sub_dim_);

    // Train m subquantizers independently
    #pragma omp parallel for
    for (int sq = 0; sq < static_cast<int>(m_); ++sq) {
        // Extract subvector data
        std::vector<float> sub_data(num_vectors * sub_dim_);
        for (size_t i = 0; i < num_vectors; ++i) {
            for (size_t d = 0; d < sub_dim_; ++d) {
                sub_data[i * sub_dim_ + d] = data[i * dim_ + sq * sub_dim_ + d];
            }
        }

        auto sub_centroids = train_kmeans(sub_data.data(), num_vectors, sub_dim_, k_sub_, metric);
        
        // Copy back to main centroids array
        for (size_t k = 0; k < k_sub_; ++k) {
            for (size_t d = 0; d < sub_dim_; ++d) {
                centroids_[sq * k_sub_ * sub_dim_ + k * sub_dim_ + d] = sub_centroids[k * sub_dim_ + d];
            }
        }
    }
    is_trained_ = true;
}

std::vector<uint8_t> ProductQuantizer::encode(const float* vector) const {
    if (!is_trained_) throw std::runtime_error("PQ not trained");

    std::vector<uint8_t> code(m_);
    for (size_t sq = 0; sq < m_; ++sq) {
        float min_dist = std::numeric_limits<float>::max();
        uint8_t best_k = 0;
        const float* sub_vec = vector + sq * sub_dim_;
        const float* sub_centroids = centroids_.data() + sq * k_sub_ * sub_dim_;

        for (size_t k = 0; k < k_sub_; ++k) {
            float dist = compute_distance(sub_vec, sub_centroids + k * sub_dim_, sub_dim_, Metric::L2); // Internal PQ dist is usually L2
            if (dist < min_dist) {
                min_dist = dist;
                best_k = static_cast<uint8_t>(k);
            }
        }
        code[sq] = best_k;
    }
    return code;
}

std::vector<float> ProductQuantizer::compute_lut(const float* query, Metric metric) const {
    if (!is_trained_) throw std::runtime_error("PQ not trained");

    std::vector<float> lut(m_ * k_sub_);
    for (size_t sq = 0; sq < m_; ++sq) {
        const float* sub_query = query + sq * sub_dim_;
        const float* sub_centroids = centroids_.data() + sq * k_sub_ * sub_dim_;

        for (size_t k = 0; k < k_sub_; ++k) {
            lut[sq * k_sub_ + k] = compute_distance(sub_query, sub_centroids + k * sub_dim_, sub_dim_, metric);
        }
    }
    return lut;
}

float ProductQuantizer::compute_adc(const uint8_t* code, const float* lut) const {
    float dist = 0.0f;
    if (m_ == 8) {
        dist += lut[0 * k_sub_ + code[0]];
        dist += lut[1 * k_sub_ + code[1]];
        dist += lut[2 * k_sub_ + code[2]];
        dist += lut[3 * k_sub_ + code[3]];
        dist += lut[4 * k_sub_ + code[4]];
        dist += lut[5 * k_sub_ + code[5]];
        dist += lut[6 * k_sub_ + code[6]];
        dist += lut[7 * k_sub_ + code[7]];
    } else if (m_ == 16) {
        dist += lut[0 * k_sub_ + code[0]];
        dist += lut[1 * k_sub_ + code[1]];
        dist += lut[2 * k_sub_ + code[2]];
        dist += lut[3 * k_sub_ + code[3]];
        dist += lut[4 * k_sub_ + code[4]];
        dist += lut[5 * k_sub_ + code[5]];
        dist += lut[6 * k_sub_ + code[6]];
        dist += lut[7 * k_sub_ + code[7]];
        dist += lut[8 * k_sub_ + code[8]];
        dist += lut[9 * k_sub_ + code[9]];
        dist += lut[10 * k_sub_ + code[10]];
        dist += lut[11 * k_sub_ + code[11]];
        dist += lut[12 * k_sub_ + code[12]];
        dist += lut[13 * k_sub_ + code[13]];
        dist += lut[14 * k_sub_ + code[14]];
        dist += lut[15 * k_sub_ + code[15]];
    } else {
        #pragma omp simd reduction(+:dist)
        for (size_t sq = 0; sq < m_; ++sq) {
            dist += lut[sq * k_sub_ + code[sq]];
        }
    }
    return dist;
}

} // namespace vectorforge
