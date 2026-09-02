#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <cstdint>

namespace vectorforge {

class ProductQuantizer {
public:
    ProductQuantizer(size_t dim, size_t m, size_t k_sub = 256);

    void train(const std::vector<Vector>& training_data, Metric metric);
    void train_flat(const float* data, size_t num_vectors, Metric metric);

    std::vector<uint8_t> encode(const float* vector) const;
    
    // Computes a Lookup Table (LUT) of size (m * k_sub) for a given query vector.
    // lut[i * k_sub + j] = distance from query subvector i to centroid j of subquantizer i.
    std::vector<float> compute_lut(const float* query, Metric metric) const;

    // Fast asymmetric distance computation using the precomputed LUT
    float compute_adc(const uint8_t* code, const float* lut) const;

    size_t get_m() const { return m_; }
    size_t get_k_sub() const { return k_sub_; }
    size_t get_sub_dim() const { return sub_dim_; }
    bool is_trained() const { return is_trained_; }

    const std::vector<float>& get_centroids() const { return centroids_; }
    void set_centroids(const std::vector<float>& centroids) { 
        centroids_ = centroids; 
        is_trained_ = true; 
    }

private:
    size_t dim_;
    size_t m_;
    size_t k_sub_;
    size_t sub_dim_;
    bool is_trained_;

    // Flattened centroids: m * k_sub * sub_dim floats
    std::vector<float> centroids_;
};

} // namespace vectorforge
