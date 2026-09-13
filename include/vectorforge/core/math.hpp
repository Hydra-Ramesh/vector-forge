#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <cstddef>

namespace vectorforge {

typedef float (*DistanceFunction)(const float*, const float*, size_t, Metric);

DistanceFunction get_distance_function();

// Computes the distance between two vectors of given dimension
float compute_distance(const std::vector<float>& a, const std::vector<float>& b, size_t dim, Metric metric = Metric::L2);

// Computes the distance between a raw float array and a vector
float compute_distance(const float* a, const std::vector<float>& b, size_t dim, Metric metric = Metric::L2);

float compute_distance(const float* a, const float* b, size_t dim, Metric metric = Metric::L2);

// Lloyd's algorithm for K-Means clustering
// Returns a flattened array of size (k * dim) containing the cluster centroids.
std::vector<float> train_kmeans(const float* data, size_t vector_count, size_t dimension, size_t cluster_count, Metric metric, int max_iterations = 50);

// Binarize a float vector into packed uint8_t for Binary Quantization
std::vector<uint8_t> binarize_vector(const float* vec, size_t dim);

// Compute Hamming distance between two binarized vectors
float compute_distance_hamming(const uint8_t* a, const uint8_t* b, size_t original_dim);

} // namespace vectorforge
