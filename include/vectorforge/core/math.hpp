#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <cstddef>

namespace vectorforge {

// Core distance calculation used across indices
float compute_distance(const float* left_vector, const float* right_vector, size_t dimension, Metric metric);

// Lloyd's algorithm for K-Means clustering
// Returns a flattened array of size (k * dim) containing the cluster centroids.
std::vector<float> train_kmeans(const float* data, size_t vector_count, size_t dimension, size_t cluster_count, Metric metric, int max_iterations = 50);

} // namespace vectorforge
