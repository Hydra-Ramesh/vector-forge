#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <cstddef>

namespace vectorforge {

// Core distance calculation used across indices
float compute_distance(const float* a, const float* b, size_t dim, Metric metric);

// Lloyd's algorithm for K-Means clustering
// Returns a flattened array of size (k * dim) containing the cluster centroids.
std::vector<float> train_kmeans(const float* data, size_t num_vectors, size_t dim, size_t k, Metric metric, int max_iter = 50);

} // namespace vectorforge
