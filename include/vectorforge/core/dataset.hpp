#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>

namespace vectorforge {

class DatasetGenerator {
public:
    // Generate a vector of vectors with deterministic random values
    static std::vector<Vector> generate(size_t num_vectors, size_t dim, int seed = 42);
    
    // Normalize vectors in-place (useful for Cosine similarity)
    static void normalize(std::vector<Vector>& vectors);
};

} // namespace vectorforge
