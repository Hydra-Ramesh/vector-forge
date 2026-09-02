#include "vectorforge/core/dataset.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

namespace vectorforge {

std::vector<Vector> DatasetGenerator::generate(size_t num_vectors, size_t dim, int seed) {
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<Vector> dataset(num_vectors, Vector(dim));
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            dataset[i][j] = dist(gen);
        }
    }
    return dataset;
}

void DatasetGenerator::normalize(std::vector<Vector>& vectors) {
    for (auto& vec : vectors) {
        float norm_sq = 0.0f;
        for (float v : vec) {
            norm_sq += v * v;
        }
        if (norm_sq > 0.0f) {
            float norm = std::sqrt(norm_sq);
            for (float& v : vec) {
                v /= norm;
            }
        }
    }
}

} // namespace vectorforge
