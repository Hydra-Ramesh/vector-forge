#include "vectorforge/core/dataset.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

namespace vectorforge {

std::vector<Vector> DatasetGenerator::generate(size_t num_vectors, size_t dim, int seed) {
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
    std::mt19937 random_generator(seed);
    std::uniform_real_distribution<float> random_value(-1.0f, 1.0f);

    std::vector<Vector> dataset(num_vectors, Vector(dim));
    for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
        for (size_t dimension_index = 0; dimension_index < dim; ++dimension_index) {
            dataset[vector_index][dimension_index] = random_value(random_generator);
        }
    }
    return dataset;
}

void DatasetGenerator::normalize(std::vector<Vector>& vectors) {
    for (auto& vector : vectors) {
        float norm_squared = 0.0f;
        for (float value : vector) {
            norm_squared += value * value;
        }
        if (norm_squared > 0.0f) {
            float norm = std::sqrt(norm_squared);
            for (float& value : vector) {
                value /= norm;
            }
        }
    }
}

} // namespace vectorforge
