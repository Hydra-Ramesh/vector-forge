#pragma once
#include "vectorforge/core/types.hpp"
#include <vector>
#include <string>

namespace vectorforge {

class FvecsReader {
public:
    static std::vector<Vector> read(const std::string& filepath, size_t max_vectors = 0);
};

class BvecsReader {
public:
    static std::vector<Vector> read(const std::string& filepath, size_t max_vectors = 0);
};

} // namespace vectorforge
