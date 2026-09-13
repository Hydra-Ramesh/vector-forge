#include "vectorforge/core/fvecs_reader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace vectorforge {

std::vector<Vector> FvecsReader::read(const std::string& filepath, size_t max_vectors) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open fvecs file: " + filepath);
    }

    std::vector<Vector> vectors;
    int32_t dim;
    
    while (in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
        if (dim <= 0) break;
        
        Vector v(dim);
        in.read(reinterpret_cast<char*>(v.data()), dim * sizeof(float));
        vectors.push_back(std::move(v));
        
        if (max_vectors > 0 && vectors.size() >= max_vectors) {
            break;
        }
    }
    
    return vectors;
}

std::vector<Vector> BvecsReader::read(const std::string& filepath, size_t max_vectors) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open bvecs file: " + filepath);
    }

    std::vector<Vector> vectors;
    int32_t dim;
    
    while (in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) {
        if (dim <= 0) break;
        
        std::vector<uint8_t> buffer(dim);
        in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(uint8_t));
        
        Vector v(dim);
        for(size_t i = 0; i < dim; ++i) {
            v[i] = static_cast<float>(buffer[i]);
        }
        vectors.push_back(std::move(v));
        
        if (max_vectors > 0 && vectors.size() >= max_vectors) {
            break;
        }
    }
    
    return vectors;
}

} // namespace vectorforge
