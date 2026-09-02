#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace vectorforge {

class VamanaIndex {
public:
    // dim: vector dimensionality
    // R: maximum out-degree of the graph
    // L: size of the candidate list for search
    // alpha: threshold parameter for robust pruning
    VamanaIndex(size_t dim, size_t max_degree = 64, size_t L = 100, float alpha = 1.2f);
    ~VamanaIndex();

    // Add vectors to the index. Graph isn't fully optimized until build() is called.
    void add(uint64_t id, const std::vector<float>& vec);
    
    // Builds the Vamana graph (generates random graph, then refines via robust prune).
    void build(); 
    
    std::vector<SearchResult> search(const std::vector<float>& query, const SearchOptions& opts) const;
    
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    size_t dim_;
    size_t R_;
    size_t L_;
    float alpha_;
    size_t medoid_idx_; // index in data_, not the actual ID

    // We store all node data contiguously in a massive byte array.
    // This allows trivial zero-copy mmap() loading in the future.
    // Memory layout per node: 
    // [uint64_t id] [uint32_t num_neighbors] [float*dim vec] [size_t*R neighbors]
    size_t node_size_bytes_;
    std::vector<uint8_t> data_; 
    size_t num_nodes_;
    
    // Helper accessors. Since data_ can reallocate, we ALWAYS use indices, never bare pointers.
    inline float* get_vector(size_t idx) {
        return (float*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t));
    }
    inline const float* get_vector(size_t idx) const {
        return (const float*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t));
    }
    
    inline uint32_t& get_num_neighbors(size_t idx) {
        return *(uint32_t*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t));
    }
    inline uint32_t get_num_neighbors(size_t idx) const {
        return *(const uint32_t*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t));
    }
    
    inline size_t* get_neighbors(size_t idx) {
        return (size_t*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t) + dim_ * sizeof(float));
    }
    inline const size_t* get_neighbors(size_t idx) const {
        return (const size_t*)(data_.data() + idx * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t) + dim_ * sizeof(float));
    }
    
    inline uint64_t& get_id(size_t idx) {
        return *(uint64_t*)(data_.data() + idx * node_size_bytes_);
    }
    inline uint64_t get_id(size_t idx) const {
        return *(const uint64_t*)(data_.data() + idx * node_size_bytes_);
    }

    float distance(const float* a, const float* b) const;
    void robust_prune(size_t idx, std::vector<std::pair<float, size_t>>& candidates, float alpha, size_t R);
    std::vector<std::pair<float, size_t>> greedy_search(const float* query, size_t start_idx, size_t L) const;
    size_t calculate_medoid() const;
};

} // namespace vectorforge
