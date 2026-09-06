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
    VamanaIndex(size_t dimension, size_t max_degree = 64, size_t candidate_list_size = 100, float pruning_alpha = 1.2f);
    ~VamanaIndex();

    // Add vectors to the index. Graph isn't fully optimized until build() is called.
    void add(uint64_t id, const std::vector<float>& vec);
    
    // Builds the Vamana graph (generates random graph, then refines via robust prune).
    void build(); 
    
    std::vector<SearchResult> search(const std::vector<float>& query, const SearchOptions& opts) const;
    
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    size_t dimension_;
    size_t max_degree_;
    size_t candidate_list_size_;
    float pruning_alpha_;
    size_t medoid_index_;

    size_t node_size_bytes_;
    std::vector<uint8_t> data_;
    size_t num_nodes_;
    
    inline float* get_vector(size_t node_index) {
        return (float*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t));
    }
    inline const float* get_vector(size_t node_index) const {
        return (const float*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t));
    }
    
    inline uint32_t& get_num_neighbors(size_t node_index) {
        return *(uint32_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t));
    }
    inline uint32_t get_num_neighbors(size_t node_index) const {
        return *(const uint32_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t));
    }
    
    inline size_t* get_neighbors(size_t node_index) {
        return (size_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t) + dimension_ * sizeof(float));
    }
    inline const size_t* get_neighbors(size_t node_index) const {
        return (const size_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint32_t) + dimension_ * sizeof(float));
    }
    
    inline uint64_t& get_id(size_t node_index) {
        return *(uint64_t*)(data_.data() + node_index * node_size_bytes_);
    }
    inline uint64_t get_id(size_t node_index) const {
        return *(const uint64_t*)(data_.data() + node_index * node_size_bytes_);
    }

    float distance(const float* left_vector, const float* right_vector) const;
    void robust_prune(size_t node_index, std::vector<std::pair<float, size_t>>& candidates, float pruning_alpha, size_t max_degree);
    std::vector<std::pair<float, size_t>> greedy_search(const float* query_vector, size_t start_index, size_t candidate_list_size) const;
    size_t calculate_medoid() const;
};

} // namespace vectorforge
