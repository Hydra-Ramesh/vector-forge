#pragma once

#include "vectorforge/core/types.hpp"
#include "vectorforge/core/pq.hpp"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <string>

namespace vectorforge {

class QuantizedVamanaIndex {
public:
    QuantizedVamanaIndex(size_t dimension, size_t m, size_t max_degree = 64, size_t candidate_list_size = 100, float pruning_alpha = 1.2f);
    ~QuantizedVamanaIndex();

    void train(const std::vector<std::vector<float>>& training_data, Metric metric = Metric::L2);

    void add(uint64_t id, const std::vector<float>& vec, uint64_t mask = 0);
    
    void remove(uint64_t id);
    void compact();
    void build(); 
    
    std::vector<SearchResult> search(const std::vector<float>& query, const SearchOptions& opts) const;

private:
    size_t dimension_;
    size_t m_; // PQ m
    size_t max_degree_;
    size_t candidate_list_size_;
    float pruning_alpha_;
    size_t medoid_index_;
    ProductQuantizer pq_;
    bool is_trained_;

    size_t node_size_bytes_;
    std::vector<uint8_t> data_;
    size_t num_nodes_;
    
    std::vector<bool> deleted_;
    std::unordered_map<uint64_t, size_t> id_to_index_;
    
    inline uint8_t* get_code(size_t node_index) {
        return (uint8_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t));
    }
    inline const uint8_t* get_code(size_t node_index) const {
        return (const uint8_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t));
    }
    
    inline uint32_t& get_num_neighbors(size_t node_index) {
        return *(uint32_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t));
    }
    inline uint32_t get_num_neighbors(size_t node_index) const {
        return *(const uint32_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t));
    }
    
    inline size_t* get_neighbors(size_t node_index) {
        return (size_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + m_);
    }
    inline const size_t* get_neighbors(size_t node_index) const {
        return (const size_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + m_);
    }
    
    inline uint64_t& get_id(size_t node_index) {
        return *(uint64_t*)(data_.data() + node_index * node_size_bytes_);
    }
    inline uint64_t get_id(size_t node_index) const {
        return *(const uint64_t*)(data_.data() + node_index * node_size_bytes_);
    }

    inline uint64_t& get_mask(size_t node_index) {
        return *(uint64_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t));
    }
    inline uint64_t get_mask(size_t node_index) const {
        return *(const uint64_t*)(data_.data() + node_index * node_size_bytes_ + sizeof(uint64_t));
    }

    // Distance between a query's LUT and a compressed node's code (ADC)
    float distance_adc(const float* lut, const uint8_t* code) const;
    // Distance between two compressed nodes (Symmetric Distance Computation - SDC)
    float distance_sdc(const uint8_t* left_code, const uint8_t* right_code) const;

    void robust_prune(size_t node_index, std::vector<std::pair<float, size_t>>& candidates, float pruning_alpha, size_t max_degree);
    std::vector<std::pair<float, size_t>> greedy_search(const float* lut, size_t start_index, size_t candidate_list_size, uint64_t filter_mask = 0) const;
    size_t calculate_medoid() const;
};

} // namespace vectorforge
