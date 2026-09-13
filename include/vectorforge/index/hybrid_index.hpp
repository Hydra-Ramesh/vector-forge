#pragma once

#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/index/sparse_index.hpp"
#include <vector>

namespace vectorforge {

class HybridIndex {
public:
    HybridIndex(size_t dense_dim, size_t max_degree = 64, size_t candidate_list_size = 100, float pruning_alpha = 1.2f);
    
    void add(uint64_t id, const std::vector<float>& dense_vec, const std::unordered_map<uint32_t, float>& sparse_vec, uint64_t mask = 0);
    void build();
    
    std::vector<SearchResult> search(const std::vector<float>& dense_query, const std::unordered_map<uint32_t, float>& sparse_query, const SearchOptions& opts, float rrf_k = 60.0f) const;

    VamanaIndex& get_dense_index() { return dense_; }
    SparseIndex& get_sparse_index() { return sparse_; }

private:
    VamanaIndex dense_;
    SparseIndex sparse_;
};

} // namespace vectorforge
