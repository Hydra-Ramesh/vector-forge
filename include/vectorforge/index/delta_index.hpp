#pragma once

#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include <unordered_set>

namespace vectorforge {

class DeltaIndex {
public:
    DeltaIndex(size_t dimension, size_t max_degree = 64, size_t candidate_list_size = 100, float pruning_alpha = 1.2f);
    ~DeltaIndex();

    void add(uint64_t id, const std::vector<float>& vec);
    void remove(uint64_t id);
    
    // Merges delta vectors into vamana and reconstructs the graph
    void merge();
    
    std::vector<SearchResult> search(const std::vector<float>& query, const SearchOptions& opts) const;

private:
    size_t dimension_;
    VamanaIndex vamana_;
    BruteForceIndex delta_;
    
    // Tombstones for items in Vamana that were deleted, or updated
    std::unordered_set<uint64_t> tombstones_;
    
    // A separate store for vectors in Delta since BruteForceIndex doesn't let us extract them easily
    std::unordered_map<uint64_t, std::vector<float>> delta_vectors_;
};

} // namespace vectorforge
