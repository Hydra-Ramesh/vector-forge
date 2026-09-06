#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <queue>

namespace vectorforge {

struct HNSWNode {
    VectorId id;
    int max_level;
    // neighbors[level] contains a list of internal indices of neighbors
    std::vector<std::vector<int32_t>> neighbors;
};

class HNSWIndex {
public:
    HNSWIndex(size_t dimension, int max_connections = 16, int construction_expansion = 100);

    // Adds a vector to the index. In HNSW, adding actually inserts it into the graph immediately.
    void add(VectorId id, const Vector& vector);
    
    // Does nothing in this implementation since `add` builds the graph incrementally,
    // but provided to match the interface.
    void build();

    std::vector<SearchResult> search(const Vector& query, const SearchOptions& options) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t size() const { return num_vectors_; }
    size_t dimension() const { return dim_; }

private:
    size_t dim_;
    int max_connections_;
    int max_layer_zero_connections_;
    int construction_expansion_;
    double level_multiplier_;

    size_t num_vectors_;
    int max_level_;
    int32_t enterpoint_node_; // internal index of the enterpoint

    std::vector<VectorId> owned_ids_;
    std::vector<float> owned_vectors_;
    std::vector<HNSWNode> nodes_;

    std::default_random_engine rng_;

    int generate_random_level();
    
    // internal methods
    float distance(const float* left_vector, const float* right_vector) const;
    
    // search layer returns the nearest neighbors found in the layer
    void search_layer(
        const float* query_vector,
        std::vector<int32_t>& entry_points,
        int search_expansion,
        int level,
        std::priority_queue<std::pair<float, int32_t>>& candidate_queue) const;

    // select neighbors using simple distance logic (can be upgraded to heuristic later)
    std::vector<int32_t> select_neighbors(
        const float* query_vector,
        std::priority_queue<std::pair<float, int32_t>>& candidates,
        int max_neighbors,
        int level);

    void insert(int32_t internal_index, const float* vector);
};

} // namespace vectorforge
