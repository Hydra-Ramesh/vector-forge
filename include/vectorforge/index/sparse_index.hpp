#pragma once

#include "vectorforge/core/types.hpp"
#include <unordered_map>
#include <vector>

namespace vectorforge {

class SparseIndex {
public:
    SparseIndex() = default;
    
    // Add a sparse vector (e.g. TF-IDF or BM25 precomputed weights from Python)
    void add(uint64_t id, const std::unordered_map<uint32_t, float>& sparse_vec);
    void remove(uint64_t id);
    
    // Dot product search over sparse vectors
    std::vector<SearchResult> search(const std::unordered_map<uint32_t, float>& query, const SearchOptions& opts) const;

private:
    // Inverted Index: token_id -> list of (document_id, weight)
    std::unordered_map<uint32_t, std::vector<std::pair<uint64_t, float>>> inverted_index_;
    // Forward Index (needed for easy removal)
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, float>> forward_index_;
};

} // namespace vectorforge
