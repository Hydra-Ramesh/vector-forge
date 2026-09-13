#include "vectorforge/index/hybrid_index.hpp"
#include <algorithm>

namespace vectorforge {

HybridIndex::HybridIndex(size_t dense_dim, size_t max_degree, size_t candidate_list_size, float pruning_alpha)
    : dense_(dense_dim, max_degree, candidate_list_size, pruning_alpha) {
}

void HybridIndex::add(uint64_t id, const std::vector<float>& dense_vec, const std::unordered_map<uint32_t, float>& sparse_vec, uint64_t mask) {
    dense_.add(id, dense_vec, mask);
    sparse_.add(id, sparse_vec);
}

void HybridIndex::build() {
    dense_.build();
}

std::vector<SearchResult> HybridIndex::search(const std::vector<float>& dense_query, const std::unordered_map<uint32_t, float>& sparse_query, const SearchOptions& opts, float rrf_k) const {
    // Increase internal top_k for RRF to work effectively
    SearchOptions internal_opts = opts;
    internal_opts.top_k = opts.top_k * 5; 
    
    auto dense_results = dense_.search(dense_query, internal_opts);
    auto sparse_results = sparse_.search(sparse_query, internal_opts);
    
    std::unordered_map<uint64_t, float> rrf_scores;
    
    // Rank Dense
    for (size_t i = 0; i < dense_results.size(); ++i) {
        rrf_scores[dense_results[i].id] += 1.0f / (rrf_k + (i + 1));
    }
    
    // Rank Sparse
    for (size_t i = 0; i < sparse_results.size(); ++i) {
        rrf_scores[sparse_results[i].id] += 1.0f / (rrf_k + (i + 1));
    }
    
    std::vector<SearchResult> fused_results;
    fused_results.reserve(rrf_scores.size());
    for (const auto& kv : rrf_scores) {
        // Negate score because SearchResult sorts by smallest distance
        fused_results.push_back({kv.first, -kv.second});
    }
    
    std::sort(fused_results.begin(), fused_results.end());
    
    if (fused_results.size() > static_cast<size_t>(opts.top_k)) {
        fused_results.resize(opts.top_k);
    }
    
    return fused_results;
}

} // namespace vectorforge
