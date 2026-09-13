#include "vectorforge/index/delta_index.hpp"
#include <algorithm>

namespace vectorforge {

DeltaIndex::DeltaIndex(size_t dimension, size_t max_degree, size_t candidate_list_size, float pruning_alpha)
    : dimension_(dimension), vamana_(dimension, max_degree, candidate_list_size, pruning_alpha), delta_(dimension) {
}

DeltaIndex::~DeltaIndex() {}

void DeltaIndex::add(uint64_t id, const std::vector<float>& vec) {
    if (vec.size() != dimension_) throw std::invalid_argument("Vector dimension mismatch");
    
    // If updating an existing item in Vamana, tombstone it
    tombstones_.insert(id);
    
    // Add to delta buffer
    delta_vectors_[id] = vec;
    delta_.add(id, vec);
    delta_.build(); // BruteForce build is a no-op but required by API
}

void DeltaIndex::remove(uint64_t id) {
    tombstones_.insert(id);
    delta_vectors_.erase(id);
    
    // We would remove from BruteForceIndex here, but BruteForceIndex in this repo 
    // doesn't have a remove() method. So we just filter tombstones in search.
}

void DeltaIndex::merge() {
    for (uint64_t id : tombstones_) {
        vamana_.remove(id);
    }
    
    for (const auto& kv : delta_vectors_) {
        // Technically we should check if it's already in Vamana and update, but Vamana doesn't support 
        // in-place updates. So we just add as a new node.
        vamana_.add(kv.first, kv.second);
    }
    
    vamana_.compact(); // This rebuilds the graph
    
    delta_vectors_.clear();
    tombstones_.clear();
    
    // Reset delta index (reinitialize)
    delta_ = BruteForceIndex(dimension_);
}

std::vector<SearchResult> DeltaIndex::search(const std::vector<float>& query, const SearchOptions& opts) const {
    // 1. Search Vamana
    auto vamana_results = vamana_.search(query, opts);
    
    // 2. Search Delta
    auto delta_results = delta_.search(query, opts);
    
    // 3. Merge and filter tombstones
    std::vector<SearchResult> merged;
    merged.reserve(vamana_results.size() + delta_results.size());
    
    for (const auto& r : vamana_results) {
        if (tombstones_.find(r.id) == tombstones_.end()) {
            merged.push_back(r);
        }
    }
    
    for (const auto& r : delta_results) {
        if (tombstones_.find(r.id) == tombstones_.end() || delta_vectors_.find(r.id) != delta_vectors_.end()) {
            merged.push_back(r);
        }
    }
    
    // Sort combined results
    std::sort(merged.begin(), merged.end());
    
    // Take Top-K
    if (merged.size() > static_cast<size_t>(opts.top_k)) {
        merged.resize(opts.top_k);
    }
    
    return merged;
}

} // namespace vectorforge
