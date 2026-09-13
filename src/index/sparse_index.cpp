#include "vectorforge/index/sparse_index.hpp"
#include <algorithm>

namespace vectorforge {

void SparseIndex::add(uint64_t id, const std::unordered_map<uint32_t, float>& sparse_vec) {
    forward_index_[id] = sparse_vec;
    for (const auto& kv : sparse_vec) {
        inverted_index_[kv.first].push_back({id, kv.second});
    }
}

void SparseIndex::remove(uint64_t id) {
    auto it = forward_index_.find(id);
    if (it != forward_index_.end()) {
        for (const auto& kv : it->second) {
            auto& posting_list = inverted_index_[kv.first];
            posting_list.erase(std::remove_if(posting_list.begin(), posting_list.end(), 
                [id](const std::pair<uint64_t, float>& p) { return p.first == id; }), posting_list.end());
        }
        forward_index_.erase(it);
    }
}

std::vector<SearchResult> SparseIndex::search(const std::unordered_map<uint32_t, float>& query, const SearchOptions& opts) const {
    std::unordered_map<uint64_t, float> scores;
    
    for (const auto& q_kv : query) {
        auto it = inverted_index_.find(q_kv.first);
        if (it != inverted_index_.end()) {
            for (const auto& doc_kv : it->second) {
                scores[doc_kv.first] += q_kv.second * doc_kv.second;
            }
        }
    }
    
    std::vector<SearchResult> results;
    results.reserve(scores.size());
    for (const auto& kv : scores) {
        // We invert the score because SearchResult expects a distance (smaller is better).
        // Since sparse search returns similarity score (higher is better), we make it negative.
        results.push_back({kv.first, -kv.second});
    }
    
    std::sort(results.begin(), results.end());
    
    if (results.size() > static_cast<size_t>(opts.top_k)) {
        results.resize(opts.top_k);
    }
    
    return results;
}

} // namespace vectorforge
