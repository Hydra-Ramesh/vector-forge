#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace vectorforge {

using VectorId = uint64_t;
using Vector = std::vector<float>;

enum class Metric {
    L2,
    Cosine,
    Hamming // Used for Binary Quantization (BQ)
};

struct SearchStats {
    double centroid_search_ms = 0;
    double lut_compute_ms = 0;
    double list_scan_ms = 0;
    double rerank_ms = 0;
    double total_ms = 0;
};

struct SearchOptions {
    int top_k = 10;
    Metric metric = Metric::L2;
    uint64_t filter_mask = 0; // 0 means no filtering
    
    // Temporal parameters for time-aware retrieval
    float temporal_decay_factor = 0.0f; 
    uint64_t query_timestamp = 0;

    SearchStats* stats = nullptr;
};

struct SearchResult {
    VectorId id;
    float distance;
    
    SearchResult() = default;
    SearchResult(VectorId i, float d) : id(i), distance(d) {}
    
    // For max-heap (to keep the smallest distances, we reverse the comparator)
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

} // namespace vectorforge
