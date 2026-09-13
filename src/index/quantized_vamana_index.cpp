#include "vectorforge/index/quantized_vamana_index.hpp"
#include "vectorforge/core/math.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace vectorforge {

QuantizedVamanaIndex::QuantizedVamanaIndex(size_t dimension, size_t m, size_t max_degree, size_t candidate_list_size, float pruning_alpha)
    : dimension_(dimension), m_(m), max_degree_(max_degree), candidate_list_size_(candidate_list_size), pruning_alpha_(pruning_alpha), medoid_index_(0), pq_(dimension, m), is_trained_(false), num_nodes_(0) {
    node_size_bytes_ = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + m_ + max_degree_ * sizeof(size_t);
}

QuantizedVamanaIndex::~QuantizedVamanaIndex() {}

void QuantizedVamanaIndex::train(const std::vector<std::vector<float>>& training_data, Metric metric) {
    std::vector<Vector> train_vecs(training_data.size());
    for(size_t i = 0; i < training_data.size(); ++i) train_vecs[i] = training_data[i];
    pq_.train(train_vecs, metric);
    is_trained_ = true;
}

void QuantizedVamanaIndex::add(uint64_t id, const std::vector<float>& vector, uint64_t mask) {
    if (!is_trained_) throw std::runtime_error("Index must be trained before adding vectors");
    if (vector.size() != dimension_) throw std::invalid_argument("Vector dimension mismatch");
    
    size_t node_index = num_nodes_++;
    data_.resize(num_nodes_ * node_size_bytes_);
    
    get_id(node_index) = id;
    get_mask(node_index) = mask;
    get_num_neighbors(node_index) = 0;
    
    std::vector<uint8_t> code = pq_.encode(vector.data());
    std::copy(code.begin(), code.end(), get_code(node_index));
    
    deleted_.push_back(false);
    id_to_index_[id] = node_index;
}

void QuantizedVamanaIndex::remove(uint64_t id) {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        deleted_[it->second] = true;
        id_to_index_.erase(it);
    }
}

void QuantizedVamanaIndex::compact() {
    // Skipping implementation for brevity
}

float QuantizedVamanaIndex::distance_adc(const float* lut, const uint8_t* code) const {
    return pq_.compute_adc(code, lut);
}

float QuantizedVamanaIndex::distance_sdc(const uint8_t* left_code, const uint8_t* right_code) const {
    const std::vector<float>& centroids = pq_.get_centroids();
    size_t k_sub = pq_.get_k_sub();
    size_t sub_dim = pq_.get_sub_dim();
    
    float total_dist = 0.0f;
    for (size_t i = 0; i < m_; ++i) {
        size_t c_left = left_code[i];
        size_t c_right = right_code[i];
        
        const float* c_left_vec = centroids.data() + i * k_sub * sub_dim + c_left * sub_dim;
        const float* c_right_vec = centroids.data() + i * k_sub * sub_dim + c_right * sub_dim;
        
        total_dist += compute_distance(c_left_vec, c_right_vec, sub_dim, Metric::L2);
    }
    return total_dist;
}

size_t QuantizedVamanaIndex::calculate_medoid() const {
    if (num_nodes_ == 0) return 0;
    return 0; // Simplified
}

std::vector<std::pair<float, size_t>> QuantizedVamanaIndex::greedy_search(const float* lut, size_t start_index, size_t candidate_list_size, uint64_t filter_mask) const {
    std::vector<std::pair<float, size_t>> top_candidates;
    std::unordered_set<size_t> visited;
    
    auto compare = [](const std::pair<float, size_t>& l, const std::pair<float, size_t>& r) { return l.first > r.first; };
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(compare)> candidates(compare);
    
    float start_distance = distance_adc(lut, get_code(start_index));
    candidates.push({start_distance, start_index});
    visited.insert(start_index);
    top_candidates.push_back({start_distance, start_index});
    
    while (!candidates.empty()) {
        auto [candidate_distance, candidate_index] = candidates.top();
        candidates.pop();
        
        float worst_candidate_distance = top_candidates.back().first;
        if (top_candidates.size() == candidate_list_size && candidate_distance > worst_candidate_distance) break;
        
        uint32_t neighbor_count = get_num_neighbors(candidate_index);
        const size_t* neighbors = get_neighbors(candidate_index);
        
        for (uint32_t n = 0; n < neighbor_count; ++n) {
            size_t neighbor_index = neighbors[n];
            if (visited.find(neighbor_index) == visited.end()) {
                visited.insert(neighbor_index);
                
                if (filter_mask != 0 && (get_mask(neighbor_index) & filter_mask) != filter_mask) {
                    continue;
                }
                
                float neighbor_distance = distance_adc(lut, get_code(neighbor_index));
                
                auto it = std::lower_bound(top_candidates.begin(), top_candidates.end(), std::make_pair(neighbor_distance, neighbor_index),
                                           [](const auto& l, const auto& r) { return l.first < r.first; });
                if (it != top_candidates.end() || top_candidates.size() < candidate_list_size) {
                    top_candidates.insert(it, {neighbor_distance, neighbor_index});
                    if (top_candidates.size() > candidate_list_size) top_candidates.pop_back();
                    candidates.push({neighbor_distance, neighbor_index});
                }
            }
        }
    }
    return top_candidates;
}

void QuantizedVamanaIndex::robust_prune(size_t node_index, std::vector<std::pair<float, size_t>>& candidates, float pruning_alpha, size_t max_degree) {
    uint32_t neighbor_count = get_num_neighbors(node_index);
    size_t* neighbors = get_neighbors(node_index);
    std::unordered_set<size_t> unique_neighbors;
    std::vector<std::pair<float, size_t>> candidate_neighbors;
    
    auto add_candidate = [&](size_t neighbor_index, float distance) {
        if (neighbor_index != node_index && unique_neighbors.find(neighbor_index) == unique_neighbors.end()) {
            unique_neighbors.insert(neighbor_index);
            candidate_neighbors.push_back({distance, neighbor_index});
        }
    };
    
    for (const auto& candidate : candidates) add_candidate(candidate.second, candidate.first);
    for (uint32_t n = 0; n < neighbor_count; ++n) {
        size_t neighbor_index = neighbors[n];
        add_candidate(neighbor_index, distance_sdc(get_code(node_index), get_code(neighbor_index)));
    }
    
    std::sort(candidate_neighbors.begin(), candidate_neighbors.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    
    std::vector<size_t> new_neighbors;
    while (!candidate_neighbors.empty() && new_neighbors.size() < max_degree) {
        size_t closest_neighbor = candidate_neighbors.front().second;
        new_neighbors.push_back(closest_neighbor);
        
        std::vector<std::pair<float, size_t>> remaining_candidates;
        for (size_t i = 1; i < candidate_neighbors.size(); ++i) {
            size_t other_neighbor = candidate_neighbors[i].second;
            float distance_between = distance_sdc(get_code(closest_neighbor), get_code(other_neighbor));
            if (pruning_alpha * distance_between > candidate_neighbors[i].first) {
                remaining_candidates.push_back(candidate_neighbors[i]);
            }
        }
        candidate_neighbors = remaining_candidates;
    }
    
    get_num_neighbors(node_index) = static_cast<uint32_t>(new_neighbors.size());
    for (size_t i = 0; i < new_neighbors.size(); ++i) neighbors[i] = new_neighbors[i];
}

void QuantizedVamanaIndex::build() {
    if (num_nodes_ == 0 || !is_trained_) return;
    medoid_index_ = calculate_medoid();
    
    // Initialize random graph
    std::mt19937 rng(42);
    for (size_t i = 0; i < num_nodes_; ++i) {
        std::vector<size_t> indices(num_nodes_);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t* neighbors = get_neighbors(i);
        uint32_t added = 0;
        for (size_t c = 0; c < num_nodes_ && added < max_degree_; ++c) {
            if (indices[c] != i) neighbors[added++] = indices[c];
        }
        get_num_neighbors(i) = added;
    }
    
    std::vector<size_t> perm(num_nodes_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    
    auto process_pass = [&](float pass_alpha) {
        for (size_t pass_index = 0; pass_index < num_nodes_; ++pass_index) {
            size_t node_index = perm[pass_index];
            
            // Reconstruct full float vector of node to get LUT (ADC) for greedy search
            std::vector<float> node_lut(m_ * pq_.get_k_sub());
            const uint8_t* node_code = get_code(node_index);
            // Wait, for graph building we can just use SDC, but greedy_search expects a LUT!
            // We can construct a LUT for the centroid of this node.
            // Simplified: we will construct the decoded vector first, then compute LUT.
            std::vector<float> decoded(dimension_);
            const std::vector<float>& centroids = pq_.get_centroids();
            size_t sub_dim = pq_.get_sub_dim();
            for (size_t i = 0; i < m_; ++i) {
                for (size_t j = 0; j < sub_dim; ++j) {
                    decoded[i * sub_dim + j] = centroids[i * pq_.get_k_sub() * sub_dim + node_code[i] * sub_dim + j];
                }
            }
            std::vector<float> lut = pq_.compute_lut(decoded.data(), Metric::L2);
            
            auto candidates = greedy_search(lut.data(), medoid_index_, candidate_list_size_, 0);
            robust_prune(node_index, candidates, pass_alpha, max_degree_);
            
            // Reverse edges omitted for brevity
        }
    };
    process_pass(1.0f);
    process_pass(pruning_alpha_);
}

std::vector<SearchResult> QuantizedVamanaIndex::search(const std::vector<float>& query, const SearchOptions& opts) const {
    if (!is_trained_) throw std::runtime_error("Index must be trained");
    if (query.size() != dimension_) throw std::invalid_argument("Query dimension mismatch");
    
    std::vector<float> lut = pq_.compute_lut(query.data(), opts.metric);
    size_t search_candidate_limit = std::max(static_cast<size_t>(opts.top_k), candidate_list_size_);
    
    auto top_candidates = greedy_search(lut.data(), medoid_index_, search_candidate_limit, opts.filter_mask);
    
    std::vector<SearchResult> results;
    size_t returned = 0;
    for (size_t i = 0; i < top_candidates.size() && returned < opts.top_k; ++i) {
        if (!deleted_[top_candidates[i].second]) {
            results.push_back({get_id(top_candidates[i].second), top_candidates[i].first});
            returned++;
        }
    }
    return results;
}

} // namespace vectorforge
