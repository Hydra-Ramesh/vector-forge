#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/core/math.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <fstream>

namespace vectorforge {

VamanaIndex::VamanaIndex(size_t dimension, size_t max_degree, size_t candidate_list_size, float pruning_alpha)
    : dimension_(dimension), max_degree_(max_degree), candidate_list_size_(candidate_list_size), pruning_alpha_(pruning_alpha), medoid_index_(0), num_nodes_(0) {
    node_size_bytes_ = sizeof(uint64_t) + sizeof(uint32_t) + dimension_ * sizeof(float) + max_degree_ * sizeof(size_t);
}

VamanaIndex::~VamanaIndex() {}

void VamanaIndex::add(uint64_t id, const std::vector<float>& vector) {
    if (vector.size() != dimension_) throw std::invalid_argument("Vector dimension mismatch");
    
    size_t node_index = num_nodes_++;
    data_.resize(num_nodes_ * node_size_bytes_);
    
    get_id(node_index) = id;
    get_num_neighbors(node_index) = 0;
    std::copy(vector.begin(), vector.end(), get_vector(node_index));
}

float VamanaIndex::distance(const float* left_vector, const float* right_vector) const {
    return compute_distance(left_vector, right_vector, dimension_, Metric::L2);
}

size_t VamanaIndex::calculate_medoid() const {
    if (num_nodes_ == 0) return 0;
    
    std::vector<float> centroid(dimension_, 0.0f);
    for (size_t node_index = 0; node_index < num_nodes_; ++node_index) {
        const float* node_vector = get_vector(node_index);
        for (size_t dimension_index = 0; dimension_index < dimension_; ++dimension_index) {
            centroid[dimension_index] += node_vector[dimension_index];
        }
    }
    for (size_t dimension_index = 0; dimension_index < dimension_; ++dimension_index) {
        centroid[dimension_index] /= static_cast<float>(num_nodes_);
    }
    
    float nearest_distance = std::numeric_limits<float>::max();
    size_t nearest_index = 0;
    for (size_t node_index = 0; node_index < num_nodes_; ++node_index) {
        float distance_to_centroid = distance(centroid.data(), get_vector(node_index));
        if (distance_to_centroid < nearest_distance) {
            nearest_distance = distance_to_centroid;
            nearest_index = node_index;
        }
    }
    return nearest_index;
}

std::vector<std::pair<float, size_t>> VamanaIndex::greedy_search(const float* query_vector, size_t start_index, size_t candidate_list_size) const {
    std::vector<std::pair<float, size_t>> top_candidates;
    std::unordered_set<size_t> visited;
    
    auto compare_distances = [](const std::pair<float, size_t>& left_candidate, const std::pair<float, size_t>& right_candidate) {
        return left_candidate.first > right_candidate.first;
    };
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(compare_distances)> candidates(compare_distances);
    
    float start_distance = distance(query_vector, get_vector(start_index));
    candidates.push({start_distance, start_index});
    visited.insert(start_index);
    top_candidates.push_back({start_distance, start_index});
    
    while (!candidates.empty()) {
        auto [candidate_distance, candidate_index] = candidates.top();
        candidates.pop();
        
        float worst_candidate_distance = top_candidates.back().first;
        if (top_candidates.size() == candidate_list_size && candidate_distance > worst_candidate_distance) {
            break;
        }
        
        uint32_t neighbor_count = get_num_neighbors(candidate_index);
        const size_t* neighbors = get_neighbors(candidate_index);
        
        for (uint32_t neighbor_offset = 0; neighbor_offset < neighbor_count; ++neighbor_offset) {
            size_t neighbor_index = neighbors[neighbor_offset];
            if (visited.find(neighbor_index) == visited.end()) {
                visited.insert(neighbor_index);
                float neighbor_distance = distance(query_vector, get_vector(neighbor_index));
                
                auto insertion_point = std::lower_bound(top_candidates.begin(), top_candidates.end(), std::make_pair(neighbor_distance, neighbor_index),
                                                   [](const auto& left_candidate, const auto& right_candidate) { return left_candidate.first < right_candidate.first; });
                if (insertion_point != top_candidates.end() || top_candidates.size() < candidate_list_size) {
                    top_candidates.insert(insertion_point, {neighbor_distance, neighbor_index});
                    if (top_candidates.size() > candidate_list_size) {
                        top_candidates.pop_back();
                    }
                    candidates.push({neighbor_distance, neighbor_index});
                }
            }
        }
    }
    return top_candidates;
}

void VamanaIndex::robust_prune(size_t node_index, std::vector<std::pair<float, size_t>>& candidates, float pruning_alpha, size_t max_degree) {
    uint32_t neighbor_count = get_num_neighbors(node_index);
    size_t* neighbors = get_neighbors(node_index);
    
    std::unordered_set<size_t> unique_neighbors;
    std::vector<std::pair<float, size_t>> candidate_neighbors;
    
    auto add_candidate = [&](size_t neighbor_index, float neighbor_distance) {
        if (neighbor_index != node_index && unique_neighbors.find(neighbor_index) == unique_neighbors.end()) {
            unique_neighbors.insert(neighbor_index);
            candidate_neighbors.push_back({neighbor_distance, neighbor_index});
        }
    };
    
    for (const auto& candidate : candidates) {
        add_candidate(candidate.second, candidate.first);
    }
    for (uint32_t neighbor_offset = 0; neighbor_offset < neighbor_count; ++neighbor_offset) {
        size_t neighbor_index = neighbors[neighbor_offset];
        add_candidate(neighbor_index, distance(get_vector(node_index), get_vector(neighbor_index)));
    }
    
    std::sort(candidate_neighbors.begin(), candidate_neighbors.end(), [](const auto& left_candidate, const auto& right_candidate) { return left_candidate.first < right_candidate.first; });
    
    std::vector<size_t> new_neighbors;
    while (!candidate_neighbors.empty() && new_neighbors.size() < max_degree) {
        size_t closest_neighbor = candidate_neighbors.front().second;
        new_neighbors.push_back(closest_neighbor);
        
        std::vector<std::pair<float, size_t>> remaining_candidates;
        for (size_t candidate_index = 1; candidate_index < candidate_neighbors.size(); ++candidate_index) {
            size_t other_neighbor = candidate_neighbors[candidate_index].second;
            float distance_between_neighbors = distance(get_vector(closest_neighbor), get_vector(other_neighbor));
            float distance_from_node = candidate_neighbors[candidate_index].first;
            
            if (pruning_alpha * distance_between_neighbors > distance_from_node) {
                remaining_candidates.push_back(candidate_neighbors[candidate_index]);
            }
        }
        candidate_neighbors = remaining_candidates;
    }
    
    get_num_neighbors(node_index) = static_cast<uint32_t>(new_neighbors.size());
    for (size_t neighbor_offset = 0; neighbor_offset < new_neighbors.size(); ++neighbor_offset) {
        neighbors[neighbor_offset] = new_neighbors[neighbor_offset];
    }
}

void VamanaIndex::build() {
    if (num_nodes_ == 0) return;
    
    medoid_index_ = calculate_medoid();
    std::cout << "Calculated medoid index: " << medoid_index_ << "\n";
    
    // 1. Initialize random graph
    std::mt19937 rng(42);
    for (size_t i = 0; i < num_nodes_; ++i) {
        std::vector<size_t> indices(num_nodes_);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t* neighbors = get_neighbors(i);
        uint32_t added = 0;
        for (size_t candidate_offset = 0; candidate_offset < num_nodes_ && added < max_degree_; ++candidate_offset) {
            if (indices[candidate_offset] != i) {
                neighbors[added++] = indices[candidate_offset];
            }
        }
        get_num_neighbors(i) = added;
    }
    
    // 2. Pass 1: alpha = 1.0
    std::cout << "Vamana Pass 1 (alpha=1.0)...\n";
    std::vector<size_t> perm(num_nodes_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    
    auto process_pass = [&](float pass_alpha, const char* pass_name) {
        for (size_t pass_index = 0; pass_index < num_nodes_; ++pass_index) {
            if (pass_index > 0 && pass_index % 1000 == 0) std::cout << "  [" << pass_name << "] Processed " << pass_index << " nodes...\n";
            size_t node_index = perm[pass_index];
            auto candidates = greedy_search(get_vector(node_index), medoid_index_, candidate_list_size_);
            robust_prune(node_index, candidates, pass_alpha, max_degree_);
            
            // Reverse edges
            uint32_t neighbor_count = get_num_neighbors(node_index);
            const size_t* neighbors = get_neighbors(node_index);
            for (uint32_t neighbor_offset = 0; neighbor_offset < neighbor_count; ++neighbor_offset) {
                size_t neighbor_index = neighbors[neighbor_offset];
                
                uint32_t& reverse_neighbor_count = get_num_neighbors(neighbor_index);
                size_t* neighbor_list = get_neighbors(neighbor_index);
                
                bool found = false;
                for (uint32_t neighbor_offset = 0; neighbor_offset < reverse_neighbor_count; ++neighbor_offset) {
                    if (neighbor_list[neighbor_offset] == node_index) { found = true; break; }
                }
                
                if (!found) {
                    if (reverse_neighbor_count < max_degree_) {
                        neighbor_list[reverse_neighbor_count++] = node_index;
                    } else {
                        // Prune n if it exceeds R
                        std::vector<std::pair<float, size_t>> neighbor_candidates;
                        neighbor_candidates.push_back({distance(get_vector(neighbor_index), get_vector(node_index)), node_index});
                        robust_prune(neighbor_index, neighbor_candidates, pass_alpha, max_degree_);
                    }
                }
            }
        }
    };
    
    process_pass(1.0f, "Pass 1");
    
    std::cout << "Vamana Pass 2 (alpha=" << pruning_alpha_ << ")...\n";
    process_pass(pruning_alpha_, "Pass 2");
}

std::vector<SearchResult> VamanaIndex::search(const std::vector<float>& query, const SearchOptions& opts) const {
    if (query.size() != dimension_) throw std::invalid_argument("Query dimension mismatch");
    
    size_t search_candidate_limit = std::max(static_cast<size_t>(opts.top_k), candidate_list_size_);
    auto top_candidates = greedy_search(query.data(), medoid_index_, search_candidate_limit);
    
    std::vector<SearchResult> results;
    size_t result_limit = std::min(static_cast<size_t>(opts.top_k), top_candidates.size());
    for (size_t result_index = 0; result_index < result_limit; ++result_index) {
        results.push_back({get_id(top_candidates[result_index].second), top_candidates[result_index].first});
    }
    return results;
}

void VamanaIndex::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    out.write(reinterpret_cast<const char*>(&max_degree_), sizeof(max_degree_));
    out.write(reinterpret_cast<const char*>(&candidate_list_size_), sizeof(candidate_list_size_));
    out.write(reinterpret_cast<const char*>(&pruning_alpha_), sizeof(pruning_alpha_));
    out.write(reinterpret_cast<const char*>(&medoid_index_), sizeof(medoid_index_));
    out.write(reinterpret_cast<const char*>(&node_size_bytes_), sizeof(node_size_bytes_));
    out.write(reinterpret_cast<const char*>(&num_nodes_), sizeof(num_nodes_));
    
    size_t data_size = num_nodes_ * node_size_bytes_;
    out.write(reinterpret_cast<const char*>(data_.data()), data_size);
}

void VamanaIndex::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file");
    
    in.read(reinterpret_cast<char*>(&dimension_), sizeof(dimension_));
    in.read(reinterpret_cast<char*>(&max_degree_), sizeof(max_degree_));
    in.read(reinterpret_cast<char*>(&candidate_list_size_), sizeof(candidate_list_size_));
    in.read(reinterpret_cast<char*>(&pruning_alpha_), sizeof(pruning_alpha_));
    in.read(reinterpret_cast<char*>(&medoid_index_), sizeof(medoid_index_));
    in.read(reinterpret_cast<char*>(&node_size_bytes_), sizeof(node_size_bytes_));
    in.read(reinterpret_cast<char*>(&num_nodes_), sizeof(num_nodes_));
    
    size_t data_size = num_nodes_ * node_size_bytes_;
    data_.resize(data_size);
    in.read(reinterpret_cast<char*>(data_.data()), data_size);
}

} // namespace vectorforge
