#include "vectorforge/index/hnsw_index.hpp"
#include "vectorforge/core/math.hpp"
#include <stdexcept>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace vectorforge {

HNSWIndex::HNSWIndex(size_t dimension, int max_connections, int construction_expansion)
    : dim_(dimension), max_connections_(max_connections), max_layer_zero_connections_(2 * max_connections), construction_expansion_(construction_expansion),
      level_multiplier_(1 / log(1.0 * max_connections)), num_vectors_(0), max_level_(-1), enterpoint_node_(-1) {
}

float HNSWIndex::distance(const float* left_vector, const float* right_vector) const {
    return compute_distance(left_vector, right_vector, dim_, Metric::L2);
}

int HNSWIndex::generate_random_level() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_value = -log(distribution(rng_)) * level_multiplier_;
    return static_cast<int>(random_value);
}

void HNSWIndex::add(VectorId id, const Vector& vector) {
    if (vector.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    int32_t internal_index = static_cast<int32_t>(num_vectors_);
    owned_ids_.push_back(id);
    owned_vectors_.insert(owned_vectors_.end(), vector.begin(), vector.end());
    
    int level = generate_random_level();
    HNSWNode node;
    node.id = id;
    node.max_level = level;
    node.neighbors.resize(level + 1);
    nodes_.push_back(node);
    
    num_vectors_++;
    
    insert(internal_index, owned_vectors_.data() + internal_index * dim_);
}

void HNSWIndex::build() {
    // HNSW graph is built incrementally inside `add`
}

void HNSWIndex::insert(int32_t internal_index, const float* query_vector) {
    HNSWNode& node = nodes_[internal_index];
    int level = node.max_level;

    if (enterpoint_node_ == -1) {
        enterpoint_node_ = internal_index;
        max_level_ = level;
        return;
    }

    int32_t curr_obj = enterpoint_node_;
    float current_distance = distance(query_vector, owned_vectors_.data() + curr_obj * dim_);
    
    // Phase 1: greedy search from max_level to level + 1
    for (int lc = max_level_; lc > level; lc--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& neighbors = nodes_[curr_obj].neighbors[lc];
            for (int32_t neighbor_index : neighbors) {
                float neighbor_distance = distance(query_vector, owned_vectors_.data() + neighbor_index * dim_);
                if (neighbor_distance < current_distance) {
                    current_distance = neighbor_distance;
                    curr_obj = neighbor_index;
                    changed = true;
                }
            }
        }
    }
    
    std::vector<int32_t> eps = {curr_obj};
    
    // Phase 2: insert at layers level down to 0
    for (int lc = std::min(max_level_, level); lc >= 0; lc--) {
        std::priority_queue<std::pair<float, int32_t>> top_candidates;
        search_layer(query_vector, eps, construction_expansion_, lc, top_candidates);
        
        std::vector<int32_t> selected = select_neighbors(query_vector, top_candidates, lc == 0 ? max_layer_zero_connections_ : max_connections_, lc);
        
        // Add connections
        node.neighbors[lc] = selected;
        for (int32_t neighbor : selected) {
            auto& n_neighbors = nodes_[neighbor].neighbors[lc];
            n_neighbors.push_back(internal_index);
            
            int connection_limit = (lc == 0) ? max_layer_zero_connections_ : max_connections_;
            // Prune connections if needed
            if (n_neighbors.size() > connection_limit) {
                std::priority_queue<std::pair<float, int32_t>> candidates;
                for (int32_t n : n_neighbors) {
                    float d = distance(owned_vectors_.data() + neighbor * dim_, owned_vectors_.data() + n * dim_);
                    candidates.push({d, n});
                }
                auto new_conn = select_neighbors(owned_vectors_.data() + neighbor * dim_, candidates, connection_limit, lc);
                n_neighbors = new_conn;
            }
        }
        
        // eps for next layer
        eps.clear();
        for (int32_t n : selected) {
            eps.push_back(n);
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        enterpoint_node_ = internal_index;
    }
}

void HNSWIndex::search_layer(
    const float* query_vector,
    std::vector<int32_t>& entry_points,
    int search_expansion,
    int level,
    std::priority_queue<std::pair<float, int32_t>>& candidate_queue) const
{
    std::priority_queue<std::pair<float, int32_t>, std::vector<std::pair<float, int32_t>>, std::greater<std::pair<float, int32_t>>> candidates;
    
    std::unordered_set<int32_t> visited;
    
    for (int32_t entry_point : entry_points) {
        float distance_to_entry = distance(query_vector, owned_vectors_.data() + entry_point * dim_);
        candidates.push({distance_to_entry, entry_point});
        candidate_queue.push({distance_to_entry, entry_point});
        visited.insert(entry_point);
    }
    
    while (!candidates.empty()) {
        auto [candidate_distance, candidate_index] = candidates.top();
        candidates.pop();
        
        if (candidate_queue.size() >= search_expansion && candidate_distance > candidate_queue.top().first) {
            break;
        }
        
        for (int32_t neighbor_index : nodes_[candidate_index].neighbors[level]) {
            if (visited.find(neighbor_index) == visited.end()) {
                visited.insert(neighbor_index);
                float neighbor_distance = distance(query_vector, owned_vectors_.data() + neighbor_index * dim_);
                
                if (candidate_queue.size() < search_expansion || neighbor_distance < candidate_queue.top().first) {
                    candidates.push({neighbor_distance, neighbor_index});
                    candidate_queue.push({neighbor_distance, neighbor_index});
                    
                    if (candidate_queue.size() > search_expansion) {
                        candidate_queue.pop();
                    }
                }
            }
        }
    }
}

// select neighbors
std::vector<int32_t> HNSWIndex::select_neighbors(
    const float* query_vector,
    std::priority_queue<std::pair<float, int32_t>>& candidates, 
    int max_neighbors,
    int level) 
{
    std::vector<int32_t> result;
    while (candidates.size() > max_neighbors) {
        candidates.pop();
    }
    
    while (!candidates.empty()) {
        result.push_back(candidates.top().second);
        candidates.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<SearchResult> HNSWIndex::search(const Vector& query, const SearchOptions& options) const {
    std::vector<SearchResult> results;
    if (num_vectors_ == 0) return results;
    
    int32_t current_node = enterpoint_node_;
    float current_distance = distance(query.data(), owned_vectors_.data() + current_node * dim_);
    
    // search to layer 1
    for (int lc = max_level_; lc > 0; lc--) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int32_t neighbor_index : nodes_[current_node].neighbors[lc]) {
                float neighbor_distance = distance(query.data(), owned_vectors_.data() + neighbor_index * dim_);
                if (neighbor_distance < current_distance) {
                    current_distance = neighbor_distance;
                    current_node = neighbor_index;
                    changed = true;
                }
            }
        }
    }
    
    std::vector<int32_t> entry_points = {current_node};
    std::priority_queue<std::pair<float, int32_t>> candidate_queue;
    int search_expansion = std::max(options.top_k, 50);
    
    search_layer(query.data(), entry_points, search_expansion, 0, candidate_queue);
    
    while (candidate_queue.size() > options.top_k) {
        candidate_queue.pop();
    }
    
    while (!candidate_queue.empty()) {
        auto [distance, node_index] = candidate_queue.top();
        candidate_queue.pop();
        results.push_back({owned_ids_[node_index], distance});
    }
    
    std::reverse(results.begin(), results.end());
    
    return results;
}

void HNSWIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for saving");
    
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&max_connections_), sizeof(max_connections_));
    out.write(reinterpret_cast<const char*>(&max_layer_zero_connections_), sizeof(max_layer_zero_connections_));
    out.write(reinterpret_cast<const char*>(&construction_expansion_), sizeof(construction_expansion_));
    out.write(reinterpret_cast<const char*>(&num_vectors_), sizeof(num_vectors_));
    out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
    out.write(reinterpret_cast<const char*>(&enterpoint_node_), sizeof(enterpoint_node_));
    
    if (num_vectors_ > 0) {
        out.write(reinterpret_cast<const char*>(owned_ids_.data()), num_vectors_ * sizeof(VectorId));
        out.write(reinterpret_cast<const char*>(owned_vectors_.data()), num_vectors_ * dim_ * sizeof(float));
    }
    
    for (size_t i = 0; i < num_vectors_; ++i) {
        const HNSWNode& node = nodes_[i];
        out.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        out.write(reinterpret_cast<const char*>(&node.max_level), sizeof(node.max_level));
        for (int l = 0; l <= node.max_level; ++l) {
            size_t n_size = node.neighbors[l].size();
            out.write(reinterpret_cast<const char*>(&n_size), sizeof(n_size));
            if (n_size > 0) {
                out.write(reinterpret_cast<const char*>(node.neighbors[l].data()), n_size * sizeof(int32_t));
            }
        }
    }
}

void HNSWIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for loading");
    
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&max_connections_), sizeof(max_connections_));
    in.read(reinterpret_cast<char*>(&max_layer_zero_connections_), sizeof(max_layer_zero_connections_));
    in.read(reinterpret_cast<char*>(&construction_expansion_), sizeof(construction_expansion_));
    in.read(reinterpret_cast<char*>(&num_vectors_), sizeof(num_vectors_));
    in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
    in.read(reinterpret_cast<char*>(&enterpoint_node_), sizeof(enterpoint_node_));
    
    if (num_vectors_ > 0) {
        owned_ids_.resize(num_vectors_);
        in.read(reinterpret_cast<char*>(owned_ids_.data()), num_vectors_ * sizeof(VectorId));
        
        owned_vectors_.resize(num_vectors_ * dim_);
        in.read(reinterpret_cast<char*>(owned_vectors_.data()), num_vectors_ * dim_ * sizeof(float));
    }
    
    nodes_.resize(num_vectors_);
    for (size_t i = 0; i < num_vectors_; ++i) {
        HNSWNode& node = nodes_[i];
        in.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        in.read(reinterpret_cast<char*>(&node.max_level), sizeof(node.max_level));
        node.neighbors.resize(node.max_level + 1);
        for (int l = 0; l <= node.max_level; ++l) {
            size_t n_size;
            in.read(reinterpret_cast<char*>(&n_size), sizeof(n_size));
            if (n_size > 0) {
                node.neighbors[l].resize(n_size);
                in.read(reinterpret_cast<char*>(node.neighbors[l].data()), n_size * sizeof(int32_t));
            }
        }
    }
}

} // namespace vectorforge
