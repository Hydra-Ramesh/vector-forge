#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/core/math.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <fstream>

namespace vectorforge {

VamanaIndex::VamanaIndex(size_t dim, size_t max_degree, size_t L, float alpha)
    : dim_(dim), R_(max_degree), L_(L), alpha_(alpha), medoid_idx_(0), num_nodes_(0) {
    node_size_bytes_ = sizeof(uint64_t) + sizeof(uint32_t) + dim_ * sizeof(float) + R_ * sizeof(size_t);
}

VamanaIndex::~VamanaIndex() {}

void VamanaIndex::add(uint64_t id, const std::vector<float>& vec) {
    if (vec.size() != dim_) throw std::invalid_argument("Vector dimension mismatch");
    
    size_t idx = num_nodes_++;
    data_.resize(num_nodes_ * node_size_bytes_);
    
    get_id(idx) = id;
    get_num_neighbors(idx) = 0;
    std::copy(vec.begin(), vec.end(), get_vector(idx));
}

float VamanaIndex::distance(const float* a, const float* b) const {
    return compute_distance(a, b, dim_, Metric::L2);
}

size_t VamanaIndex::calculate_medoid() const {
    if (num_nodes_ == 0) return 0;
    
    std::vector<float> centroid(dim_, 0.0f);
    for (size_t i = 0; i < num_nodes_; ++i) {
        const float* vec = get_vector(i);
        for (size_t d = 0; d < dim_; ++d) {
            centroid[d] += vec[d];
        }
    }
    for (size_t d = 0; d < dim_; ++d) {
        centroid[d] /= static_cast<float>(num_nodes_);
    }
    
    float min_dist = std::numeric_limits<float>::max();
    size_t best_idx = 0;
    for (size_t i = 0; i < num_nodes_; ++i) {
        float d = distance(centroid.data(), get_vector(i));
        if (d < min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

std::vector<std::pair<float, size_t>> VamanaIndex::greedy_search(const float* query, size_t start_idx, size_t L) const {
    std::vector<std::pair<float, size_t>> top_L;
    std::unordered_set<size_t> visited;
    
    // Min-heap for candidates to explore
    auto cmp = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;
    };
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, decltype(cmp)> candidates(cmp);
    
    float start_dist = distance(query, get_vector(start_idx));
    candidates.push({start_dist, start_idx});
    visited.insert(start_idx);
    top_L.push_back({start_dist, start_idx});
    
    while (!candidates.empty()) {
        auto [dist_c, c] = candidates.top();
        candidates.pop();
        
        // If the closest candidate is further than the worst in top_L, we can't improve
        float worst_in_L = top_L.back().first;
        if (top_L.size() == L && dist_c > worst_in_L) {
            break; // Stop exploring if we are expanding nodes worse than our worst candidate
        }
        
        uint32_t num_neighbors = get_num_neighbors(c);
        const size_t* neighbors = get_neighbors(c);
        
        for (uint32_t i = 0; i < num_neighbors; ++i) {
            size_t n = neighbors[i];
            if (visited.find(n) == visited.end()) {
                visited.insert(n);
                float dist_n = distance(query, get_vector(n));
                
                // Add to top_L and keep sorted
                auto it = std::lower_bound(top_L.begin(), top_L.end(), std::make_pair(dist_n, n),
                                           [](const auto& a, const auto& b) { return a.first < b.first; });
                if (it != top_L.end() || top_L.size() < L) {
                    top_L.insert(it, {dist_n, n});
                    if (top_L.size() > L) {
                        top_L.pop_back();
                    }
                    candidates.push({dist_n, n});
                }
            }
        }
    }
    return top_L;
}

void VamanaIndex::robust_prune(size_t idx, std::vector<std::pair<float, size_t>>& candidates, float alpha, size_t R) {
    // Add current neighbors to candidates
    uint32_t num_neighbors = get_num_neighbors(idx);
    size_t* neighbors = get_neighbors(idx);
    
    std::unordered_set<size_t> V_set;
    std::vector<std::pair<float, size_t>> V;
    
    // Helper to add uniquely
    auto add_to_V = [&](size_t n, float dist) {
        if (n != idx && V_set.find(n) == V_set.end()) {
            V_set.insert(n);
            V.push_back({dist, n});
        }
    };
    
    for (const auto& c : candidates) {
        add_to_V(c.second, c.first);
    }
    for (uint32_t i = 0; i < num_neighbors; ++i) {
        size_t n = neighbors[i];
        add_to_V(n, distance(get_vector(idx), get_vector(n)));
    }
    
    // Sort V by distance from idx
    std::sort(V.begin(), V.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<size_t> new_neighbors;
    while (!V.empty() && new_neighbors.size() < R) {
        // p_star is the closest in V
        size_t p_star = V.front().second;
        new_neighbors.push_back(p_star);
        
        // Remove from V any p_prime where alpha * dist(p_star, p_prime) <= dist(idx, p_prime)
        std::vector<std::pair<float, size_t>> remaining_V;
        for (size_t i = 1; i < V.size(); ++i) {
            size_t p_prime = V[i].second;
            float dist_p_star_p_prime = distance(get_vector(p_star), get_vector(p_prime));
            float dist_idx_p_prime = V[i].first;
            
            if (alpha * dist_p_star_p_prime > dist_idx_p_prime) {
                remaining_V.push_back(V[i]);
            }
        }
        V = remaining_V;
    }
    
    // Write back new neighbors
    get_num_neighbors(idx) = static_cast<uint32_t>(new_neighbors.size());
    for (size_t i = 0; i < new_neighbors.size(); ++i) {
        neighbors[i] = new_neighbors[i];
    }
}

void VamanaIndex::build() {
    if (num_nodes_ == 0) return;
    
    medoid_idx_ = calculate_medoid();
    std::cout << "Calculated medoid index: " << medoid_idx_ << "\n";
    
    // 1. Initialize random graph
    std::mt19937 rng(42);
    for (size_t i = 0; i < num_nodes_; ++i) {
        std::vector<size_t> indices(num_nodes_);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        size_t* neighbors = get_neighbors(i);
        uint32_t added = 0;
        for (size_t j = 0; j < num_nodes_ && added < R_; ++j) {
            if (indices[j] != i) {
                neighbors[added++] = indices[j];
            }
        }
        get_num_neighbors(i) = added;
    }
    
    // 2. Pass 1: alpha = 1.0
    std::cout << "Vamana Pass 1 (alpha=1.0)...\n";
    std::vector<size_t> perm(num_nodes_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    
    auto process_pass = [&](float cur_alpha, const char* pass_name) {
        for (size_t i = 0; i < num_nodes_; ++i) {
            if (i > 0 && i % 1000 == 0) std::cout << "  [" << pass_name << "] Processed " << i << " nodes...\n";
            size_t idx = perm[i];
            auto candidates = greedy_search(get_vector(idx), medoid_idx_, L_);
            robust_prune(idx, candidates, cur_alpha, R_);
            
            // Reverse edges
            uint32_t num_neighbors = get_num_neighbors(idx);
            const size_t* neighbors = get_neighbors(idx);
            for (uint32_t j = 0; j < num_neighbors; ++j) {
                size_t n = neighbors[j];
                
                // Add idx to n's neighbors
                uint32_t& n_num = get_num_neighbors(n);
                size_t* n_neighbors = get_neighbors(n);
                
                bool found = false;
                for (uint32_t k = 0; k < n_num; ++k) {
                    if (n_neighbors[k] == idx) { found = true; break; }
                }
                
                if (!found) {
                    if (n_num < R_) {
                        n_neighbors[n_num++] = idx;
                    } else {
                        // Prune n if it exceeds R
                        std::vector<std::pair<float, size_t>> n_candidates;
                        n_candidates.push_back({distance(get_vector(n), get_vector(idx)), idx});
                        robust_prune(n, n_candidates, cur_alpha, R_);
                    }
                }
            }
        }
    };
    
    process_pass(1.0f, "Pass 1");
    
    // 3. Pass 2: alpha = alpha_ (typically 1.2)
    std::cout << "Vamana Pass 2 (alpha=" << alpha_ << ")...\n";
    process_pass(alpha_, "Pass 2");
}

std::vector<SearchResult> VamanaIndex::search(const std::vector<float>& query, const SearchOptions& opts) const {
    if (query.size() != dim_) throw std::invalid_argument("Query dimension mismatch");
    
    size_t L_search = std::max(static_cast<size_t>(opts.top_k), L_);
    auto top_L = greedy_search(query.data(), medoid_idx_, L_search);
    
    std::vector<SearchResult> results;
    size_t limit = std::min(static_cast<size_t>(opts.top_k), top_L.size());
    for (size_t i = 0; i < limit; ++i) {
        results.push_back({get_id(top_L[i].second), top_L[i].first});
    }
    return results;
}

void VamanaIndex::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&R_), sizeof(R_));
    out.write(reinterpret_cast<const char*>(&L_), sizeof(L_));
    out.write(reinterpret_cast<const char*>(&alpha_), sizeof(alpha_));
    out.write(reinterpret_cast<const char*>(&medoid_idx_), sizeof(medoid_idx_));
    out.write(reinterpret_cast<const char*>(&node_size_bytes_), sizeof(node_size_bytes_));
    out.write(reinterpret_cast<const char*>(&num_nodes_), sizeof(num_nodes_));
    
    size_t data_size = num_nodes_ * node_size_bytes_;
    out.write(reinterpret_cast<const char*>(data_.data()), data_size);
}

void VamanaIndex::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file");
    
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&R_), sizeof(R_));
    in.read(reinterpret_cast<char*>(&L_), sizeof(L_));
    in.read(reinterpret_cast<char*>(&alpha_), sizeof(alpha_));
    in.read(reinterpret_cast<char*>(&medoid_idx_), sizeof(medoid_idx_));
    in.read(reinterpret_cast<char*>(&node_size_bytes_), sizeof(node_size_bytes_));
    in.read(reinterpret_cast<char*>(&num_nodes_), sizeof(num_nodes_));
    
    size_t data_size = num_nodes_ * node_size_bytes_;
    data_.resize(data_size);
    in.read(reinterpret_cast<char*>(data_.data()), data_size);
}

} // namespace vectorforge
