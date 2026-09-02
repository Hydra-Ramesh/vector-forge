#pragma once

#include "vectorforge/core/types.hpp"
#include "vectorforge/core/pq.hpp"
#include "vectorforge/core/mmap_reader.hpp"
#include <vector>
#include <string>

namespace vectorforge {

class IVFPQIndex {
public:
    IVFPQIndex(size_t dim, size_t m, size_t k_sub = 256, bool store_raw_vectors = true);

    void train(const std::vector<Vector>& training_data, size_t nlist, Metric metric);
    void add(VectorId id, const Vector& vector);
    void build();
    
    // search with optional reranking
    std::vector<SearchResult> search(const Vector& query, const SearchOptions& options, size_t nprobe, size_t rerank_n = 0) const;
    std::vector<std::vector<SearchResult>> search_batch(const std::vector<Vector>& queries, const SearchOptions& options, size_t nprobe, size_t rerank_n = 0) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    void load_mmap(const std::string& path);

    size_t size() const { return num_vectors_; }
    size_t dimension() const { return dim_; }

private:
    size_t dim_;
    size_t m_;
    size_t k_sub_;
    size_t num_vectors_;
    size_t nlist_;
    Metric metric_;
    bool is_trained_;
    bool store_raw_vectors_;

    ProductQuantizer pq_;
    std::vector<float> centroids_;
    
    // Inverted lists (used when building or purely RAM loaded)
    std::vector<std::vector<VectorId>> list_ids_;
    std::vector<std::vector<uint8_t>> list_codes_;
    std::vector<std::vector<float>> list_raw_vectors_; 

    // Active pointers for polymorphic memory (RAM vs Mmap)
    std::vector<const VectorId*> active_list_ids_;
    std::vector<const uint8_t*> active_list_codes_;
    std::vector<const float*> active_list_raw_;
    std::vector<size_t> active_list_sizes_;

    MmapReader mmap_reader_;
};

} // namespace vectorforge
