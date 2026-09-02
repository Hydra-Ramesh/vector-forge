#pragma once

#include "vectorforge/core/types.hpp"
#include <vector>
#include <string>
#include <memory>

namespace vectorforge {

class IVFIndex {
public:
    IVFIndex(size_t dim);

    void train(const std::vector<Vector>& training_data, size_t nlist, Metric metric);
    void add(VectorId id, const Vector& vector);
    void build();
    std::vector<SearchResult> search(const Vector& query, const SearchOptions& options, size_t nprobe) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t size() const { return num_vectors_; }
    size_t dimension() const { return dim_; }
    size_t nlist() const { return nlist_; }

private:
    size_t dim_;
    size_t num_vectors_;
    size_t nlist_;
    Metric metric_;
    bool is_trained_;

    std::vector<float> centroids_;
    std::vector<std::vector<VectorId>> list_ids_;
    std::vector<std::vector<float>> list_vectors_;
};

} // namespace vectorforge
