#pragma once

#include "vectorforge/core/types.hpp"
#include "vectorforge/core/mmap_reader.hpp"
#include <vector>
#include <string>
#include <memory>

namespace vectorforge {

class BruteForceIndex {
public:
    BruteForceIndex(size_t dim);
    BruteForceIndex(const BruteForceIndex& other);
    BruteForceIndex& operator=(const BruteForceIndex& other);
    BruteForceIndex(BruteForceIndex&& other) noexcept;
    BruteForceIndex& operator=(BruteForceIndex&& other) noexcept;

    void add(VectorId id, const Vector& vector);
    void build();
    std::vector<SearchResult> search(const Vector& query, const SearchOptions& options) const;
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    void load_mmap(const std::string& path);

    size_t size() const { return num_vectors_; }
    size_t dimension() const { return dim_; }

private:
    size_t dim_;
    size_t num_vectors_;
    
    // Owned data (used when building or standard loading)
    std::vector<VectorId> owned_ids_;
    std::vector<float> owned_vectors_;

    // Pointers to active data (can point to owned data or mmap data)
    const VectorId* active_ids_;
    const float* active_vectors_;

    // Mmap reader for memory-mapped mode
    std::unique_ptr<MmapReader> mmap_reader_;
};

} // namespace vectorforge
