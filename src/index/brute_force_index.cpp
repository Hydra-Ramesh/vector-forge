#include <omp.h>
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/math.hpp"
#include <queue>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

namespace vectorforge {

static const char MAGIC_BYTES[8] = {'V','F','O','R','G','E','0','1'};
static const uint32_t FORMAT_VERSION = 1;

struct IndexHeader {
    char magic[8];
    uint32_t version;
    uint32_t dimension;
    uint64_t count;
};

BruteForceIndex::BruteForceIndex(size_t dim) : dim_(dim), num_vectors_(0), active_ids_(nullptr), active_vectors_(nullptr) {
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
}

BruteForceIndex::BruteForceIndex(const BruteForceIndex& other) 
    : dim_(other.dim_), num_vectors_(other.num_vectors_),
      owned_ids_(other.owned_ids_), owned_vectors_(other.owned_vectors_) {
    if (num_vectors_ > 0) {
        active_ids_ = owned_ids_.data();
        active_vectors_ = owned_vectors_.data();
    } else {
        active_ids_ = nullptr;
        active_vectors_ = nullptr;
    }
    // Note: We don't copy mmap_reader_
}

BruteForceIndex& BruteForceIndex::operator=(const BruteForceIndex& other) {
    if (this != &other) {
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
        owned_ids_ = other.owned_ids_;
        owned_vectors_ = other.owned_vectors_;
        if (num_vectors_ > 0) {
            active_ids_ = owned_ids_.data();
            active_vectors_ = owned_vectors_.data();
        } else {
            active_ids_ = nullptr;
            active_vectors_ = nullptr;
        }
        mmap_reader_.reset();
    }
    return *this;
}

BruteForceIndex::BruteForceIndex(BruteForceIndex&& other) noexcept
    : dim_(other.dim_), num_vectors_(other.num_vectors_),
      owned_ids_(std::move(other.owned_ids_)), owned_vectors_(std::move(other.owned_vectors_)),
      mmap_reader_(std::move(other.mmap_reader_)) {
    if (num_vectors_ > 0) {
        active_ids_ = owned_ids_.data();
        active_vectors_ = owned_vectors_.data();
    } else {
        active_ids_ = nullptr;
        active_vectors_ = nullptr;
    }
    other.num_vectors_ = 0;
    other.active_ids_ = nullptr;
    other.active_vectors_ = nullptr;
}

BruteForceIndex& BruteForceIndex::operator=(BruteForceIndex&& other) noexcept {
    if (this != &other) {
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
        owned_ids_ = std::move(other.owned_ids_);
        owned_vectors_ = std::move(other.owned_vectors_);
        mmap_reader_ = std::move(other.mmap_reader_);
        if (num_vectors_ > 0) {
            active_ids_ = owned_ids_.data();
            active_vectors_ = owned_vectors_.data();
        } else {
            active_ids_ = nullptr;
            active_vectors_ = nullptr;
        }
        other.num_vectors_ = 0;
        other.active_ids_ = nullptr;
        other.active_vectors_ = nullptr;
    }
    return *this;
}

void BruteForceIndex::add(VectorId id, const Vector& vector) {
    if (mmap_reader_ && mmap_reader_->is_open()) {
        throw std::runtime_error("Cannot add to a memory-mapped index. Load normally instead.");
    }
    if (vector.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    owned_ids_.push_back(id);
    owned_vectors_.insert(owned_vectors_.end(), vector.begin(), vector.end());
    
    num_vectors_++;
    active_ids_ = owned_ids_.data();
    active_vectors_ = owned_vectors_.data();
}

void BruteForceIndex::build() {
    active_ids_ = owned_ids_.data();
    active_vectors_ = owned_vectors_.data();
}

std::vector<SearchResult> BruteForceIndex::search(const Vector& query, const SearchOptions& options) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }

    if (num_vectors_ == 0) return {};

    int thread_count = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        thread_count = omp_get_num_threads();
    }
#endif

    std::vector<std::priority_queue<SearchResult>> local_queues(thread_count);
    const float* query_data = query.data();

    #pragma omp parallel for
    for (int64_t vector_index = 0; vector_index < static_cast<int64_t>(num_vectors_); ++vector_index) {
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        float distance = compute_distance(query_data, active_vectors_ + vector_index * dim_, dim_, options.metric);
        auto& local_queue = local_queues[thread_index];
        
        if (local_queue.size() < static_cast<size_t>(options.top_k)) {
            local_queue.push({active_ids_[vector_index], distance});
        } else if (distance < local_queue.top().distance) {
            local_queue.pop();
            local_queue.push({active_ids_[vector_index], distance});
        }
    }

    std::priority_queue<SearchResult> global_queue;
    for (auto& local_queue : local_queues) {
        while (!local_queue.empty()) {
            if (global_queue.size() < static_cast<size_t>(options.top_k)) {
                global_queue.push(local_queue.top());
            } else if (local_queue.top().distance < global_queue.top().distance) {
                global_queue.pop();
                global_queue.push(local_queue.top());
            }
            local_queue.pop();
        }
    }

    std::vector<SearchResult> results;
    results.reserve(global_queue.size());
    while (!global_queue.empty()) {
        results.push_back(global_queue.top());
        global_queue.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

void BruteForceIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    IndexHeader header;
    std::memcpy(header.magic, MAGIC_BYTES, 8);
    header.version = FORMAT_VERSION;
    header.dimension = static_cast<uint32_t>(dim_);
    header.count = static_cast<uint64_t>(num_vectors_);

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    if (num_vectors_ > 0) {
        out.write(reinterpret_cast<const char*>(active_ids_), num_vectors_ * sizeof(VectorId));
        out.write(reinterpret_cast<const char*>(active_vectors_), num_vectors_ * dim_ * sizeof(float));
    }
}

void BruteForceIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(IndexHeader))) {
        throw std::runtime_error("File too small to contain header");
    }

    IndexHeader header;
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("Failed to read header");
    }

    if (std::memcmp(header.magic, MAGIC_BYTES, 8) != 0) {
        throw std::runtime_error("Invalid magic bytes, not a VectorForge file");
    }
    if (header.version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported file version");
    }

    size_t expected_size = sizeof(IndexHeader) + 
                           header.count * sizeof(VectorId) + 
                           header.count * header.dimension * sizeof(float);
    if (static_cast<size_t>(file_size) != expected_size) {
        throw std::runtime_error("File size does not match expected size from header");
    }

    dim_ = header.dimension;
    num_vectors_ = header.count;

    owned_ids_.resize(num_vectors_);
    owned_vectors_.resize(num_vectors_ * dim_);

    if (num_vectors_ > 0) {
        in.read(reinterpret_cast<char*>(owned_ids_.data()), num_vectors_ * sizeof(VectorId));
        in.read(reinterpret_cast<char*>(owned_vectors_.data()), num_vectors_ * dim_ * sizeof(float));
    }

    active_ids_ = owned_ids_.data();
    active_vectors_ = owned_vectors_.data();
    
    if (mmap_reader_) {
        mmap_reader_->close();
    }
}

void BruteForceIndex::load_mmap(const std::string& path) {
    if (!mmap_reader_) {
        mmap_reader_ = std::make_unique<MmapReader>();
    }
    mmap_reader_->open(path);

    const uint8_t* mapped_data = mmap_reader_->data();
    size_t mapped_size = mmap_reader_->size();

    if (mapped_size < sizeof(IndexHeader)) {
        throw std::runtime_error("File too small to contain header");
    }

    const IndexHeader* header = reinterpret_cast<const IndexHeader*>(mapped_data);
    
    if (std::memcmp(header->magic, MAGIC_BYTES, 8) != 0) {
        throw std::runtime_error("Invalid magic bytes, not a VectorForge file");
    }
    if (header->version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported file version");
    }

    size_t expected_size = sizeof(IndexHeader) + 
                           header->count * sizeof(VectorId) + 
                           header->count * header->dimension * sizeof(float);
    if (mapped_size != expected_size) {
        throw std::runtime_error("File size does not match expected size from header");
    }

    dim_ = header->dimension;
    num_vectors_ = header->count;

    if (num_vectors_ > 0) {
        active_ids_ = reinterpret_cast<const VectorId*>(mapped_data + sizeof(IndexHeader));
        active_vectors_ = reinterpret_cast<const float*>(mapped_data + sizeof(IndexHeader) + num_vectors_ * sizeof(VectorId));
    } else {
        active_ids_ = nullptr;
        active_vectors_ = nullptr;
    }

    // Clear owned memory to save space
    owned_ids_.clear();
    owned_vectors_.clear();
    owned_ids_.shrink_to_fit();
    owned_vectors_.shrink_to_fit();
}

} // namespace vectorforge
