#include <omp.h>
#include "vectorforge/index/ivf_index.hpp"
#include "vectorforge/core/math.hpp"
#include <queue>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <limits>

namespace vectorforge {

static const char MAGIC_BYTES_IVF[8] = {'V','F','I','V','F','0','1',' '};
static const uint32_t FORMAT_VERSION = 1;

struct IVFHeader {
    char magic[8];
    uint32_t version;
    uint32_t dimension;
    uint64_t count;
    uint32_t nlist;
    uint32_t metric; // 0 for L2, 1 for Cosine
};

IVFIndex::IVFIndex(size_t dim) : dim_(dim), num_vectors_(0), nlist_(0), metric_(Metric::L2), is_trained_(false) {
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
}

void IVFIndex::train(const std::vector<Vector>& training_data, size_t nlist, Metric metric) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    nlist_ = nlist;
    metric_ = metric;

    size_t training_vector_count = training_data.size();
    std::vector<float> flat_data(training_vector_count * dim_);
    for (size_t vector_index = 0; vector_index < training_vector_count; ++vector_index) {
        std::copy(training_data[vector_index].begin(), training_data[vector_index].end(), flat_data.begin() + vector_index * dim_);
    }

    centroids_ = train_kmeans(flat_data.data(), training_vector_count, dim_, nlist_, metric_);
    list_ids_.resize(nlist_);
    list_vectors_.resize(nlist_);
    is_trained_ = true;
}

void IVFIndex::add(VectorId id, const Vector& vector) {
    if (!is_trained_) {
        throw std::runtime_error("Index must be trained before adding vectors");
    }
    if (vector.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    float nearest_distance = std::numeric_limits<float>::max();
    size_t nearest_list = 0;
    const float* vector_data = vector.data();
    for (size_t list_index = 0; list_index < nlist_; ++list_index) {
        float distance = compute_distance(vector_data, centroids_.data() + list_index * dim_, dim_, metric_);
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_list = list_index;
        }
    }

    list_ids_[nearest_list].push_back(id);
    list_vectors_[nearest_list].insert(list_vectors_[nearest_list].end(), vector.begin(), vector.end());
    num_vectors_++;
}

void IVFIndex::build() {
    // Already organized in inverted lists.
}

std::vector<SearchResult> IVFIndex::search(const Vector& query, const SearchOptions& options, size_t nprobe) const {
    if (!is_trained_) {
        throw std::runtime_error("Index not trained");
    }
    if (query.size() != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    if (nprobe == 0 || nprobe > nlist_) {
        nprobe = nlist_;
    }

    const float* q_data = query.data();

    // 1. Find top `nprobe` centroids
    std::priority_queue<SearchResult> centroid_queue;
    for (size_t centroid_index = 0; centroid_index < nlist_; ++centroid_index) {
        float distance = compute_distance(q_data, centroids_.data() + centroid_index * dim_, dim_, options.metric);
        if (centroid_queue.size() < nprobe) {
            centroid_queue.push({centroid_index, distance});
        } else if (distance < centroid_queue.top().distance) {
            centroid_queue.pop();
            centroid_queue.push({c, d});
        }
    }

    std::vector<size_t> target_lists;
    while (!centroid_queue.empty()) {
        target_lists.push_back(static_cast<size_t>(centroid_queue.top().id));
        centroid_queue.pop();
    }

    // 2. Search only within target lists
    int thread_count = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        thread_count = omp_get_num_threads();
    }
#endif

    std::vector<std::priority_queue<SearchResult>> local_queues(thread_count);

    #pragma omp parallel for
    for (int64_t target_offset = 0; target_offset < static_cast<int64_t>(target_lists.size()); ++target_offset) {
        size_t list_index = target_lists[target_offset];
        int thread_index = 0;
#ifdef _OPENMP
        thread_index = omp_get_thread_num();
#endif
        auto& local_queue = local_queues[thread_index];
        size_t list_size = list_ids_[list_index].size();
        const VectorId* ids_data = list_ids_[list_index].data();
        const float* vectors_data = list_vectors_[list_index].data();

        for (size_t vector_index = 0; vector_index < list_size; ++vector_index) {
            float distance = compute_distance(q_data, vectors_data + vector_index * dim_, dim_, options.metric);
            if (local_queue.size() < static_cast<size_t>(options.top_k)) {
                local_queue.push({ids_data[vector_index], distance});
            } else if (distance < local_queue.top().distance) {
                local_queue.pop();
                local_queue.push({ids_data[vector_index], distance});
            }
        }
    }

    std::priority_queue<SearchResult> global_queue;
    for (auto& q : local_queues) {
        while (!q.empty()) {
            if (global_queue.size() < static_cast<size_t>(options.top_k)) {
                global_queue.push(q.top());
            } else if (q.top().distance < global_queue.top().distance) {
                global_queue.pop();
                global_queue.push(q.top());
            }
            q.pop();
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

void IVFIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    IVFHeader header;
    std::memcpy(header.magic, MAGIC_BYTES_IVF, 8);
    header.version = FORMAT_VERSION;
    header.dimension = static_cast<uint32_t>(dim_);
    header.count = static_cast<uint64_t>(num_vectors_);
    header.nlist = static_cast<uint32_t>(nlist_);
    header.metric = static_cast<uint32_t>(metric_);

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write centroids
    out.write(reinterpret_cast<const char*>(centroids_.data()), nlist_ * dim_ * sizeof(float));

    // Write inverted lists
    for (size_t c = 0; c < nlist_; ++c) {
        uint32_t list_sz = static_cast<uint32_t>(list_ids_[c].size());
        out.write(reinterpret_cast<const char*>(&list_sz), sizeof(list_sz));
        if (list_sz > 0) {
            out.write(reinterpret_cast<const char*>(list_ids_[c].data()), list_sz * sizeof(VectorId));
            out.write(reinterpret_cast<const char*>(list_vectors_[c].data()), list_sz * dim_ * sizeof(float));
        }
    }
}

void IVFIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    IVFHeader header;
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("Failed to read header");
    }

    if (std::memcmp(header.magic, MAGIC_BYTES_IVF, 8) != 0) {
        throw std::runtime_error("Invalid magic bytes, not a VectorForge IVF file");
    }
    
    dim_ = header.dimension;
    num_vectors_ = header.count;
    nlist_ = header.nlist;
    metric_ = static_cast<Metric>(header.metric);
    is_trained_ = true;

    centroids_.resize(nlist_ * dim_);
    in.read(reinterpret_cast<char*>(centroids_.data()), nlist_ * dim_ * sizeof(float));

    list_ids_.resize(nlist_);
    list_vectors_.resize(nlist_);

    for (size_t c = 0; c < nlist_; ++c) {
        uint32_t list_sz = 0;
        in.read(reinterpret_cast<char*>(&list_sz), sizeof(list_sz));
        if (list_sz > 0) {
            list_ids_[c].resize(list_sz);
            list_vectors_[c].resize(list_sz * dim_);
            in.read(reinterpret_cast<char*>(list_ids_[c].data()), list_sz * sizeof(VectorId));
            in.read(reinterpret_cast<char*>(list_vectors_[c].data()), list_sz * dim_ * sizeof(float));
        }
    }
}

} // namespace vectorforge
