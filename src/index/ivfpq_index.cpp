#include <omp.h>
#include "vectorforge/index/ivfpq_index.hpp"
#include "vectorforge/core/math.hpp"
#include "vectorforge/core/sys_utils.hpp"
#include <queue>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <limits>

namespace vectorforge {

static const char MAGIC_BYTES_IVFPQ[8] = {'V','F','P','Q','I','0','1',' '};
static const uint32_t FORMAT_VERSION = 1;

struct IVFPQHeader {
    char magic[8];
    uint32_t version;
    uint32_t dimension;
    uint64_t count;
    uint32_t nlist;
    uint32_t m;
    uint32_t k_sub;
    uint32_t metric;
    uint32_t store_raw;
    uint32_t padding;
};

IVFPQIndex::IVFPQIndex(size_t dim, size_t m, size_t k_sub, bool store_raw_vectors) 
    : dim_(dim), m_(m), k_sub_(k_sub), num_vectors_(0), nlist_(0), metric_(Metric::L2), 
      is_trained_(false), store_raw_vectors_(store_raw_vectors), pq_(dim, m, k_sub) {
}

void IVFPQIndex::train(const std::vector<Vector>& training_data, size_t nlist, Metric metric) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    nlist_ = nlist;
    metric_ = metric;

    size_t num_train = training_data.size();
    std::vector<float> flat_data(num_train * dim_);
    for (size_t i = 0; i < num_train; ++i) {
        std::copy(training_data[i].begin(), training_data[i].end(), flat_data.begin() + i * dim_);
    }

    std::cout << "Training IVF centroids...\n";
    centroids_ = train_kmeans(flat_data.data(), num_train, dim_, nlist_, metric_);
    
    // Typically for IVFPQ, PQ is trained on the residuals (data - centroid). 
    // For simplicity, we can train PQ on the absolute vectors.
    std::cout << "Training PQ codebooks...\n";
    pq_.train_flat(flat_data.data(), num_train, metric_);

    list_ids_.resize(nlist_);
    list_codes_.resize(nlist_);
    if (store_raw_vectors_) {
        list_raw_vectors_.resize(nlist_);
    }
    is_trained_ = true;
}

void IVFPQIndex::add(VectorId id, const Vector& vector) {
    if (!is_trained_) {
        throw std::runtime_error("Index must be trained before adding vectors");
    }
    
    const float* v_data = vector.data();
    float min_dist = std::numeric_limits<float>::max();
    size_t best_c = 0;
    
    for (size_t c = 0; c < nlist_; ++c) {
        float d = compute_distance(v_data, centroids_.data() + c * dim_, dim_, metric_);
        if (d < min_dist) {
            min_dist = d;
            best_c = c;
        }
    }

    std::vector<uint8_t> code = pq_.encode(v_data);

    list_ids_[best_c].push_back(id);
    list_codes_[best_c].insert(list_codes_[best_c].end(), code.begin(), code.end());
    if (store_raw_vectors_) {
        list_raw_vectors_[best_c].insert(list_raw_vectors_[best_c].end(), vector.begin(), vector.end());
    }
    num_vectors_++;
}

void IVFPQIndex::build() {
    active_list_ids_.resize(nlist_);
    active_list_codes_.resize(nlist_);
    active_list_raw_.resize(nlist_);
    active_list_sizes_.resize(nlist_);

    for (size_t c = 0; c < nlist_; ++c) {
        active_list_sizes_[c] = list_ids_[c].size();
        active_list_ids_[c] = list_ids_[c].empty() ? nullptr : list_ids_[c].data();
        active_list_codes_[c] = list_codes_[c].empty() ? nullptr : list_codes_[c].data();
        if (store_raw_vectors_) {
            active_list_raw_[c] = list_raw_vectors_[c].empty() ? nullptr : list_raw_vectors_[c].data();
        } else {
            active_list_raw_[c] = nullptr;
        }
    }
}

std::vector<SearchResult> IVFPQIndex::search(const Vector& query, const SearchOptions& options, size_t nprobe, size_t rerank_n) const {
    Timer total_timer;
    
    if (!is_trained_) throw std::runtime_error("Index not trained");
    if (nprobe == 0 || nprobe > nlist_) nprobe = nlist_;

    const float* q_data = query.data();

    // 1. Find top nprobe centroids using std::nth_element (O(N) instead of O(N log K))
    Timer centroid_timer;
    std::vector<SearchResult> centroid_dists(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
        centroid_dists[c].id = c;
        centroid_dists[c].distance = compute_distance(q_data, centroids_.data() + c * dim_, dim_, options.metric);
    }
    
    std::nth_element(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end());
    
    std::vector<size_t> target_lists(nprobe);
    for (size_t i = 0; i < nprobe; ++i) {
        target_lists[i] = static_cast<size_t>(centroid_dists[i].id);
    }
    
    if (options.stats) options.stats->centroid_search_ms += centroid_timer.elapsed_ms();

    // 2. Precompute LUT for ADC
    Timer lut_timer;
    std::vector<float> lut = pq_.compute_lut(q_data, options.metric);
    if (options.stats) options.stats->lut_compute_ms += lut_timer.elapsed_ms();

    // We might need to keep more candidates if we want to rerank top N
    size_t keep_k = std::max(static_cast<size_t>(options.top_k), rerank_n);

    Timer scan_timer;
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
#endif

    std::vector<std::priority_queue<SearchResult>> local_queues(num_threads);

    // 3. Search target lists using ADC
    #pragma omp parallel for
    for (int64_t idx = 0; idx < static_cast<int64_t>(target_lists.size()); ++idx) {
        size_t c = target_lists[idx];
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& q = local_queues[tid];
        size_t list_size = active_list_sizes_[c];
        const VectorId* ids_ptr = active_list_ids_[c];
        const uint8_t* codes_ptr = active_list_codes_[c];
        
        for (size_t i = 0; i < list_size; ++i) {
            // Hardware Prefetch: Hide memory latency by requesting the CPU to fetch future codes into L1 cache
            if (i + 16 < list_size) {
                VFORGE_PREFETCH(codes_ptr + (i + 16) * m_, 0, 1);
            }

            float dist = pq_.compute_adc(codes_ptr + i * m_, lut.data());
            VectorId compound_id = (static_cast<VectorId>(c) << 32) | static_cast<VectorId>(i);
            
            if (q.size() < keep_k) {
                q.push({compound_id, dist});
            } else if (dist < q.top().distance) {
                q.pop();
                q.push({compound_id, dist});
            }
        }
    }

    std::priority_queue<SearchResult> global_queue;
    for (auto& q : local_queues) {
        while (!q.empty()) {
            if (global_queue.size() < keep_k) {
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
    if (options.stats) options.stats->list_scan_ms += scan_timer.elapsed_ms();

    // 4. Optional exact reranking
    Timer rerank_timer;
    if (rerank_n > 0 && store_raw_vectors_) {
        size_t rerank_limit = std::min(rerank_n, results.size());
        for (size_t i = 0; i < rerank_limit; ++i) {
            size_t c = results[i].id >> 32;
            size_t idx = results[i].id & 0xFFFFFFFF;
            
            // Prefetch the raw vector for the NEXT candidate to completely hide RAM latency
            if (i + 1 < rerank_limit) {
                size_t next_c = results[i+1].id >> 32;
                size_t next_idx = results[i+1].id & 0xFFFFFFFF;
                VFORGE_PREFETCH(active_list_raw_[next_c] + next_idx * dim_, 0, 1);
            }

            float exact_dist = compute_distance(q_data, active_list_raw_[c] + idx * dim_, dim_, options.metric);
            results[i].distance = exact_dist;
        }
        // Resort the top rerank_n items based on exact distances
        std::sort(results.begin(), results.begin() + std::min(rerank_n, results.size()));
    }
    if (options.stats) options.stats->rerank_ms += rerank_timer.elapsed_ms();

    // 5. Restore original IDs and truncate to top_k
    for (auto& res : results) {
        size_t c = res.id >> 32;
        size_t idx = res.id & 0xFFFFFFFF;
        res.id = active_list_ids_[c][idx];
    }
    
    if (results.size() > static_cast<size_t>(options.top_k)) {
        results.resize(options.top_k);
    }
    
    if (options.stats) options.stats->total_ms += total_timer.elapsed_ms();

    return results;
}

std::vector<std::vector<SearchResult>> IVFPQIndex::search_batch(
    const std::vector<Vector>& queries, const SearchOptions& options, size_t nprobe, size_t rerank_n) const {
    
    std::vector<std::vector<SearchResult>> all_results(queries.size());
    
    // Batch parallelization: Thread across queries instead of across inverted lists for a single query.
    // This perfectly saturates CPUs without inner thread spin-up overhead.
    #pragma omp parallel for schedule(dynamic)
    for (int64_t q_idx = 0; q_idx < static_cast<int64_t>(queries.size()); ++q_idx) {
        // Internal OpenMP pragmas in search() will naturally serialize (nested parallel off by default),
        // guaranteeing max QPS for batch queries.
        SearchOptions local_opts = options;
        local_opts.stats = nullptr; // Ignore stats for batch throughput
        all_results[q_idx] = search(queries[q_idx], local_opts, nprobe, rerank_n);
    }
    
    return all_results;
}

void IVFPQIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing: " + path);

    IVFPQHeader header;
    std::memcpy(header.magic, MAGIC_BYTES_IVFPQ, 8);
    header.version = FORMAT_VERSION;
    header.dimension = static_cast<uint32_t>(dim_);
    header.count = static_cast<uint64_t>(num_vectors_);
    header.nlist = static_cast<uint32_t>(nlist_);
    header.m = static_cast<uint32_t>(m_);
    header.k_sub = static_cast<uint32_t>(k_sub_);
    header.metric = static_cast<uint32_t>(metric_);
    header.store_raw = store_raw_vectors_ ? 1 : 0;
    header.padding = 0;

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(centroids_.data()), nlist_ * dim_ * sizeof(float));
    out.write(reinterpret_cast<const char*>(pq_.get_centroids().data()), m_ * k_sub_ * (dim_ / m_) * sizeof(float));

    // Write list sizes
    std::vector<uint32_t> list_sizes(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
        list_sizes[c] = static_cast<uint32_t>(active_list_sizes_[c]);
    }
    out.write(reinterpret_cast<const char*>(list_sizes.data()), nlist_ * sizeof(uint32_t));

    // Write list offsets so we can map easily
    std::vector<uint64_t> list_offsets(nlist_);
    uint64_t current_offset = sizeof(header) + 
                              nlist_ * dim_ * sizeof(float) + 
                              m_ * k_sub_ * (dim_ / m_) * sizeof(float) +
                              nlist_ * sizeof(uint32_t) +
                              nlist_ * sizeof(uint64_t);
                              
    for (size_t c = 0; c < nlist_; ++c) {
        list_offsets[c] = current_offset;
        uint32_t sz = list_sizes[c];
        if (sz > 0) {
            current_offset += sz * sizeof(VectorId);
            current_offset += sz * m_ * sizeof(uint8_t);
            if (store_raw_vectors_) {
                current_offset += sz * dim_ * sizeof(float);
            }
        }
    }
    out.write(reinterpret_cast<const char*>(list_offsets.data()), nlist_ * sizeof(uint64_t));

    for (size_t c = 0; c < nlist_; ++c) {
        uint32_t sz = list_sizes[c];
        if (sz > 0) {
            out.write(reinterpret_cast<const char*>(active_list_ids_[c]), sz * sizeof(VectorId));
            out.write(reinterpret_cast<const char*>(active_list_codes_[c]), sz * m_ * sizeof(uint8_t));
            if (store_raw_vectors_) {
                out.write(reinterpret_cast<const char*>(active_list_raw_[c]), sz * dim_ * sizeof(float));
            }
        }
    }
}

void IVFPQIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for reading: " + path);

    IVFPQHeader header;
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header))) throw std::runtime_error("Failed to read header");
    if (std::memcmp(header.magic, MAGIC_BYTES_IVFPQ, 8) != 0) throw std::runtime_error("Invalid magic bytes");

    dim_ = header.dimension;
    num_vectors_ = header.count;
    nlist_ = header.nlist;
    m_ = header.m;
    k_sub_ = header.k_sub;
    metric_ = static_cast<Metric>(header.metric);
    store_raw_vectors_ = (header.store_raw == 1);
    is_trained_ = true;

    pq_ = ProductQuantizer(dim_, m_, k_sub_);
    
    centroids_.resize(nlist_ * dim_);
    in.read(reinterpret_cast<char*>(centroids_.data()), nlist_ * dim_ * sizeof(float));
    
    std::vector<float> pq_centroids(m_ * k_sub_ * (dim_ / m_));
    in.read(reinterpret_cast<char*>(pq_centroids.data()), pq_centroids.size() * sizeof(float));
    pq_.set_centroids(pq_centroids);

    list_ids_.resize(nlist_);
    list_codes_.resize(nlist_);
    if (store_raw_vectors_) list_raw_vectors_.resize(nlist_);

    std::vector<uint32_t> list_sizes(nlist_);
    in.read(reinterpret_cast<char*>(list_sizes.data()), nlist_ * sizeof(uint32_t));

    std::vector<uint64_t> list_offsets(nlist_);
    in.read(reinterpret_cast<char*>(list_offsets.data()), nlist_ * sizeof(uint64_t));

    for (size_t c = 0; c < nlist_; ++c) {
        uint32_t list_sz = list_sizes[c];
        if (list_sz > 0) {
            list_ids_[c].resize(list_sz);
            list_codes_[c].resize(list_sz * m_);
            in.read(reinterpret_cast<char*>(list_ids_[c].data()), list_sz * sizeof(VectorId));
            in.read(reinterpret_cast<char*>(list_codes_[c].data()), list_sz * m_ * sizeof(uint8_t));
            
            if (store_raw_vectors_) {
                list_raw_vectors_[c].resize(list_sz * dim_);
                in.read(reinterpret_cast<char*>(list_raw_vectors_[c].data()), list_sz * dim_ * sizeof(float));
            }
        }
    }
    
    build(); // Setup active pointers
}

void IVFPQIndex::load_mmap(const std::string& path) {
    mmap_reader_.open(path);
    const uint8_t* ptr = mmap_reader_.data();

    const IVFPQHeader* header = reinterpret_cast<const IVFPQHeader*>(ptr);
    if (std::memcmp(header->magic, MAGIC_BYTES_IVFPQ, 8) != 0) throw std::runtime_error("Invalid magic bytes");

    dim_ = header->dimension;
    num_vectors_ = header->count;
    nlist_ = header->nlist;
    m_ = header->m;
    k_sub_ = header->k_sub;
    metric_ = static_cast<Metric>(header->metric);
    store_raw_vectors_ = (header->store_raw == 1);
    is_trained_ = true;
    
    ptr += sizeof(IVFPQHeader);

    centroids_.resize(nlist_ * dim_);
    std::memcpy(centroids_.data(), ptr, nlist_ * dim_ * sizeof(float));
    ptr += nlist_ * dim_ * sizeof(float);

    pq_ = ProductQuantizer(dim_, m_, k_sub_);
    std::vector<float> pq_centroids(m_ * k_sub_ * (dim_ / m_));
    std::memcpy(pq_centroids.data(), ptr, pq_centroids.size() * sizeof(float));
    pq_.set_centroids(pq_centroids);
    ptr += pq_centroids.size() * sizeof(float);

    const uint32_t* sizes_ptr = reinterpret_cast<const uint32_t*>(ptr);
    ptr += nlist_ * sizeof(uint32_t);

    const uint64_t* offsets_ptr = reinterpret_cast<const uint64_t*>(ptr);
    ptr += nlist_ * sizeof(uint64_t);

    active_list_sizes_.resize(nlist_);
    active_list_ids_.resize(nlist_);
    active_list_codes_.resize(nlist_);
    active_list_raw_.resize(nlist_);

    const uint8_t* base = mmap_reader_.data();
    for (size_t c = 0; c < nlist_; ++c) {
        active_list_sizes_[c] = sizes_ptr[c];
        if (sizes_ptr[c] > 0) {
            uint64_t offset = offsets_ptr[c];
            active_list_ids_[c] = reinterpret_cast<const VectorId*>(base + offset);
            active_list_codes_[c] = reinterpret_cast<const uint8_t*>(base + offset + sizes_ptr[c] * sizeof(VectorId));
            if (store_raw_vectors_) {
                active_list_raw_[c] = reinterpret_cast<const float*>(base + offset + sizes_ptr[c] * sizeof(VectorId) + sizes_ptr[c] * m_ * sizeof(uint8_t));
            } else {
                active_list_raw_[c] = nullptr;
            }
        } else {
            active_list_ids_[c] = nullptr;
            active_list_codes_[c] = nullptr;
            active_list_raw_[c] = nullptr;
        }
    }
}

} // namespace vectorforge
