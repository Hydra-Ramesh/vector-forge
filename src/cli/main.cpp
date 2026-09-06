#include <iostream>
#include <string>
#include "vectorforge/core/types.hpp"
#include "vectorforge/core/dataset.hpp"
#include "vectorforge/index/ivf_index.hpp"
#include "vectorforge/index/ivfpq_index.hpp"
#include "vectorforge/index/hnsw_index.hpp"
#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/core/sys_utils.hpp"
#include <chrono>

using namespace vectorforge;

int main(int argc, char** argv) {
    std::cout << "VectorForge CLI\n";
    if (argc < 2) {
        std::cerr << "Usage: vectorforge <command> [options]\n";
        return 1;
    }
    
    std::string command = argv[1];
    if (command == "benchmark") {
        size_t num_vectors = 10000;
        size_t dim = 128;
        size_t nlist = 100;
        size_t nprobe = 10;

        std::cout << "Generating dataset...\n";
        auto data = DatasetGenerator::generate(num_vectors, dim);
        
        std::cout << "Training IVF Index (nlist=" << nlist << ")...\n";
        auto start = std::chrono::high_resolution_clock::now();
        IVFIndex index(dim);
        index.train(data, nlist, Metric::L2);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Training took " << diff.count() << " s\n";
        
        std::cout << "Adding vectors...\n";
        start = std::chrono::high_resolution_clock::now();
        for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
            index.add(vector_index, data[vector_index]);
        }
        index.build();
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Adding took " << diff.count() << " s\n";

        std::cout << "Running benchmark search (nprobe=" << nprobe << ")...\n";
        SearchOptions opts;
        opts.top_k = 10;
        opts.metric = Metric::L2;

        start = std::chrono::high_resolution_clock::now();
        for (size_t query_index = 0; query_index < 100; ++query_index) {
            index.search(data[query_index], opts, nprobe);
        }
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "100 searches took " << diff.count() << " s\n";
        
    } else if (command == "benchmark-pq") {
        size_t num_vectors = 100000;
        size_t num_queries = 1000;
        size_t dim = 128;
        size_t nlist = 256;
        size_t nprobe = 16;
        size_t m = 8;
        size_t rerank_n = 50;

        double initial_mem = MemoryProfiler::get_current_rss_mb();
        std::cout << "[Memory] Base process: " << initial_mem << " MB\n";

        std::cout << "Generating dataset (" << num_vectors << " vectors)...\n";
        auto data = DatasetGenerator::generate(num_vectors, dim);
        auto query_data = DatasetGenerator::generate(num_queries, dim, 42); // queries

        double data_mem = MemoryProfiler::get_current_rss_mb();
        std::cout << "[Memory] Raw dataset size in RAM: " << (data_mem - initial_mem) << " MB\n";
        
        std::cout << "Training IVFPQ Index (nlist=" << nlist << ", m=" << m << ")...\n";
        Timer t;
        IVFPQIndex index(dim, m, 256, true);
        index.train(data, nlist, Metric::L2);
        std::cout << "Training took " << t.elapsed_ms() << " ms\n";
        
        std::cout << "Adding vectors...\n";
        t.reset();
        for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
            index.add(vector_index, data[vector_index]);
        }
        index.build();
        std::cout << "Adding took " << t.elapsed_ms() << " ms\n";

        double final_mem = MemoryProfiler::get_current_rss_mb();
        std::cout << "[Memory] Peak process RSS: " << MemoryProfiler::get_peak_rss_mb() << " MB\n";
        std::cout << "[Memory] RAM Index overhead: " << (final_mem - data_mem) << " MB\n";

        std::string index_path = "benchmark_ivfpq.bin";
        std::cout << "\nSaving index to disk (" << index_path << ")...\n";
        t.reset();
        index.save(index_path);
        std::cout << "Saved in " << t.elapsed_ms() << " ms\n";

        std::cout << "Loading index via zero-copy MMAP...\n";
        IVFPQIndex mmap_index(dim, m, 256, true);
        t.reset();
        mmap_index.load_mmap(index_path);
        std::cout << "MMAP load took " << t.elapsed_ms() << " ms\n";

        std::cout << "\nRunning benchmark search (" << num_queries << " queries, nprobe=" << nprobe << ", rerank_n=" << rerank_n << ")...\n";
        SearchOptions opts;
        opts.top_k = 10;
        opts.metric = Metric::L2;
        SearchStats agg_stats;
        opts.stats = &agg_stats;

        t.reset();
        auto all_results = mmap_index.search_batch(query_data, opts, nprobe, rerank_n);
        double total_time_ms = t.elapsed_ms();
        
        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Total search time: " << total_time_ms << " ms\n";
        std::cout << "Throughput (QPS) : " << (num_queries / (total_time_ms / 1000.0)) << " req/s\n";
        std::cout << "Average latency  : " << (total_time_ms / num_queries) << " ms/query\n";
        
        std::cout << "\n=== Latency Breakdown (Averages) ===\n";
        std::cout << "Centroid Search : " << (agg_stats.centroid_search_ms / num_queries) << " ms\n";
        std::cout << "LUT Compute     : " << (agg_stats.lut_compute_ms / num_queries) << " ms\n";
        std::cout << "List Scan (ADC) : " << (agg_stats.list_scan_ms / num_queries) << " ms\n";
        std::cout << "Exact Reranking : " << (agg_stats.rerank_ms / num_queries) << " ms\n";
    } else if (command == "benchmark-hnsw") {
        size_t num_vectors = 10000;
        size_t num_queries = 1000;
        size_t dim = 128;
        int M = 16;
        int ef_construction = 100;
        int ef_search = 50;

        std::cout << "Generating dataset (" << num_vectors << " vectors)...\n";
        auto data = DatasetGenerator::generate(num_vectors, dim);
        auto query_data = DatasetGenerator::generate(num_queries, dim, 42); // queries

        std::cout << "Building HNSW Index (M=" << M << ", efConstruction=" << ef_construction << ")...\n";
        Timer t;
        HNSWIndex index(dim, M, ef_construction);
        
        for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
            index.add(vector_index, data[vector_index]);
            if (vector_index % 2000 == 0 && vector_index > 0) std::cout << "Added " << vector_index << " vectors...\n";
        }
        std::cout << "Building took " << t.elapsed_ms() << " ms\n";

        std::cout << "\nRunning benchmark search (" << num_queries << " queries, efSearch=" << ef_search << ")...\n";
        SearchOptions opts;
        opts.top_k = 10;
        opts.metric = Metric::L2;

        t.reset();
        for (size_t query_index = 0; query_index < num_queries; ++query_index) {
            index.search(query_data[query_index], opts);
        }
        double total_time_ms = t.elapsed_ms();
        
        std::cout << "\n=== HNSW Benchmark Results ===\n";
        std::cout << "Total search time: " << total_time_ms << " ms\n";
        std::cout << "Throughput (QPS) : " << (num_queries / (total_time_ms / 1000.0)) << " req/s\n";
        std::cout << "Average latency  : " << (total_time_ms / num_queries) << " ms/query\n";
    } else if (command == "benchmark-vamana") {
        size_t num_vectors = 10000;
        size_t num_queries = 1000;
        size_t dim = 128;
        size_t R = 64;
        size_t L = 100;
        float alpha = 1.2f;

        std::cout << "Generating dataset (" << num_vectors << " vectors)...\n";
        auto data = DatasetGenerator::generate(num_vectors, dim);
        auto query_data = DatasetGenerator::generate(num_queries, dim, 42);

        std::cout << "Adding vectors to Vamana Index...\n";
        Timer t;
        VamanaIndex index(dim, R, L, alpha);
        
        for (size_t vector_index = 0; vector_index < num_vectors; ++vector_index) {
            index.add(vector_index, data[vector_index]);
        }
        std::cout << "Adding took " << t.elapsed_ms() << " ms\n";
        
        std::cout << "Building Vamana Graph (R=" << R << ", L=" << L << ", alpha=" << alpha << ")...\n";
        t.reset();
        index.build();
        std::cout << "Graph Build took " << t.elapsed_ms() << " ms\n";

        std::cout << "\nRunning benchmark search (" << num_queries << " queries)...\n";
        SearchOptions opts;
        opts.top_k = 10;
        opts.metric = Metric::L2;

        t.reset();
        for (size_t query_index = 0; query_index < num_queries; ++query_index) {
            index.search(query_data[query_index], opts);
        }
        double total_time_ms = t.elapsed_ms();
        
        std::cout << "\n=== Vamana (DiskANN) Benchmark Results ===\n";
        std::cout << "Total search time: " << total_time_ms << " ms\n";
        std::cout << "Throughput (QPS) : " << (num_queries / (total_time_ms / 1000.0)) << " req/s\n";
        std::cout << "Average latency  : " << (total_time_ms / num_queries) << " ms/query\n";
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }
    
    return 0;
}
