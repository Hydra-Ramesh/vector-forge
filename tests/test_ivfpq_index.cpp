#include <gtest/gtest.h>
#include "vectorforge/index/ivfpq_index.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(IVFPQIndexTest, TrainAddSearchRerank) {
    size_t dim = 16;
    size_t num_vectors = 1000;
    size_t nlist = 10;
    size_t m = 4; // 4 subquantizers (sub_dim = 4)
    size_t k_sub = 256;
    size_t nprobe = 4;
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    IVFPQIndex ivfpq(dim, m, k_sub, true); // store_raw_vectors = true
    ivfpq.train(dataset, nlist, Metric::L2);
    
    BruteForceIndex exact(dim);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        ivfpq.add(i, dataset[i]);
        exact.add(i, dataset[i]);
    }
    
    ivfpq.build();
    exact.build();
    
    SearchOptions opts;
    opts.top_k = 10;
    opts.metric = Metric::L2;
    
    size_t num_queries = 10;
    
    // Test without reranking (approximate)
    float total_recall_no_rerank = 0.0f;
    for (size_t i = 0; i < num_queries; ++i) {
        auto query = dataset[i];
        auto res_approx = ivfpq.search(query, opts, nprobe, 0); // rerank = 0
        auto exact_res = exact.search(query, opts);
        
        int matches = 0;
        for (const auto& r_approx : res_approx) {
            for (const auto& r_exact : exact_res) {
                if (r_approx.id == r_exact.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall_no_rerank += static_cast<float>(matches) / opts.top_k;
    }
    
    // Test with reranking (should be higher or equal)
    float total_recall_rerank = 0.0f;
    for (size_t i = 0; i < num_queries; ++i) {
        auto query = dataset[i];
        auto res_rerank = ivfpq.search(query, opts, nprobe, 50); // rerank top 50
        auto exact_res = exact.search(query, opts);
        
        int matches = 0;
        for (const auto& r_rr : res_rerank) {
            for (const auto& r_exact : exact_res) {
                if (r_rr.id == r_exact.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall_rerank += static_cast<float>(matches) / opts.top_k;
    }
    
    float avg_recall_no_rerank = total_recall_no_rerank / num_queries;
    float avg_recall_rerank = total_recall_rerank / num_queries;
    
    EXPECT_GT(avg_recall_no_rerank, 0.4f); // PQ is approximate, so recall might drop a bit depending on compression
    EXPECT_GE(avg_recall_rerank, avg_recall_no_rerank); // Reranking should improve or maintain recall
}
