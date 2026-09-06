#include <gtest/gtest.h>
#include "vectorforge/index/ivfpq_index.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(IVFPQIndexTest, TrainAddSearchRerank) {
    size_t dim = 16;
    size_t num_vectors = 1000;
    size_t nlist = 10;
    size_t subquantizer_count = 4;
    size_t k_sub = 256;
    size_t nprobe = 4;
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    IVFPQIndex ivfpq(dim, subquantizer_count, k_sub, true);
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
    
    float total_recall_without_reranking = 0.0f;
    for (size_t query_index = 0; query_index < num_queries; ++query_index) {
        auto query = dataset[query_index];
        auto approximate_results = ivfpq.search(query, opts, nprobe, 0);
        auto exact_res = exact.search(query, opts);
        
        int matches = 0;
        for (const auto& approximate_result : approximate_results) {
            for (const auto& exact_result : exact_res) {
                if (approximate_result.id == exact_result.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall_without_reranking += static_cast<float>(matches) / opts.top_k;
    }
    
    float total_recall_with_reranking = 0.0f;
    for (size_t query_index = 0; query_index < num_queries; ++query_index) {
        auto query = dataset[query_index];
        auto reranked_results = ivfpq.search(query, opts, nprobe, 50);
        auto exact_res = exact.search(query, opts);
        
        int matches = 0;
        for (const auto& reranked_result : reranked_results) {
            for (const auto& exact_result : exact_res) {
                if (reranked_result.id == exact_result.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall_with_reranking += static_cast<float>(matches) / opts.top_k;
    }
    
    float average_recall_without_reranking = total_recall_without_reranking / num_queries;
    float average_recall_with_reranking = total_recall_with_reranking / num_queries;
    
    EXPECT_GT(average_recall_without_reranking, 0.4f);
    EXPECT_GE(average_recall_with_reranking, average_recall_without_reranking);
}
