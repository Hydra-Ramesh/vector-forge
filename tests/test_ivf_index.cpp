#include <gtest/gtest.h>
#include "vectorforge/index/ivf_index.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(IVFIndexTest, TrainAddSearchL2) {
    size_t dim = 16;
    size_t num_vectors = 1000;
    size_t nlist = 10;
    size_t nprobe = 4;
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    IVFIndex ivf(dim);
    ivf.train(dataset, nlist, Metric::L2);
    
    BruteForceIndex exact(dim);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        ivf.add(i, dataset[i]);
        exact.add(i, dataset[i]);
    }
    
    ivf.build();
    exact.build();
    
    SearchOptions search_options;
    search_options.top_k = 10;
    search_options.metric = Metric::L2;
    
    size_t query_count = 10;
    float total_recall = 0.0f;
    for (size_t query_index = 0; query_index < query_count; ++query_index) {
        auto query = dataset[query_index];
        auto approximate_results = ivf.search(query, search_options, nprobe);
        auto exact_results = exact.search(query, search_options);
        
        int matches = 0;
        for (const auto& approximate_result : approximate_results) {
            for (const auto& exact_result : exact_results) {
                if (approximate_result.id == exact_result.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall += static_cast<float>(matches) / search_options.top_k;
    }
    
    float average_recall = total_recall / query_count;
    EXPECT_GT(average_recall, 0.8f);
}
