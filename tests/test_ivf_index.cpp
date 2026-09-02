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
    
    SearchOptions opts;
    opts.top_k = 10;
    opts.metric = Metric::L2;
    
    // Test recall
    size_t num_queries = 10;
    float total_recall = 0.0f;
    for (size_t i = 0; i < num_queries; ++i) {
        auto query = dataset[i];
        auto ivf_res = ivf.search(query, opts, nprobe);
        auto exact_res = exact.search(query, opts);
        
        int matches = 0;
        for (const auto& r_ivf : ivf_res) {
            for (const auto& r_exact : exact_res) {
                if (r_ivf.id == r_exact.id) {
                    matches++;
                    break;
                }
            }
        }
        total_recall += static_cast<float>(matches) / opts.top_k;
    }
    
    float avg_recall = total_recall / num_queries;
    // With nprobe=4 and nlist=10, recall should be extremely high, close to 1.0
    EXPECT_GT(avg_recall, 0.8f);
}
