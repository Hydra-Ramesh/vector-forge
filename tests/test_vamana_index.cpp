#include <gtest/gtest.h>
#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(VamanaIndexTest, BuildAndSearchL2) {
    size_t dim = 16;
    size_t num_vectors = 1000;
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    VamanaIndex index(dim, 32, 50, 1.2f);
    BruteForceIndex exact(dim);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        index.add(i, dataset[i]);
        exact.add(i, dataset[i]);
    }
    
    index.build();
    exact.build();
    
    SearchOptions search_options;
    search_options.top_k = 10;
    search_options.metric = Metric::L2;
    
    size_t query_count = 10;
    float total_recall = 0.0f;
    for (size_t query_index = 0; query_index < query_count; ++query_index) {
        auto query = dataset[query_index];
        auto approximate_results = index.search(query, search_options);
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
    EXPECT_GT(average_recall, 0.9f);
}

TEST(VamanaIndexTest, RemoveAndCompact) {
    size_t dim = 16;
    size_t num_vectors = 100;
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    VamanaIndex index(dim, 32, 50, 1.2f);
    for (size_t i = 0; i < num_vectors; ++i) {
        index.add(i, dataset[i]);
    }
    index.build();
    
    // Remove exactly half of the vectors
    for (size_t i = 0; i < num_vectors; i += 2) {
        index.remove(i);
    }
    
    SearchOptions search_options;
    search_options.top_k = 10;
    search_options.metric = Metric::L2;
    
    // Test that searching DOES NOT return removed vectors
    auto results_before_compact = index.search(dataset[0], search_options);
    for (const auto& result : results_before_compact) {
        EXPECT_NE(result.id % 2, 0); // Removed all even IDs, so results must be odd
    }
    
    // Perform compaction
    index.compact();
    
    // Test search works after compact
    auto results_after_compact = index.search(dataset[0], search_options);
    for (const auto& result : results_after_compact) {
        EXPECT_NE(result.id % 2, 0); // Still shouldn't contain even IDs
    }
}

TEST(VamanaIndexTest, SaveAndLoadWithTombstones) {
    size_t dim = 8;
    size_t num_vectors = 50;
    std::string test_file = "test_vamana_tombstones.bin";
    
    auto dataset = DatasetGenerator::generate(num_vectors, dim, 42);
    
    {
        VamanaIndex index(dim, 16, 20, 1.2f);
        for (size_t i = 0; i < num_vectors; ++i) {
            index.add(i, dataset[i]);
        }
        index.build();
        
        index.remove(10);
        index.remove(20);
        
        index.save(test_file);
    }
    
    {
        VamanaIndex loaded_index(dim, 16, 20, 1.2f);
        loaded_index.load(test_file);
        
        SearchOptions search_options;
        search_options.top_k = 20;
        search_options.metric = Metric::L2;
        
        // Ensure 10 and 20 are NOT in the search results
        auto results = loaded_index.search(dataset[10], search_options);
        for (const auto& res : results) {
            EXPECT_NE(res.id, 10);
            EXPECT_NE(res.id, 20);
        }
    }
    
    std::remove(test_file.c_str());
}
