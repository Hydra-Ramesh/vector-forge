#include <gtest/gtest.h>
#include "vectorforge/index/brute_force_index.hpp"
#include <cstdio>

using namespace vectorforge;

TEST(StorageTest, SaveAndLoad) {
    std::string test_file = "test_index.bin";

    {
        BruteForceIndex index(2);
        index.add(10, {1.0f, 0.0f});
        index.add(20, {0.0f, 1.0f});
        index.build();
        index.save(test_file);
    }

    {
        BruteForceIndex index(2);
        index.load(test_file);
        
        EXPECT_EQ(index.size(), 2);
        EXPECT_EQ(index.dimension(), 2);

        SearchOptions search_options;
        search_options.top_k = 1;
        search_options.metric = Metric::L2;

        auto results = index.search({1.0f, 0.0f}, search_options);
        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].id, 10);
        EXPECT_NEAR(results[0].distance, 0.0f, 1e-5);
    }
    
    std::remove(test_file.c_str());
}

TEST(StorageTest, SaveAndLoadMmap) {
    std::string test_file = "test_index_mmap.bin";

    {
        BruteForceIndex index(3);
        index.add(100, {1.0f, 2.0f, 3.0f});
        index.add(200, {4.0f, 5.0f, 6.0f});
        index.build();
        index.save(test_file);
    }

    {
        BruteForceIndex index(1); 
        index.load_mmap(test_file);
        
        EXPECT_EQ(index.size(), 2);
        EXPECT_EQ(index.dimension(), 3);

        SearchOptions search_options;
        search_options.top_k = 2;
        search_options.metric = Metric::L2;

        auto results = index.search({1.0f, 2.0f, 3.0f}, search_options);
        ASSERT_EQ(results.size(), 2);
        EXPECT_EQ(results[0].id, 100);
        EXPECT_NEAR(results[0].distance, 0.0f, 1e-5);
    }
    
    std::remove(test_file.c_str());
}
