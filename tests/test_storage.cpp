#include <gtest/gtest.h>
#include "vectorforge/index/brute_force_index.hpp"
#include <cstdio>

using namespace vectorforge;

TEST(StorageTest, SaveAndLoad) {
    std::string test_file = "test_index.bin";

    // 1. Create and save an index
    {
        BruteForceIndex index(2);
        index.add(10, {1.0f, 0.0f});
        index.add(20, {0.0f, 1.0f});
        index.build();
        index.save(test_file);
    }

    // 2. Load the index
    {
        BruteForceIndex index(2); // Initial dim can be overridden by load
        index.load(test_file);
        
        EXPECT_EQ(index.size(), 2);
        EXPECT_EQ(index.dimension(), 2);

        SearchOptions opts;
        opts.top_k = 1;
        opts.metric = Metric::L2;

        auto results = index.search({1.0f, 0.0f}, opts);
        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].id, 10);
        EXPECT_NEAR(results[0].distance, 0.0f, 1e-5);
    }
    
    std::remove(test_file.c_str());
}

TEST(StorageTest, SaveAndLoadMmap) {
    std::string test_file = "test_index_mmap.bin";

    // 1. Create and save an index
    {
        BruteForceIndex index(3);
        index.add(100, {1.0f, 2.0f, 3.0f});
        index.add(200, {4.0f, 5.0f, 6.0f});
        index.build();
        index.save(test_file);
    }

    // 2. Load the index via mmap
    {
        BruteForceIndex index(1); 
        index.load_mmap(test_file);
        
        EXPECT_EQ(index.size(), 2);
        EXPECT_EQ(index.dimension(), 3);

        SearchOptions opts;
        opts.top_k = 2;
        opts.metric = Metric::L2;

        auto results = index.search({1.0f, 2.0f, 3.0f}, opts);
        ASSERT_EQ(results.size(), 2);
        EXPECT_EQ(results[0].id, 100);
        EXPECT_NEAR(results[0].distance, 0.0f, 1e-5);
    }
    
    std::remove(test_file.c_str());
}
