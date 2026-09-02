#include <gtest/gtest.h>
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(BruteForceIndexTest, AddAndSearchL2) {
    BruteForceIndex index(4);
    index.add(1, {0.0f, 0.0f, 0.0f, 0.0f});
    index.add(2, {1.0f, 0.0f, 0.0f, 0.0f});
    index.add(3, {0.0f, 2.0f, 0.0f, 0.0f});
    index.build();

    SearchOptions opts;
    opts.top_k = 2;
    opts.metric = Metric::L2;

    Vector query = {0.1f, 0.0f, 0.0f, 0.0f};
    auto results = index.search(query, opts);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 1); // dist = 0.01
    EXPECT_EQ(results[1].id, 2); // dist = 0.81
}

TEST(BruteForceIndexTest, AddAndSearchCosine) {
    BruteForceIndex index(2);
    index.add(1, {1.0f, 0.0f});
    index.add(2, {0.0f, 1.0f});
    index.add(3, {-1.0f, 0.0f});
    index.build();

    SearchOptions opts;
    opts.top_k = 2;
    opts.metric = Metric::Cosine;

    Vector query = {1.0f, 0.0f}; // Same as vector 1
    auto results = index.search(query, opts);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].id, 1);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-5);
    EXPECT_EQ(results[1].id, 2); // Orthogonal, dist = 1.0
}
