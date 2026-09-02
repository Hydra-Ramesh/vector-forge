#include <gtest/gtest.h>
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(DatasetTest, GenerateHasCorrectDimensions) {
    auto dataset = DatasetGenerator::generate(100, 128);
    EXPECT_EQ(dataset.size(), 100);
    EXPECT_EQ(dataset[0].size(), 128);
}

TEST(DatasetTest, DeterministicGeneration) {
    auto dataset1 = DatasetGenerator::generate(10, 16, 42);
    auto dataset2 = DatasetGenerator::generate(10, 16, 42);
    
    EXPECT_EQ(dataset1, dataset2);
}

TEST(DatasetTest, Normalize) {
    auto dataset = DatasetGenerator::generate(10, 16);
    DatasetGenerator::normalize(dataset);
    
    for (const auto& vec : dataset) {
        float norm_sq = 0.0f;
        for (float v : vec) {
            norm_sq += v * v;
        }
        // Should be close to 1.0
        EXPECT_NEAR(norm_sq, 1.0f, 1e-5);
    }
}
