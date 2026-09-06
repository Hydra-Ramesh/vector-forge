#include <gtest/gtest.h>
#include "vectorforge/core/dataset.hpp"

using namespace vectorforge;

TEST(DatasetTest, GenerateHasCorrectDimensions) {
    auto dataset = DatasetGenerator::generate(100, 128);
    EXPECT_EQ(dataset.size(), 100);
    EXPECT_EQ(dataset[0].size(), 128);
}

TEST(DatasetTest, DeterministicGeneration) {
    auto first_dataset = DatasetGenerator::generate(10, 16, 42);
    auto second_dataset = DatasetGenerator::generate(10, 16, 42);
    
    EXPECT_EQ(first_dataset, second_dataset);
}

TEST(DatasetTest, Normalize) {
    auto dataset = DatasetGenerator::generate(10, 16);
    DatasetGenerator::normalize(dataset);
    
    for (const auto& vector : dataset) {
        float norm_squared = 0.0f;
        for (float value : vector) {
            norm_squared += value * value;
        }
        EXPECT_NEAR(norm_squared, 1.0f, 1e-5);
    }
}
