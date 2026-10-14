#pragma once

#include <filesystem>

#include "HybridChaserNet.hpp"
#include "TrajectoryDataset.hpp"

namespace stai {

struct TrainConfig {
    int epochs = 1000;
    int64_t batchSize = 4096;
    double learningRate = 1e-3;
    double weightDecay = 1e-4;
    double valFraction = 0.05;
    bool bf16Autocast = false;
    int logEvery = 50;
    uint64_t seed = 1234;
};

// Joint objective: L = MSE(active head, teacher) on positives
//                    + 0.5 * sum_l [softplus(-(G+ - theta)) + softplus(G- - theta)]
void train(HybridChaserNet& net, const TrajectoryDataset& data, const TrainConfig& cfg,
           const torch::Device& device);

} // namespace stai
