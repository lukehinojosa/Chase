#pragma once

#include <filesystem>

#include <torch/torch.h>

namespace stai {

// Whole dataset held as tensors (100k samples is ~350 MB, fits on the GPU).
struct TrajectoryDataset {
    torch::Tensor inputs; // [N, 877] float
    torch::Tensor moves;  // [N, 2]   float, teacher (move_x, move_z)
    torch::Tensor roles;  // [N]      int64, head index
    torch::Tensor labels; // [N]      float, +1 positive / -1 negative

    int64_t size() const { return inputs.size(0); }
    TrajectoryDataset to(const torch::Device& device) const;
    TrajectoryDataset index(const torch::Tensor& idx) const;
};

// Reads training_data.bin (format in Topology.hpp). Throws on malformed files.
TrajectoryDataset loadTrajectoryFile(const std::filesystem::path& path);

// Random data with a learnable structure, for exercising the pipeline before
// Unity has recorded anything.
TrajectoryDataset makeSyntheticDataset(int64_t count, uint64_t seed);

} // namespace stai
