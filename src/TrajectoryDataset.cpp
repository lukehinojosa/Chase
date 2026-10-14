#include "TrajectoryDataset.hpp"

#include <fstream>
#include <stdexcept>

#include "Topology.hpp"

namespace stai {

TrajectoryDataset TrajectoryDataset::to(const torch::Device& device) const {
    return {inputs.to(device), moves.to(device), roles.to(device), labels.to(device)};
}

TrajectoryDataset TrajectoryDataset::index(const torch::Tensor& idx) const {
    return {inputs.index_select(0, idx), moves.index_select(0, idx),
            roles.index_select(0, idx), labels.index_select(0, idx)};
}

TrajectoryDataset loadTrajectoryFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open " + path.string());

    uint32_t header[4];
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)))
        throw std::runtime_error("truncated header in " + path.string());
    if (header[0] != kDatasetMagic)
        throw std::runtime_error("bad magic in " + path.string());
    if (header[1] != kDatasetVersion)
        throw std::runtime_error("unsupported dataset version " + std::to_string(header[1]));
    if (header[2] != static_cast<uint32_t>(kInputDim))
        throw std::runtime_error("dataset input dim " + std::to_string(header[2]) +
                                 " != " + std::to_string(kInputDim));

    const int64_t count = header[3];
    auto raw = torch::empty({count, kRecordWords}, torch::kFloat32);
    if (!file.read(reinterpret_cast<char*>(raw.data_ptr()), count * kRecordBytes))
        throw std::runtime_error("dataset truncated: expected " + std::to_string(count) + " records");

    // The trailing two words are int32 stored in the same buffer; reinterpret them.
    auto ints = raw.view(torch::kInt32);
    return {
        raw.slice(1, 0, kInputDim).clone(),
        raw.slice(1, kInputDim, kInputDim + kOutDim).clone(),
        ints.select(1, kInputDim + kOutDim).to(torch::kInt64),
        ints.select(1, kInputDim + kOutDim + 1).to(torch::kFloat32),
    };
}

TrajectoryDataset makeSyntheticDataset(int64_t count, uint64_t seed) {
    auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);

    // Sparse voxel grid with walls (1) and drop-offs (-1).
    auto u = torch::rand({count, kVoxelCount}, gen, opts);
    auto voxels = (u < 0.08f).to(torch::kFloat32) - (u > 0.96f).to(torch::kFloat32);

    auto slots = torch::zeros({count, kInputDim - kVoxelCount}, opts);
    auto dir = torch::randn({count, 3}, gen, opts);
    dir.select(1, 1).mul_(0.2f);
    dir = dir / dir.norm(2, 1, true).clamp_min(1e-6);

    auto roles = torch::randint(0, 2, {count}, gen, torch::kInt64); // Chaser / Runner only
    auto isChaser = (roles == 0).to(torch::kFloat32).unsqueeze(1);

    // Chaser: target in the attract slot. Runner: threat in the repel slot.
    const int a = kOffAttract - kVoxelCount, r = kOffRepel - kVoxelCount, s = kOffSelf - kVoxelCount;
    slots.select(1, a).copy_(isChaser.squeeze(1));
    slots.slice(1, a + 1, a + 4).copy_(dir * isChaser);
    slots.select(1, r).copy_(1.0f - isChaser.squeeze(1));
    slots.slice(1, r + 1, r + 4).copy_(dir * (1.0f - isChaser));
    slots.select(1, s + 2).copy_(isChaser.squeeze(1) * 2.0f - 1.0f); // role_state +1 / -1

    // Teacher: chase toward, run away, in the XZ plane.
    auto xz = torch::stack({dir.select(1, 0), dir.select(1, 2)}, 1);
    xz = xz / xz.norm(2, 1, true).clamp_min(1e-6);
    auto moves = xz * (isChaser * 2.0f - 1.0f);

    auto inputs = torch::cat({voxels, slots}, 1);

    // Negatives: half the records get scrambled sensor data.
    auto labels = torch::where(torch::rand({count}, gen, opts) < 0.5f, 1.0f, -1.0f);
    auto neg = (labels < 0).unsqueeze(1);
    auto scrambled = inputs.index_select(0, torch::randperm(count, gen, torch::kInt64));
    inputs = torch::where(neg, scrambled * -1.0f, inputs);

    return {inputs, moves, roles, labels};
}

} // namespace stai
