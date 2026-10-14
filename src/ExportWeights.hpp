#pragma once

#include <filesystem>

#include "HybridChaserNet.hpp"

namespace stai {

// Writes chaser_master_fp32.bytes in the flat layout documented in Topology.hpp.
void exportWeights(HybridChaserNet& net, const std::filesystem::path& path);

// Reads a flat weight file back into the network (round-trip check / warm start).
void importWeights(HybridChaserNet& net, const std::filesystem::path& path);

} // namespace stai
