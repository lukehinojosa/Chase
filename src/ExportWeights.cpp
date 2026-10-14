#include "ExportWeights.hpp"

#include <fstream>
#include <stdexcept>

namespace stai {
namespace {

// The single source of truth for the flat file order.
std::vector<torch::Tensor> flatOrder(HybridChaserNet& net) {
    std::vector<torch::Tensor> order;
    for (auto* layer : {&net->l1, &net->l2, &net->l3}) {
        auto& l = *layer;
        order.insert(order.end(), {l->fc->weight, l->fc->bias, l->norm->weight, l->norm->bias});
    }
    for (auto& head : net->heads)
        order.insert(order.end(), {head->weight, head->bias});
    return order;
}

} // namespace

void exportWeights(HybridChaserNet& net, const std::filesystem::path& path) {
    torch::NoGradGuard noGrad;
    std::vector<torch::Tensor> flat;
    for (auto& t : flatOrder(net))
        flat.push_back(t.detach().to(torch::kCPU, torch::kFloat32).contiguous().view(-1));
    auto all = torch::cat(flat);
    if (all.numel() != kTotalParams)
        throw std::runtime_error("parameter count " + std::to_string(all.numel()) + " != " +
                                 std::to_string(kTotalParams));

    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.write(reinterpret_cast<const char*>(all.data_ptr<float>()), kWeightFileBytes))
        throw std::runtime_error("failed writing " + path.string());
}

void importWeights(HybridChaserNet& net, const std::filesystem::path& path) {
    if (std::filesystem::file_size(path) != kWeightFileBytes)
        throw std::runtime_error(path.string() + " is not " + std::to_string(kWeightFileBytes) + " bytes");

    auto all = torch::empty({kTotalParams}, torch::kFloat32);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(all.data_ptr<float>()), kWeightFileBytes))
        throw std::runtime_error("failed reading " + path.string());

    torch::NoGradGuard noGrad;
    int64_t offset = 0;
    for (auto& t : flatOrder(net)) {
        t.copy_(all.slice(0, offset, offset + t.numel()).view(t.sizes()));
        offset += t.numel();
    }
}

} // namespace stai
