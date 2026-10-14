#include "HybridChaserNet.hpp"

namespace stai {

FFLayerImpl::FFLayerImpl(int in, int out)
    : fc(register_module("fc", torch::nn::Linear(in, out))),
      norm(register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({out})))) {}

FFLayerImpl::Output FFLayerImpl::forward(const torch::Tensor& x) {
    auto y = torch::relu(fc->forward(x));
    auto g = y.square().mean(1);
    return {y, g, norm->forward(y)};
}

HybridChaserNetImpl::HybridChaserNetImpl()
    : l1(register_module("l1", FFLayer(kInputDim, kH1))),
      l2(register_module("l2", FFLayer(kH1, kH2))),
      l3(register_module("l3", FFLayer(kH2, kH3))) {
    for (int h = 0; h < kNumHeads; ++h)
        heads[h] = register_module("head" + std::to_string(h), torch::nn::Linear(kH3, kOutDim));
}

HybridChaserNetImpl::Output HybridChaserNetImpl::forward(const torch::Tensor& input,
                                                         const torch::Tensor& role) {
    auto o1 = l1->forward(input);
    auto o2 = l2->forward(o1.normalized);
    auto o3 = l3->forward(o2.normalized);

    // Evaluate all heads ([B, 3, 2]) and gather the one matching each sample's role.
    std::vector<torch::Tensor> all;
    all.reserve(kNumHeads);
    for (auto& head : heads)
        all.push_back(torch::tanh(head->forward(o3.normalized)));
    auto stacked = torch::stack(all, 1);
    auto idx = role.view({-1, 1, 1}).expand({-1, 1, kOutDim});
    auto move = stacked.gather(1, idx).squeeze(1);

    return {move, {o1.goodness, o2.goodness, o3.goodness}};
}

} // namespace stai
