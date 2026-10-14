#pragma once

#include <array>

#include <torch/torch.h>

#include "Topology.hpp"

namespace stai {

// One Forward-Forward hidden layer: Linear -> ReLU -> (goodness) -> LayerNorm.
// Goodness is measured on the raw post-ReLU activity; the LayerNorm output is what
// the next layer sees, so it cannot trivially read the previous layer's goodness.
struct FFLayerImpl : torch::nn::Module {
    FFLayerImpl(int in, int out);

    struct Output {
        torch::Tensor activity;   // post-ReLU y, [B, out]
        torch::Tensor goodness;   // mean(y^2) per sample, [B]
        torch::Tensor normalized; // LayerNorm(y), [B, out]
    };
    Output forward(const torch::Tensor& x);

    torch::nn::Linear fc{nullptr};
    torch::nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(FFLayer);

struct HybridChaserNetImpl : torch::nn::Module {
    HybridChaserNetImpl();

    struct Output {
        torch::Tensor move;                     // active role head output, tanh, [B, 2]
        std::array<torch::Tensor, 3> goodness;  // per hidden layer, [B]
    };

    // input: [B, 877] float, role: [B] int64 head index.
    Output forward(const torch::Tensor& input, const torch::Tensor& role);

    FFLayer l1{nullptr}, l2{nullptr}, l3{nullptr};
    std::array<torch::nn::Linear, kNumHeads> heads{nullptr, nullptr, nullptr};
};
TORCH_MODULE(HybridChaserNet);

} // namespace stai
