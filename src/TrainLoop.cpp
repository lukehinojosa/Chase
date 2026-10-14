#include "TrainLoop.hpp"

#include <chrono>
#include <cstdio>

#include <ATen/autocast_mode.h>

namespace stai {
namespace {

struct Losses {
    torch::Tensor total, bp, ff;
};

Losses jointLoss(const HybridChaserNetImpl::Output& out, const TrajectoryDataset& batch) {
    auto pos = (batch.labels > 0).to(torch::kFloat32);
    auto neg = 1.0f - pos;
    auto nPos = pos.sum().clamp_min(1.0f);
    auto nNeg = neg.sum().clamp_min(1.0f);

    // Imitation only makes sense on teacher-approved (positive) samples.
    auto sqErr = (out.move.to(torch::kFloat32) - batch.moves).square().mean(1);
    auto bp = (sqErr * pos).sum() / nPos;

    auto ff = torch::zeros({}, bp.options());
    for (const auto& g : out.goodness) {
        auto gf = g.to(torch::kFloat32) - kGoodnessThreshold;
        ff = ff + (torch::softplus(-gf) * pos).sum() / nPos + (torch::softplus(gf) * neg).sum() / nNeg;
    }
    return {bp + kFFLossWeight * ff, bp, ff};
}

void logValidation(HybridChaserNet& net, const TrajectoryDataset& val, int epoch,
                   double trainBp, double trainFf, double secs) {
    torch::NoGradGuard noGrad;
    net->eval();
    auto out = net->forward(val.inputs, val.roles);
    auto losses = jointLoss(out, val);
    auto pos = val.labels > 0;

    std::printf("epoch %5d | %6.2fs | train bp %.5f ff %.4f | val bp %.5f ff %.4f |", epoch, secs,
                trainBp, trainFf, losses.bp.item<double>(), losses.ff.item<double>());
    for (size_t l = 0; l < out.goodness.size(); ++l) {
        const auto& g = out.goodness[l];
        auto gPos = g.masked_select(pos).mean().item<double>();
        auto gNeg = g.masked_select(~pos).mean().item<double>();
        auto acc = ((g > kGoodnessThreshold) == pos).to(torch::kFloat32).mean().item<double>();
        std::printf(" L%zu G+ %.2f G- %.2f acc %.3f |", l + 1, gPos, gNeg, acc);
    }
    std::printf("\n");
    std::fflush(stdout);
    net->train();
}

} // namespace

void train(HybridChaserNet& net, const TrajectoryDataset& data, const TrainConfig& cfg,
           const torch::Device& device) {
    torch::manual_seed(cfg.seed);
    net->to(device);
    net->train();

    auto perm = torch::randperm(data.size(), torch::kInt64);
    const int64_t nVal = std::max<int64_t>(1, static_cast<int64_t>(data.size() * cfg.valFraction));
    auto val = data.index(perm.slice(0, 0, nVal)).to(device);
    auto trn = data.index(perm.slice(0, nVal)).to(device);
    const int64_t n = trn.size();
    std::printf("train %lld / val %lld samples on %s\n", static_cast<long long>(n),
                static_cast<long long>(nVal), device.str().c_str());

    torch::optim::AdamW opt(net->parameters(),
                            torch::optim::AdamWOptions(cfg.learningRate).weight_decay(cfg.weightDecay));

    auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        auto order = torch::randperm(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto bpSum = torch::zeros({}, torch::TensorOptions().device(device));
        auto ffSum = torch::zeros({}, torch::TensorOptions().device(device));
        int64_t steps = 0;

        for (int64_t start = 0; start < n; start += cfg.batchSize, ++steps) {
            auto batch = trn.index(order.slice(0, start, std::min(n, start + cfg.batchSize)));

            if (cfg.bf16Autocast) {
                at::autocast::set_autocast_enabled(device.type(), true);
                at::autocast::set_autocast_dtype(device.type(), at::kBFloat16);
            }
            auto out = net->forward(batch.inputs, batch.roles);
            if (cfg.bf16Autocast) {
                at::autocast::set_autocast_enabled(device.type(), false);
                at::autocast::clear_cache();
            }

            auto losses = jointLoss(out, batch);
            opt.zero_grad();
            losses.total.backward();
            opt.step();

            bpSum += losses.bp.detach();
            ffSum += losses.ff.detach();
        }

        if (epoch % cfg.logEvery == 0 || epoch == 1 || epoch == cfg.epochs) {
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            logValidation(net, val, epoch, bpSum.item<double>() / steps, ffSum.item<double>() / steps, secs);
        }
    }
}

} // namespace stai
