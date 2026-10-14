// Offline pretrainer for the Student network (Plan.md, Phase 2).
//
//   StudentTeacherAI --data <training_data.bin> [options]
//   StudentTeacherAI --synthetic 100000 [options]     (pipeline smoke test)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <torch/cuda.h>

#include "ExportWeights.hpp"
#include "TrainLoop.hpp"

namespace {

void usage() {
    std::printf(
        "options:\n"
        "  --data <path>         training_data.bin recorded by Unity\n"
        "  --synthetic <n>       generate n synthetic samples instead of --data\n"
        "  --epochs <n>          default 1000\n"
        "  --batch <n>           default 4096\n"
        "  --lr <x>              default 1e-3\n"
        "  --bf16                bf16 autocast on CUDA\n"
        "  --cpu                 force CPU\n"
        "  --log-every <n>       default 50\n"
        "  --init <path>         warm start from a flat fp32 weight file\n"
        "  --out <path>          weight file (default <Unity>/Assets/StreamingAssets/chaser_master_fp32.bytes)\n"
        "  --checkpoint <path>   also save a LibTorch checkpoint (.pt)\n");
}

} // namespace

int main(int argc, char** argv) {
    stai::TrainConfig cfg;
    std::string dataPath, initPath, checkpointPath;
    std::string outPath = std::string(STAI_DEFAULT_UNITY_DIR) + "/Assets/StreamingAssets/chaser_master_fp32.bytes";
    int64_t synthetic = 0;
    bool forceCpu = false;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc; };
        if (arg("--data")) dataPath = argv[++i];
        else if (arg("--synthetic")) synthetic = std::atoll(argv[++i]);
        else if (arg("--epochs")) cfg.epochs = std::atoi(argv[++i]);
        else if (arg("--batch")) cfg.batchSize = std::atoll(argv[++i]);
        else if (arg("--lr")) cfg.learningRate = std::atof(argv[++i]);
        else if (arg("--log-every")) cfg.logEvery = std::atoi(argv[++i]);
        else if (arg("--init")) initPath = argv[++i];
        else if (arg("--out")) outPath = argv[++i];
        else if (arg("--checkpoint")) checkpointPath = argv[++i];
        else if (std::strcmp(argv[i], "--bf16") == 0) cfg.bf16Autocast = true;
        else if (std::strcmp(argv[i], "--cpu") == 0) forceCpu = true;
        else { usage(); return 2; }
    }
    if (dataPath.empty() == (synthetic == 0)) {
        std::printf("specify exactly one of --data or --synthetic\n");
        usage();
        return 2;
    }

    try {
        const bool cuda = !forceCpu && torch::cuda::is_available();
        torch::Device device = cuda ? torch::kCUDA : torch::kCPU;
        std::printf("LibTorch %s | CUDA available: %s (%d device)\n", TORCH_VERSION,
                    torch::cuda::is_available() ? "yes" : "no", torch::cuda::device_count());

        auto data = synthetic ? stai::makeSyntheticDataset(synthetic, cfg.seed)
                              : stai::loadTrajectoryFile(dataPath);

        stai::HybridChaserNet net;
        int64_t params = 0;
        for (auto& p : net->parameters()) params += p.numel();
        std::printf("HybridChaserNet: %lld parameters\n", static_cast<long long>(params));
        if (!initPath.empty()) stai::importWeights(net, initPath);

        stai::train(net, data, cfg, device);

        net->to(torch::kCPU);
        stai::exportWeights(net, outPath);
        std::printf("wrote %s (%d bytes)\n", outPath.c_str(), stai::kWeightFileBytes);
        if (!checkpointPath.empty()) torch::save(net, checkpointPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
