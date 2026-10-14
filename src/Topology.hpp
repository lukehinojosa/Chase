#pragma once

// Network topology and on-disk formats shared with Unity.
// Mirror: Chase/Assets/Scripts/AI/BrainTopology.cs. Keep both in sync.

#include <cstdint>

namespace stai {

// ---- Input vector layout (877 floats, bot-local yaw space) ----
inline constexpr int kVoxelX = 11, kVoxelY = 7, kVoxelZ = 11;
inline constexpr int kVoxelCount = kVoxelX * kVoxelY * kVoxelZ; // 847
inline constexpr int kAttractSlot = 8;
inline constexpr int kRepelSlot = 8;
inline constexpr int kProtectSlot = 5;
inline constexpr int kAllySlot = 5;
inline constexpr int kSelfContext = 4;

inline constexpr int kOffVoxel = 0;
inline constexpr int kOffAttract = kOffVoxel + kVoxelCount;
inline constexpr int kOffRepel = kOffAttract + kAttractSlot;
inline constexpr int kOffProtect = kOffRepel + kRepelSlot;
inline constexpr int kOffAlly = kOffProtect + kProtectSlot;
inline constexpr int kOffSelf = kOffAlly + kAllySlot;
inline constexpr int kInputDim = kOffSelf + kSelfContext;
static_assert(kInputDim == 877);

// ---- Layers ----
inline constexpr int kH1 = 128, kH2 = 128, kH3 = 32;
inline constexpr int kNumHeads = 3; // 0 = Chaser, 1 = Runner, 2 = Defender
inline constexpr int kOutDim = 2;   // (move_x, move_z)
inline constexpr int kActivationsPerBot = kH1 + kH2 + kH3;
static_assert(kActivationsPerBot == 288);

// Linear W + b, plus LayerNorm gamma + beta.
constexpr int ffLayerParams(int in, int out) { return in * out + out + 2 * out; }
inline constexpr int kHeadParams = kH3 * kOutDim + kOutDim;
inline constexpr int kTotalParams = ffLayerParams(kInputDim, kH1) + ffLayerParams(kH1, kH2) +
                                    ffLayerParams(kH2, kH3) + kNumHeads * kHeadParams;
static_assert(ffLayerParams(kInputDim, kH1) == 112640);
static_assert(ffLayerParams(kH1, kH2) == 16768);
static_assert(ffLayerParams(kH2, kH3) == 4192);
static_assert(kNumHeads * kHeadParams == 198);
static_assert(kTotalParams == 133798);

// ---- FF hyperparameters ----
inline constexpr float kGoodnessThreshold = 2.0f;
inline constexpr float kFFLossWeight = 0.5f;

// ---- training_data.bin ----
// Header (16 bytes, little-endian):
//   uint32 magic = 'STTD', uint32 version, uint32 inputDim, uint32 recordCount
// Then recordCount records of kRecordFloats 32-bit words:
//   float input[877], float teacherMove[2], int32 roleIndex, int32 label (+1 positive / -1 negative)
inline constexpr uint32_t kDatasetMagic = 0x44545453; // "STTD"
inline constexpr uint32_t kDatasetVersion = 1;
inline constexpr int kRecordWords = kInputDim + kOutDim + 2;
inline constexpr int kRecordBytes = kRecordWords * 4;
static_assert(kRecordBytes == 3524);

// ---- chaser_master_fp32.bytes ----
// Raw little-endian fp32, no header, exactly kTotalParams floats, in this order:
//   for layer in (L1, L2, L3): W[out][in] (row-major), b[out], gamma[out], beta[out]
//   for head in (Chaser, Runner, Defender): W[2][32] (row-major), b[2]
inline constexpr int kWeightFileBytes = kTotalParams * 4;

} // namespace stai
