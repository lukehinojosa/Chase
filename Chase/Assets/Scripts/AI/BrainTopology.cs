namespace StudentTeacherAI
{
    /// <summary>
    /// Network topology and on-disk formats shared with the C++ pretrainer.
    /// Mirror: src/Topology.hpp in the repo root. Keep both in sync.
    /// </summary>
    public static class BrainTopology
    {
        // ---- Input vector layout (877 floats, bot-local yaw space) ----
        public const int VoxelX = 11, VoxelY = 7, VoxelZ = 11;
        public const int VoxelCount = VoxelX * VoxelY * VoxelZ; // 847
        public const int AttractSlot = 8; // has_LOS, dir_xyz, dist_norm, vel_xz, time_since_seen
        public const int RepelSlot = 8;
        public const int ProtectSlot = 5; // is_active, dir_xyz, dist_norm
        public const int AllySlot = 5;
        public const int SelfContext = 4; // self_vel_xz, role_state, territory_state

        public const int OffVoxel = 0;
        public const int OffAttract = OffVoxel + VoxelCount;
        public const int OffRepel = OffAttract + AttractSlot;
        public const int OffProtect = OffRepel + RepelSlot;
        public const int OffAlly = OffProtect + ProtectSlot;
        public const int OffSelf = OffAlly + AllySlot;
        public const int InputDim = OffSelf + SelfContext; // 877

        // ---- Layers ----
        public const int H1 = 128, H2 = 128, H3 = 32;
        public const int NumHeads = 3; // see Role
        public const int OutDim = 2;   // (move_x, move_z)
        public const int ActivationsPerBot = H1 + H2 + H3; // 288

        static int FFLayerParams(int inDim, int outDim) => inDim * outDim + outDim + 2 * outDim;
        public const int HeadParams = H3 * OutDim + OutDim;
        public static readonly int TotalParams =
            FFLayerParams(InputDim, H1) + FFLayerParams(H1, H2) + FFLayerParams(H2, H3) + NumHeads * HeadParams; // 133798
        public static int PackedFp16Words => (TotalParams + 1) / 2; // 66899

        // ---- FF hyperparameters ----
        public const float GoodnessThreshold = 2.0f;

        // ---- training_data.bin (see TrajectoryWriter) ----
        public const uint DatasetMagic = 0x44545453; // "STTD"
        public const uint DatasetVersion = 1;
        public const int RecordWords = InputDim + OutDim + 2;
        public const int RecordBytes = RecordWords * 4; // 3524

        // ---- chaser_master_fp32.bytes ----
        // Raw little-endian fp32, no header, TotalParams floats:
        //   for layer in (L1, L2, L3): W[out][in] (row-major), b[out], gamma[out], beta[out]
        //   for head in (Chaser, Runner, Defender): W[2][32] (row-major), b[2]
        public const string MasterWeightsFile = "chaser_master_fp32.bytes";
    }

    /// <summary>Readout head index. Matches the head order in the weight file.</summary>
    public enum Role
    {
        Chaser = 0,
        Runner = 1,
        Defender = 2,
    }
}
