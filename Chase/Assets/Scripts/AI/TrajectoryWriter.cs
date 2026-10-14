using System;
using System.IO;

namespace StudentTeacherAI
{
    /// <summary>
    /// Writes training_data.bin for the C++ pretrainer (src/TrajectoryDataset.cpp).
    ///
    /// Header (16 bytes, little-endian): uint magic 'STTD', uint version, uint inputDim, uint recordCount.
    /// Record: float input[877], float teacherMove[2], int roleIndex, int label (+1 positive / -1 negative).
    /// recordCount is patched on Dispose, so always dispose the writer.
    /// </summary>
    public sealed class TrajectoryWriter : IDisposable
    {
        readonly BinaryWriter _writer;
        uint _count;

        public uint Count => _count;

        public TrajectoryWriter(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            _writer = new BinaryWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20));
            _writer.Write(BrainTopology.DatasetMagic);
            _writer.Write(BrainTopology.DatasetVersion);
            _writer.Write((uint)BrainTopology.InputDim);
            _writer.Write(0u); // recordCount, patched on Dispose
        }

        public void Write(ReadOnlySpan<float> input, float moveX, float moveZ, Role role, bool positive)
        {
            if (input.Length != BrainTopology.InputDim)
                throw new ArgumentException($"input has {input.Length} floats, expected {BrainTopology.InputDim}");

            foreach (var v in input)
                _writer.Write(v);
            _writer.Write(moveX);
            _writer.Write(moveZ);
            _writer.Write((int)role);
            _writer.Write(positive ? 1 : -1);
            _count++;
        }

        public void Dispose()
        {
            _writer.Seek(12, SeekOrigin.Begin);
            _writer.Write(_count);
            _writer.Dispose();
        }
    }
}
