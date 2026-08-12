namespace DeepCompare.Engine;

/// <summary>読み込んだ重み 1 本。行優先で並んだ f32。</summary>
public sealed record Tensor(float[] Data, int[] Shape)
{
    public int Rows => Shape[0];
    public int Cols => Shape.Length > 1 ? Shape[1] : 1;

    /// <summary>行 r の先頭からの連続領域。行列積はこの単位で内積を取る。</summary>
    public ReadOnlySpan<float> Row(int r) => Data.AsSpan(r * Cols, Cols);
}

/// <summary>
/// exe に埋め込む重みの容器を読む。
///
/// 目的は配布サイズを小さくすることだけで、推論を速くすることではない。よって行ごとの
/// 対称 int8 で保存し、読み込み時に f32 へ戻す。演算は f32 のままなので、量子化された
/// 行列積を書き起こす必要がない。
///
/// 書き出しは DeepCompare.ModelPrep、読み込みはこの中の Load。片方だけを直すと
/// 無言で壊れるので、定数の定義をここに集めてある。
///
/// 配布物として出回っている量子化 ONNX と違い、量子化されるのは重みだけで活性値は
/// f32 のままなので、束ねた相手によって結果が変わることがない。
/// </summary>
public static class DcmWeights
{
    private static ReadOnlySpan<byte> Magic => "DCM1"u8;

    /// <summary>書き出し側（model-prep）が使う。読み書きで同じ値を指す必要がある。</summary>
    public static byte[] MagicBytes => "DCM1"u8.ToArray();

    /// <summary>f32 をそのまま。バイアスや LayerNorm など、小さくて精度が効く物に使う。</summary>
    public const byte KindF32 = 0;

    /// <summary>行ごとの対称 int8。value = q * scale[row]。</summary>
    public const byte KindQ8PerRow = 1;

    /// <summary>
    /// int8 化する下限。これを下回る物は f32 のままでも合計サイズにほぼ効かず、
    /// 量子化誤差だけが乗るので触らない。
    /// </summary>
    public const int QuantizeMinElements = 4096;

    public static Dictionary<string, Tensor> Load(ReadOnlySpan<byte> data)
    {
        var reader = new Reader(data);
        if (!reader.Take(4).SequenceEqual(Magic))
        {
            throw new InvalidDataException("重みの形式が違う（MAGIC 不一致）");
        }

        var count = (int)reader.U32();
        var result = new Dictionary<string, Tensor>(count, StringComparer.Ordinal);
        for (var i = 0; i < count; i++)
        {
            var name = reader.String();
            var kind = reader.U8();
            var rank = reader.U8();
            var shape = new int[rank];
            var numel = 1;
            for (var d = 0; d < rank; d++)
            {
                shape[d] = (int)reader.U32();
                numel *= shape[d];
            }

            float[] values;
            switch (kind)
            {
                case KindF32:
                    values = reader.F32Array(numel);
                    break;

                case KindQ8PerRow:
                    if (rank != 2)
                    {
                        throw new InvalidDataException($"{name}: 行ごと量子化は 2 次元のみ");
                    }
                    var rows = shape[0];
                    var cols = shape[1];
                    var scales = reader.F32Array(rows);
                    var quantized = reader.Take(numel);
                    values = new float[numel];
                    for (var r = 0; r < rows; r++)
                    {
                        var scale = scales[r];
                        var offset = r * cols;
                        for (var c = 0; c < cols; c++)
                        {
                            values[offset + c] = (sbyte)quantized[offset + c] * scale;
                        }
                    }
                    break;

                default:
                    throw new InvalidDataException($"{name}: 未知の保存形式 {kind}");
            }
            result[name] = new Tensor(values, shape);
        }

        if (!reader.AtEnd)
        {
            throw new InvalidDataException("重みの末尾に余分なバイトがある");
        }
        return result;
    }

    /// <summary>先頭から順に読むだけの薄いカーソル。長さ確認を一箇所に集約する。</summary>
    private ref struct Reader(ReadOnlySpan<byte> data)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _pos = 0;

        public readonly bool AtEnd => _pos == _data.Length;

        public ReadOnlySpan<byte> Take(int n)
        {
            if (_pos + n > _data.Length)
            {
                throw new InvalidDataException("重みの終端を越えて読もうとした");
            }
            var slice = _data.Slice(_pos, n);
            _pos += n;
            return slice;
        }

        public byte U8() => Take(1)[0];

        public uint U32() => BitConverter.ToUInt32(Take(4));

        public string String()
        {
            var length = (int)U32();
            return System.Text.Encoding.UTF8.GetString(Take(length));
        }

        public float[] F32Array(int count)
        {
            var bytes = Take(count * 4);
            var values = new float[count];
            for (var i = 0; i < count; i++)
            {
                values[i] = BitConverter.ToSingle(bytes.Slice(i * 4, 4));
            }
            return values;
        }
    }
}
