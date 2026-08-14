using System.Text;
using System.Text.Json;
using DeepCompare.Engine;

namespace DeepCompare.ModelPrep;

/// <summary>
/// MiniLM の safetensors を int8 量子化して、exe に埋め込む形式へ変換する開発用ツール。
///
/// 使い方:
///   model-prep assets/src/model.safetensors assets/minilm.dcm [--no-quantize]
///
/// --no-quantize は検証用。同じ経路を f32 のまま通した物を作れるので、
/// 「実装が間違っている」のか「量子化で精度が落ちた」のかを切り分けられる。
/// </summary>
internal static class Program
{
    /// <summary>
    /// 文埋め込みでは使わない重み。
    ///
    /// pooler は sentence-transformers の MiniLM が平均プーリングを使うので出番が無い。
    /// cls / predictions は事前学習のマスク言語モデル用ヘッドで、語彙サイズの行列を
    /// 抱えているため、残すと配布物が無駄に膨らむ。
    /// </summary>
    private static bool ShouldSkip(string name)
        => name.StartsWith("pooler.", StringComparison.Ordinal)
           || name.Contains(".pooler.", StringComparison.Ordinal)
           || name.StartsWith("cls.", StringComparison.Ordinal)
           || name.Contains(".predictions.", StringComparison.Ordinal)
           || name.EndsWith(".position_ids", StringComparison.Ordinal);

    public static int Main(string[] args)
    {
        var quantize = !args.Contains("--no-quantize");
        var keepRowsPath = ValueOf(args, "--keep-rows");
        var positional = Positional(args).ToArray();
        if (positional.Length < 2)
        {
            Console.Error.WriteLine(
                "使い方: model-prep <入力.safetensors> <出力.dcm> [--no-quantize] "
                + "[--keep-rows <残す行番号の一覧>]");
            return 2;
        }

        try
        {
            var keepRows = keepRowsPath is null ? null : ReadKeepRows(keepRowsPath);
            Convert(positional[0], positional[1], quantize, keepRows);
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"エラー: {error.Message}");
            return 1;
        }
    }

    /// <summary>オプションの値。<c>--keep-rows 一覧.txt</c> の形で受ける。</summary>
    private static string? ValueOf(string[] args, string name)
    {
        var at = Array.IndexOf(args, name);
        return at >= 0 && at + 1 < args.Length ? args[at + 1] : null;
    }

    /// <summary>値を伴うオプションの「値の側」を、位置引数と取り違えない。</summary>
    private static IEnumerable<string> Positional(string[] args)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == "--keep-rows")
            {
                i++;
                continue;
            }
            if (!args[i].StartsWith('-'))
            {
                yield return args[i];
            }
        }
    }

    private static int[] ReadKeepRows(string path)
        => [.. File.ReadLines(path)
            .Where(line => line.Trim().Length > 0)
            .Select(line => int.Parse(line.Trim()))];

    private static void Convert(
        string inputPath, string outputPath, bool quantize, int[]? keepRows)
    {
        var raw = File.ReadAllBytes(inputPath);
        var tensors = SafeTensors.Parse(raw);

        var body = new MemoryStream();
        var written = 0u;
        var report = new Report();

        foreach (var name in tensors.Keys.Order(StringComparer.Ordinal))
        {
            if (ShouldSkip(name))
            {
                report.Skipped.Add(name);
                continue;
            }

            var tensor = tensors[name];
            var values = tensor.ToFloats(raw);
            var shape = tensor.Shape;

            // **語彙の刈り込み。** 埋め込み行列はモデルの 8 割を占めるが、
            // その大半は使わない言語の行。残す行だけ抜き出す。
            // **抜いた行は、対象の言語の文には一致し得ないもの**なので、
            // トークン化の結果は変わらない（unigram は語彙全体から最良の
            // 分割を選ぶが、候補になり得ないものを消しても選択は動かない）。
            if (keepRows is not null && IsWordEmbedding(name))
            {
                (values, shape) = TakeRows(values, shape, keepRows, name);
            }

            WriteString(body, name);

            if (quantize && shape.Length == 2 && values.Length >= DcmWeights.QuantizeMinElements)
            {
                body.WriteByte(DcmWeights.KindQ8PerRow);
                WriteShape(body, shape);
                report.RecordQuantized(name, values,
                    WriteQ8PerRow(body, values, shape[0], shape[1]));
            }
            else
            {
                body.WriteByte(DcmWeights.KindF32);
                WriteShape(body, shape);
                foreach (var v in values)
                {
                    body.Write(BitConverter.GetBytes(v));
                }
                report.KeptF32Bytes += values.Length * 4;
            }
            written++;
        }

        using var output = File.Create(outputPath);
        output.Write(DcmWeights.MagicBytes);
        output.Write(BitConverter.GetBytes(written));
        body.Position = 0;
        body.CopyTo(output);
        output.Flush();

        report.Print(raw.Length, output.Length, written, outputPath);
    }

    /// <summary>
    /// 行ごとに scale = max|w| / 127 を取り、round(w / scale) を i8 で書く。
    ///
    /// 行ごとにするのは、BERT の線形層は行（出力チャネル）ごとに大きさの桁が違うため。
    /// 行列全体で 1 つのスケールにすると、小さい行がまるごと 0 に潰れる。
    /// </summary>
    private static QuantStats WriteQ8PerRow(Stream output, float[] values, int rows, int cols)
    {
        var scales = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var maxAbs = 0f;
            for (var c = 0; c < cols; c++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(values[r * cols + c]));
            }
            // 全要素 0 の行でも 0 除算にしない。
            scales[r] = maxAbs > 0f ? maxAbs / 127f : 1f;
        }
        foreach (var s in scales)
        {
            output.Write(BitConverter.GetBytes(s));
        }

        var stats = new QuantStats();
        var quantized = new byte[values.Length];
        for (var r = 0; r < rows; r++)
        {
            var scale = scales[r];
            for (var c = 0; c < cols; c++)
            {
                var index = r * cols + c;
                var v = values[index];
                // MathF.Round の既定は偶数丸めで、0.5 を 0 に落とす。Rust の round() は
                // 0 から遠い側へ丸めるので、そのままだと同じ重みから別のバイト列が出る。
                var q = (sbyte)Math.Clamp(
                    MathF.Round(v / scale, MidpointRounding.AwayFromZero), -127f, 127f);
                quantized[index] = (byte)q;
                var error = q * scale - v;
                stats.MaxAbsError = Math.Max(stats.MaxAbsError, Math.Abs(error));
                stats.SumSquaredError += (double)error * error;
                stats.SumSquaredReference += (double)v * v;
            }
        }
        output.Write(quantized);
        return stats;
    }

    private static bool IsWordEmbedding(string name)
        => name.EndsWith("word_embeddings.weight", StringComparison.Ordinal);

    /// <summary>
    /// 指定された行だけを、指定された順で抜き出す。
    ///
    /// **順番は残す側の並びに従う。** 語彙ファイルの行番号がそのまま
    /// トークン ID になるので、ここがずれると全部の語が別の意味になる。
    /// </summary>
    private static (float[] Values, int[] Shape) TakeRows(
        float[] values, int[] shape, int[] keep, string name)
    {
        if (shape.Length != 2)
        {
            throw new InvalidOperationException(
                $"{name} は 2 次元ではないので刈り込めない（形 {string.Join('×', shape)}）。");
        }

        var (rows, cols) = (shape[0], shape[1]);
        var taken = new float[(long)keep.Length * cols];
        for (var i = 0; i < keep.Length; i++)
        {
            var source = keep[i];
            if (source < 0 || source >= rows)
            {
                throw new InvalidOperationException(
                    $"残す行 {source} が範囲外（{name} は {rows} 行）。");
            }
            Array.Copy(values, (long)source * cols, taken, (long)i * cols, cols);
        }

        Console.WriteLine($"  刈り込み: {name} {rows} 行 -> {keep.Length} 行");
        return (taken, [keep.Length, cols]);
    }

    private static void WriteString(Stream output, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        output.Write(BitConverter.GetBytes((uint)bytes.Length));
        output.Write(bytes);
    }

    private static void WriteShape(Stream output, int[] shape)
    {
        output.WriteByte((byte)shape.Length);
        foreach (var d in shape)
        {
            output.Write(BitConverter.GetBytes((uint)d));
        }
    }

    private struct QuantStats
    {
        public float MaxAbsError;
        public double SumSquaredError;
        public double SumSquaredReference;
    }

    private sealed class Report
    {
        public List<string> Skipped { get; } = [];
        public int KeptF32Bytes;
        private int _quantized;
        private (string Name, float Error)? _worst;
        private double _totalSquaredError;
        private double _totalSquaredReference;

        public void RecordQuantized(string name, float[] values, QuantStats stats)
        {
            _quantized++;
            _totalSquaredError += stats.SumSquaredError;
            _totalSquaredReference += stats.SumSquaredReference;
            if (_worst is null || stats.MaxAbsError > _worst.Value.Error)
            {
                _worst = (name, stats.MaxAbsError);
            }
        }

        public void Print(long inputBytes, long outputBytes, uint written, string outputPath)
        {
            static double Mib(long b) => b / 1024.0 / 1024.0;
            Console.WriteLine($"出力: {outputPath}");
            Console.WriteLine($"  {Mib(inputBytes):F1} MiB -> {Mib(outputBytes):F1} MiB "
                + $"({(double)outputBytes / inputBytes * 100:F1}%)");
            Console.WriteLine($"  テンソル {written} 本 (int8 {_quantized} 本 / "
                + $"f32 のまま {Mib(KeptF32Bytes):F2} MiB)");
            if (Skipped.Count > 0)
            {
                Console.WriteLine($"  除外 {Skipped.Count} 本: {string.Join(", ", Skipped)}");
            }
            // 相対誤差。埋め込みのコサイン類似度への影響は別途、実文で測る。
            var relative = Math.Sqrt(_totalSquaredError / Math.Max(_totalSquaredReference, double.Epsilon));
            Console.WriteLine($"  量子化の相対二乗誤差: {relative * 100:F4}%");
            if (_worst is { } worst)
            {
                Console.WriteLine($"  最大絶対誤差: {worst.Error:F6} ({worst.Name})");
            }
        }
    }
}

/// <summary>safetensors の最小限の読み取り。必要なのは名前・形・dtype・実体だけ。</summary>
internal sealed record SafeTensorEntry(string Dtype, int[] Shape, long Begin, long End)
{
    public float[] ToFloats(byte[] file)
    {
        var span = file.AsSpan((int)Begin, (int)(End - Begin));
        return Dtype switch
        {
            "F32" => Read(span, 4, s => BitConverter.ToSingle(s)),
            "F16" => Read(span, 2, s => (float)BitConverter.ToHalf(s)),
            "BF16" => Read(span, 2, s =>
            {
                // bf16 は f32 の上位 16 ビットそのもの。
                var bits = (uint)BitConverter.ToUInt16(s) << 16;
                return BitConverter.UInt32BitsToSingle(bits);
            }),
            _ => throw new NotSupportedException($"未対応の dtype: {Dtype}"),
        };
    }

    private static float[] Read(ReadOnlySpan<byte> span, int width, ReadFloat convert)
    {
        var result = new float[span.Length / width];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = convert(span.Slice(i * width, width));
        }
        return result;
    }

    private delegate float ReadFloat(ReadOnlySpan<byte> bytes);
}

internal static class SafeTensors
{
    public static Dictionary<string, SafeTensorEntry> Parse(byte[] file)
    {
        // 先頭 8 バイトがヘッダ長、続く JSON がテンソルの一覧、その後が実体。
        var headerLength = (int)BitConverter.ToUInt64(file, 0);
        var dataStart = 8 + headerLength;
        using var doc = JsonDocument.Parse(file.AsMemory(8, headerLength));

        var result = new Dictionary<string, SafeTensorEntry>(StringComparer.Ordinal);
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__")
            {
                continue;
            }
            var offsets = property.Value.GetProperty("data_offsets");
            result[property.Name] = new SafeTensorEntry(
                property.Value.GetProperty("dtype").GetString()!,
                property.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                dataStart + offsets[0].GetInt64(),
                dataStart + offsets[1].GetInt64());
        }
        return result;
    }
}
