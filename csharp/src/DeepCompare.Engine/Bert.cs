using System.Numerics.Tensors;

namespace DeepCompare.Engine;

/// <summary>
/// MiniLM（BERT）の前向き計算と平均プーリング。
///
/// ONNX Runtime を使わず自前で持っている理由は配布サイズ。libonnxruntime は 28.5MB あり、
/// ネイティブライブラリなので NativeAOT でも実行ファイルに畳み込めず、別ファイルとして
/// 付いて回る。必要なのは 6 層の BERT encoder 一本だけなので、そこだけ書く。
///
/// 併せて、配布されている量子化 ONNX が持っていた問題も消える。あちらは活性値まで
/// 実行時に量子化するため、詰め物が入ると同じ行でも結果が 1% ほど動いていた。ここでは
/// 重みだけを int8 で持ち、演算は f32 なので、束ねた相手に依存しない。
///
/// 形の定数は可能な限り重みの次元から導く。設定値を別途持つと、モデルを差し替えた
/// ときに設定だけが古いまま残り、無言で誤った出力を出す。
/// </summary>
public sealed class Bert
{
    private readonly Tensor _wordEmbeddings;
    private readonly Tensor _positionEmbeddings;
    private readonly Tensor _tokenTypeEmbeddings;
    private readonly LayerNorm _embeddingNorm;
    private readonly Layer[] _layers;

    public int HiddenSize { get; }
    public int MaxPosition { get; }

    private readonly int _numHeads;
    private readonly int _headDim;

    public Bert(Dictionary<string, Tensor> weights, int numAttentionHeads = 12, float layerNormEps = 1e-12f)
    {
        _wordEmbeddings = Get(weights, "embeddings.word_embeddings.weight");
        _positionEmbeddings = Get(weights, "embeddings.position_embeddings.weight");
        _tokenTypeEmbeddings = Get(weights, "embeddings.token_type_embeddings.weight");

        HiddenSize = _wordEmbeddings.Cols;
        MaxPosition = _positionEmbeddings.Rows;
        if (HiddenSize % numAttentionHeads != 0)
        {
            throw new InvalidDataException(
                $"hidden_size {HiddenSize} が注意ヘッド数 {numAttentionHeads} で割り切れない");
        }
        _numHeads = numAttentionHeads;
        _headDim = HiddenSize / numAttentionHeads;
        _embeddingNorm = new LayerNorm(weights, "embeddings.LayerNorm", layerNormEps);

        // 層数も数えて決める。
        var layers = new List<Layer>();
        for (var i = 0; weights.ContainsKey($"encoder.layer.{i}.attention.self.query.weight"); i++)
        {
            layers.Add(new Layer(weights, $"encoder.layer.{i}", layerNormEps));
        }
        if (layers.Count == 0)
        {
            throw new InvalidDataException("encoder の層が一つも見つからない");
        }
        _layers = layers.ToArray();
    }

    private static Tensor Get(Dictionary<string, Tensor> weights, string name)
        => weights.TryGetValue(name, out var tensor)
            ? tensor
            : throw new InvalidDataException($"重みが足りない: {name}");

    /// <summary>
    /// 1 つの系列を埋め込む。詰め物は使わない（呼び出し側が長さの揃った行だけを渡す
    /// 必要もない）ので、マスクの扱いを誤る余地がそもそも無い。
    /// 戻り値は平均プーリング後に L2 正規化した長さ <see cref="HiddenSize"/> のベクトル。
    /// </summary>
    public float[] Embed(ReadOnlySpan<int> tokenIds)
    {
        var seq = Math.Min(tokenIds.Length, MaxPosition);
        var hidden = new float[seq * HiddenSize];

        // 語・位置・種別の埋め込みを足して正規化する。単一の文しか扱わないので
        // token_type は常に 0 行目。
        var tokenType = _tokenTypeEmbeddings.Row(0);
        for (var t = 0; t < seq; t++)
        {
            var row = hidden.AsSpan(t * HiddenSize, HiddenSize);
            _wordEmbeddings.Row(tokenIds[t]).CopyTo(row);
            TensorPrimitives.Add(row, _positionEmbeddings.Row(t), row);
            TensorPrimitives.Add(row, tokenType, row);
        }
        _embeddingNorm.ApplyInPlace(hidden, seq, HiddenSize);

        foreach (var layer in _layers)
        {
            layer.Forward(hidden, seq, HiddenSize, _numHeads, _headDim);
        }

        // 平均プーリング。sentence-transformers の MiniLM の既定のプーリング。
        var pooled = new float[HiddenSize];
        for (var t = 0; t < seq; t++)
        {
            TensorPrimitives.Add(pooled, hidden.AsSpan(t * HiddenSize, HiddenSize), pooled);
        }
        TensorPrimitives.Divide(pooled, Math.Max(1, seq), pooled);

        // ここで正規化まで済ませておけば、後段の類似度は単なる内積になる。
        var norm = MathF.Sqrt(TensorPrimitives.Dot<float>(pooled, pooled));
        if (norm > 1e-12f)
        {
            TensorPrimitives.Divide(pooled, norm, pooled);
        }
        return pooled;
    }

    /// <summary>
    /// y[m] = x[m] @ W^T + b。W は [out, in] の行優先なので行同士の内積で足りる。
    ///
    /// 出力側の n を外側に置いてある。素直に m を外側にすると、入力行ごとに重み行列を
    /// 頭から読み直すことになる。全結合部の重みは 2.3MB あるので、20 行の入力なら
    /// 46MB を流すことになり、演算ではなくメモリ帯域で頭打ちになる。n を外側にすれば
    /// 重み 1 行（1.5KB か 6KB）が L1 に載ったまま全入力行と内積を取れて、
    /// 重み全体を読むのは 1 回で済む。
    ///
    /// 内積そのものは同じ組み合わせを同じ順で足すので、結果はビット単位で変わらない。
    /// </summary>
    internal static void Linear(
        ReadOnlySpan<float> input, int rows, int inDim,
        Tensor weight, Tensor bias,
        Span<float> output)
    {
        var outDim = weight.Rows;
        for (var n = 0; n < outDim; n++)
        {
            var w = weight.Row(n);
            var b = bias.Data[n];
            for (var m = 0; m < rows; m++)
            {
                // 手書きの多累算器版も試したが 1.6 倍遅かった。TensorPrimitives.Dot は
                // FMA と複数の累算器を既に使っている。ここは置き換えないこと。
                output[m * outDim + n] = TensorPrimitives.Dot(input.Slice(m * inDim, inDim), w) + b;
            }
        }
    }

    private sealed class LayerNorm(Dictionary<string, Tensor> weights, string prefix, float eps)
    {
        private readonly float[] _weight = Get(weights, $"{prefix}.weight").Data;
        private readonly float[] _bias = Get(weights, $"{prefix}.bias").Data;

        public void ApplyInPlace(Span<float> data, int rows, int dim)
        {
            for (var m = 0; m < rows; m++)
            {
                var row = data.Slice(m * dim, dim);
                var mean = TensorPrimitives.Sum(row) / dim;
                var variance = 0f;
                foreach (var v in row)
                {
                    var d = v - mean;
                    variance += d * d;
                }
                variance /= dim;
                var inv = 1f / MathF.Sqrt(variance + eps);
                for (var i = 0; i < dim; i++)
                {
                    row[i] = (row[i] - mean) * inv * _weight[i] + _bias[i];
                }
            }
        }
    }

    private sealed class Layer
    {
        private readonly Tensor _qw, _qb, _kw, _kb, _vw, _vb;
        private readonly Tensor _attnOutW, _attnOutB;
        private readonly LayerNorm _attnNorm;
        private readonly Tensor _interW, _interB;
        private readonly Tensor _outW, _outB;
        private readonly LayerNorm _outNorm;

        public Layer(Dictionary<string, Tensor> w, string p, float eps)
        {
            _qw = Get(w, $"{p}.attention.self.query.weight");
            _qb = Get(w, $"{p}.attention.self.query.bias");
            _kw = Get(w, $"{p}.attention.self.key.weight");
            _kb = Get(w, $"{p}.attention.self.key.bias");
            _vw = Get(w, $"{p}.attention.self.value.weight");
            _vb = Get(w, $"{p}.attention.self.value.bias");
            _attnOutW = Get(w, $"{p}.attention.output.dense.weight");
            _attnOutB = Get(w, $"{p}.attention.output.dense.bias");
            _attnNorm = new LayerNorm(w, $"{p}.attention.output.LayerNorm", eps);
            _interW = Get(w, $"{p}.intermediate.dense.weight");
            _interB = Get(w, $"{p}.intermediate.dense.bias");
            _outW = Get(w, $"{p}.output.dense.weight");
            _outB = Get(w, $"{p}.output.dense.bias");
            _outNorm = new LayerNorm(w, $"{p}.output.LayerNorm", eps);
        }

        public void Forward(float[] hidden, int seq, int dim, int heads, int headDim)
        {
            var q = new float[seq * dim];
            var k = new float[seq * dim];
            var v = new float[seq * dim];
            Linear(hidden, seq, dim, _qw, _qb, q);
            Linear(hidden, seq, dim, _kw, _kb, k);
            Linear(hidden, seq, dim, _vw, _vb, v);

            var context = new float[seq * dim];
            var scale = 1f / MathF.Sqrt(headDim);
            var scores = new float[seq];

            for (var h = 0; h < heads; h++)
            {
                var offset = h * headDim;
                for (var i = 0; i < seq; i++)
                {
                    var qi = q.AsSpan(i * dim + offset, headDim);
                    for (var j = 0; j < seq; j++)
                    {
                        scores[j] = TensorPrimitives.Dot(qi, k.AsSpan(j * dim + offset, headDim)) * scale;
                    }
                    Softmax(scores);

                    var ctx = context.AsSpan(i * dim + offset, headDim);
                    for (var j = 0; j < seq; j++)
                    {
                        TensorPrimitives.MultiplyAdd(
                            v.AsSpan(j * dim + offset, headDim), scores[j], ctx, ctx);
                    }
                }
            }

            // 自己注意の残差接続。
            var projected = new float[seq * dim];
            Linear(context, seq, dim, _attnOutW, _attnOutB, projected);
            TensorPrimitives.Add(projected, hidden, hidden);
            _attnNorm.ApplyInPlace(hidden, seq, dim);

            // 全結合部。HF の "gelu" は誤差関数版なので、tanh 近似ではなくそちらを使う。
            // ここを取り違えると出力が微妙にずれる。
            var intermediate = new float[seq * _interW.Rows];
            Linear(hidden, seq, dim, _interW, _interB, intermediate);
            for (var i = 0; i < intermediate.Length; i++)
            {
                intermediate[i] = GeluErf(intermediate[i]);
            }

            var output = new float[seq * dim];
            Linear(intermediate, seq, _interW.Rows, _outW, _outB, output);
            TensorPrimitives.Add(output, hidden, hidden);
            _outNorm.ApplyInPlace(hidden, seq, dim);
        }

        private static void Softmax(Span<float> values)
        {
            var max = TensorPrimitives.Max(values);
            var sum = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Exp(values[i] - max);
                sum += values[i];
            }
            TensorPrimitives.Divide(values, sum, values);
        }

        /// <summary>誤差関数版の GELU。tanh 近似ではないので、参照実装と同じ値になる。</summary>
        private static float GeluErf(float x)
            => 0.5f * x * (1f + Erf(x * 0.70710678f));

        /// <summary>Abramowitz &amp; Stegun 7.1.26。単精度なら十分な精度が出る。</summary>
        private static float Erf(float x)
        {
            var sign = MathF.Sign(x);
            x = MathF.Abs(x);
            var t = 1f / (1f + 0.3275911f * x);
            var y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
            return sign * y;
        }
    }
}
