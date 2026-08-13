using System.Reflection;

namespace DeepCompare.Engine;

/// <summary>
/// 行から埋め込みベクトルを作る。
///
/// 推論は行数に比例して重くなるので、ここで無駄な推論を落としきる。ソースコードには
/// `}` や空行のように同一内容の行が大量に出るため、同じ文字列は一度しか通さない。
///
/// 詰め物は使わない。系列ごとに独立して計算するので、束ねた相手によって結果が変わる
/// ことがなく、マスクの扱いを誤る余地も無い。
/// </summary>
public sealed class Embedder
{
    /// <summary>位置埋め込みの上限。これを越える行は切り詰める。</summary>
    private const int MaxTokens = 512;

    private readonly Bert _bert;
    private readonly WordPieceTokenizer _tokenizer;

    private Embedder(Bert bert, WordPieceTokenizer tokenizer)
    {
        _bert = bert;
        _tokenizer = tokenizer;
    }

    /// <summary>既定の重みファイルの名前。実行ファイルと同じ場所に置く。</summary>
    public const string DefaultWeightsFileName = "minilm.dcm";

    /// <summary>モデルの置き場所を指す環境変数。</summary>
    public const string ModelEnvironmentVariable = "DEEPCOMPARE_MODEL";

    /// <summary>
    /// 使うモデルの場所を決める。
    ///
    /// 探す順は ①明示された場所 ②環境変数 ③実行ファイルの隣。
    /// **環境変数を入れてあるのは、入れ替えて試すのに再配置が要らないから。**
    /// 日本語向けのモデルを試すとき、この 1 つで切り替えられる。
    /// </summary>
    public static string ResolveModelPath(string? explicitPath = null)
    {
        if (explicitPath is { Length: > 0 })
        {
            // 名前だけ渡されたら、実行ファイルの隣を探す。
            return File.Exists(explicitPath)
                ? explicitPath
                : Path.Combine(AppContext.BaseDirectory, explicitPath);
        }

        var fromEnvironment = Environment.GetEnvironmentVariable(ModelEnvironmentVariable);
        if (fromEnvironment is { Length: > 0 })
        {
            return fromEnvironment;
        }

        return Path.Combine(AppContext.BaseDirectory, DefaultWeightsFileName);
    }

    /// <summary>
    /// 実行ファイルの隣にあるモデルの一覧。**選ばせるために要る。**
    /// 置いてあるものが分からないと、名前を当てて打ち込むことになる。
    /// </summary>
    public static IReadOnlyList<string> AvailableModels()
    {
        try
        {
            return [.. Directory.EnumerateFiles(AppContext.BaseDirectory, "*.dcm")
                .Select(Path.GetFileName)
                .Where(name => name is { Length: > 0 })
                .Select(name => name!)
                .Order(StringComparer.OrdinalIgnoreCase)];
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    /// <summary>
    /// 重みと、埋め込みの語彙から組み立てる。
    ///
    /// 重みを埋め込まないのは、22MB を実行ファイルから追い出すためと、再ビルド無しで
    /// モデルを差し替えられるようにするため。
    /// </summary>
    public static Embedder CreateFromDefaultAssets(string? modelPath = null)
    {
        var path = ResolveModelPath(modelPath);
        if (!File.Exists(path))
        {
            var available = AvailableModels();
            throw new FileNotFoundException(
                $"モデルが見つからない: {path}"
                + (available.Count > 0
                    ? $"（隣にあるのは {string.Join(", ", available)}）"
                    : $"（実行ファイルと同じ場所に {DefaultWeightsFileName} を置く）"),
                path);
        }

        using var vocab = OpenResource(typeof(Embedder).Assembly, "vocab.txt");
        return Create(File.ReadAllBytes(path), vocab);
    }

    /// <summary>ファイルから組み立てる。試験や、重みを差し替えて試すときに使う。</summary>
    public static Embedder CreateFromFiles(string weightsPath, string vocabPath)
    {
        using var vocab = File.OpenRead(vocabPath);
        return Create(File.ReadAllBytes(weightsPath), vocab);
    }

    private static Embedder Create(byte[] weights, Stream vocab)
        => new(new Bert(DcmWeights.Load(weights)), WordPieceTokenizer.FromVocab(vocab));

    private static Stream OpenResource(Assembly assembly, string name)
        => assembly.GetManifestResourceStream(name)
           ?? throw new InvalidOperationException(
               $"埋め込み資材が見つからない: {name}（ビルド時に Assets/ へ配置されているか確認）");

    /// <summary>
    /// 各行の埋め込みを、入力と同じ順序で返す。すべて L2 正規化済みなので、
    /// 類似度は内積で取れる。
    /// </summary>
    public IReadOnlyList<float[]> EmbedLines(IReadOnlyList<string> lines)
    {
        if (lines.Count == 0)
        {
            return [];
        }

        // 同じ文字列は一度だけ通す。
        var unique = new List<string>();
        var indexOf = new Dictionary<string, int>(StringComparer.Ordinal);
        var assignment = new int[lines.Count];
        for (var i = 0; i < lines.Count; i++)
        {
            if (!indexOf.TryGetValue(lines[i], out var slot))
            {
                slot = unique.Count;
                unique.Add(lines[i]);
                indexOf[lines[i]] = slot;
            }
            assignment[i] = slot;
        }

        var embeddings = new float[unique.Count][];
        // 行どうしは独立なので並列に回せる。
        Parallel.For(0, unique.Count, i =>
        {
            var ids = _tokenizer.Encode(unique[i], MaxTokens);
            embeddings[i] = _bert.Embed(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids));
        });

        var result = new float[lines.Count][];
        for (var i = 0; i < lines.Count; i++)
        {
            result[i] = embeddings[assignment[i]];
        }
        return result;
    }

    /// <summary>正規化済みベクトル同士のコサイン類似度。内積そのもの。</summary>
    public static float CosineSimilarity(float[] a, float[] b)
        => System.Numerics.Tensors.TensorPrimitives.Dot<float>(a, b);
}
