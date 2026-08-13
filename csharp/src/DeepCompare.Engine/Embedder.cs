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
    private readonly ITokenizer _tokenizer;

    private Embedder(Bert bert, ITokenizer tokenizer)
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

        using var vocab = OpenVocabFor(path);
        return Create(File.ReadAllBytes(path), vocab);
    }

    /// <summary>モデルに対応する語彙ファイルの拡張子。</summary>
    public const string VocabExtension = ".vocab";

    /// <summary>
    /// そのモデルの語彙ファイルの場所。<c>minilm.dcm</c> なら <c>minilm.vocab</c>。
    ///
    /// **名前で対応させる。** 語彙と重みは必ず対で使う物で、片方だけ差し替えると
    /// 番号は付くのにモデルが学習した番号とは別物になり、**エラーにならないまま
    /// 無意味な結果**が出る。同じ名前にしておけば、置き忘れに気づける。
    /// </summary>
    public static string VocabPathFor(string modelPath)
        => Path.ChangeExtension(modelPath, VocabExtension);

    /// <summary>
    /// 語彙を開く。隣に無ければ、exe に埋め込んである英語モデル用の物を使う。
    ///
    /// **埋め込みの方を残す。** 既定の minilm.dcm は語彙を別ファイルに持たない
    /// 形で配ってきたので、これを消すと**既存の置き方が壊れる**。
    /// </summary>
    private static Stream OpenVocabFor(string modelPath)
    {
        var beside = VocabPathFor(modelPath);
        return File.Exists(beside)
            ? File.OpenRead(beside)
            : OpenResource(typeof(Embedder).Assembly, "vocab.txt");
    }

    /// <summary>ファイルから組み立てる。試験や、重みを差し替えて試すときに使う。</summary>
    public static Embedder CreateFromFiles(string weightsPath, string vocabPath)
    {
        using var vocab = File.OpenRead(vocabPath);
        return Create(File.ReadAllBytes(weightsPath), vocab);
    }

    private static Embedder Create(byte[] weights, Stream vocab)
    {
        var tensors = DcmWeights.Load(weights);
        var tokenizer = TokenizerLoader.Load(vocab);

        // **語彙と重みが噛み合っているかを、使う前に確かめる。**
        // 合っていないと、番号が語彙表の外を指して例外になるか、
        // 範囲内なら黙って別の語の埋め込みを引く。後者は気づけない。
        if (tensors.TryGetValue("embeddings.word_embeddings.weight", out var wordEmbeddings)
            && wordEmbeddings.Rows != tokenizer.Count)
        {
            throw new InvalidDataException(
                $"語彙 {tokenizer.Count:N0} 語に対して、モデルは {wordEmbeddings.Rows:N0} 語を"
                + $" 期待しています。モデルと語彙（{VocabExtension}）を対で置いてください。");
        }
        return new Embedder(new Bert(tensors), tokenizer);
    }

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
