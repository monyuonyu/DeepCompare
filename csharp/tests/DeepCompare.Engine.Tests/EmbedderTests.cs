using Xunit;
using System.Text.Json;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 埋め込みが Rust 版と一致することを確認する。
///
/// 参照値 tests/reference_embeddings.json は Python + ONNX Runtime（fp32）で作ったもので、
/// Rust 版もこれと突き合わせて f32 で 0.999999 の一致を確認済み。同じ参照値に対して
/// C# 版も一致すれば、言語を跨いでも同じ結果が出ていることになる。
///
/// こちらは量子化済み（quint8）の ONNX を使うので、fp32 参照値との差はその分だけ出る。
/// 行の対応付けは 0.5 前後を境に判断するため、0.99 を保てば判断が変わることはまず無い。
/// </summary>
public class EmbedderTests
{
    private static string RepositoryRoot
    {
        get
        {
            var dir = AppContext.BaseDirectory;
            while (dir is not null && !File.Exists(Path.Combine(dir, "README.md")))
            {
                dir = Path.GetDirectoryName(dir);
            }
            return dir ?? throw new InvalidOperationException("リポジトリの根が見つからない");
        }
    }

    private static (List<string> Texts, List<float[]> Vectors)? LoadReference()
    {
        var path = Path.Combine(RepositoryRoot, "tests", "reference_embeddings.json");
        if (!File.Exists(path))
        {
            return null;
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var texts = doc.RootElement.GetProperty("texts")
            .EnumerateArray().Select(e => e.GetString()!).ToList();
        var vectors = doc.RootElement.GetProperty("vectors")
            .EnumerateArray()
            .Select(row => row.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray())
            .ToList();
        return (texts, vectors);
    }

    [Fact]
    public void EmbeddingsMatchTheReferenceImplementation()
    {
        var reference = LoadReference();
        if (reference is null)
        {
            // 参照値が無い環境では黙って飛ばす。
            return;
        }

        var embedder = Embedder.CreateFromDefaultAssets();
        var ours = embedder.EmbedLines(reference.Value.Texts);
        Assert.Equal(reference.Value.Vectors.Count, ours.Count);

        var min = float.PositiveInfinity;
        var sum = 0f;
        for (var i = 0; i < ours.Count; i++)
        {
            var similarity = Embedder.CosineSimilarity(ours[i], reference.Value.Vectors[i]);
            min = Math.Min(min, similarity);
            sum += similarity;
            Console.WriteLine($"{similarity:F6}  {reference.Value.Texts[i]}");
        }
        Console.WriteLine($"最小 {min:F6} / 平均 {sum / ours.Count:F6}");

        // 参照値は fp32 なので、ここでの差は量子化そのものによるもの。
        // Rust 版と同じ重み（行ごと int8、演算は f32）を使っているので、
        // 同じ水準（Rust 版の実測は最小 0.9991）に収まるはず。
        Assert.True(min > 0.99f, $"参照実装と離れすぎている（最小 {min:F6}）");
    }

    /// <summary>
    /// 量子化していない f32 の重みを同じ経路に通す。
    ///
    /// これが一致するなら実装（トークナイズ・マスク・プーリング・正規化）は正しく、
    /// ずれの原因は量子化にあると言い切れる。逆にここで外れるなら実装の誤り。
    /// 切り分けができないと「なんとなく合っていない」で止まってしまう。
    /// </summary>
    [Fact]
    public void Fp32ModelMatchesTheReferenceAlmostExactly()
    {
        var reference = LoadReference();
        var modelPath = Path.Combine(RepositoryRoot, "assets", "minilm-f32.dcm");
        var vocabPath = Path.Combine(RepositoryRoot, "assets", "src", "vocab.txt");
        if (reference is null || !File.Exists(modelPath) || !File.Exists(vocabPath))
        {
            return;
        }

        var embedder = Embedder.CreateFromFiles(modelPath, vocabPath);
        var ours = embedder.EmbedLines(reference.Value.Texts);

        var min = float.PositiveInfinity;
        for (var i = 0; i < ours.Count; i++)
        {
            var similarity = Embedder.CosineSimilarity(ours[i], reference.Value.Vectors[i]);
            min = Math.Min(min, similarity);
            Console.WriteLine($"fp32 {similarity:F6}  {reference.Value.Texts[i]}");
        }
        Console.WriteLine($"fp32 最小 {min:F6}");
        Assert.True(min > 0.9999f, $"実装のどこかが違う（fp32 で最小 {min:F6}）");
    }

    [Fact]
    public void IdenticalTextsProduceIdenticalVectors()
    {
        var embedder = Embedder.CreateFromDefaultAssets();
        // 重複除去が効いていれば、同じ文字列は同じ配列を指すか、少なくとも同値になる。
        var vectors = embedder.EmbedLines(["import os", "import os", "import sys"]);
        Assert.Equal(vectors[0], vectors[1]);
        Assert.NotEqual(vectors[0], vectors[2]);
    }

    [Fact]
    public void EmbeddingsAreL2Normalized()
    {
        var embedder = Embedder.CreateFromDefaultAssets();
        foreach (var vector in embedder.EmbedLines(["def main():", "", "    return x + 1"]))
        {
            var norm = MathF.Sqrt(vector.Sum(v => v * v));
            Assert.InRange(norm, 0.999f, 1.001f);
        }
    }

    /// <summary>
    /// 長さの違う行を混ぜても、同じ行の埋め込みが変わらないこと。
    ///
    /// 量子化 ONNX は活性値のスケールを実行時に決めるので、詰め物が入ると同じ行でも
    /// 結果が動く（長さを揃えずに束ねると実測 0.9907 まで落ちた）。トークン長が
    /// 完全に一致する行だけを束ねることで詰め物をなくし、束ねた相手に依存しないようにした。
    /// これが崩れると、無関係な行を 1 行足しただけでスコアが変わる。
    /// </summary>
    [Fact]
    public void BatchCompositionDoesNotAffectResults()
    {
        var embedder = Embedder.CreateFromDefaultAssets();
        var alone = embedder.EmbedLines(["x = 1"])[0];
        var mixed = embedder.EmbedLines(["x = 1", new string('y', 400), "z"])[0];
        Assert.Equal(alone, mixed);
    }

    [Fact]
    public void EmptyInputReturnsNothing()
    {
        var embedder = Embedder.CreateFromDefaultAssets();
        Assert.Empty(embedder.EmbedLines([]));
    }
}
