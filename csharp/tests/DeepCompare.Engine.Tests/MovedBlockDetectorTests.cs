using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 移動したブロックの検出。
///
/// 短い一致を拾うと「移動」だらけになって本当に動いたものが埋もれるので、
/// 拾いすぎないことを重点的に見る。
/// </summary>
public sealed class MovedBlockDetectorTests
{
    private static DecodedText Text(params string[] lines)
        => TextDecoder.Decode(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines)));

    private static List<MovedBlock> Detect(string[] left, string[] right, int minimum = 3)
    {
        var l = Text(left);
        var r = Text(right);
        return MovedBlockDetector.Detect(DiffComparer.Compare(l, r, embedder: null), l, r, minimum);
    }

    private static string[] Function(string name) =>
    [
        $"void {name}() {{",
        $"    doWork({name});",
        $"    log(\"{name}\");",
        "}",
    ];

    [Fact]
    public void IdenticalFilesHaveNoMoves()
    {
        Assert.Empty(Detect(["a", "b", "c"], ["a", "b", "c"]));
    }

    /// <summary>関数がファイル内で下へ動いた場合。</summary>
    [Fact]
    public void AFunctionMovedDownIsDetected()
    {
        string[] left = [.. Function("first"), .. Function("second"), "// 末尾"];
        string[] right = [.. Function("second"), .. Function("first"), "// 末尾"];

        var moved = Detect(left, right);

        Assert.NotEmpty(moved);
        Assert.All(moved, m => Assert.True(m.Length >= 3));
        Assert.All(moved, m => Assert.True(m.Exact));
    }

    /// <summary>短い一致は拾わないこと。`}` だけの行で「移動」と言われても困る。</summary>
    [Fact]
    public void ShortRunsAreIgnored()
    {
        var moved = Detect(["a", "}", "b"], ["b", "}", "a"], minimum: 3);

        Assert.Empty(moved);
    }

    [Fact]
    public void TheMinimumLengthIsRespected()
    {
        string[] left = ["x1", "x2", "keep", "m1", "m2"];
        string[] right = ["m1", "m2", "keep", "x1", "x2"];

        Assert.Empty(Detect(left, right, minimum: 3));
        Assert.NotEmpty(Detect(left, right, minimum: 2));
    }

    /// <summary>
    /// 移動した行の位置が正しいこと。
    ///
    /// **どちらが「動いた」かは、残った側の方が長いかで決まる。** Myers はより長い
    /// 共通部分を軸に取るので、動かない側を長くしておかないと「stay が動いた」と
    /// 解釈される。ここでは stay を 4 行にして軸を固定する。
    /// </summary>
    [Fact]
    public void ThePositionsPointAtTheMovedLines()
    {
        string[] left = ["moved1", "moved2", "moved3", "stay1", "stay2", "stay3", "stay4"];
        string[] right = ["stay1", "stay2", "stay3", "stay4", "moved1", "moved2", "moved3"];

        var block = Assert.Single(Detect(left, right));

        Assert.Equal(0, block.LeftStart);
        Assert.Equal(4, block.RightStart);
        Assert.Equal(3, block.Length);
    }

    /// <summary>同じ右側の範囲を 2 つの移動に割り当てないこと。</summary>
    [Fact]
    public void OneTargetRangeIsUsedOnce()
    {
        string[] left = ["a1", "a2", "a3", "keep", "a1", "a2", "a3"];
        string[] right = ["keep", "a1", "a2", "a3"];

        var moved = Detect(left, right);

        Assert.All(moved, m => Assert.Equal(3, m.Length));
        // 右側は 1 箇所しか無いので、割り当ては 1 つまで。
        Assert.True(moved.Count <= 1, $"{moved.Count} 件になった");
    }

    /// <summary>ただ追加されただけの行を移動と言わないこと。</summary>
    [Fact]
    public void PureAdditionsAreNotMoves()
    {
        Assert.Empty(Detect(["a", "b"], ["a", "b", "new1", "new2", "new3"]));
    }

    [Fact]
    public void PureDeletionsAreNotMoves()
    {
        Assert.Empty(Detect(["a", "b", "old1", "old2", "old3"], ["a", "b"]));
    }
}
