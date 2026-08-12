using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 差分の反映。
///
/// 「見るだけ」から「直せる」へ移る部分なので、間違えるとファイルを壊す。片側が空の塊
/// （純粋な追加・削除）で挿入位置を取り違えるのが最も起きやすい壊れ方なので、
/// そこを厚く固定する。
/// </summary>
public sealed class MergeTests
{
    private static DecodedText Text(params string[] lines)
        => TextDecoder.Decode(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines)));

    private static (Comparison Comparison, DecodedText Left, DecodedText Right) Compare(
        string[] left, string[] right)
    {
        var l = Text(left);
        var r = Text(right);
        return (DiffComparer.Compare(l, r, embedder: null), l, r);
    }

    [Fact]
    public void IdenticalFilesHaveNoBlocks()
    {
        var (comparison, _, _) = Compare(["a", "b"], ["a", "b"]);
        Assert.Empty(Merge.Blocks(comparison));
    }

    [Fact]
    public void ConsecutiveDifferencesFormOneBlock()
    {
        var (comparison, _, _) = Compare(
            ["same", "x1", "x2", "same2"],
            ["same", "y1", "y2", "same2"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(2, block.LeftCount);
        Assert.Equal(2, block.RightCount);
        Assert.Equal(1, block.LeftStart);
        Assert.Equal(1, block.RightStart);
    }

    [Fact]
    public void SeparatedDifferencesFormSeparateBlocks()
    {
        var (comparison, _, _) = Compare(
            ["a", "x", "b", "y", "c"],
            ["a", "X", "b", "Y", "c"]);

        Assert.Equal(2, Merge.Blocks(comparison).Count);
    }

    /// <summary>
    /// 右にしか無い行（純粋な追加）。左は 0 行だが、左のどこへ入るかは決まっていないと
    /// 反映できない。
    /// </summary>
    [Fact]
    public void RightOnlyBlockCarriesTheLeftInsertionPoint()
    {
        var (comparison, left, right) = Compare(["a", "b"], ["a", "inserted", "b"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(0, block.LeftCount);
        Assert.Equal(1, block.RightCount);
        Assert.Equal(1, block.LeftStart);

        // 右を左へ写すと、その位置に挿入される。
        Assert.Equal(
            ["a", "inserted", "b"],
            Merge.CopyToLeft(block, left.Lines, right.Lines));

        // 左を右へ写すと、その行が消える。
        Assert.Equal(
            ["a", "b"],
            Merge.CopyToRight(block, left.Lines, right.Lines));
    }

    [Fact]
    public void LeftOnlyBlockCarriesTheRightInsertionPoint()
    {
        var (comparison, left, right) = Compare(["a", "removed", "b"], ["a", "b"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(1, block.LeftCount);
        Assert.Equal(0, block.RightCount);
        Assert.Equal(1, block.RightStart);

        Assert.Equal(
            ["a", "removed", "b"],
            Merge.CopyToRight(block, left.Lines, right.Lines));
    }

    /// <summary>先頭に挿入された場合。挿入位置が 0 になること。</summary>
    [Fact]
    public void InsertionAtTheStartAnchorsAtZero()
    {
        var (comparison, left, right) = Compare(["a"], ["head", "a"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(0, block.LeftStart);
        Assert.Equal(["head", "a"], Merge.CopyToLeft(block, left.Lines, right.Lines));
    }

    /// <summary>末尾に追加された場合。挿入位置が末尾になること。</summary>
    [Fact]
    public void InsertionAtTheEndAnchorsAtTheEnd()
    {
        var (comparison, left, right) = Compare(["a"], ["a", "tail"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(1, block.LeftStart);
        Assert.Equal(["a", "tail"], Merge.CopyToLeft(block, left.Lines, right.Lines));
    }

    /// <summary>行数が違う塊。1 行が 3 行に増えるような変更。</summary>
    [Fact]
    public void BlocksWithDifferentLineCountsCopyWholesale()
    {
        var (comparison, left, right) = Compare(
            ["a", "one", "z"],
            ["a", "1", "2", "3", "z"]);

        var block = Assert.Single(Merge.Blocks(comparison));
        Assert.Equal(["a", "1", "2", "3", "z"], Merge.CopyToLeft(block, left.Lines, right.Lines));
        Assert.Equal(["a", "one", "z"], Merge.CopyToRight(block, left.Lines, right.Lines));
    }

    /// <summary>
    /// 全部の塊を写せば、写した側と完全に同じ並びになること。反映が「一部だけ効く」
    /// 壊れ方を捕まえる。塊は後ろから当てる（前から当てると行番号がずれる）。
    /// </summary>
    [Fact]
    public void CopyingEveryBlockMakesTheSidesEqual()
    {
        string[] leftLines = ["keep", "old1", "old2", "shared", "gone", "tail"];
        string[] rightLines = ["keep", "new1", "shared", "added", "tail"];
        var (comparison, left, right) = Compare(leftLines, rightLines);

        var result = (IReadOnlyList<string>)right.Lines;
        foreach (var block in Merge.Blocks(comparison).AsEnumerable().Reverse())
        {
            result = Merge.Replace(
                result, block.RightStart, block.RightCount,
                left.Lines, block.LeftStart, block.LeftCount);
        }

        Assert.Equal(leftLines, result);
    }

    [Fact]
    public void ReplaceRejectsRangesOutsideTheDocument()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Merge.Replace(["a"], 0, 5, ["b"], 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Merge.Replace(["a"], 0, 1, ["b"], 0, 5));
    }
}
