using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>Rust 版 align.rs の試験を移したもの。</summary>
public class AlignerTests
{
    private static List<string> Lines(string text)
        => text.Split('\n', StringSplitOptions.None).ToList();

    private static List<Block> ChangedBlocks(IReadOnlyList<string> a, IReadOnlyList<string> b)
        => Aligner.Split(a, b).OfType<Segment.Changed>().Select(s => s.Block).ToList();

    [Fact]
    public void IdenticalFilesProduceASingleIdenticalSegment()
    {
        var a = Lines("one\ntwo\nthree");
        var segments = Aligner.Split(a, a);
        var only = Assert.IsType<Segment.Identical>(Assert.Single(segments));
        Assert.Equal(0, only.LeftStart);
        Assert.Equal(0, only.RightStart);
        Assert.Equal(3, only.Length);
    }

    /// <summary>
    /// 二段構えの肝。1000 行のうち 1 行だけ違うなら、モデルに渡すのも DP にかけるのも
    /// その 1 行分だけで済まなければ意味がない。
    /// </summary>
    [Fact]
    public void OnlyTheChangedRegionBecomesABlock()
    {
        var blocks = ChangedBlocks(Lines("a\nb\nc\nd\ne"), Lines("a\nb\nX\nd\ne"));
        var block = Assert.Single(blocks);
        Assert.Equal(2, block.LeftStart);
        Assert.Equal(1, block.LeftLength);
        Assert.Equal(2, block.RightStart);
        Assert.Equal(1, block.RightLength);
    }

    /// <summary>分けて流すと、消えた行と増えた行を対応付ける機会が失われる。</summary>
    [Fact]
    public void AdjacentDeleteAndInsertMergeIntoOneBlock()
    {
        var blocks = ChangedBlocks(Lines("keep\nold1\nold2\ntail"), Lines("keep\nnew1\ntail"));
        var block = Assert.Single(blocks);
        Assert.Equal(1, block.LeftStart);
        Assert.Equal(2, block.LeftLength);
        Assert.Equal(1, block.RightStart);
        Assert.Equal(1, block.RightLength);
    }

    [Fact]
    public void SegmentsCoverEveryLineExactlyOnce()
    {
        var a = Lines("a\nb\nc\nd\ne\nf");
        var b = Lines("a\nX\nc\nY\nZ\nf");
        var leftSeen = new int[a.Count];
        var rightSeen = new int[b.Count];
        foreach (var segment in Aligner.Split(a, b))
        {
            switch (segment)
            {
                case Segment.Identical id:
                    for (var k = 0; k < id.Length; k++)
                    {
                        leftSeen[id.LeftStart + k]++;
                        rightSeen[id.RightStart + k]++;
                    }
                    break;
                case Segment.Changed ch:
                    for (var i = 0; i < ch.Block.LeftLength; i++)
                    {
                        leftSeen[ch.Block.LeftStart + i]++;
                    }
                    for (var j = 0; j < ch.Block.RightLength; j++)
                    {
                        rightSeen[ch.Block.RightStart + j]++;
                    }
                    break;
            }
        }
        Assert.All(leftSeen, c => Assert.Equal(1, c));
        Assert.All(rightSeen, c => Assert.Equal(1, c));
    }

    [Fact]
    public void EmptySideYieldsOnlyGaps()
    {
        var right = Aligner.NeedlemanWunsch(0, 3, Aligner.DefaultPairThreshold, (_, _) => 0f);
        Assert.Equal([Pair.RightOnly(0), Pair.RightOnly(1), Pair.RightOnly(2)], right);

        var left = Aligner.NeedlemanWunsch(2, 0, Aligner.DefaultPairThreshold, (_, _) => 0f);
        Assert.Equal([Pair.LeftOnly(0), Pair.LeftOnly(1)], left);
    }

    [Fact]
    public void HighSimilarityPairsAreMatchedDiagonally()
    {
        var pairs = Aligner.NeedlemanWunsch(3, 3, Aligner.DefaultPairThreshold, (i, j) => i == j ? 1f : 0f);
        Assert.Equal(3, pairs.Count);
        for (var k = 0; k < 3; k++)
        {
            Assert.Equal(k, pairs[k].Left);
            Assert.Equal(k, pairs[k].Right);
        }
    }

    /// <summary>何も似ていないなら、無理に対応させず左右を空きで並べる方が読める。</summary>
    [Fact]
    public void DissimilarLinesBecomeGapsRatherThanForcedPairs()
    {
        var pairs = Aligner.NeedlemanWunsch(2, 2, Aligner.DefaultPairThreshold, (_, _) => 0f);
        Assert.All(pairs, p => Assert.True(p.Left is null || p.Right is null));
        Assert.Equal(2, pairs.Count(p => p.Left is not null));
        Assert.Equal(2, pairs.Count(p => p.Right is not null));
    }

    [Fact]
    public void AnInsertedLineShiftsTheRestInsteadOfMisaligningIt()
    {
        int[] mapping = [0, 2, 3];
        var pairs = Aligner.NeedlemanWunsch(3, 4, Aligner.DefaultPairThreshold,
            (i, j) => mapping[i] == j ? 1f : 0f);
        var matched = pairs.Where(p => p.Left is not null && p.Right is not null)
            .Select(p => (p.Left!.Value, p.Right!.Value)).ToList();
        Assert.Equal([(0, 0), (1, 2), (2, 3)], matched);
        Assert.Equal(1, pairs.Count(p => p.Left is null));
    }

    /// <summary>
    /// 左 [A, B]、右 [B', C] で、対応するのは B と B' だけという形。
    /// 旧実装の罰則の置き方では、長さが揃っている限り必ず対にされるため
    /// A↔B' と B↔C という誤った並びになっていた。
    /// </summary>
    [Fact]
    public void ASingleCorrespondingLineIsNotDraggedOutOfPlace()
    {
        var pairs = Aligner.NeedlemanWunsch(2, 2, Aligner.DefaultPairThreshold,
            (i, j) => i == 1 && j == 0 ? 0.95f : 0.1f);
        var matched = pairs.Where(p => p.Left is not null && p.Right is not null)
            .Select(p => (p.Left!.Value, p.Right!.Value)).ToList();
        Assert.Equal([(1, 0)], matched);
        Assert.Equal(1, pairs.Count(p => p.Right is null));
        Assert.Equal(1, pairs.Count(p => p.Left is null));
    }

    /// <summary>
    /// 対にするほど似ていない 1 行同士は、左右それぞれの空きとして並ぶ。
    /// その並び順は削除が先。逆だと「増えてから消えた」ように読めてしまう。
    /// </summary>
    [Fact]
    public void ARemovedLineIsListedBeforeTheAddedOne()
    {
        var pairs = Aligner.NeedlemanWunsch(1, 1, Aligner.DefaultPairThreshold, (_, _) => 0f);
        Assert.Equal(2, pairs.Count);
        Assert.Equal(Pair.LeftOnly(0), pairs[0]);
        Assert.Equal(Pair.RightOnly(0), pairs[1]);
    }

    [Fact]
    public void PairsReportTheRawSimilarityNotTheThresholdedOne()
    {
        var pairs = Aligner.NeedlemanWunsch(1, 1, Aligner.DefaultPairThreshold, (_, _) => 0.8f);
        Assert.Equal(0.8f, pairs[0].Score);
    }

    /// <summary>
    /// 行が消えたり重複したりしないことの担保。旧実装は経路が途切れると
    /// 残りの行が黙って消えていた。
    /// </summary>
    [Fact]
    public void EveryLineAppearsExactlyOnceAndInOrder()
    {
        var pairs = Aligner.NeedlemanWunsch(5, 4, Aligner.DefaultPairThreshold,
            (i, j) => (i + j) % 3 * 0.4f);
        Assert.Equal([0, 1, 2, 3, 4], pairs.Where(p => p.Left is not null).Select(p => p.Left!.Value));
        Assert.Equal([0, 1, 2, 3], pairs.Where(p => p.Right is not null).Select(p => p.Right!.Value));
    }

    [Fact]
    public void OversizedBlockFallsBackToPlainListing()
    {
        var pairs = Aligner.WithoutScoring(new Block(0, 2, 0, 3));
        Assert.Equal(2, pairs.Count(p => p.Left is not null));
        Assert.Equal(3, pairs.Count(p => p.Right is not null));
    }
}
