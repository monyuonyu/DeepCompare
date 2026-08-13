using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// コミットグラフの列の割り当て。
///
/// **一覧に並べるだけでは、どこで分かれてどこで合流したのかが読み取れない。**
/// </summary>
public sealed class GitGraphTests
{
    private static GitCommit C(string hash, params string[] parents)
        => new(hash, hash[..Math.Min(7, hash.Length)], "著者", DateTimeOffset.UnixEpoch, hash, parents);

    [Fact]
    public void 一本道は同じ列に並ぶ()
    {
        var rows = GitGraph.Build([C("c", "b"), C("b", "a"), C("a")]);

        Assert.All(rows, r => Assert.Equal(0, r.Lane));
        Assert.All(rows, r => Assert.Equal(1, r.Width));
    }

    [Fact]
    public void 枝分かれで列が増える()
    {
        // c と d が同じ親 b を持つ（分岐）。
        var rows = GitGraph.Build([C("d", "b"), C("c", "b"), C("b", "a"), C("a")]);

        Assert.Equal(0, rows[0].Lane);
        Assert.Equal(1, rows[1].Lane);   // 別の列へ
        // 合流した後は 1 本に戻る。
        Assert.Equal(1, rows[3].Width);
    }

    [Fact]
    public void マージは印を持つ()
    {
        var rows = GitGraph.Build([C("m", "a", "b"), C("b", "r"), C("a", "r"), C("r")]);

        Assert.True(rows[0].IsMerge);
        Assert.False(rows[1].IsMerge);
    }

    [Fact]
    public void マージから親へ線が二本伸びる()
    {
        var rows = GitGraph.Build([C("m", "a", "b"), C("a", "r"), C("b", "r"), C("r")]);

        Assert.Equal(2, rows[0].Edges.Count);
        // 最初の親は自分の列を引き継ぐ。**主要な枝が横へ動くと追いにくい。**
        Assert.Equal(rows[0].Lane, rows[0].Edges[0].ToLane);
        Assert.NotEqual(rows[0].Lane, rows[0].Edges[1].ToLane);
    }

    [Fact]
    public void 素通りする線を数える()
    {
        // m が a と b を持ち、a の行では b の列が素通りする。
        var rows = GitGraph.Build([C("m", "a", "b"), C("a", "r"), C("b", "r"), C("r")]);

        Assert.Contains(rows[1].Passing, lane => lane == rows[0].Edges[1].ToLane);
    }

    [Fact]
    public void 根で列が終わる()
    {
        var rows = GitGraph.Build([C("a")]);

        Assert.Empty(rows[0].Edges);
        Assert.Empty(rows[0].Passing);
    }

    [Fact]
    public void 列の数に上限を持つ()
    {
        // **20 列も並べば目では追えない。** 上限を超えたら重ねる。
        var commits = new List<GitCommit>();
        for (var i = 0; i < 30; i++)
        {
            commits.Add(C($"h{i}", "root"));
        }
        commits.Add(C("root"));

        var rows = GitGraph.Build(commits, maximumLanes: 5);

        Assert.All(rows, r => Assert.True(r.Lane < 5, $"列 {r.Lane} が上限を超えた"));
    }

    [Fact]
    public void 空でも落ちない()
    {
        Assert.Empty(GitGraph.Build([]));
    }

    [Fact]
    public void 実際の履歴の形で列が増えすぎない()
    {
        // 一本道に時々マージが入る、よくある形。
        var commits = new List<GitCommit>();
        for (var i = 0; i < 20; i++)
        {
            commits.Add(i % 5 == 0 && i > 0
                ? C($"h{i}", $"h{i + 1}", $"f{i}")
                : C($"h{i}", $"h{i + 1}"));
            if (i % 5 == 0 && i > 0)
            {
                commits.Add(C($"f{i}", $"h{i + 1}"));
            }
        }
        commits.Add(C("h20"));

        var rows = GitGraph.Build(commits);

        // 枝が閉じれば列は戻る。**閉じた枝の列を放置すると、幅だけが伸びる。**
        Assert.True(rows[^1].Width <= 2, $"最後の幅が {rows[^1].Width}");
    }
}
