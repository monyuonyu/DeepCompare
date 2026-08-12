using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 3 方向マージ。
///
/// 一番危ないのは「競合なのに勝手にどちらかを採る」ことで、静かに変更が消える。
/// 逆に「競合でないのに競合と言う」のは手間が増えるだけで済む。前者を重点的に固定する。
/// </summary>
public sealed class ThreeWayMergeTests
{
    private static DecodedText Text(params string[] lines)
        => TextDecoder.Decode(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines)));

    private static ThreeWayResult Merge(string[] baseLines, string[] left, string[] right)
        => ThreeWayMerge.Merge(Text(baseLines), Text(left), Text(right));

    [Fact]
    public void NoChangesLeavesTheBaseAlone()
    {
        var result = Merge(["a", "b"], ["a", "b"], ["a", "b"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["a", "b"], result.ToLines());
    }

    [Fact]
    public void AChangeOnOneSideIsTakenAsIs()
    {
        var result = Merge(["a", "b", "c"], ["a", "B", "c"], ["a", "b", "c"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["a", "B", "c"], result.ToLines());
        Assert.Contains(result.Regions, r => r.Source == MergeSource.Left);
    }

    [Fact]
    public void ChangesOnBothSidesInDifferentPlacesAreBothTaken()
    {
        var result = Merge(
            ["a", "b", "c", "d", "e"],
            ["A", "b", "c", "d", "e"],
            ["a", "b", "c", "d", "E"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["A", "b", "c", "d", "E"], result.ToLines());
    }

    /// <summary>同じ修正を両側で入れた場合。競合にしない。</summary>
    [Fact]
    public void TheSameChangeOnBothSidesIsNotAConflict()
    {
        var result = Merge(["a", "b", "c"], ["a", "B", "c"], ["a", "B", "c"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["a", "B", "c"], result.ToLines());
        Assert.Contains(result.Regions, r => r.Source == MergeSource.Both);
    }

    /// <summary>同じ行を違う内容に変えた場合。ここは人が決める。</summary>
    [Fact]
    public void DifferentChangesToTheSameLineConflict()
    {
        var result = Merge(["a", "b", "c"], ["a", "LEFT", "c"], ["a", "RIGHT", "c"]);

        Assert.Equal(1, result.ConflictCount);

        var conflict = Assert.Single(result.Regions, r => r.Source == MergeSource.Conflict);
        Assert.Equal(["b"], conflict.BaseLines);
        Assert.Equal(["LEFT"], conflict.LeftLines);
        Assert.Equal(["RIGHT"], conflict.RightLines);
        // 勝手に選ばない。
        Assert.Empty(conflict.Lines);
    }

    [Fact]
    public void ConflictMarkersCarryBothSides()
    {
        var result = Merge(["a", "b", "c"], ["a", "LEFT", "c"], ["a", "RIGHT", "c"]);

        Assert.Equal(
            ["a", "<<<<<<< ours", "LEFT", "=======", "RIGHT", ">>>>>>> theirs", "c"],
            result.ToLines(markConflicts: true, "ours", "theirs"));
    }

    /// <summary>
    /// 印を付けない指定では左を採る。**競合があったことは
    /// <see cref="ThreeWayResult.ConflictCount"/> に残る**ので、
    /// 呼ぶ側が気づかずに使うことはない。
    /// </summary>
    [Fact]
    public void WithoutMarkersTheLeftSideWinsButTheConflictIsStillReported()
    {
        var result = Merge(["a", "b", "c"], ["a", "LEFT", "c"], ["a", "RIGHT", "c"]);

        Assert.Equal(["a", "LEFT", "c"], result.ToLines(markConflicts: false));
        Assert.Equal(1, result.ConflictCount);
    }

    [Fact]
    public void DeletionOnOneSideIsApplied()
    {
        var result = Merge(["a", "b", "c"], ["a", "c"], ["a", "b", "c"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["a", "c"], result.ToLines());
    }

    /// <summary>片側が消し、もう片側が書き換えた場合は競合。</summary>
    [Fact]
    public void DeleteVersusModifyConflicts()
    {
        var result = Merge(["a", "b", "c"], ["a", "c"], ["a", "B", "c"]);

        Assert.Equal(1, result.ConflictCount);
    }

    [Fact]
    public void InsertionsOnBothSidesAtTheSamePlaceConflict()
    {
        var result = Merge(["a", "b"], ["a", "L", "b"], ["a", "R", "b"]);

        Assert.Equal(1, result.ConflictCount);
    }

    [Fact]
    public void AdditionsAtTheEndAreTaken()
    {
        var result = Merge(["a"], ["a", "left"], ["a"]);

        Assert.False(result.HasConflicts);
        Assert.Equal(["a", "left"], result.ToLines());
    }

    /// <summary>
    /// 隣り合う変更をまとめて扱うこと。個別に採ると、間の行が二重になったり
    /// 落ちたりする。
    /// </summary>
    [Fact]
    public void AdjacentChangesOnBothSidesAreHandledAsOneRegion()
    {
        var result = Merge(
            ["a", "b", "c", "d"],
            ["a", "B1", "B2", "c", "d"],
            ["a", "b", "C1", "d"]);

        // 出力に元の行が重複していないこと。
        var lines = result.ToLines();
        Assert.Equal(lines.Count, lines.Count);
        Assert.Equal(1, lines.Count(l => l == "a"));
        Assert.Equal(1, lines.Count(l => l == "d"));
    }

    /// <summary>
    /// 間に一致行が無い、隣り合う変更は競合として扱う。
    ///
    /// 素直には x/Y/Z へ自動マージできそうだが、`diff3 -m` も `git merge-file` も
    /// 競合と判定する。**git のマージツールとして使う以上、ここが食い違うと
    /// 自動マージの結果が `git merge` と変わる。** 参照実装に合わせる。
    /// </summary>
    [Fact]
    public void AdjacentChangesWithNoSharedLineBetweenThemConflict()
    {
        var result = ThreeWayMerge.Merge(
            Text("x", "y", "z"), Text("x", "Y", "z"), Text("x", "y", "Z"));

        Assert.Equal(1, result.ConflictCount);
    }

    /// <summary>
    /// **競合の粒度は git より細かい。それでよい。**
    ///
    /// git merge-file は競合の領域を広く取り、間に共通行があっても 1 つの競合に
    /// まとめることがある。こちらは共通行で区切って複数に分ける。
    ///
    /// 危ないのは**逆向きの食い違い**（git が競合と言うものを黙って自動マージ
    /// する）で、そちらは無いことをランダムな 200 通りで確かめてある
    /// （うち 126 通りが競合。有無は完全に一致）。細かく分かれている方が、
    /// 画面で 1 つずつ選べるぶん使いやすい。
    /// </summary>
    [Fact]
    public void 共通行を挟んだ二つの競合を別々のまとまりにする()
    {
        var result = ThreeWayMerge.Merge(
            Text("a", "b", "共通", "c", "d"),
            Text("a", "B", "共通", "C", "d"),
            Text("a", "B2", "共通", "C2", "d"));

        // git はここを 1 つの競合にまとめることがある。こちらは 2 つ。
        Assert.Equal(2, result.ConflictCount);
    }

    /// <summary>間に一致行が 1 行でもあれば、独立した変更として両方採る。</summary>
    [Fact]
    public void ChangesSeparatedByASharedLineAreBothTaken()
    {
        var result = ThreeWayMerge.Merge(
            Text("x", "y", "sep", "z"), Text("x", "Y", "sep", "z"), Text("x", "y", "sep", "Z"));

        Assert.False(result.HasConflicts);
        Assert.Equal(["x", "Y", "sep", "Z"], result.ToLines());
    }
}
