using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public class FolderComparerTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "dc-folder-" + Guid.NewGuid().ToString("N"));

    private string Left => Path.Combine(_root, "left");
    private string Right => Path.Combine(_root, "right");

    public FolderComparerTests()
    {
        Directory.CreateDirectory(Left);
        Directory.CreateDirectory(Right);
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch
        {
            // 後片付けに失敗しても試験の結果には関係ない。
        }
    }

    /// <summary>side は "left" か "right"。試験用の一時領域の下に書く。</summary>
    private void Write(string side, string relative, string content)
    {
        var root = side switch
        {
            "left" => Left,
            "right" => Right,
            _ => throw new ArgumentException($"left か right のみ: {side}", nameof(side)),
        };
        var path = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }

    private FolderComparison Run(FolderCompareOptions? options = null)
        => FolderComparer.Compare(Left, Right, options);

    [Fact]
    public void IdenticalFilesAreReportedAsIdentical()
    {
        Write("left", "a.txt", "same");
        Write("right", "a.txt", "same");
        var entry = Assert.Single(Run().Entries);
        Assert.Equal(EntryStatus.Identical, entry.Status);
        Assert.False(entry.IsDirectory);
    }

    [Fact]
    public void SameSizeButDifferentContentIsDetected()
    {
        // サイズだけで判断していると取り違える組み合わせ。
        Write("left", "a.txt", "abcd");
        Write("right", "a.txt", "abce");
        Assert.Equal(EntryStatus.Different, Assert.Single(Run().Entries).Status);
    }

    [Fact]
    public void FilesOnOneSideOnlyAreMarked()
    {
        Write("left", "only-left.txt", "x");
        Write("right", "only-right.txt", "y");
        var entries = Run().Entries.ToDictionary(e => e.Name);
        Assert.Equal(EntryStatus.LeftOnly, entries["only-left.txt"].Status);
        Assert.Equal(EntryStatus.RightOnly, entries["only-right.txt"].Status);
    }

    [Fact]
    public void SubdirectoriesAreWalkedAndReported()
    {
        Write("left", "sub/deep/a.txt", "same");
        Write("right", "sub/deep/a.txt", "same");
        var entries = Run().Entries;

        Assert.Contains(entries, e => e is { Name: "sub", IsDirectory: true, Depth: 0 });
        Assert.Contains(entries, e => e is { Name: "deep", IsDirectory: true, Depth: 1 });
        var file = Assert.Single(entries, e => e.Name == "a.txt");
        Assert.Equal("sub/deep/a.txt", file.RelativePath);
        Assert.Equal(2, file.Depth);
    }

    [Fact]
    public void RecursionCanBeTurnedOff()
    {
        Write("left", "sub/a.txt", "x");
        Write("right", "sub/a.txt", "x");
        var entries = Run(new FolderCompareOptions { Recursive = false }).Entries;
        Assert.Single(entries);
        Assert.True(entries[0].IsDirectory);
    }

    /// <summary>
    /// 生成物や版管理の内部まで比べると、数だけ膨れて本当に見たい差分が埋もれる。
    /// </summary>
    [Fact]
    public void ExcludedDirectoriesAreSkipped()
    {
        Write("left", ".git/config", "x");
        Write("left", "node_modules/pkg/index.js", "x");
        Write("left", "src/main.cs", "x");
        Write("right", "src/main.cs", "x");

        var names = Run().Entries.Select(e => e.Name).ToList();
        Assert.DoesNotContain(".git", names);
        Assert.DoesNotContain("node_modules", names);
        Assert.Contains("src", names);
    }

    [Fact]
    public void EmptyFilesOnBothSidesAreIdentical()
    {
        Write("left", "empty.txt", string.Empty);
        Write("right", "empty.txt", string.Empty);
        Assert.Equal(EntryStatus.Identical, Assert.Single(Run().Entries).Status);
    }

    /// <summary>読み込みの塊をまたぐ位置の違いを取りこぼさないこと。</summary>
    [Fact]
    public void DifferenceBeyondTheFirstChunkIsDetected()
    {
        var baseText = new string('a', 200_000);
        Write("left", "big.txt", baseText + "X" + new string('b', 1000));
        Write("right", "big.txt", baseText + "Y" + new string('b', 1000));
        Assert.Equal(EntryStatus.Different, Assert.Single(Run().Entries).Status);
    }

    [Fact]
    public void StatsCountEachCategory()
    {
        Write("left", "same.txt", "x");
        Write("right", "same.txt", "x");
        Write("left", "diff.txt", "a");
        Write("right", "diff.txt", "b");
        Write("left", "gone.txt", "x");
        Write("right", "new.txt", "x");

        var stats = Run().Stats;
        Assert.Equal(1, stats.Identical);
        Assert.Equal(1, stats.Different);
        Assert.Equal(1, stats.LeftOnly);
        Assert.Equal(1, stats.RightOnly);
    }

    [Fact]
    public void IdenticalEntriesCanBeOmitted()
    {
        Write("left", "same.txt", "x");
        Write("right", "same.txt", "x");
        Write("left", "diff.txt", "a");
        Write("right", "diff.txt", "b");

        var entries = Run(new FolderCompareOptions { IncludeIdentical = false }).Entries;
        Assert.Equal("diff.txt", Assert.Single(entries).Name);
    }

    [Fact]
    public void SizesAndTimestampsAreReported()
    {
        Write("left", "a.txt", "12345");
        Write("right", "a.txt", "123");
        var entry = Assert.Single(Run().Entries);
        Assert.Equal(5, entry.LeftSize);
        Assert.Equal(3, entry.RightSize);
        Assert.NotNull(entry.LeftModified);
        Assert.NotNull(entry.RightModified);
    }

    [Fact]
    public void EntriesAreOrderedByNameSoTheTwoSidesLineUp()
    {
        foreach (var name in new[] { "c.txt", "a.txt", "b.txt" })
        {
            Write("left", name, "x");
            Write("right", name, "x");
        }
        var names = Run().Entries.Select(e => e.Name).ToList();
        Assert.Equal(["a.txt", "b.txt", "c.txt"], names);
    }

    [Fact]
    public void CancellationStopsTheWalk()
    {
        for (var i = 0; i < 50; i++)
        {
            Write("left", $"f{i}.txt", "x");
            Write("right", $"f{i}.txt", "x");
        }
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(
            () => FolderComparer.Compare(Left, Right, null, null, cts.Token));
    }
}
