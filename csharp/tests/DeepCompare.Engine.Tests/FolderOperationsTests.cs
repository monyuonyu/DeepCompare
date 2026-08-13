using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 戻せない操作なので、条件の判定をここで確かめる。
///
/// **一時領域の中だけで動かす。** 試験が実際にファイルを作って消すので、
/// 場所を間違えると本物を壊す。
/// </summary>
public class FolderOperationsTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "dc-folderops-" + Guid.NewGuid().ToString("N")[..8]);

    public FolderOperationsTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // 消せなくても試験の結果は変わらない。
        }
        GC.SuppressFinalize(this);
    }

    private string File_(string name, string content = "x")
    {
        var path = Path.Combine(_root, name);
        System.IO.File.WriteAllText(path, content);
        return path;
    }

    private string Dir_(string name)
    {
        var path = Path.Combine(_root, name);
        Directory.CreateDirectory(path);
        return path;
    }

    // ---- 名前の判定 ----

    [Theory]
    [InlineData("普通の名前.txt")]
    [InlineData("a")]
    [InlineData("日本語のファイル")]
    public void 使える名前を通す(string name)
        => Assert.Null(FolderOperations.WhyInvalidName(name));

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(".")]
    [InlineData("..")]
    public void 使えない名前を断る(string name)
        => Assert.NotNull(FolderOperations.WhyInvalidName(name));

    [Fact]
    public void 区切り文字を含む名前を断る()
    {
        // **「名前を変える」つもりで別の場所へ移すのを防ぐ。**
        Assert.NotNull(FolderOperations.WhyInvalidName("a/b"));
        Assert.NotNull(FolderOperations.WhyInvalidName("a\\b"));
    }

    [Theory]
    [InlineData("a:b")]
    [InlineData("a*b")]
    [InlineData("a?b")]
    [InlineData("a\"b")]
    [InlineData("a<b")]
    [InlineData("a|b")]
    public void Windows_で開けなくなる文字を断る(string name)
    {
        // **Linux でも断る。** 作れてしまうと、Windows へ持って行ったときに
        // 開けないファイルができる。往復する前提の道具なので、ここで止める。
        Assert.NotNull(FolderOperations.WhyInvalidName(name));
    }

    // ---- 名前を変える ----

    [Fact]
    public void ファイルの名前を変える()
    {
        var path = File_("old.txt", "中身");
        var result = FolderOperations.Rename(path, "new.txt", isDirectory: false);

        Assert.True(result.Ok);
        Assert.False(File.Exists(path));
        Assert.Equal("中身", File.ReadAllText(Path.Combine(_root, "new.txt")));
    }

    [Fact]
    public void フォルダーの名前を変える()
    {
        var path = Dir_("old");
        File.WriteAllText(Path.Combine(path, "中身.txt"), "x");

        var result = FolderOperations.Rename(path, "new", isDirectory: true);

        Assert.True(result.Ok);
        Assert.False(Directory.Exists(path));
        Assert.True(File.Exists(Path.Combine(_root, "new", "中身.txt")));
    }

    [Fact]
    public void 同じ名前への変更は何もしない()
    {
        // **「変えました」と出さない。** 変わったと思われる。
        var path = File_("same.txt");
        var result = FolderOperations.Rename(path, "same.txt", isDirectory: false);

        Assert.False(result.Ok);
        Assert.True(File.Exists(path));
    }

    [Fact]
    public void 既にある名前へは変えない()
    {
        var path = File_("a.txt", "こちら");
        File_("b.txt", "あちら");

        var result = FolderOperations.Rename(path, "b.txt", isDirectory: false);

        Assert.False(result.Ok);
        // **上書きしていない。**
        Assert.Equal("あちら", File.ReadAllText(Path.Combine(_root, "b.txt")));
        Assert.True(File.Exists(path));
    }

    [Fact]
    public void 区切り文字を含む名前へは変えない()
    {
        var path = File_("a.txt");
        var result = FolderOperations.Rename(path, "sub/a.txt", isDirectory: false);

        Assert.False(result.Ok);
        Assert.True(File.Exists(path));
    }

    // ---- 移す ----

    [Fact]
    public void ファイルを移すと元から消える()
    {
        var source = File_("move.txt", "中身");
        var destination = Path.Combine(Dir_("dest"), "move.txt");

        var result = FolderOperations.Move(source, destination, isDirectory: false);

        Assert.True(result.Ok);
        Assert.False(File.Exists(source));
        Assert.Equal("中身", File.ReadAllText(destination));
    }

    [Fact]
    public void 移す先が無ければ作る()
    {
        var source = File_("move.txt", "中身");
        // **まだ無い階層。** 比較の相手側にそのフォルダーが無いことは普通にある。
        var destination = Path.Combine(_root, "deep", "deeper", "move.txt");

        var result = FolderOperations.Move(source, destination, isDirectory: false);

        Assert.True(result.Ok);
        Assert.Equal("中身", File.ReadAllText(destination));
    }

    [Fact]
    public void 移す先にあるものは上書きする()
    {
        var source = File_("a.txt", "新しい");
        var destination = Path.Combine(Dir_("dest"), "a.txt");
        File.WriteAllText(destination, "古い");

        var result = FolderOperations.Move(source, destination, isDirectory: false);

        Assert.True(result.Ok);
        Assert.Equal("新しい", File.ReadAllText(destination));
    }

    [Fact]
    public void 無いものは移せない()
    {
        var result = FolderOperations.Move(
            Path.Combine(_root, "無い.txt"),
            Path.Combine(_root, "先.txt"), isDirectory: false);
        Assert.False(result.Ok);
    }

    [Fact]
    public void 自分の中へは移せない()
    {
        // **無限に潜るか、途中で失敗して中身が壊れる。**
        var source = Dir_("dir");
        File.WriteAllText(Path.Combine(source, "中身.txt"), "x");
        var destination = Path.Combine(source, "sub");

        var result = FolderOperations.Move(source, destination, isDirectory: true);

        Assert.False(result.Ok);
        // **元が無事。**
        Assert.True(File.Exists(Path.Combine(source, "中身.txt")));
    }

    [Fact]
    public void 名前が似ているだけの隣は自分の中ではない()
    {
        // `/a/bc` を `/a/b` の中と判定してはいけない。
        var a = Dir_("b");
        var b = Path.Combine(_root, "bc");

        Assert.False(FolderOperations.IsInside(b, a));
        Assert.True(FolderOperations.IsInside(Path.Combine(a, "sub"), a));
    }

    // ---- フォルダーを作る ----

    [Fact]
    public void フォルダーを作る()
    {
        var result = FolderOperations.NewFolder(_root, "新しい");
        Assert.True(result.Ok);
        Assert.True(Directory.Exists(Path.Combine(_root, "新しい")));
    }

    [Fact]
    public void 既にあるフォルダーは作らない()
    {
        Dir_("ある");
        var result = FolderOperations.NewFolder(_root, "ある");
        Assert.False(result.Ok);
    }

    [Fact]
    public void 同じ名前のファイルがあれば作らない()
    {
        File_("かぶる");
        var result = FolderOperations.NewFolder(_root, "かぶる");
        Assert.False(result.Ok);
    }

    [Fact]
    public void 場所が無ければ作らない()
    {
        var result = FolderOperations.NewFolder(
            Path.Combine(_root, "存在しない"), "新しい");
        Assert.False(result.Ok);
    }

    [Fact]
    public void 使えない名前のフォルダーは作らない()
    {
        var result = FolderOperations.NewFolder(_root, "a/b");
        Assert.False(result.Ok);
        Assert.False(Directory.Exists(Path.Combine(_root, "a")));
    }
}

/// <summary>
/// 差異だけを出すとき、中身が残らなかったディレクトリの行を落とす。
///
/// **一致だけのディレクトリの見出しが並ぶと、差異を探している目には
/// 「ここに何かある」と読めてしまう。**
/// </summary>
public class DropEmptyDirectoriesTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "dc-emptydir-" + Guid.NewGuid().ToString("N")[..8]);

    public DropEmptyDirectoriesTests()
    {
        Directory.CreateDirectory(_root);
        foreach (var side in new[] { "L", "R" })
        {
            Directory.CreateDirectory(Path.Combine(_root, side, "same"));
            Directory.CreateDirectory(Path.Combine(_root, side, "diff"));
            Directory.CreateDirectory(Path.Combine(_root, side, "deep", "inner"));
            File.WriteAllText(Path.Combine(_root, side, "same", "a.txt"), "同じ");
            File.WriteAllText(Path.Combine(_root, side, "deep", "inner", "c.txt"), "同じ");
        }
        // ここだけ中身を変える。
        File.WriteAllText(Path.Combine(_root, "L", "diff", "b.txt"), "こちら");
        File.WriteAllText(Path.Combine(_root, "R", "diff", "b.txt"), "あちら");
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); }
        catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    private FolderComparison Compare(bool includeIdentical) => FolderComparer.Compare(
        Path.Combine(_root, "L"), Path.Combine(_root, "R"),
        new FolderCompareOptions { IncludeIdentical = includeIdentical });

    [Fact]
    public void 差異だけのときは中身の無いディレクトリを出さない()
    {
        var names = Compare(includeIdentical: false).Entries
            .Select(e => e.RelativePath).ToList();

        Assert.Contains("diff", names);
        Assert.Contains("diff/b.txt", names);
        // **中身が全部一致だったディレクトリは消える。**
        Assert.DoesNotContain("same", names);
        Assert.DoesNotContain("deep", names);
        Assert.DoesNotContain("deep/inner", names);
    }

    [Fact]
    public void 全部出すときはディレクトリを残す()
    {
        var names = Compare(includeIdentical: true).Entries
            .Select(e => e.RelativePath).ToList();

        Assert.Contains("same", names);
        Assert.Contains("deep", names);
        Assert.Contains("deep/inner", names);
    }

    [Fact]
    public void 落としたディレクトリは集計にも入れない()
    {
        // **一覧と数を食い違わせない。** 「フォルダー 4」と出ているのに
        // 一覧に 1 つしかないと、残りをどこかで見落としたのかと疑わせる。
        // ファイルの数（一致 2 / 違う 1）は走査した実数のまま。
        var stats = Compare(includeIdentical: false).Stats;
        Assert.Equal(2, stats.Identical);
        Assert.Equal(1, stats.Different);
        Assert.Equal(1, stats.Directories);
    }
}
