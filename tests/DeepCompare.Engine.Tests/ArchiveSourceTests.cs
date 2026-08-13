using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 書庫の展開。展開したものを確実に消すこと、フォルダーと同じように比べられることを見る。
/// </summary>
public sealed class ArchiveSourceTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "dc-archive-test-" + Guid.NewGuid().ToString("N"));

    public ArchiveSourceTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch
        {
            // 後片付けの失敗は結果に関係ない。
        }
    }

    private string MakeFolder(string name, params (string Path, string Content)[] files)
    {
        var folder = Path.Combine(_root, name);
        foreach (var (relative, content) in files)
        {
            var path = Path.Combine(folder, relative.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, content);
        }
        Directory.CreateDirectory(folder);
        return folder;
    }

    private string MakeZip(string name, params (string Path, string Content)[] files)
    {
        var folder = MakeFolder(name + "-src", files);
        var zip = Path.Combine(_root, name + ".zip");
        System.IO.Compression.ZipFile.CreateFromDirectory(folder, zip);
        return zip;
    }

    [Theory]
    [InlineData("a.zip", true)]
    [InlineData("a.tar", true)]
    [InlineData("a.tar.gz", true)]
    [InlineData("a.tgz", true)]
    [InlineData("A.ZIP", true)]
    [InlineData("a.txt", false)]
    [InlineData("a.7z", false)]
    public void ArchivesAreRecognisedByExtension(string name, bool expected)
    {
        Assert.Equal(expected, ArchiveSource.LooksLikeArchive(name));
    }

    [Fact]
    public void AFolderIsUsedAsIsAndNotDeleted()
    {
        var folder = MakeFolder("plain", ("a.txt", "x"));

        using (var source = ArchiveSource.Open(folder))
        {
            Assert.False(source.IsExtracted);
            Assert.Equal(folder, source.Path);
        }

        // 実体のフォルダーを消してしまわないこと。
        Assert.True(Directory.Exists(folder));
    }

    [Fact]
    public void AZipIsExtractedAndCleanedUp()
    {
        var zip = MakeZip("data", ("a.txt", "hello"), ("sub/b.txt", "world"));
        string extracted;

        using (var source = ArchiveSource.Open(zip))
        {
            Assert.True(source.IsExtracted);
            extracted = source.Path;
            Assert.Equal("hello", File.ReadAllText(Path.Combine(extracted, "a.txt")));
            Assert.Equal("world", File.ReadAllText(Path.Combine(extracted, "sub", "b.txt")));
        }

        Assert.False(Directory.Exists(extracted));
    }

    /// <summary>書庫とフォルダーを混ぜて比べられること。乗り換えの実用場面はこれ。</summary>
    [Fact]
    public void AnArchiveCanBeComparedAgainstAFolder()
    {
        var zip = MakeZip("released", ("a.txt", "same"), ("b.txt", "old"));
        var folder = MakeFolder("working", ("a.txt", "same"), ("b.txt", "new"));

        using var leftSource = ArchiveSource.Open(zip);
        using var rightSource = ArchiveSource.Open(folder);
        var result = FolderComparer.Compare(leftSource.Path, rightSource.Path);

        Assert.Equal(1, result.Stats.Identical);
        Assert.Equal(1, result.Stats.Different);
    }

    [Fact]
    public void MissingPathsFailClearly()
    {
        Assert.Throws<DirectoryNotFoundException>(
            () => ArchiveSource.Open(Path.Combine(_root, "does-not-exist")));
    }

    [Fact]
    public void UnsupportedFormatsFailClearly()
    {
        var path = Path.Combine(_root, "a.7z");
        File.WriteAllText(path, "not really an archive");

        var error = Assert.Throws<NotSupportedException>(() => ArchiveSource.Open(path));
        Assert.Contains("zip", error.Message);
    }

    /// <summary>壊れた書庫でも一時領域を残さないこと。</summary>
    [Fact]
    public void ABrokenArchiveLeavesNothingBehind()
    {
        var path = Path.Combine(_root, "broken.zip");
        File.WriteAllText(path, "これは zip ではない");

        var before = Directory.GetDirectories(Path.GetTempPath(), "dc-archive-*").Length;
        Assert.ThrowsAny<Exception>(() => ArchiveSource.Open(path));
        var after = Directory.GetDirectories(Path.GetTempPath(), "dc-archive-*").Length;

        Assert.Equal(before, after);
    }
}
