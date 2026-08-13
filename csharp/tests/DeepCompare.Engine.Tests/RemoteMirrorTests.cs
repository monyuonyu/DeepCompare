using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>覚え書きの上に作った、偽の置き場。**本物のサーバーを立てずに経路を確かめる。**</summary>
internal sealed class FakeFileSource : IFileSource
{
    private readonly Dictionary<string, byte[]> _files = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTimeOffset> _times = new(StringComparer.Ordinal);

    public string Display { get; init; } = "fake://置き場";
    public bool CanWrite { get; init; } = true;

    /// <summary>読んだ回数。**余計に取りに行っていないか**を見る。</summary>
    public int Reads { get; private set; }

    public FakeFileSource Add(string path, string content, DateTimeOffset? when = null)
    {
        _files[path] = new UTF8Encoding(false).GetBytes(content);
        if (when is { } value)
        {
            _times[path] = value;
        }
        return this;
    }

    public FakeFileSource AddBig(string path, int size)
    {
        _files[path] = new byte[size];
        return this;
    }

    public IReadOnlyList<RemoteEntry> List(
        string relativePath, CancellationToken cancellationToken = default)
    {
        var prefix = relativePath.Length == 0 ? string.Empty : relativePath.TrimEnd('/') + "/";
        var result = new List<RemoteEntry>();
        var directories = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (path, content) in _files)
        {
            if (!path.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            var rest = path[prefix.Length..];
            var slash = rest.IndexOf('/');
            if (slash < 0)
            {
                result.Add(new RemoteEntry(path, false, content.Length,
                    _times.TryGetValue(path, out var when) ? when : null));
            }
            else if (directories.Add(rest[..slash]))
            {
                result.Add(new RemoteEntry(prefix + rest[..slash], true, 0, null));
            }
        }
        return result;
    }

    public byte[] Read(string relativePath, CancellationToken cancellationToken = default)
    {
        Reads++;
        return _files[relativePath];
    }

    public void Write(string relativePath, byte[] content, CancellationToken cancellationToken = default)
        => _files[relativePath] = content;

    public void Delete(string relativePath, CancellationToken cancellationToken = default)
        => _files.Remove(relativePath);

    public bool Exists(string relativePath, CancellationToken cancellationToken = default)
        => _files.ContainsKey(relativePath);

    public void Dispose()
    {
    }
}

public sealed class RemoteMirrorTests
{
    [Fact]
    public void 取ってきて普通のフォルダーとして扱える()
    {
        var source = new FakeFileSource()
            .Add("a.txt", "あ")
            .Add("下/b.txt", "い")
            .Add("下/さらに/c.txt", "う");

        using var mirror = RemoteMirror.Fetch(source);

        Assert.Equal(3, mirror.Result.Files);
        Assert.True(Directory.Exists(mirror.Path));
        Assert.Equal("あ", File.ReadAllText(Path.Combine(mirror.Path, "a.txt")));
        Assert.Equal("う", File.ReadAllText(
            Path.Combine(mirror.Path, "下", "さらに", "c.txt")));
    }

    [Fact]
    public void 時刻を合わせる()
    {
        // **合わせないと、時刻で比べる経路が「全部違う」としか言わなくなる。**
        var when = new DateTimeOffset(2026, 8, 13, 5, 0, 0, TimeSpan.Zero);
        var source = new FakeFileSource().Add("a.txt", "あ", when);

        using var mirror = RemoteMirror.Fetch(source);

        Assert.Equal(when.UtcDateTime,
            File.GetLastWriteTimeUtc(Path.Combine(mirror.Path, "a.txt")));
    }

    [Fact]
    public void 大きすぎるものは取ってこない()
    {
        // リモートに 2GB の記録が 1 つあるだけで、比較を始める前に
        // 回線と時間を使い切る。
        var source = new FakeFileSource()
            .Add("小さい.txt", "あ")
            .AddBig("大きい.bin", 10 * 1024);

        using var mirror = RemoteMirror.Fetch(source,
            new MirrorOptions { MaximumFileSize = 1024 });

        Assert.Equal(1, mirror.Result.Files);
        Assert.True(mirror.Result.HasSkipped);
        Assert.Contains(mirror.Result.Skipped, s => s.Contains("大きい.bin"));
        // **取らなかったことを黙らない。**
        Assert.Contains("取っていません", mirror.Describe());
    }

    [Fact]
    public void 全体の上限で止まる()
    {
        // 相手の大きさは事前に分からない。ここで止められないと、途中で
        // disk full になって何も残らない。
        var source = new FakeFileSource()
            .AddBig("a.bin", 600)
            .AddBig("b.bin", 600)
            .AddBig("c.bin", 600);

        using var mirror = RemoteMirror.Fetch(source,
            new MirrorOptions { MaximumTotalSize = 1000 });

        Assert.Equal(1, mirror.Result.Files);
        Assert.Equal(2, mirror.Result.Skipped.Count);
    }

    [Fact]
    public void 絞り込みが効く()
    {
        var source = new FakeFileSource()
            .Add("a.txt", "あ")
            .Add("b.log", "い");

        using var mirror = RemoteMirror.Fetch(source,
            new MirrorOptions { Filter = new NameFilter(Exclude: ["*.log"]) });

        Assert.Equal(1, mirror.Result.Files);
        Assert.False(File.Exists(Path.Combine(mirror.Path, "b.log")));
    }

    [Fact]
    public void 深すぎる木で止まる()
    {
        var source = new FakeFileSource();
        source.Add(string.Join('/', Enumerable.Repeat("下", 40)) + "/深.txt", "x");

        using var mirror = RemoteMirror.Fetch(source, new MirrorOptions { MaximumDepth = 5 });

        Assert.Equal(0, mirror.Result.Files);
        Assert.Contains(mirror.Result.Skipped, s => s.Contains("深すぎます"));
    }

    [Fact]
    public void 後始末で一時領域を消す()
    {
        var source = new FakeFileSource().Add("a.txt", "あ");

        string path;
        using (var mirror = RemoteMirror.Fetch(source))
        {
            path = mirror.Path;
            Assert.True(Directory.Exists(path));
        }
        Assert.False(Directory.Exists(path));
    }

    [Fact]
    public void 元の場所を見出しに使う()
    {
        // 一時領域の名前を出しても、どこを見ているのか分からない。
        var source = new FakeFileSource { Display = "davs://例.com/dav/" }.Add("a.txt", "あ");

        using var mirror = RemoteMirror.Fetch(source);

        Assert.Equal("davs://例.com/dav/", mirror.Display);
        Assert.Contains("davs://例.com/dav/", mirror.Describe());
    }
}

public sealed class RemoteLocationTests
{
    [Theory]
    [InlineData("dav://例.com/dav/", true)]
    [InlineData("davs://例.com/dav/", true)]
    [InlineData("s3://入口/バケツ", true)]
    [InlineData("/手元/の/場所", false)]
    [InlineData(@"C:\手元\の\場所", false)]
    [InlineData("https://例.com/", false)]     // 素の HTTP は置き場ではない
    public void リモートかどうかを見分ける(string location, bool expected)
        => Assert.Equal(expected, RemoteLocation.IsRemote(location));

    [Fact]
    public void 合言葉を伏せる()
    {
        // **画面・記録・レポートに出す前に必ず通す。** 場所の文字列はそのまま
        // 履歴やセッションに残るので、そこに合言葉が入ると後から漏れる。
        Assert.Equal("davs://利用者:***@例.com/dav/",
            RemoteLocation.Redact("davs://利用者:ひみつ@例.com/dav/"));
        Assert.Equal("s3://AKIA123:***@入口/バケツ",
            RemoteLocation.Redact("s3://AKIA123:とても秘密@入口/バケツ"));

        // 合言葉が無ければそのまま。
        Assert.Equal("davs://例.com/dav/", RemoteLocation.Redact("davs://例.com/dav/"));
        Assert.Equal("/手元/の/場所", RemoteLocation.Redact("/手元/の/場所"));
    }

    [Fact]
    public void 手元のパスはそのまま開く()
    {
        using var source = RemoteLocation.Open(Path.GetTempPath());
        Assert.IsType<LocalFileSource>(source);
    }

    [Fact]
    public void 鍵が無ければ断る()
    {
        // **黙って繋がらないより、何が足りないかを言う。**
        var previous = (
            Environment.GetEnvironmentVariable("AWS_ACCESS_KEY_ID"),
            Environment.GetEnvironmentVariable("AWS_SECRET_ACCESS_KEY"));
        Environment.SetEnvironmentVariable("AWS_ACCESS_KEY_ID", null);
        Environment.SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY", null);
        try
        {
            var error = Assert.Throws<ArgumentException>(
                () => RemoteLocation.Open("s3://入口/バケツ"));
            Assert.Contains("AWS_ACCESS_KEY_ID", error.Message);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AWS_ACCESS_KEY_ID", previous.Item1);
            Environment.SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY", previous.Item2);
        }
    }

    [Fact]
    public void S3の場所を分解する()
    {
        using var source = RemoteLocation.Open("s3://KEY:SECRET@例.com/バケツ/接頭辞/深く");
        Assert.Equal("s3://バケツ/接頭辞/深く", source.Display);
    }

    [Fact]
    public void 入口にschemeを書いても壊れない()
    {
        // MinIO や試験用の置き場は `http://主機:番号` の形で書く。
        // **先に外さないと `/` で切ったときに壊れる。**
        using var source = RemoteLocation.Open("s3://KEY:SECRET@http://127.0.0.1:9000/バケツ/接頭辞");
        Assert.Equal("s3://バケツ/接頭辞", source.Display);
    }

    [Fact]
    public void 形が足りなければ断る()
    {
        var error = Assert.Throws<ArgumentException>(
            () => RemoteLocation.Open("s3://KEY:SECRET@例.com"));
        // **合言葉を伏せたうえで**理由を出す。
        Assert.Contains("KEY:***@", error.Message);
        Assert.DoesNotContain("SECRET", error.Message);
    }
}

public sealed class SessionRedactionTests : IDisposable
{
    private readonly string _path =
        Path.Combine(Path.GetTempPath(), "dc-session-" + Guid.NewGuid().ToString("N")[..8] + ".json");

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void 合言葉を保存しない()
    {
        // **平文で残り続ける（そして本人も忘れる）。**
        var store = new SessionStore(_path);
        store.Upsert(new Session
        {
            Name = "本番",
            LeftPath = "/手元",
            RightPath = "ftps://利用者:ひみつ@主機/場所",
        });

        var saved = store.Load().Single();
        Assert.Equal("ftps://利用者:***@主機/場所", saved.RightPath);

        // ファイルの中身にも残っていない。
        Assert.DoesNotContain("ひみつ", File.ReadAllText(_path));
    }
}
