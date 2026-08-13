using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class SessionStoreTests : IDisposable
{
    private readonly string _path = Path.Combine(
        Path.GetTempPath(), "dc-session-" + Guid.NewGuid().ToString("N"), "sessions.json");

    public void Dispose()
    {
        try
        {
            Directory.Delete(Path.GetDirectoryName(_path)!, recursive: true);
        }
        catch
        {
            // 後片付けの失敗は結果に関係ない。
        }
    }

    private SessionStore Store() => new(_path);

    [Fact]
    public void LoadingWhenNothingIsSavedReturnsEmpty()
    {
        Assert.Empty(Store().Load());
    }

    [Fact]
    public void SavedSessionsSurviveARoundTrip()
    {
        var session = new Session
        {
            Name = "作業中",
            Kind = SessionKind.Text,
            LeftPath = "/tmp/a.cs",
            RightPath = "/tmp/b.cs",
            PairThreshold = 0.42f,
            Whitespace = WhitespaceMode.CollapseRuns,
            IgnoreCase = true,
            IgnoredPatterns = [@"\d{4}-\d{2}-\d{2}"],
        };

        Store().Upsert(session);
        var loaded = Assert.Single(Store().Load());

        Assert.Equal("作業中", loaded.Name);
        Assert.Equal(0.42f, loaded.PairThreshold);
        Assert.Equal(WhitespaceMode.CollapseRuns, loaded.Whitespace);
        Assert.True(loaded.IgnoreCase);
        Assert.Equal([@"\d{4}-\d{2}-\d{2}"], loaded.IgnoredPatterns);
    }

    /// <summary>
    /// 保存した設定がそのまま比較の指定になること。ここが繋がっていないと、
    /// 覚えているだけで使われない。
    /// </summary>
    [Fact]
    public void SavedSettingsBecomeCompareOptions()
    {
        var session = new Session
        {
            Whitespace = WhitespaceMode.IgnoreAll,
            IgnoreCase = true,
            IgnoredPatterns = ["v[0-9]+"],
        };

        var options = session.ToCompareOptions();

        Assert.Equal(options.Ignoring.Normalize("A v1 B"), options.Ignoring.Normalize("a v22 b"));
    }

    [Fact]
    public void SavedFolderSettingsBecomeFolderOptions()
    {
        var session = new Session
        {
            IncludeNames = ["*.cs"],
            FolderMode = FolderComparisonMode.SizeAndTimestamp,
            TimestampToleranceSeconds = 2,
        };

        var options = session.ToFolderOptions();

        Assert.True(options.Filter.Allows("a.cs", isDirectory: false));
        Assert.False(options.Filter.Allows("a.txt", isDirectory: false));
        Assert.Equal(FolderComparisonMode.SizeAndTimestamp, options.Mode);
        Assert.Equal(2, options.TimestampToleranceSeconds);
    }

    [Fact]
    public void UpsertReplacesTheSameNameInsteadOfAdding()
    {
        var store = Store();
        store.Upsert(new Session { Name = "作業", LeftPath = "old" });
        store.Upsert(new Session { Name = "作業", LeftPath = "new" });

        var loaded = Assert.Single(store.Load());
        Assert.Equal("new", loaded.LeftPath);
    }

    [Fact]
    public void MostRecentlyUsedComesFirst()
    {
        var store = Store();
        store.Upsert(new Session { Name = "古い" });
        Thread.Sleep(10);
        store.Upsert(new Session { Name = "新しい" });

        Assert.Equal("新しい", store.Load()[0].Name);
    }

    [Fact]
    public void RemoveDropsTheNamedSession()
    {
        var store = Store();
        store.Upsert(new Session { Name = "a" });
        store.Upsert(new Session { Name = "b" });

        store.Remove("a");

        var loaded = Assert.Single(store.Load());
        Assert.Equal("b", loaded.Name);
    }

    /// <summary>
    /// 壊れた設定ファイルで起動できなくならないこと。設定のせいで道具ごと
    /// 使えなくなるのは割に合わない。
    /// </summary>
    [Fact]
    public void ACorruptFileIsTreatedAsEmpty()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        File.WriteAllText(_path, "{ これは JSON ではない");

        Assert.Empty(Store().Load());
    }

    /// <summary>書き込みの途中で落ちても元の設定が残るよう、一時ファイル経由で置き換える。</summary>
    [Fact]
    public void SavingDoesNotLeaveATemporaryFileBehind()
    {
        var store = Store();
        store.Upsert(new Session { Name = "a" });

        Assert.False(File.Exists(_path + ".tmp"));
        Assert.True(File.Exists(_path));
    }
}
