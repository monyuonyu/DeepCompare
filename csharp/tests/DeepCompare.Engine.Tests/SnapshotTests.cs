using System.Text;
using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class SnapshotTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "dc-snapshot-" + Guid.NewGuid().ToString("N")[..8]);

    public SnapshotTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // 後始末に失敗しても試験の結果には関係しない。
        }
    }

    private void Write(string relative, string content)
    {
        var path = Path.Combine(_root, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content, new UTF8Encoding(false));
    }

    [Fact]
    public void 写しを取って読み直せる()
    {
        Write("a.txt", "あ\n");
        Write("下/b.txt", "い\n");

        var snapshot = Snapshots.Take(_root, withHashes: true);
        var restored = Snapshots.Load(Snapshots.Save(snapshot));

        Assert.Equal(snapshot.Entries.Count, restored.Entries.Count);
        Assert.Equal(2, restored.FileCount);
        Assert.Equal(1, restored.DirectoryCount);
        Assert.True(restored.HasHashes);

        var original = snapshot.Entries.Single(e => e.RelativePath == "a.txt");
        var loaded = restored.Entries.Single(e => e.RelativePath == "a.txt");
        Assert.Equal(original.Size, loaded.Size);
        Assert.Equal(original.Hash, loaded.Hash);
        // 時刻は往復しても丸めで狂わない（"o" 書式で書いている）。
        Assert.Equal(original.Modified, loaded.Modified);
    }

    [Fact]
    public void 名前にタブが入っていても壊れない()
    {
        // **名前を行の最後に置いてある**ので、後ろを全部名前として読めば
        // 区切りと衝突しない。
        var snapshot = new Snapshot("/tmp", DateTimeOffset.Now,
        [
            new SnapshotEntry("変\tな\t名前.txt", false, 3, new DateTime(2026, 8, 13, 1, 2, 3, DateTimeKind.Utc)),
        ]);

        var loaded = Snapshots.Load(Snapshots.Save(snapshot));

        Assert.Equal("変\tな\t名前.txt", Assert.Single(loaded.Entries).RelativePath);
    }

    [Fact]
    public void 増えた減った変わったを見分ける()
    {
        Write("残る.txt", "1\n");
        Write("消える.txt", "2\n");
        Write("変わる.txt", "3\n");
        var before = Snapshots.Take(_root, withHashes: true);

        File.Delete(Path.Combine(_root, "消える.txt"));
        Write("変わる.txt", "3 を書き換えた\n");
        Write("増える.txt", "4\n");
        var after = Snapshots.Take(_root, withHashes: true);

        var result = Snapshots.Compare(before, after);

        Assert.Equal(EntryStatus.Identical,
            result.Entries.Single(e => e.RelativePath == "残る.txt").Status);
        Assert.Equal(EntryStatus.LeftOnly,
            result.Entries.Single(e => e.RelativePath == "消える.txt").Status);
        Assert.Equal(EntryStatus.RightOnly,
            result.Entries.Single(e => e.RelativePath == "増える.txt").Status);
        Assert.Equal(EntryStatus.Different,
            result.Entries.Single(e => e.RelativePath == "変わる.txt").Status);

        Assert.Equal(1, result.Stats.Different);
        Assert.Equal(1, result.Stats.LeftOnly);
        Assert.Equal(1, result.Stats.RightOnly);
    }

    [Fact]
    public void 大きさも時刻も同じで中身だけ違う場合を指紋で見分ける()
    {
        // **これが指紋を取る理由。** 同じ秒の内に書き換えると、大きさも時刻も
        // 同じまま中身だけが変わる。指紋が無ければ「同じ」としか言えない。
        Write("a.txt", "あいう\n");
        var stamp = new DateTime(2026, 8, 13, 12, 0, 0, DateTimeKind.Utc);
        var path = Path.Combine(_root, "a.txt");
        File.SetLastWriteTimeUtc(path, stamp);
        var before = Snapshots.Take(_root, withHashes: true);

        Write("a.txt", "かきく\n");            // 同じ長さ
        File.SetLastWriteTimeUtc(path, stamp); // 同じ時刻に戻す
        var after = Snapshots.Take(_root, withHashes: true);

        Assert.Equal(before.Entries[0].Size, after.Entries[0].Size);
        Assert.Equal(before.Entries[0].Modified, after.Entries[0].Modified);
        Assert.NotEqual(before.Entries[0].Hash, after.Entries[0].Hash);

        Assert.Equal(EntryStatus.Different,
            Snapshots.Compare(before, after).Entries.Single(e => !e.IsDirectory).Status);
    }

    [Fact]
    public void 指紋を取らなければ大きさと時刻で比べる()
    {
        Write("a.txt", "あいう\n");
        var stamp = new DateTime(2026, 8, 13, 12, 0, 0, DateTimeKind.Utc);
        var path = Path.Combine(_root, "a.txt");
        File.SetLastWriteTimeUtc(path, stamp);
        var before = Snapshots.Take(_root);

        Write("a.txt", "かきく\n");
        File.SetLastWriteTimeUtc(path, stamp);
        var after = Snapshots.Take(_root);

        Assert.False(before.HasHashes);
        // 指紋が無いので見分けられない。**そう振る舞うと決めたことを固定する。**
        Assert.Equal(EntryStatus.Identical,
            Snapshots.Compare(before, after).Entries.Single(e => !e.IsDirectory).Status);
    }

    [Fact]
    public void 並びは取るたびに同じになる()
    {
        // 写し自体をバージョン管理に入れる人がいる。順序が揺れると、
        // 中身が変わっていなくても差分が出る。
        Write("z.txt", "1\n");
        Write("a.txt", "2\n");
        Write("m/n.txt", "3\n");

        var first = Snapshots.Take(_root);
        var second = Snapshots.Take(_root);

        Assert.Equal(
            first.Entries.Select(e => e.RelativePath),
            second.Entries.Select(e => e.RelativePath));
        Assert.Equal(Snapshots.Save(first).Split("---\n")[1], Snapshots.Save(second).Split("---\n")[1]);
    }

    [Fact]
    public void 絞り込みが効く()
    {
        Write("a.txt", "1\n");
        Write("b.log", "2\n");

        var snapshot = Snapshots.Take(_root, filter: new NameFilter(Exclude: ["*.log"]));

        Assert.DoesNotContain(snapshot.Entries, e => e.RelativePath.EndsWith(".log"));
        Assert.Contains(snapshot.Entries, e => e.RelativePath == "a.txt");
    }

    [Fact]
    public void 写しでないものを読ませたら断る()
    {
        Assert.Throws<InvalidDataException>(() => Snapshots.Load("これは写しではない\n"));
    }
}
