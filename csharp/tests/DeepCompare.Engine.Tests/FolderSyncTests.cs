using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// フォルダーの同期。
///
/// **必ず「予定」を作ってから実行する。** 何が起きるかを見ずに走らせる同期は、
/// 取り返しのつかない事故を生む。
/// </summary>
public sealed class FolderSyncTests : IDisposable
{
    private readonly string _root =
        Path.Combine(Path.GetTempPath(), "dc-sync-" + Guid.NewGuid().ToString("N")[..8]);

    private string Left => Path.Combine(_root, "left");
    private string Right => Path.Combine(_root, "right");

    public FolderSyncTests()
    {
        Directory.CreateDirectory(Left);
        Directory.CreateDirectory(Right);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private void Write(string side, string name, string content, DateTime? modified = null)
    {
        var path = Path.Combine(side, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        if (modified is { } time)
        {
            File.SetLastWriteTimeUtc(path, time);
        }
    }

    private SyncPlan Plan(SyncOptions options)
        => FolderSync.Plan(FolderComparer.Compare(Left, Right, new FolderCompareOptions()), options);

    private SyncResult Apply(SyncPlan plan) => FolderSync.Apply(plan, Left, Right);

    [Fact]
    public void 左にしか無いものを右へ写す()
    {
        Write("left".Replace("left", Left), "a.txt", "中身");

        var plan = Plan(new SyncOptions { Direction = SyncDirection.ToRight });

        var step = Assert.Single(plan.Steps);
        Assert.Equal(SyncAction.CopyToRight, step.Action);

        Apply(plan);
        Assert.True(File.Exists(Path.Combine(Right, "a.txt")));
    }

    [Fact]
    public void 既定では消さない()
    {
        // **同期の事故はほとんどこれで起きる。**「揃える」のつもりで押したら、
        // 片側にしか無い作りかけが消えた、という形。
        Write(Right, "only.txt", "x");

        var plan = Plan(new SyncOptions { Direction = SyncDirection.ToRight });

        Assert.True(plan.IsEmpty);
        Assert.Equal(0, plan.Deletions);
    }

    [Fact]
    public void 消すと言われたときだけ消す()
    {
        Write(Right, "only.txt", "x");

        var plan = Plan(new SyncOptions
        {
            Direction = SyncDirection.ToRight,
            DeleteOrphans = true,
        });

        Assert.Equal(SyncAction.DeleteRight, Assert.Single(plan.Steps).Action);

        Apply(plan);
        Assert.False(File.Exists(Path.Combine(Right, "only.txt")));
    }

    [Fact]
    public void 中身が違えば正の側から写す()
    {
        Write(Left, "a.txt", "あたらしい");
        Write(Right, "a.txt", "ふるい");

        Apply(Plan(new SyncOptions { Direction = SyncDirection.ToRight }));

        Assert.Equal("あたらしい", File.ReadAllText(Path.Combine(Right, "a.txt")));
    }

    [Fact]
    public void 両方向では新しい方を採る()
    {
        var old = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var recent = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Write(Left, "a.txt", "ふるい", old);
        Write(Right, "a.txt", "あたらしい", recent);

        var plan = Plan(new SyncOptions { Direction = SyncDirection.Both });

        Assert.Equal(SyncAction.CopyToLeft, Assert.Single(plan.Steps).Action);

        Apply(plan);
        Assert.Equal("あたらしい", File.ReadAllText(Path.Combine(Left, "a.txt")));
    }

    [Fact]
    public void 時刻が同じで中身が違えば触らない()
    {
        // **どちらが新しいか決められない。** 勝手にどちらかを採ると、
        // 片方の変更が黙って消える。
        var time = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Write(Left, "a.txt", "こちら", time);
        Write(Right, "a.txt", "あちら", time);

        var plan = Plan(new SyncOptions { Direction = SyncDirection.Both });

        Assert.True(plan.IsEmpty);
    }

    [Fact]
    public void 時刻のわずかなずれは同じとみなす()
    {
        // FAT と NTFS の間では 1〜2 秒ずれる。0 にすると毎回どちらかを写す。
        var time = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        Write(Left, "a.txt", "こちら", time);
        Write(Right, "a.txt", "あちら", time.AddSeconds(1));

        var plan = Plan(new SyncOptions
        {
            Direction = SyncDirection.Both,
            ToleranceSeconds = 2,
        });

        Assert.True(plan.IsEmpty);
    }

    [Fact]
    public void 写した後は時刻も合わせる()
    {
        // **合わせないと、次に時刻で比べたときにまた「違う」と出て、写し続ける。**
        Write(Left, "a.txt", "中身", new DateTime(2020, 5, 5, 0, 0, 0, DateTimeKind.Utc));

        Apply(Plan(new SyncOptions { Direction = SyncDirection.ToRight }));

        Assert.Equal(
            File.GetLastWriteTimeUtc(Path.Combine(Left, "a.txt")),
            File.GetLastWriteTimeUtc(Path.Combine(Right, "a.txt")));
    }

    [Fact]
    public void 入れ子の場所へも写す()
    {
        Write(Left, Path.Combine("deep", "nested", "a.txt"), "中身");

        Apply(Plan(new SyncOptions { Direction = SyncDirection.ToRight }));

        Assert.True(File.Exists(Path.Combine(Right, "deep", "nested", "a.txt")));
    }

    [Fact]
    public void 予定に理由を添える()
    {
        // **予定を人が読んで確かめられるようにする。**
        Write(Left, "a.txt", "x");

        var step = Assert.Single(Plan(new SyncOptions()).Steps);

        Assert.Contains("左にしか無い", step.Reason);
        Assert.Contains("a.txt", step.Describe());
    }

    [Fact]
    public void 消す操作があることを強く伝える()
    {
        Write(Right, "only.txt", "x");

        var text = FolderSync.Format(Plan(new SyncOptions
        {
            Direction = SyncDirection.ToRight,
            DeleteOrphans = true,
        }));

        Assert.Contains("元に戻せません", text);
    }

    [Fact]
    public void することが無ければそう言う()
    {
        Write(Left, "a.txt", "同じ");
        Write(Right, "a.txt", "同じ");

        Assert.Contains("することはありません", FolderSync.Format(Plan(new SyncOptions())));
    }

    [Fact]
    public void 一件失敗しても続ける()
    {
        // **途中で止めると、どこまで進んだのか分からない中途半端な状態が残る。**
        Write(Left, "a.txt", "1");
        Write(Left, "b.txt", "2");

        var plan = new SyncPlan([
            new SyncStep("missing.txt", SyncAction.CopyToRight, "作り話"),
            new SyncStep("a.txt", SyncAction.CopyToRight, "左にしか無い"),
            new SyncStep("b.txt", SyncAction.CopyToRight, "左にしか無い"),
        ]);

        var result = Apply(plan);

        Assert.Equal(2, result.Done);
        Assert.Single(result.Errors);
        Assert.False(result.AllSucceeded);
        Assert.True(File.Exists(Path.Combine(Right, "b.txt")));
    }
}
