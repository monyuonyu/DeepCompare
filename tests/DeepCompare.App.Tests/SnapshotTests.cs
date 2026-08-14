using Avalonia.Headless.XUnit;
using DeepCompare.Engine;
using Xunit;

namespace DeepCompare.App.Tests;

public class SnapshotTests
{
    [AvaloniaFact]
    public async Task フォルダーでない場所を写そうとすると理由が出る()
    {
        var model = new SnapshotViewModel(TestShell.Create())
        {
            FolderPath = Path.Combine(Path.GetTempPath(), "存在しない-" + Guid.NewGuid()),
        };

        await model.TakeAsync();

        Assert.NotEmpty(model.Message);
    }

    /// <summary>
    /// 写しを取ると、**そのまま比較の相手に据わる**（入れ直させない）。
    /// </summary>
    [AvaloniaFact]
    public async Task 写しを取ると比較の相手に据わる()
    {
        using var folder = new TempFolder();
        using var snapshot = new TempFile(string.Empty, ".dcsnap");
        File.WriteAllText(Path.Combine(folder.Path, "あ.txt"), "あ\n");

        var model = new SnapshotViewModel(TestShell.Create(savePath: snapshot.Path))
        {
            FolderPath = folder.Path,
        };
        await model.TakeAsync();

        Assert.Equal(snapshot.Path, model.SnapshotPath);
        Assert.Contains("1 ファイル", model.Summary);
        Assert.Empty(model.Message);
    }

    [AvaloniaFact]
    public async Task 写した後に増えたファイルが増えたとして出る()
    {
        using var folder = new TempFolder();
        using var snapshot = new TempFile(string.Empty, ".dcsnap");
        File.WriteAllText(Path.Combine(folder.Path, "もとから.txt"), "あ\n");

        var model = new SnapshotViewModel(TestShell.Create(savePath: snapshot.Path))
        {
            FolderPath = folder.Path,
        };
        await model.TakeAsync();

        File.WriteAllText(Path.Combine(folder.Path, "あとから.txt"), "い\n");
        await model.CompareAsync();

        var row = Assert.Single(model.Rows, r => r.Path.Contains("あとから.txt"));
        Assert.Equal(EntryStatus.RightOnly, row.Entry.Status);
    }

    [AvaloniaFact]
    public async Task 写した後に消えたファイルが消えたとして出る()
    {
        using var folder = new TempFolder();
        using var snapshot = new TempFile(string.Empty, ".dcsnap");
        var path = Path.Combine(folder.Path, "消える.txt");
        File.WriteAllText(path, "あ\n");

        var model = new SnapshotViewModel(TestShell.Create(savePath: snapshot.Path))
        {
            FolderPath = folder.Path,
        };
        await model.TakeAsync();

        File.Delete(path);
        await model.CompareAsync();

        var row = Assert.Single(model.Rows, r => r.Path.Contains("消える.txt"));
        Assert.Equal(EntryStatus.LeftOnly, row.Entry.Status);
    }

    /// <summary>
    /// **指紋ありで写せば、中身の変化まで見分けられる。**
    /// 大きさが同じまま中身だけ変わった場合が、これでしか捕まらない。
    /// </summary>
    [AvaloniaFact]
    public async Task 指紋ありなら大きさが同じでも中身の変化が出る()
    {
        using var folder = new TempFolder();
        using var snapshot = new TempFile(string.Empty, ".dcsnap");
        var path = Path.Combine(folder.Path, "同じ大きさ.txt");
        File.WriteAllText(path, "あいう\n");

        var model = new SnapshotViewModel(TestShell.Create(savePath: snapshot.Path))
        {
            FolderPath = folder.Path,
            WithHashes = true,
        };
        await model.TakeAsync();

        File.WriteAllText(path, "かきく\n");
        await model.CompareAsync();

        var row = Assert.Single(model.Rows, r => r.Path.Contains("同じ大きさ.txt"));
        Assert.Equal(EntryStatus.Different, row.Entry.Status);
        Assert.False(model.WithoutHashes);
    }

    [AvaloniaFact]
    public async Task 何も変わっていなければ何も出ない()
    {
        using var folder = new TempFolder();
        using var snapshot = new TempFile(string.Empty, ".dcsnap");
        File.WriteAllText(Path.Combine(folder.Path, "そのまま.txt"), "あ\n");

        var model = new SnapshotViewModel(TestShell.Create(savePath: snapshot.Path))
        {
            FolderPath = folder.Path,
        };
        await model.TakeAsync();
        await model.CompareAsync();

        Assert.Empty(model.Rows);
        Assert.Contains("同じ", model.Summary);
    }
}
