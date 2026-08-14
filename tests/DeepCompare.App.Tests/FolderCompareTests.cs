using Avalonia.Headless.XUnit;
using DeepCompare.Engine;
using Xunit;

namespace DeepCompare.App.Tests;

public class FolderCompareTests
{
    private static FolderCompareViewModel Open(TempFolder left, TempFolder right)
        => new(TestShell.Create()) { LeftRoot = left.Path, RightRoot = right.Path };

    [AvaloniaFact]
    public async Task 片側にしかないファイルが片側だけとして出る()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        File.WriteAllText(Path.Combine(left.Path, "ひとり.txt"), "あ\n");

        var model = Open(left, right);
        await model.RunAsync();

        var row = Assert.Single(model.Rows, r => r.Entry.Name == "ひとり.txt");
        Assert.Equal(EntryStatus.LeftOnly, row.Entry.Status);
    }

    [AvaloniaFact]
    public async Task 中身が違えば違うとして出る()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        File.WriteAllText(Path.Combine(left.Path, "同名.txt"), "あ\n");
        File.WriteAllText(Path.Combine(right.Path, "同名.txt"), "い\n");

        var model = Open(left, right);
        await model.RunAsync();

        var row = Assert.Single(model.Rows, r => r.Entry.Name == "同名.txt");
        Assert.Equal(EntryStatus.Different, row.Entry.Status);
    }

    /// <summary>
    /// **同じものは既定で隠す。** 隠したものが「無い」ことにならないよう、
    /// 出す設定にすれば現れることまで見る。
    /// </summary>
    [AvaloniaFact]
    public async Task 同じファイルは既定で隠れ出す設定にすれば現れる()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        File.WriteAllText(Path.Combine(left.Path, "同じ.txt"), "あ\n");
        File.WriteAllText(Path.Combine(right.Path, "同じ.txt"), "あ\n");

        var model = Open(left, right);
        await model.RunAsync();
        var hidden = model.Rows.Any(r => r.Entry.Name == "同じ.txt");

        model.ShowIdentical = true;
        await model.RunAsync();
        var shown = model.Rows.Any(r => r.Entry.Name == "同じ.txt");

        Assert.False(hidden);
        Assert.True(shown);
    }

    [AvaloniaFact]
    public async Task 入れ子のフォルダーもたどる()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        Directory.CreateDirectory(Path.Combine(left.Path, "奥"));
        File.WriteAllText(Path.Combine(left.Path, "奥", "深い.txt"), "あ\n");

        var model = Open(left, right);
        await model.RunAsync();

        Assert.Contains(model.Rows, r => r.Entry.Name == "深い.txt");
    }

    [AvaloniaFact]
    public async Task ファイルを右へ写すと相手側に現れる()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        File.WriteAllText(Path.Combine(left.Path, "写す.txt"), "あ\n");

        var model = Open(left, right);
        await model.RunAsync();
        var row = model.Rows.Single(r => r.Entry.Name == "写す.txt");
        await model.CopyFileAsync(row, toRight: true);

        Assert.True(File.Exists(Path.Combine(right.Path, "写す.txt")));
        Assert.Equal("あ\n", File.ReadAllText(Path.Combine(right.Path, "写す.txt")));
    }

    [AvaloniaFact]
    public async Task 承知すればファイルが消え一覧からも消える()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        var path = Path.Combine(left.Path, "消す.txt");
        File.WriteAllText(path, "あ\n");

        var model = Open(left, right);
        model.Confirm = _ => Task.FromResult(true);
        await model.RunAsync();
        var row = model.Rows.Single(r => r.Entry.Name == "消す.txt");
        await model.DeleteAsync(row, left: true);

        Assert.False(File.Exists(path));
        Assert.DoesNotContain(model.Rows, r => r.Entry.Name == "消す.txt");
    }

    /// <summary>
    /// **尋ねる先が無ければ消さない。** 元に戻せない操作なので、
    /// 確認の経路が繋がっていないときに黙って実行してはいけない。
    /// </summary>
    [AvaloniaFact]
    public async Task 確認の受け取り手が無ければ消さない()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        var path = Path.Combine(left.Path, "消えないで.txt");
        File.WriteAllText(path, "あ\n");

        var model = Open(left, right);
        await model.RunAsync();
        var row = model.Rows.Single(r => r.Entry.Name == "消えないで.txt");
        await model.DeleteAsync(row, left: true);

        Assert.True(File.Exists(path));
    }

    [AvaloniaFact]
    public async Task 断ればファイルは残る()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        var path = Path.Combine(left.Path, "やめる.txt");
        File.WriteAllText(path, "あ\n");

        var model = Open(left, right);
        var asked = false;
        model.Confirm = _ => { asked = true; return Task.FromResult(false); };
        await model.RunAsync();
        var row = model.Rows.Single(r => r.Entry.Name == "やめる.txt");
        await model.DeleteAsync(row, left: true);

        Assert.True(asked);
        Assert.True(File.Exists(path));
    }

    [AvaloniaFact]
    public async Task 空どうしなら何も出ない()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();

        var model = Open(left, right);
        await model.RunAsync();

        Assert.Empty(model.Rows);
    }
}
