using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

/// <summary>
/// **「押した結果」を確かめる。** ここまでは試験が届いていなかった範囲で、
/// エンジンが正しくても画面の経路で落ちるものを捕まえる。
/// </summary>
public class TextCompareTests
{
    [AvaloniaFact]
    public async Task 二つのファイルを比べると行が並び差分が見つかる()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        Assert.NotEmpty(model.VisibleRows);
        Assert.True(model.HasDifferences);
        Assert.False(model.IsBusy);
    }

    [AvaloniaFact]
    public async Task 中身が同じなら差分は出ない()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nい\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        Assert.NotEmpty(model.VisibleRows);
        Assert.False(model.HasDifferences);
    }

    /// <summary>
    /// Windows で踏んだ不具合の再発防止。フォルダーをそのまま読みに行って
    /// 「Access to the path ... is denied」を出していた。
    /// </summary>
    [AvaloniaFact]
    public async Task 両方フォルダーならフォルダー比較へ移る()
    {
        using var left = new TempFolder();
        using var right = new TempFolder();
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        Assert.Contains(shell.Tabs, tab => tab.Content is FolderCompareViewModel);
    }

    [AvaloniaFact]
    public async Task 片方だけフォルダーなら理由を出して止まる()
    {
        using var left = new TempFolder();
        using var right = new TempFile("あ\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        Assert.Contains("フォルダー", model.Placeholder);
        Assert.Empty(model.VisibleRows);
    }

    /// <summary>片方だけ渡されたら、そのファイルを 1 枚で開く。</summary>
    [AvaloniaFact]
    public async Task 片方だけでも中身が出る()
    {
        using var left = new TempFile("あ\nい\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, string.Empty, run: false);
        await model.RunCompareAsync();

        Assert.Equal(3, model.VisibleRows.Count);
    }

    [AvaloniaFact]
    public async Task 何も指定しなければ促す文が出る()
    {
        var shell = TestShell.Create();

        var model = shell.ShowText(string.Empty, string.Empty, run: false);
        await model.RunCompareAsync();

        Assert.NotEmpty(model.Placeholder);
        Assert.Empty(model.VisibleRows);
    }

    [AvaloniaFact]
    public async Task 塊を右へ写すと差分が消え右が変更済みになる()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        var block = model.VisibleRows.First(row => row.IsBlockStart);
        await model.ApplyBlockAsync(block, toRight: true);

        Assert.False(model.HasDifferences);
        Assert.True(model.RightModified);
        Assert.False(model.LeftModified);
    }

    [AvaloniaFact]
    public async Task 写した後に取り消すと元へ戻る()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        var block = model.VisibleRows.First(row => row.IsBlockStart);
        await model.ApplyBlockAsync(block, toRight: true);
        await model.UndoAsync();

        Assert.True(model.HasDifferences);
    }

    /// <summary>
    /// **写しただけではファイルは変わらない。** 保存して初めて書かれる。
    /// </summary>
    [AvaloniaFact]
    public async Task 保存するまでファイルは変わらない()
    {
        using var left = new TempFile("あ\nい\nう\n");
        using var right = new TempFile("あ\nZ\nう\n");
        var shell = TestShell.Create();

        var model = shell.ShowText(left.Path, right.Path, run: false);
        await model.RunCompareAsync();

        var block = model.VisibleRows.First(row => row.IsBlockStart);
        await model.ApplyBlockAsync(block, toRight: true);

        Assert.Contains("Z", right.Read());

        await model.SaveAsync(left: false);

        Assert.DoesNotContain("Z", right.Read());
        Assert.Contains("い", right.Read());
        Assert.False(model.RightModified);
    }
}
