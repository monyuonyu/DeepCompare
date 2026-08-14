using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

public class MergeTests
{
    private static MergeViewModel Open(TempFile basis, TempFile left, TempFile right)
        => new(TestShell.Create())
        {
            BasePath = basis.Path,
            LeftPath = left.Path,
            RightPath = right.Path,
        };

    [AvaloniaFact]
    public async Task 別々の場所を直した二つは競合せず両方入る()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("A\nい\nう\n");
        using var right = new TempFile("あ\nい\nC\n");

        var model = Open(basis, left, right);
        await model.MergeAsync();

        Assert.DoesNotContain(model.Regions, r => r.IsConflict);
        var result = string.Join("\n", model.Regions.Select(r => r.ResultText));
        Assert.Contains("A", result);
        Assert.Contains("C", result);
    }

    [AvaloniaFact]
    public async Task 同じ場所を別々に直すと競合として残る()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("あ\nL\nう\n");
        using var right = new TempFile("あ\nR\nう\n");

        var model = Open(basis, left, right);
        await model.MergeAsync();

        var conflict = Assert.Single(model.Regions, r => r.IsConflict);
        Assert.Equal(ConflictChoice.Undecided, conflict.Choice);
        Assert.False(conflict.IsDecided);
    }

    [AvaloniaFact]
    public async Task 競合で左を選べば左の中身が結果になる()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("あ\nL\nう\n");
        using var right = new TempFile("あ\nR\nう\n");

        var model = Open(basis, left, right);
        await model.MergeAsync();
        var conflict = model.Regions.Single(r => r.IsConflict);
        conflict.Choice = ConflictChoice.Left;

        Assert.True(conflict.IsDecided);
        Assert.Contains("L", conflict.ResultText);
        Assert.DoesNotContain("R", conflict.ResultText);
    }

    /// <summary>
    /// 「どちらも採らない」は **元（祖先）へ戻す**。空にするのではない
    /// ——どちらの変更も要らない、という意味なので。
    /// </summary>
    [AvaloniaFact]
    public async Task どちらも採らないを選ぶと元の行へ戻る()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("あ\nL\nう\n");
        using var right = new TempFile("あ\nR\nう\n");

        var model = Open(basis, left, right);
        await model.MergeAsync();
        var conflict = model.Regions.Single(r => r.IsConflict);
        conflict.Choice = ConflictChoice.Neither;

        Assert.True(conflict.IsDecided);
        Assert.Contains("い", conflict.ResultText);
        Assert.DoesNotContain("L", conflict.ResultText);
        Assert.DoesNotContain("R", conflict.ResultText);
    }

    /// <summary>
    /// 決めたものが結果へ載ることを、書き出した中身で確かめる。
    /// **書き出し先は人に選ばせる経路なので、受け取り手を差し込む。**
    /// </summary>
    [AvaloniaFact]
    public async Task 決めた結果が書き出す中身に載る()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("あ\nL\nう\n");
        using var right = new TempFile("あ\nR\nう\n");

        IReadOnlyList<string> written = [];
        var model = Open(basis, left, right);
        model.SaveHandler = lines => { written = lines; return Task.CompletedTask; };
        await model.MergeAsync();
        model.Regions.Single(r => r.IsConflict).Choice = ConflictChoice.Right;
        await model.SaveAsync();

        Assert.Contains("R", written);
        Assert.DoesNotContain("L", written);
    }

    /// <summary>
    /// **未決が残ったまま索引へ載せない。** 印の付いた行がそのままコミットに
    /// 入る事故は、git を使っていて一番起きやすい失敗。受け取り手がある経路
    /// （git の競合解決）では、書き出さずに拒む。
    /// </summary>
    [AvaloniaFact]
    public async Task 未決が残っていると確定を拒む()
    {
        using var basis = new TempFile("あ\nい\nう\n");
        using var left = new TempFile("あ\nL\nう\n");
        using var right = new TempFile("あ\nR\nう\n");

        var called = false;
        var model = Open(basis, left, right);
        model.SaveHandler = _ => { called = true; return Task.CompletedTask; };
        await model.MergeAsync();
        await model.SaveAsync();

        Assert.False(called);
        Assert.Contains("未決", model.Message);
    }
}
