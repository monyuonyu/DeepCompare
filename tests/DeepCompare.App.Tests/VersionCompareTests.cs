using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

/// <summary>
/// 版の比較。**ここは画面側の分岐だけを見る。**
/// PE の読み取りそのものはエンジン側の試験（合成した実行ファイルで
/// 網羅済み）に任せる — 合成の組み立てが engine の試験の中にあり、
/// こちらから呼べないため。
/// </summary>
public class VersionCompareTests
{
    [AvaloniaFact]
    public async Task 指定が無ければ促す()
    {
        var model = new VersionCompareViewModel(TestShell.Create());

        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
        Assert.Empty(model.Rows);
    }

    [AvaloniaFact]
    public async Task 実行ファイルでなければ理由を出して止まる()
    {
        using var left = new TempFile("これは実行ファイルではありません");
        using var right = new TempFile("これも違います");

        var model = new VersionCompareViewModel(TestShell.Create())
        {
            LeftPath = left.Path,
            RightPath = right.Path,
        };
        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
        Assert.Empty(model.Rows);
        Assert.False(model.Busy);
    }
}
