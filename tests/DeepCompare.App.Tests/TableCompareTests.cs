using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

public class TableCompareTests
{
    private static TableCompareViewModel Open(TempFile left, TempFile right)
        => new(TestShell.Create()) { LeftPath = left.Path, RightPath = right.Path };

    [AvaloniaFact]
    public async Task 見出しが列として並ぶ()
    {
        using var left = new TempFile("id,名前,値\n1,あ,10\n", ".csv");
        using var right = new TempFile("id,名前,値\n1,あ,20\n", ".csv");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.Equal(3, model.Header.Count);
        Assert.Equal("id", model.Header[0].Text);
    }

    [AvaloniaFact]
    public async Task 変わった行だけが既定で出る()
    {
        using var left = new TempFile("id,値\n1,10\n2,20\n", ".csv");
        using var right = new TempFile("id,値\n1,10\n2,99\n", ".csv");

        var model = Open(left, right);
        await model.CompareAsync();

        var row = Assert.Single(model.Rows);
        Assert.Contains("99", row.RightText);
    }

    /// <summary>
    /// **キー列を決めれば、行の順序が変わっても同じ行として対応付く。**
    /// 位置で突き合わせると、並べ替えただけで全行が違って見える。
    /// </summary>
    [AvaloniaFact]
    public async Task キー列を決めれば並べ替えても差分にならない()
    {
        using var left = new TempFile("id,値\n1,10\n2,20\n", ".csv");
        using var right = new TempFile("id,値\n2,20\n1,10\n", ".csv");

        var model = Open(left, right);
        model.KeyColumns = "id";
        await model.CompareAsync();

        Assert.Empty(model.Rows);
    }

    [AvaloniaFact]
    public async Task 見ない列を指定すればその差は消える()
    {
        using var left = new TempFile("id,更新日,値\n1,2026-01-01,10\n", ".csv");
        using var right = new TempFile("id,更新日,値\n1,2026-08-14,10\n", ".csv");

        var model = Open(left, right);
        await model.CompareAsync();
        var before = model.Rows.Count;

        model.IgnoredColumns = "更新日";
        await model.CompareAsync();

        Assert.Equal(1, before);
        Assert.Empty(model.Rows);
    }

    [AvaloniaFact]
    public async Task 同じ行も出す設定にすれば現れる()
    {
        using var left = new TempFile("id,値\n1,10\n2,20\n", ".csv");
        using var right = new TempFile("id,値\n1,10\n2,99\n", ".csv");

        var model = Open(left, right);
        model.ShowUnchanged = true;
        await model.CompareAsync();

        Assert.Equal(2, model.Rows.Count);
    }

    [AvaloniaFact]
    public async Task 片側にしかない行は片側だけとして出る()
    {
        using var left = new TempFile("id,値\n1,10\n2,20\n", ".csv");
        using var right = new TempFile("id,値\n1,10\n", ".csv");

        var model = Open(left, right);
        model.KeyColumns = "id";
        await model.CompareAsync();

        var row = Assert.Single(model.Rows);
        Assert.True(row.HasLeft);
        Assert.False(row.HasRight);
    }

    [AvaloniaFact]
    public async Task TSVも読める()
    {
        using var left = new TempFile("id\t値\n1\t10\n", ".tsv");
        using var right = new TempFile("id\t値\n1\t99\n", ".tsv");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.Equal(2, model.Header.Count);
        Assert.Single(model.Rows);
    }
}
