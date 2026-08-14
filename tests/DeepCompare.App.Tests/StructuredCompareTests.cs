using Avalonia.Headless.XUnit;
using Xunit;

namespace DeepCompare.App.Tests;

public class StructuredCompareTests
{
    private static StructuredCompareViewModel Open(TempFile left, TempFile right)
        => new(TestShell.Create()) { LeftPath = left.Path, RightPath = right.Path };

    /// <summary>
    /// **この画面の存在理由。** キーを並べ替えただけ・整形しただけの差は
    /// 差分にしない。テキストとして比べると全行が違って見えるもの。
    /// </summary>
    [AvaloniaFact]
    public async Task キーの順序と整形の違いは差分にしない()
    {
        using var left = new TempFile("{\"a\":1,\"b\":2}", ".json");
        using var right = new TempFile("{\n  \"b\": 2,\n  \"a\": 1\n}", ".json");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.Empty(model.Changes);
    }

    [AvaloniaFact]
    public async Task 値が変わればその位置が出る()
    {
        using var left = new TempFile("{\"a\":1,\"b\":2}", ".json");
        using var right = new TempFile("{\"a\":1,\"b\":99}", ".json");

        var model = Open(left, right);
        await model.CompareAsync();

        var change = Assert.Single(model.Changes);
        Assert.Contains("b", change.Path);
        Assert.Contains("2", change.Left);
        Assert.Contains("99", change.Right);
    }

    /// <summary>型が変わったことは明示する（"1" と 1 は別もの）。</summary>
    [AvaloniaFact]
    public async Task 型の変化は注記が付く()
    {
        using var left = new TempFile("{\"a\":1}", ".json");
        using var right = new TempFile("{\"a\":\"1\"}", ".json");

        var model = Open(left, right);
        await model.CompareAsync();

        var change = Assert.Single(model.Changes);
        Assert.True(change.HasTypeNote);
    }

    [AvaloniaFact]
    public async Task 壊れたJSONは理由を出して止まる()
    {
        using var left = new TempFile("{\"a\":1}", ".json");
        using var right = new TempFile("{これは JSON ではない", ".json");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
        Assert.Empty(model.Changes);
        Assert.False(model.Busy);
    }

    [AvaloniaFact]
    public async Task 見ない位置を指定すればその差は消える()
    {
        using var left = new TempFile("{\"版\":\"1.0\",\"本体\":\"あ\"}", ".json");
        using var right = new TempFile("{\"版\":\"2.0\",\"本体\":\"あ\"}", ".json");

        var model = Open(left, right);
        await model.CompareAsync();
        var before = model.Changes.Count;

        // 位置は根（$）からの道で書く。1 行に 1 つ。
        model.IgnoredPaths = "$.版";
        await model.CompareAsync();

        Assert.Equal(1, before);
        Assert.Empty(model.Changes);
    }
}
