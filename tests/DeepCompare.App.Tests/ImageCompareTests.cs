using Avalonia.Headless.XUnit;
using SkiaSharp;
using Xunit;

namespace DeepCompare.App.Tests;

public class ImageCompareTests
{
    /// <summary>塗りつぶした PNG を書き出す。**本物の復号経路を通す。**</summary>
    private static TempFile Png(int width, int height, SKColor color, SKColor? spot = null)
    {
        using var bitmap = new SKBitmap(width, height);
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(color);
            if (spot is { } dot)
            {
                using var paint = new SKPaint { Color = dot };
                canvas.DrawRect(0, 0, 1, 1, paint);
            }
        }
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        var file = new TempFile(string.Empty, ".png");
        File.WriteAllBytes(file.Path, data.ToArray());
        return file;
    }

    private static ImageCompareViewModel Open(TempFile left, TempFile right)
        => new(TestShell.Create()) { LeftPath = left.Path, RightPath = right.Path };

    [AvaloniaFact]
    public async Task 同じ画像なら完全に同じと言う()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Red);

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.Contains("同じ", model.Summary);
        Assert.False(model.SizesDiffer);
        Assert.Empty(model.Message);
    }

    [AvaloniaFact]
    public async Task 一画素だけ違っても見つかる()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Red, spot: SKColors.Blue);

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.DoesNotContain("完全に同じ", model.Summary);
        Assert.NotNull(model.Difference);
    }

    /// <summary>
    /// **大きさが違っても重なる範囲は比べる。** 「大きさが違います」で
    /// 終わらせると、切り取っただけの画像で何も分からない。
    /// </summary>
    [AvaloniaFact]
    public async Task 大きさが違っても比べられその旨が出る()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(4, 4, SKColors.Red);

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.True(model.SizesDiffer);
        Assert.Contains("8×8", model.SizeText);
        Assert.Contains("4×4", model.SizeText);
    }

    [AvaloniaFact]
    public async Task 画像として読めなければ理由を出して止まる()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = new TempFile("これは画像ではありません", ".png");

        var model = Open(left, right);
        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
        Assert.False(model.Busy);
    }

    [AvaloniaFact]
    public async Task 指定が無ければ促す()
    {
        var model = new ImageCompareViewModel(TestShell.Create());

        await model.CompareAsync();

        Assert.NotEmpty(model.Message);
    }

    /// <summary>
    /// **違いがあれば、そこを見せる側へ自動で切り替える。** 探させない。
    /// </summary>
    [AvaloniaFact]
    public async Task 違いがあれば違いの表示へ移る()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Blue);
        var model = Open(left, right);

        await model.CompareAsync();

        Assert.True(model.IsDifference);
        Assert.False(model.IsSideBySide);
    }

    [AvaloniaFact]
    public async Task 同じ画像なら左右に並べたまま()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Red);
        var model = Open(left, right);

        await model.CompareAsync();

        Assert.True(model.IsSideBySide);
    }

    [AvaloniaFact]
    public async Task 見せ方を手で切り替えられる()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Red);
        var model = Open(left, right);
        await model.CompareAsync();

        model.SetDifferenceCommand.Execute(null);
        Assert.True(model.IsDifference);
        Assert.False(model.IsSideBySide);

        model.SetSwipeCommand.Execute(null);
        Assert.True(model.IsSwipe);

        model.SetSideBySideCommand.Execute(null);
        Assert.True(model.IsSideBySide);
    }

    [AvaloniaFact]
    public async Task 拡大と等倍が効く()
    {
        using var left = Png(8, 8, SKColors.Red);
        using var right = Png(8, 8, SKColors.Red);
        var model = Open(left, right);
        await model.CompareAsync();

        var start = model.Zoom;
        model.ZoomInCommand.Execute(null);
        Assert.True(model.Zoom > start);

        model.ZoomResetCommand.Execute(null);
        Assert.Equal(start, model.Zoom);
        Assert.Equal("100%", model.ZoomText);
    }
}
