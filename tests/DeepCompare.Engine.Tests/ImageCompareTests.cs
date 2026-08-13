using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class ImageCompareTests
{
    /// <summary>単色で埋めた画像を作る。並びは B, G, R, A。</summary>
    private static PixelImage Solid(int width, int height, byte b, byte g, byte r, byte a = 0xFF)
    {
        var pixels = new byte[width * height * 4];
        for (var i = 0; i < width * height; i++)
        {
            pixels[i * 4] = b;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = r;
            pixels[i * 4 + 3] = a;
        }
        return new PixelImage(width, height, pixels);
    }

    private static void Set(PixelImage image, int x, int y, byte b, byte g, byte r, byte a = 0xFF)
    {
        var offset = image.Offset(x, y);
        image.Pixels[offset] = b;
        image.Pixels[offset + 1] = g;
        image.Pixels[offset + 2] = r;
        image.Pixels[offset + 3] = a;
    }

    [Fact]
    public void 同じ画像は完全に同じと出る()
    {
        var result = ImageCompare.Compare(Solid(4, 3, 10, 20, 30), Solid(4, 3, 10, 20, 30));

        Assert.True(result.IsIdentical);
        Assert.Equal(12, result.SameCount);
        Assert.Equal(0, result.DifferentCount);
        Assert.Null(result.Bounds);
    }

    [Fact]
    public void 一画素だけ違えばその位置が範囲になる()
    {
        var left = Solid(5, 5, 0, 0, 0);
        var right = Solid(5, 5, 0, 0, 0);
        Set(right, 2, 3, 255, 255, 255);

        var result = ImageCompare.Compare(left, right);

        Assert.False(result.IsIdentical);
        Assert.Equal(1, result.DifferentCount);
        Assert.Equal(24, result.SameCount);
        Assert.Equal((2, 3, 2, 3), result.Bounds);
    }

    [Fact]
    public void しきい値の内側の差は違うと数えない()
    {
        // **JPEG を保存し直すだけで全画素が 1〜2 変わる。** しきい値が 0 だと
        // 「全部違う」としか出ず、どこを直したのかが読み取れない。
        var left = Solid(3, 3, 100, 100, 100);
        var right = Solid(3, 3, 104, 103, 102);

        var result = ImageCompare.Compare(left, right);

        Assert.Equal(0, result.DifferentCount);
        Assert.Equal(9, result.NearCount);
        Assert.True(result.LooksSame);
        Assert.False(result.IsIdentical);
        Assert.Null(result.Bounds);
    }

    [Fact]
    public void しきい値を超える差は違うと数える()
    {
        var result = ImageCompare.Compare(
            Solid(2, 2, 100, 100, 100), Solid(2, 2, 120, 100, 100));

        Assert.Equal(4, result.DifferentCount);
        Assert.False(result.LooksSame);
    }

    [Fact]
    public void 大きさが違ってもはみ出した分だけ片方扱いにする()
    {
        // **諦めない。** 余白を足しただけの画像で「全然違う」としか
        // 言えなくなるのを避ける。
        var left = Solid(2, 2, 50, 50, 50);
        var right = Solid(4, 2, 50, 50, 50);

        var result = ImageCompare.Compare(left, right);

        Assert.Equal(4, result.Width);
        Assert.Equal(2, result.Height);
        Assert.Equal(4, result.SameCount);      // 重なる 2×2
        Assert.Equal(4, result.MissingCount);   // はみ出した 2×2
        Assert.Equal(0, result.DifferentCount);
        Assert.Equal((2, 0, 3, 1), result.Bounds);
    }

    [Fact]
    public void 透明度だけの違いも拾う()
    {
        // 見ないと、透明にしただけの変更が「同じ」になる。
        var left = Solid(2, 2, 10, 10, 10, 0xFF);
        var right = Solid(2, 2, 10, 10, 10, 0x00);

        Assert.Equal(4, ImageCompare.Compare(left, right).DifferentCount);
        Assert.Equal(0, ImageCompare.Compare(left, right,
            new ImageCompareOptions { CompareAlpha = false }).DifferentCount);
    }

    [Fact]
    public void 違いを塗った画像は元と同じ大きさになる()
    {
        var left = Solid(3, 2, 0, 0, 0);
        var right = Solid(3, 2, 0, 0, 0);
        Set(right, 1, 1, 255, 255, 255);

        var result = ImageCompare.Compare(left, right);
        var highlighted = ImageCompare.Highlight(left, result);

        Assert.Equal(3, highlighted.Width);
        Assert.Equal(2, highlighted.Height);

        // 違う画素は指定した色で塗られる。
        var offset = highlighted.Offset(1, 1);
        Assert.Equal(0x2E, highlighted.Pixels[offset]);
        Assert.Equal(0x22, highlighted.Pixels[offset + 1]);
        Assert.Equal(0xCF, highlighted.Pixels[offset + 2]);

        // 同じ画素は薄くなる（黒 → 白寄り）。埋もれさせないため。
        var same = highlighted.Offset(0, 0);
        Assert.True(highlighted.Pixels[same] > 0x80);
    }

    [Fact]
    public void 画素が多すぎれば黙って固まらずに断る()
    {
        // 実際に確保はしない。大きさだけを申告した画像で判断させる。
        var huge = new PixelImage(10_000, 10_000, []);
        Assert.Throws<InvalidOperationException>(() => ImageCompare.Compare(huge, huge));
    }

    [Fact]
    public void 名前から画像らしさを見分ける()
    {
        Assert.True(ImageCompare.LooksLikeImage("a.PNG"));
        Assert.True(ImageCompare.LooksLikeImage("/tmp/写真.jpeg"));
        Assert.False(ImageCompare.LooksLikeImage("a.txt"));
        Assert.False(ImageCompare.LooksLikeImage("a"));
    }
}
