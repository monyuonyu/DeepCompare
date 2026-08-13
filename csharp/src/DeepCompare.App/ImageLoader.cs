using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using SkiaSharp;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 画像を読んで画素の並びにする。
///
/// **復号は Avalonia に任せる。** PNG や JPEG の復号を engine に自前で書くと、
/// 対応形式を追いかけ続けることになる。画面側は既に持っているので、
/// ここで橋渡しだけする（engine は「並んだ画素をどう比べるか」に専念する）。
/// </summary>
public static class ImageLoader
{
    /// <summary>
    /// 読んで <see cref="PixelImage"/> にする。
    ///
    /// 並びは B, G, R, A。**Avalonia の既定に合わせる。** 変換を挟むと
    /// そこが新しい間違いの場所になる。
    ///
    /// **Skia を直に呼ぶ。** Avalonia の <see cref="Bitmap"/> は描画基盤の
    /// 初期化を要求するので、画面を開かない CLI からは使えない
    /// （"Unable to locate IPlatformRenderInterface" で落ちる）。
    /// Skia は Avalonia が既に連れてきているので、依存は増えない。
    /// </summary>
    public static PixelImage Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var codec = SKCodec.Create(stream)
            ?? throw new NotSupportedException($"{path} を画像として読めませんでした。");

        var info = codec.Info;
        if ((long)info.Width * info.Height > ImageCompare.MaximumPixels)
        {
            throw new InvalidOperationException(
                $"画素が多すぎます（{info.Width}×{info.Height}）。"
                + $"{ImageCompare.MaximumPixels:N0} 画素までです。");
        }

        // **透明度を掛け合わせない形（Unpremul）で受け取る。** 掛け合わせた形だと
        // 透明な部分の色が失われ、「透明にしただけ」と「色も変えた」が
        // 区別できなくなる。
        var target = new SKImageInfo(
            info.Width, info.Height, SKColorType.Bgra8888, SKAlphaType.Unpremul);

        var pixels = new byte[target.Width * target.Height * PixelImage.BytesPerPixel];
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(
            pixels, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            var result = codec.GetPixels(target, handle.AddrOfPinnedObject());
            if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
            {
                throw new NotSupportedException($"{path} を復号できませんでした（{result}）。");
            }
        }
        finally
        {
            handle.Free();
        }

        return new PixelImage(target.Width, target.Height, pixels);
    }

    public static PixelImage FromBitmap(Bitmap bitmap)
    {
        var width = bitmap.PixelSize.Width;
        var height = bitmap.PixelSize.Height;

        if ((long)width * height > ImageCompare.MaximumPixels)
        {
            throw new InvalidOperationException(
                $"画素が多すぎます（{width}×{height}）。{ImageCompare.MaximumPixels:N0} 画素までです。");
        }

        var stride = width * PixelImage.BytesPerPixel;
        var pixels = new byte[stride * height];

        // **透明度を掛け合わせない形（Unpremul）で受け取る。** 掛け合わせた形だと
        // 透明な部分の色が失われ、「透明にしただけ」と「色も変えた」が
        // 区別できなくなる。
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(
            pixels, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            bitmap.CopyPixels(
                new PixelRect(0, 0, width, height),
                handle.AddrOfPinnedObject(), pixels.Length, stride);
        }
        finally
        {
            handle.Free();
        }

        return new PixelImage(width, height, pixels);
    }

    /// <summary>比べた結果を、そのまま画面に出せる画像にする。</summary>
    public static WriteableBitmap ToBitmap(PixelImage image)
    {
        var bitmap = new WriteableBitmap(
            new PixelSize(image.Width, image.Height),
            new Vector(96, 96),
            PixelFormats.Bgra8888,
            AlphaFormat.Unpremul);

        using var buffer = bitmap.Lock();
        var stride = image.Width * PixelImage.BytesPerPixel;
        for (var y = 0; y < image.Height; y++)
        {
            System.Runtime.InteropServices.Marshal.Copy(
                image.Pixels, y * stride,
                buffer.Address + y * buffer.RowBytes, stride);
        }
        return bitmap;
    }
}
