namespace DeepCompare.Engine;

/// <summary>
/// 画像 1 枚。**画素の並びだけを持つ。**
///
/// 読み込みはここでやらない。PNG や JPEG の復号を自前で書くと、対応形式を
/// 追いかけ続けることになるうえ、画面側（Avalonia）が既に持っている。
/// エンジンは「並んだ画素をどう比べるか」だけを引き受ける。
///
/// 並びは左上から右へ、行ごと。1 画素 4 バイト（B, G, R, A）。
/// **Avalonia の既定の並びに合わせる。** 変換を挟むと、そこが新しい間違いの場所になる。
/// </summary>
public sealed class PixelImage(int width, int height, byte[] pixels)
{
    public int Width { get; } = width;
    public int Height { get; } = height;

    /// <summary>B, G, R, A の順で 4 バイト × 幅 × 高さ。</summary>
    public byte[] Pixels { get; } = pixels;

    public const int BytesPerPixel = 4;

    public bool SameSize(PixelImage other) => Width == other.Width && Height == other.Height;

    /// <summary>(x, y) の画素の、並びの中での位置。</summary>
    public int Offset(int x, int y) => (y * Width + x) * BytesPerPixel;
}

/// <summary>画素 1 つの違いの度合い。</summary>
public enum PixelDifference
{
    /// <summary>完全に同じ。</summary>
    Same,

    /// <summary>違うが、しきい値の内側。</summary>
    Near,

    /// <summary>違う。</summary>
    Different,

    /// <summary>片方にしか無い（大きさが違う）。</summary>
    Missing,
}

public sealed record ImageCompareOptions
{
    /// <summary>
    /// これ以下の差は「同じ」とみなす。0〜255。
    ///
    /// **既定を 0 にしない。** JPEG は保存し直すだけで全画素が 1〜2 変わる。
    /// 0 だと「全部違う」としか出ず、どこを直したのかが読み取れない。
    /// </summary>
    public int Tolerance { get; init; } = 8;

    /// <summary>
    /// 透明度も見るか。
    ///
    /// 既定で見る。**見ないと、透明にしただけの変更が「同じ」になる。**
    /// </summary>
    public bool CompareAlpha { get; init; } = true;
}

/// <summary>2 枚を比べた結果。</summary>
public sealed record ImageComparison(
    int Width,
    int Height,
    /// <summary>画素ごとの違い。並びは画像と同じ（左上から行ごと）。</summary>
    PixelDifference[] Map,
    int SameCount,
    int NearCount,
    int DifferentCount,
    int MissingCount)
{
    public int Total => Width * Height;

    public bool IsIdentical => DifferentCount == 0 && MissingCount == 0 && NearCount == 0;

    /// <summary>しきい値の内側の差だけか。「見た目は同じ」に当たる。</summary>
    public bool LooksSame => DifferentCount == 0 && MissingCount == 0;

    /// <summary>違う画素の割合（0〜1）。</summary>
    public double DifferenceRatio => Total == 0 ? 0 : (double)(DifferentCount + MissingCount) / Total;

    /// <summary>
    /// 違いのある範囲を囲む四角。違いが無ければ null。
    ///
    /// **どこを見ればいいかを一目で示す。** 大きな画像で 1 か所だけ違うとき、
    /// 差分の地図をじっと眺めて探すことになるのを避ける。
    /// </summary>
    public (int Left, int Top, int Right, int Bottom)? Bounds { get; init; }
}

/// <summary>
/// 画像を画素で比べる（BC の Picture Compare に当たる）。
///
/// **大きさが違っても比較を諦めない。** 重なる範囲を比べ、はみ出した分は
/// 「片方にしか無い」とする。諦めると、余白を足しただけの画像で
/// 「全然違う」としか言えなくなる。
/// </summary>
public static class ImageCompare
{
    /// <summary>
    /// 画素数の上限。これを超えるものは比べない。
    ///
    /// 4000 万画素は 4 バイト × 2 枚 + 地図で 400MB 近くになる。
    /// **黙って固まるより、比べられないと言う方がよい。**
    /// </summary>
    public const int MaximumPixels = 40_000_000;

    public static ImageComparison Compare(
        PixelImage left, PixelImage right, ImageCompareOptions? options = null)
    {
        options ??= new ImageCompareOptions();

        // 重なる範囲だけでなく、両方を覆う大きさで地図を作る。
        // はみ出した部分も「片方にしか無い」として残す。
        var width = Math.Max(left.Width, right.Width);
        var height = Math.Max(left.Height, right.Height);

        if ((long)width * height > MaximumPixels)
        {
            throw new InvalidOperationException(
                $"画素が多すぎます（{width}×{height}）。{MaximumPixels:N0} 画素までです。");
        }

        var map = new PixelDifference[width * height];
        var same = 0;
        var near = 0;
        var different = 0;
        var missing = 0;

        var boundsLeft = int.MaxValue;
        var boundsTop = int.MaxValue;
        var boundsRight = int.MinValue;
        var boundsBottom = int.MinValue;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * width + x;
                var inLeft = x < left.Width && y < left.Height;
                var inRight = x < right.Width && y < right.Height;

                if (!inLeft || !inRight)
                {
                    map[index] = PixelDifference.Missing;
                    missing++;
                }
                else
                {
                    map[index] = ComparePixel(left, right, x, y, options);
                    switch (map[index])
                    {
                        case PixelDifference.Same: same++; break;
                        case PixelDifference.Near: near++; break;
                        default: different++; break;
                    }
                }

                if (map[index] is PixelDifference.Different or PixelDifference.Missing)
                {
                    if (x < boundsLeft) { boundsLeft = x; }
                    if (y < boundsTop) { boundsTop = y; }
                    if (x > boundsRight) { boundsRight = x; }
                    if (y > boundsBottom) { boundsBottom = y; }
                }
            }
        }

        return new ImageComparison(width, height, map, same, near, different, missing)
        {
            Bounds = boundsRight < boundsLeft
                ? null
                : (boundsLeft, boundsTop, boundsRight, boundsBottom),
        };
    }

    private static PixelDifference ComparePixel(
        PixelImage left, PixelImage right, int x, int y, ImageCompareOptions options)
    {
        var a = left.Offset(x, y);
        var b = right.Offset(x, y);

        var channels = options.CompareAlpha ? 4 : 3;
        var worst = 0;
        for (var c = 0; c < channels; c++)
        {
            var delta = Math.Abs(left.Pixels[a + c] - right.Pixels[b + c]);
            if (delta > worst)
            {
                worst = delta;
            }
        }

        return worst == 0 ? PixelDifference.Same
            : worst <= options.Tolerance ? PixelDifference.Near
            : PixelDifference.Different;
    }

    /// <summary>
    /// 違いを塗った画像を作る。
    ///
    /// 下敷きは<b>左の画像を薄くしたもの</b>。真っ黒な背景に差分だけ光らせる
    /// 見せ方もあるが、それだと「画像のどのあたりが違うのか」が分からない。
    /// </summary>
    public static PixelImage Highlight(
        PixelImage left, ImageComparison comparison,
        byte differentB = 0x2E, byte differentG = 0x22, byte differentR = 0xCF)
    {
        var pixels = new byte[comparison.Width * comparison.Height * PixelImage.BytesPerPixel];

        for (var y = 0; y < comparison.Height; y++)
        {
            for (var x = 0; x < comparison.Width; x++)
            {
                var index = y * comparison.Width + x;
                var target = index * PixelImage.BytesPerPixel;

                if (comparison.Map[index] is PixelDifference.Different)
                {
                    pixels[target] = differentB;
                    pixels[target + 1] = differentG;
                    pixels[target + 2] = differentR;
                    pixels[target + 3] = 0xFF;
                    continue;
                }

                if (comparison.Map[index] is PixelDifference.Missing)
                {
                    // 片方にしか無い部分。市松にせず、薄い灰色で塗る。
                    pixels[target] = 0xC0;
                    pixels[target + 1] = 0xC0;
                    pixels[target + 2] = 0xC0;
                    pixels[target + 3] = 0xFF;
                    continue;
                }

                // 同じ部分は左の画像を薄く敷く。
                if (x < left.Width && y < left.Height)
                {
                    var source = left.Offset(x, y);
                    pixels[target] = Fade(left.Pixels[source]);
                    pixels[target + 1] = Fade(left.Pixels[source + 1]);
                    pixels[target + 2] = Fade(left.Pixels[source + 2]);
                    pixels[target + 3] = 0xFF;
                }
            }
        }

        return new PixelImage(comparison.Width, comparison.Height, pixels);
    }

    /// <summary>白に寄せて薄くする。差分の色が上に乗ったときに埋もれないように。</summary>
    private static byte Fade(byte value) => (byte)(value + (255 - value) * 3 / 4);

    /// <summary>その名前が画像として扱えそうか。</summary>
    public static bool LooksLikeImage(string path)
    {
        var extension = Path.GetExtension(path).ToLowerInvariant();
        return extension is ".png" or ".jpg" or ".jpeg" or ".bmp" or ".gif" or ".webp" or ".ico";
    }

    /// <summary>人が読む形に整える。</summary>
    public static string Format(ImageComparison comparison)
    {
        if (comparison.IsIdentical)
        {
            return $"{comparison.Width}×{comparison.Height} — 完全に同じです。" + Environment.NewLine;
        }

        var text = new System.Text.StringBuilder();
        text.AppendLine($"大きさ {comparison.Width}×{comparison.Height}（{comparison.Total:N0} 画素）");
        text.AppendLine($"同じ {comparison.SameCount:N0}"
            + $" / しきい値の内 {comparison.NearCount:N0}"
            + $" / 違う {comparison.DifferentCount:N0}"
            + $" / 片方だけ {comparison.MissingCount:N0}");
        text.AppendLine($"違いの割合 {comparison.DifferenceRatio:P2}");

        if (comparison.Bounds is { } bounds)
        {
            // 区切りは ASCII のハイフンにする。**en dash は Windows のコンソールで
            // 「?」に化ける**（実機で確認）。記号を凝る場所ではない。
            text.AppendLine($"違いのある範囲 ({bounds.Left}, {bounds.Top}) - ({bounds.Right}, {bounds.Bottom})");
        }
        if (comparison.LooksSame && !comparison.IsIdentical)
        {
            text.AppendLine("**しきい値の内側の差だけです。** 保存し直しただけの可能性があります。");
        }
        return text.ToString();
    }
}
