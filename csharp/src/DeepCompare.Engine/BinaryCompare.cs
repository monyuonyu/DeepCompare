using System.Text;

namespace DeepCompare.Engine;

/// <summary>16 進表示の 1 行分。</summary>
public sealed record HexRow(
    /// <summary>左のオフセット。左に対応が無ければ null。</summary>
    long? LeftOffset,
    long? RightOffset,
    /// <summary>その行のバイト列。無い側は空。</summary>
    IReadOnlyList<byte> LeftBytes,
    IReadOnlyList<byte> RightBytes,
    /// <summary>左右で違うバイトの位置（行の中での添字）。</summary>
    IReadOnlyList<int> ChangedColumns)
{
    public bool IsUnchanged => LeftOffset is not null && RightOffset is not null
        && ChangedColumns.Count == 0;

    /// <summary>16 進の並び。無い側は空文字。</summary>
    public string Hex(bool left)
    {
        var bytes = left ? LeftBytes : RightBytes;
        if (bytes.Count == 0)
        {
            return string.Empty;
        }
        var builder = new StringBuilder(bytes.Count * 3);
        for (var i = 0; i < bytes.Count; i++)
        {
            if (i > 0)
            {
                // 8 バイトごとに区切りを広げる。数えやすさが段違いに変わる。
                builder.Append(i % 8 == 0 ? "  " : " ");
            }
            builder.Append(bytes[i].ToString("X2"));
        }
        return builder.ToString();
    }

    /// <summary>
    /// 文字として読める部分。読めないバイトは <c>.</c> にする。
    ///
    /// **符号化は決め打たない。** ここは「バイトを見る」ための表示なので、
    /// ASCII の範囲だけを文字にして、それ以外は点で埋める。日本語を読みたい
    /// ならテキスト比較の側を使う。
    /// </summary>
    public string Ascii(bool left)
    {
        var bytes = left ? LeftBytes : RightBytes;
        if (bytes.Count == 0)
        {
            return string.Empty;
        }
        var builder = new StringBuilder(bytes.Count);
        foreach (var b in bytes)
        {
            builder.Append(b is >= 0x20 and < 0x7F ? (char)b : '.');
        }
        return builder.ToString();
    }
}

public sealed record BinaryComparison(
    IReadOnlyList<HexRow> Rows,
    long LeftLength,
    long RightLength)
{
    public int DifferentRows => Rows.Count(r => !r.IsUnchanged);

    public bool Identical => DifferentRows == 0 && LeftLength == RightLength;
}

/// <summary>
/// バイト列としての比較（16 進表示）。
///
/// **テキストとして読めないファイルのための出口。** 実行ファイル、画像、
/// 独自形式のデータ。テキスト比較に掛けると符号化の推定が外れて意味の無い
/// 差分が出るか、そもそも読めない。
///
/// 行の対応付けは Myers で行う。**単純に先頭から 16 バイトずつ突き合わせない。**
/// 1 バイト挿入されただけで以降が全部ずれ、全行が差分になってしまう。
/// </summary>
public static class BinaryCompare
{
    /// <summary>1 行に並べるバイト数。16 が慣習で、桁が揃って読みやすい。</summary>
    public const int BytesPerRow = 16;

    /// <summary>
    /// 読み込む上限。既定 64MB。
    ///
    /// 16 進で出す以上、人が読める量には限りがある。上限を持たないと、
    /// 数 GB のファイルを開いた瞬間に落ちる。
    /// </summary>
    public const long DefaultMaximumBytes = 64L * 1024 * 1024;

    public static BinaryComparison Compare(
        ReadOnlySpan<byte> left, ReadOnlySpan<byte> right)
    {
        var leftRows = Split(left);
        var rightRows = Split(right);

        // 行の中身をそのまま比べる鍵にする。16 バイトの並びが一致すれば同じ行。
        var leftKeys = leftRows.Select(Key).ToArray();
        var rightKeys = rightRows.Select(Key).ToArray();

        var rows = new List<HexRow>();
        foreach (var op in Myers.Compute(leftKeys, rightKeys))
        {
            switch (op.Kind)
            {
                case DiffKind.Equal:
                    for (var i = 0; i < op.OldLength; i++)
                    {
                        var l = op.OldStart + i;
                        var r = op.NewStart + i;
                        rows.Add(new HexRow(
                            (long)l * BytesPerRow, (long)r * BytesPerRow,
                            leftRows[l], rightRows[r], []));
                    }
                    break;

                case DiffKind.Replace:
                {
                    // 置き換えは行どうしを組にして、違うバイトの位置まで出す。
                    var shared = Math.Min(op.OldLength, op.NewLength);
                    for (var i = 0; i < shared; i++)
                    {
                        var l = op.OldStart + i;
                        var r = op.NewStart + i;
                        rows.Add(new HexRow(
                            (long)l * BytesPerRow, (long)r * BytesPerRow,
                            leftRows[l], rightRows[r], Differing(leftRows[l], rightRows[r])));
                    }
                    for (var i = shared; i < op.OldLength; i++)
                    {
                        var l = op.OldStart + i;
                        rows.Add(new HexRow((long)l * BytesPerRow, null, leftRows[l], [], []));
                    }
                    for (var i = shared; i < op.NewLength; i++)
                    {
                        var r = op.NewStart + i;
                        rows.Add(new HexRow(null, (long)r * BytesPerRow, [], rightRows[r], []));
                    }
                    break;
                }

                case DiffKind.Delete:
                    for (var i = 0; i < op.OldLength; i++)
                    {
                        var l = op.OldStart + i;
                        rows.Add(new HexRow((long)l * BytesPerRow, null, leftRows[l], [], []));
                    }
                    break;

                case DiffKind.Insert:
                    for (var i = 0; i < op.NewLength; i++)
                    {
                        var r = op.NewStart + i;
                        rows.Add(new HexRow(null, (long)r * BytesPerRow, [], rightRows[r], []));
                    }
                    break;
            }
        }

        return new BinaryComparison(rows, left.Length, right.Length);
    }

    /// <summary>ファイルから読んで比べる。上限を超える分は切り捨て、その旨を返す。</summary>
    public static (BinaryComparison Comparison, bool Truncated) CompareFiles(
        string leftPath, string rightPath, long maximumBytes = DefaultMaximumBytes)
    {
        var (left, leftCut) = Read(leftPath, maximumBytes);
        var (right, rightCut) = Read(rightPath, maximumBytes);
        return (Compare(left, right), leftCut || rightCut);
    }

    private static (byte[] Bytes, bool Truncated) Read(string path, long maximum)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length <= maximum)
        {
            var all = new byte[stream.Length];
            stream.ReadExactly(all);
            return (all, false);
        }

        var partial = new byte[maximum];
        stream.ReadExactly(partial);
        return (partial, true);
    }

    private static List<byte[]> Split(ReadOnlySpan<byte> data)
    {
        var rows = new List<byte[]>((data.Length + BytesPerRow - 1) / BytesPerRow);
        for (var offset = 0; offset < data.Length; offset += BytesPerRow)
        {
            var length = Math.Min(BytesPerRow, data.Length - offset);
            rows.Add(data.Slice(offset, length).ToArray());
        }
        return rows;
    }

    /// <summary>行の中身を 1 本の文字列にする。Myers に渡す鍵。</summary>
    private static string Key(byte[] row) => System.Convert.ToHexString(row);

    private static List<int> Differing(byte[] left, byte[] right)
    {
        var result = new List<int>();
        var length = Math.Max(left.Length, right.Length);
        for (var i = 0; i < length; i++)
        {
            var l = i < left.Length ? left[i] : (int?)null;
            var r = i < right.Length ? right[i] : (int?)null;
            if (l != r)
            {
                result.Add(i);
            }
        }
        return result;
    }

    /// <summary>
    /// テキストとして扱えなさそうか。
    ///
    /// **判定は控えめにする。** 誤ってテキストを「バイナリ」と言うと、せっかくの
    /// 意味的な比較が使えなくなる。NUL バイトがある場合だけ確実に非テキストと
    /// みなす（UTF-16 は BOM で判別できるので、そちらは除く）。
    /// </summary>
    public static bool LooksBinary(string path, int sampleSize = 8000)
    {
        try
        {
            using var stream = File.OpenRead(path);
            var buffer = new byte[Math.Min(sampleSize, stream.Length)];
            var read = stream.Read(buffer, 0, buffer.Length);
            if (read < 2)
            {
                return false;
            }

            // UTF-16 は NUL を含むが、テキストとして読める。BOM を先に見る。
            if ((buffer[0] == 0xFF && buffer[1] == 0xFE) || (buffer[0] == 0xFE && buffer[1] == 0xFF))
            {
                return false;
            }

            return buffer.AsSpan(0, read).IndexOf((byte)0) >= 0;
        }
        catch (IOException)
        {
            return false;
        }
    }

    /// <summary>CLI 向けの整形。</summary>
    public static string Format(BinaryComparison comparison, int limit = 2000)
    {
        var text = new StringBuilder();
        text.AppendLine($"left  {comparison.LeftLength} バイト");
        text.AppendLine($"right {comparison.RightLength} バイト");
        text.AppendLine("legend = 一致 / ~ 違う / - 左のみ / + 右のみ");
        text.AppendLine("---");

        var shown = 0;
        foreach (var row in comparison.Rows)
        {
            if (shown++ >= limit)
            {
                text.AppendLine($"（残り {comparison.Rows.Count - limit} 行は省略）");
                break;
            }

            var mark = (row.LeftOffset, row.RightOffset) switch
            {
                (not null, not null) when row.IsUnchanged => '=',
                (not null, not null) => '~',
                (not null, null) => '-',
                _ => '+',
            };
            var offset = row.LeftOffset ?? row.RightOffset ?? 0;
            text.AppendLine(
                $"{mark} {offset:X8}  {row.Hex(true),-49}|{row.Ascii(true),-16}"
                + $"  {row.Hex(false),-49}|{row.Ascii(false),-16}");
        }

        text.AppendLine("---");
        text.AppendLine(comparison.Identical
            ? "同じ内容です。"
            : $"{comparison.DifferentRows} 行が違います。");
        return text.ToString();
    }
}
