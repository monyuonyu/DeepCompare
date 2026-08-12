namespace DeepCompare.Engine;

/// <summary>推定された文字符号化。表示してユーザーが誤りに気付けるよう名前を持たせている。</summary>
public enum TextEncoding
{
    Utf8,
    Utf8Bom,
    Utf16Le,
    Utf16Be,
    ShiftJis,
    EucJp,
    /// <summary>どれとしても妥当に解釈できず、不正な箇所を潰して読んだ場合。</summary>
    Utf8Lossy,
}

/// <summary>支配的な改行コード。</summary>
public enum LineEnding
{
    Lf,
    CrLf,
    Cr,
    /// <summary>混在。「中身は同じなのに全行差分」の典型的な原因なので独立した値にしてある。</summary>
    Mixed,
    /// <summary>改行が一つも無い。</summary>
    None,
}

public sealed record DecodedText(
    IReadOnlyList<string> Lines,
    TextEncoding Encoding,
    LineEnding LineEnding)
{
    /// <summary>
    /// 末尾が改行で終わっているか。
    ///
    /// 行の並びだけでは区別が付かないが、保存するときにこれを間違えると、
    /// 触っていないファイルの最終行を書き換えてしまう。unified 形式でも
    /// 「\ No newline at end of file」を出すかどうかがこれで決まる。
    /// </summary>
    public bool EndsWithNewline { get; init; } = true;
}

/// <summary>
/// ファイルのバイト列を行の並びへ落とす。
///
/// 旧 Python 実装は UTF-8 固定だったので、Windows で保存した Shift_JIS のソースを渡すと
/// 比較開始と同時に例外で終わっていた。ここで符号化を推定し、BOM と改行コードは
/// 復元できる形で剥がしておく。
/// </summary>
public static class TextDecoder
{
    static TextDecoder()
    {
        // Shift_JIS や EUC-JP は .NET Core 以降、既定では登録されていない。
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
    }

    public static string Label(TextEncoding encoding) => encoding switch
    {
        TextEncoding.Utf8 => "UTF-8",
        TextEncoding.Utf8Bom => "UTF-8 (BOM)",
        TextEncoding.Utf16Le => "UTF-16LE",
        TextEncoding.Utf16Be => "UTF-16BE",
        TextEncoding.ShiftJis => "Shift_JIS",
        TextEncoding.EucJp => "EUC-JP",
        TextEncoding.Utf8Lossy => "UTF-8 (不正バイトを置換)",
        _ => "?",
    };

    public static string Label(LineEnding ending) => ending switch
    {
        LineEnding.Lf => "LF",
        LineEnding.CrLf => "CRLF",
        LineEnding.Cr => "CR",
        LineEnding.Mixed => "混在",
        LineEnding.None => "-",
        _ => "?",
    };

    public static DecodedText Decode(ReadOnlySpan<byte> bytes)
    {
        var (text, encoding) = DecodeToString(bytes);
        return new DecodedText(SplitLines(text), encoding, DetectLineEnding(text))
        {
            EndsWithNewline = text.Length == 0 || text[^1] is '\n' or '\r',
        };
    }

    private static (string, TextEncoding) DecodeToString(ReadOnlySpan<byte> bytes)
    {
        // 1. BOM があればそれが最も確かな根拠なので優先する。
        if (bytes.StartsWith(new byte[] { 0xEF, 0xBB, 0xBF }))
        {
            return (new System.Text.UTF8Encoding(false).GetString(bytes[3..]), TextEncoding.Utf8Bom);
        }
        if (bytes.StartsWith(new byte[] { 0xFF, 0xFE }))
        {
            return (System.Text.Encoding.Unicode.GetString(bytes[2..]), TextEncoding.Utf16Le);
        }
        if (bytes.StartsWith(new byte[] { 0xFE, 0xFF }))
        {
            return (System.Text.Encoding.BigEndianUnicode.GetString(bytes[2..]), TextEncoding.Utf16Be);
        }

        // 2. UTF-8 として厳密に妥当なら UTF-8。多バイト列の妥当性は偶然には成立しにくく、
        //    Shift_JIS のテキストが UTF-8 として通ることはまず無いので、この順序で良い。
        if (TryDecodeStrict(bytes, new System.Text.UTF8Encoding(false, true), out var utf8))
        {
            return (utf8, TextEncoding.Utf8);
        }

        // 3. 日本語のレガシー符号化を、誤り無く復号できるものだけ順に試す。
        //    Windows で書かれたソースを想定して Shift_JIS を先に見る。
        // EUC-JP は 51932 を使う。20932 も "euc-jp" を名乗るが、不正なバイト列を
        // 黙って受理してしまう（例: 0xC3 0x28 を「構」として読む）。正しい日本語は
        // どちらも同じバイト列で往復するため、正常系では違いが出ず、壊れた入力を
        // 与えたときにだけ誤判定として表面化する。
        foreach (var (codePage, tag) in new[] { (932, TextEncoding.ShiftJis), (51932, TextEncoding.EucJp) })
        {
            var encoding = System.Text.Encoding.GetEncoding(
                codePage,
                System.Text.EncoderFallback.ExceptionFallback,
                System.Text.DecoderFallback.ExceptionFallback);
            // 「復号でエラーが出なかった」だけでは根拠として弱い。Shift_JIS は単バイトで
            // 受け付ける範囲が広く、latin-1 のテキストやバイナリでもほぼエラー無しに
            // 通ってしまうため、結果が日本語文として妥当かどうかまで見る。
            if (TryDecodeStrict(bytes, encoding, out var legacy) && LooksLikeJapaneseText(legacy))
            {
                return (legacy, tag);
            }
        }

        // 4. どれでもない。読めるところまで読む方が、開けないより有用。
        return (new System.Text.UTF8Encoding(false).GetString(bytes), TextEncoding.Utf8Lossy);
    }

    private static bool TryDecodeStrict(ReadOnlySpan<byte> bytes, System.Text.Encoding encoding, out string text)
    {
        try
        {
            text = encoding.GetString(bytes);
            return true;
        }
        catch (System.Text.DecoderFallbackException)
        {
            text = string.Empty;
            return false;
        }
        catch (ArgumentException)
        {
            // 一部の符号化は不正な並びを ArgumentException で返す。
            text = string.Empty;
            return false;
        }
    }

    /// <summary>
    /// レガシー符号化として復号した結果が、本当にその符号化の文章だったかを判定する。
    ///
    /// 判断材料は二つ。制御文字が混ざっていればテキストではなくバイナリを読んでいる。
    /// 非 ASCII の大半が半角カタカナなら、それは日本語ではなく誤復号の兆候で、
    /// latin-1 の上位バイトや任意のバイナリを Shift_JIS として読むと 0xA1..0xDF が
    /// すべて半角カタカナに化けるためこの形になる。
    ///
    /// 代償として、本当に半角カタカナばかりのファイルは取りこぼす。ソースコードの
    /// 比較という用途ではまず出てこない形なので、誤検出を減らす側を採る。
    /// </summary>
    private static bool LooksLikeJapaneseText(string text)
    {
        var nonAscii = 0;
        var halfwidthKatakana = 0;
        foreach (var c in text)
        {
            if (char.IsControl(c) && c is not ('\t' or '\n' or '\r'))
            {
                return false;
            }
            if (c > 0x7F)
            {
                nonAscii++;
                if (c is >= '｡' and <= 'ﾟ')
                {
                    halfwidthKatakana++;
                }
            }
        }
        // 非 ASCII が無いなら、そもそも UTF-8 として通っていたはず。ここへ来る時点で
        // 何か食い違っているので採用しない。
        if (nonAscii == 0)
        {
            return false;
        }
        return halfwidthKatakana * 2 <= nonAscii;
    }

    private static LineEnding DetectLineEnding(string text)
    {
        int crlf = 0, lf = 0, cr = 0;
        for (var i = 0; i < text.Length; i++)
        {
            if (text[i] == '\r')
            {
                if (i + 1 < text.Length && text[i + 1] == '\n')
                {
                    crlf++;
                    i++;
                    continue;
                }
                cr++;
            }
            else if (text[i] == '\n')
            {
                lf++;
            }
        }
        return (crlf > 0, lf > 0, cr > 0) switch
        {
            (false, false, false) => LineEnding.None,
            (true, false, false) => LineEnding.CrLf,
            (false, true, false) => LineEnding.Lf,
            (false, false, true) => LineEnding.Cr,
            _ => LineEnding.Mixed,
        };
    }

    /// <summary>
    /// CRLF / LF / CR のいずれでも分割する。末尾の改行は空行を生まない
    /// （Python の str.splitlines() と同じ扱い）。
    /// </summary>
    private static List<string> SplitLines(string text)
    {
        var lines = new List<string>();
        var start = 0;
        for (var i = 0; i < text.Length; i++)
        {
            if (text[i] == '\r')
            {
                lines.Add(text[start..i]);
                if (i + 1 < text.Length && text[i + 1] == '\n')
                {
                    i++;
                }
                start = i + 1;
            }
            else if (text[i] == '\n')
            {
                lines.Add(text[start..i]);
                start = i + 1;
            }
        }
        if (start < text.Length)
        {
            lines.Add(text[start..]);
        }
        return lines;
    }
}
