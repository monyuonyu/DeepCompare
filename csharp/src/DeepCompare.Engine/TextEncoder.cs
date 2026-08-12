using System.Text;

namespace DeepCompare.Engine;

/// <summary>
/// 行の並びをバイト列へ戻す。<see cref="TextDecoder"/> の逆。
///
/// 差分を反映して保存する以上、読んだときの符号化と改行コードを保って書き戻せないと
/// 使えない。Shift_JIS のファイルを開いて 1 行直したら全体が UTF-8 になった、という
/// 壊し方をしないためにここを分けてある。
///
/// 改行が混在していたファイルは、その事実だけが分かっていて元の並びは残っていない。
/// その場合は環境の既定に寄せる。混在を保つには行ごとに改行を覚える必要があり、
/// 「混在は直すべきもの」と考えて割り切っている。
/// </summary>
public static class TextEncoder
{
    static TextEncoder()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
    }

    public static string Newline(LineEnding ending) => ending switch
    {
        LineEnding.Lf => "\n",
        LineEnding.CrLf => "\r\n",
        LineEnding.Cr => "\r",
        LineEnding.None => "\n",
        _ => Environment.NewLine,
    };

    /// <summary>
    /// 元の <paramref name="source"/> と同じ符号化・改行で、<paramref name="lines"/> を
    /// バイト列にする。
    /// </summary>
    public static byte[] Encode(IReadOnlyList<string> lines, DecodedText source)
        => Encode(lines, source.Encoding, source.LineEnding, source.EndsWithNewline);

    /// <param name="endsWithNewline">
    /// 末尾に改行を置くか。読んだファイルに無かったなら足さない。勝手に足すと、
    /// 触っていない最終行が変更として現れる。
    /// </param>
    public static byte[] Encode(
        IReadOnlyList<string> lines, TextEncoding encoding, LineEnding ending,
        bool endsWithNewline = true)
    {
        var text = string.Join(Newline(ending), lines);

        if (endsWithNewline && ending != LineEnding.None && lines.Count > 0)
        {
            text += Newline(ending);
        }

        return encoding switch
        {
            TextEncoding.Utf8 => new UTF8Encoding(false).GetBytes(text),
            TextEncoding.Utf8Lossy => new UTF8Encoding(false).GetBytes(text),
            TextEncoding.Utf8Bom => [.. Encoding.UTF8.GetPreamble(), .. new UTF8Encoding(false).GetBytes(text)],
            TextEncoding.Utf16Le => [.. new UnicodeEncoding(false, true).GetPreamble(),
                                     .. new UnicodeEncoding(false, false).GetBytes(text)],
            TextEncoding.Utf16Be => [.. new UnicodeEncoding(true, true).GetPreamble(),
                                     .. new UnicodeEncoding(true, false).GetBytes(text)],
            TextEncoding.ShiftJis => GetBytesChecked(932, text),
            TextEncoding.EucJp => GetBytesChecked(51932, text),
            _ => new UTF8Encoding(false).GetBytes(text),
        };
    }

    /// <summary>
    /// 表せない文字があれば例外にする。既定の動作は「?」への置換で、黙って情報が
    /// 落ちる。Shift_JIS のファイルへ絵文字を含む行を反映した、といった場合に
    /// 気づけないのは困る。
    /// </summary>
    private static byte[] GetBytesChecked(int codePage, string text)
    {
        var encoding = Encoding.GetEncoding(
            codePage, EncoderFallback.ExceptionFallback, DecoderFallback.ExceptionFallback);
        try
        {
            return encoding.GetBytes(text);
        }
        catch (EncoderFallbackException error)
        {
            throw new InvalidOperationException(
                $"{TextDecoder.Label(codePage == 932 ? TextEncoding.ShiftJis : TextEncoding.EucJp)} "
                + $"で表せない文字がある: '{error.CharUnknown}'。"
                + "保存すると失われるので、符号化を変えるか該当箇所を直すこと。", error);
        }
    }
}
