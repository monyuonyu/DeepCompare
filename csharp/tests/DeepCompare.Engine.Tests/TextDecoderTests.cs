using Xunit;
using System.Text;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// Rust 版 text.rs の試験をそのまま移したもの。同じ入力で同じ判定になることを担保する。
/// </summary>
public class TextDecoderTests
{
    public TextDecoderTests()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
    }

    private static byte[] Encode(int codePage, string text) => Encoding.GetEncoding(codePage).GetBytes(text);

    [Fact]
    public void PlainUtf8IsDetected()
    {
        var d = TextDecoder.Decode(Encoding.UTF8.GetBytes("あいう\nかきく\n"));
        Assert.Equal(TextEncoding.Utf8, d.Encoding);
        Assert.Equal(["あいう", "かきく"], d.Lines);
        Assert.Equal(LineEnding.Lf, d.LineEnding);
    }

    /// <summary>BOM が本文に残ると 1 行目だけが必ず差分として出てしまう。</summary>
    [Fact]
    public void Utf8BomIsStrippedNotLeftOnTheFirstLine()
    {
        var bytes = new List<byte> { 0xEF, 0xBB, 0xBF };
        bytes.AddRange("fn main() {}\n"u8.ToArray());
        var d = TextDecoder.Decode(bytes.ToArray());
        Assert.Equal(TextEncoding.Utf8Bom, d.Encoding);
        Assert.Equal(["fn main() {}"], d.Lines);
    }

    /// <summary>旧実装がここで例外を投げて比較そのものを諦めていた入力。</summary>
    [Fact]
    public void ShiftJisIsReadInsteadOfFailing()
    {
        var d = TextDecoder.Decode(Encode(932, "日本語のコメント\nprint(1)\n"));
        Assert.Equal(TextEncoding.ShiftJis, d.Encoding);
        Assert.Equal(["日本語のコメント", "print(1)"], d.Lines);
    }

    [Fact]
    public void Utf16LeWithBomIsRead()
    {
        var bytes = new List<byte> { 0xFF, 0xFE };
        bytes.AddRange(Encoding.Unicode.GetBytes("abc\ndef\n"));
        var d = TextDecoder.Decode(bytes.ToArray());
        Assert.Equal(TextEncoding.Utf16Le, d.Encoding);
        Assert.Equal(["abc", "def"], d.Lines);
    }

    [Fact]
    public void Utf16BeWithBomIsRead()
    {
        var bytes = new List<byte> { 0xFE, 0xFF };
        bytes.AddRange(Encoding.BigEndianUnicode.GetBytes("abc\ndef\n"));
        var d = TextDecoder.Decode(bytes.ToArray());
        Assert.Equal(TextEncoding.Utf16Be, d.Encoding);
        Assert.Equal(["abc", "def"], d.Lines);
    }

    /// <summary>改行コードだけが違うファイルが全行差分にならないことの担保。</summary>
    [Fact]
    public void CrlfAndLfProduceIdenticalLines()
    {
        var lf = TextDecoder.Decode("a\nb\nc\n"u8);
        var crlf = TextDecoder.Decode("a\r\nb\r\nc\r\n"u8);
        Assert.Equal(lf.Lines, crlf.Lines);
        Assert.Equal(LineEnding.Lf, lf.LineEnding);
        Assert.Equal(LineEnding.CrLf, crlf.LineEnding);
    }

    [Fact]
    public void MixedLineEndingsAreReported()
        => Assert.Equal(LineEnding.Mixed, TextDecoder.Decode("a\r\nb\nc"u8).LineEnding);

    [Fact]
    public void TrailingNewlineDoesNotCreateAnEmptyLine()
    {
        Assert.Equal(["a", "b"], TextDecoder.Decode("a\nb\n"u8).Lines);
        Assert.Equal(["a", "b"], TextDecoder.Decode("a\nb"u8).Lines);
    }

    [Fact]
    public void BlankLinesInTheMiddleAreKept()
        => Assert.Equal(["a", "", "b"], TextDecoder.Decode("a\n\nb\n"u8).Lines);

    [Fact]
    public void EmptyInputYieldsNoLines()
    {
        var d = TextDecoder.Decode([]);
        Assert.Empty(d.Lines);
        Assert.Equal(LineEnding.None, d.LineEnding);
    }

    /// <summary>開けずに終わるより、読めるところまで見せる方が使える。</summary>
    [Fact]
    public void UndecodableBytesStillYieldText()
    {
        var d = TextDecoder.Decode([(byte)'a', 0xC3, 0x28, (byte)'b']);
        Assert.Equal(TextEncoding.Utf8Lossy, d.Encoding);
        Assert.Single(d.Lines);
    }

    /// <summary>
    /// Shift_JIS は 0xA1..0xDF を単バイトの半角カタカナとして受け入れるので、
    /// 「復号エラーが出ない」だけを根拠にすると latin-1 が丸ごと化けて通る。
    /// </summary>
    [Fact]
    public void Latin1IsNotMistakenForShiftJis()
    {
        byte[] latin1 = [.. "caf"u8, 0xE9, .. " na"u8, 0xEF, .. "ve r"u8, 0xE9, .. "sum"u8, 0xE9];
        Assert.NotEqual(TextEncoding.ShiftJis, TextDecoder.Decode(latin1).Encoding);
    }

    [Fact]
    public void BinaryInputIsNotMistakenForTextEncoding()
    {
        byte[] binary = [0x00, 0x01, 0xB1, 0xB2, 0x02, 0xC0];
        var encoding = TextDecoder.Decode(binary).Encoding;
        Assert.NotEqual(TextEncoding.ShiftJis, encoding);
        Assert.NotEqual(TextEncoding.EucJp, encoding);
    }

    /// <summary>誤検出を潰す過程で、本来通すべき日本語まで弾いていないことの確認。</summary>
    [Fact]
    public void JapaneseShiftJisSurvivesTheStricterCheck()
    {
        var bytes = Encode(932, "// 設定を読み込む\nlet 名前 = \"太郎\";\n");
        Assert.Equal(TextEncoding.ShiftJis, TextDecoder.Decode(bytes).Encoding);
    }

    [Fact]
    public void EucJpIsDetected()
    {
        var d = TextDecoder.Decode(Encode(51932, "日本語の文章です\n二行目\n"));
        Assert.Equal(TextEncoding.EucJp, d.Encoding);
        Assert.Equal(["日本語の文章です", "二行目"], d.Lines);
    }
}
