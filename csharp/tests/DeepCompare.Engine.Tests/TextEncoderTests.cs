using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 書き戻し。差分を反映して保存する以上、読んだ形を保てないと使えない。
/// 「往復させて元に戻る」ことを軸に固定する。
/// </summary>
public sealed class TextEncoderTests
{
    [Theory]
    [InlineData(TextEncoding.Utf8, LineEnding.Lf)]
    [InlineData(TextEncoding.Utf8, LineEnding.CrLf)]
    [InlineData(TextEncoding.Utf8Bom, LineEnding.CrLf)]
    [InlineData(TextEncoding.Utf16Le, LineEnding.Lf)]
    [InlineData(TextEncoding.Utf16Be, LineEnding.CrLf)]
    [InlineData(TextEncoding.ShiftJis, LineEnding.CrLf)]
    [InlineData(TextEncoding.EucJp, LineEnding.Lf)]
    public void RoundTripPreservesLinesEncodingAndLineEnding(
        TextEncoding encoding, LineEnding ending)
    {
        string[] lines = ["日本語の行", "ascii line", "記号 <&> \"'"];

        var bytes = TextEncoder.Encode(lines, encoding, ending);
        var decoded = TextDecoder.Decode(bytes);

        Assert.Equal(lines, decoded.Lines);
        Assert.Equal(encoding, decoded.Encoding);
        Assert.Equal(ending, decoded.LineEnding);
    }

    /// <summary>読んだものをそのまま書けばバイト列まで一致すること。</summary>
    [Fact]
    public void EncodingWhatWasDecodedReproducesTheBytes()
    {
        var original = "a\r\nb\r\n"u8.ToArray();
        var decoded = TextDecoder.Decode(original);

        Assert.Equal(original, TextEncoder.Encode(decoded.Lines, decoded));
    }

    [Fact]
    public void FilesWithoutATrailingNewlineDoNotGainOne()
    {
        var decoded = TextDecoder.Decode("only line"u8.ToArray());
        Assert.Equal(LineEnding.None, decoded.LineEnding);

        Assert.Equal("only line"u8.ToArray(), TextEncoder.Encode(decoded.Lines, decoded));
    }

    /// <summary>
    /// Shift_JIS で表せない文字は黙って落とさず、どの文字かを示して失敗すること。
    /// 既定の動作は「?」への置換で、気づかないうちに情報が消える。
    /// </summary>
    [Fact]
    public void CharactersThatCannotBeEncodedFailLoudly()
    {
        var error = Assert.Throws<InvalidOperationException>(
            () => TextEncoder.Encode(["絵文字 🙂 入り"], TextEncoding.ShiftJis, LineEnding.Lf));

        Assert.Contains("Shift_JIS", error.Message);
    }

    [Fact]
    public void EmptyInputProducesEmptyOutput()
    {
        Assert.Empty(TextEncoder.Encode([], TextEncoding.Utf8, LineEnding.Lf));
    }

    /// <summary>
    /// 末尾に改行が無いファイルへ改行を足さないこと。足すと、触っていない最終行が
    /// 変更として現れる。git が「\ No newline at end of file」と出す状態のファイルで起きる。
    /// </summary>
    [Fact]
    public void FilesEndingWithoutANewlineStayThatWay()
    {
        var original = "a\nb"u8.ToArray();
        var decoded = TextDecoder.Decode(original);

        Assert.False(decoded.EndsWithNewline);
        Assert.Equal(["a", "b"], decoded.Lines);
        Assert.Equal(original, TextEncoder.Encode(decoded.Lines, decoded));
    }

    [Fact]
    public void FilesEndingWithANewlineKeepIt()
    {
        var original = "a\nb\n"u8.ToArray();
        var decoded = TextDecoder.Decode(original);

        Assert.True(decoded.EndsWithNewline);
        Assert.Equal(original, TextEncoder.Encode(decoded.Lines, decoded));
    }

    [Fact]
    public void CrLfWithoutATrailingNewlineRoundTrips()
    {
        var original = "a\r\nb"u8.ToArray();
        var decoded = TextDecoder.Decode(original);

        Assert.Equal(original, TextEncoder.Encode(decoded.Lines, decoded));
    }
}
