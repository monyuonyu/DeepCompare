using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// トークン列そのものを Python の tokenizers（正解）と突き合わせる。
///
/// 期待値は `tokenizers` ライブラリで実際に生成したもの。埋め込みの数値だけを見ていると
/// 「なんとなく合っていない」で止まってしまい、原因がトークナイズなのかモデルなのか
/// 切り分けられない。ここで段を分けておく。
/// </summary>
public class WordPieceTokenizerTests
{
    private static WordPieceTokenizer Create()
    {
        var dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "README.md")))
        {
            dir = Path.GetDirectoryName(dir);
        }
        var vocab = Path.Combine(dir!, "assets", "src", "vocab.txt");
        return WordPieceTokenizer.FromVocab(File.OpenRead(vocab));
    }

    /// <summary>
    /// `+` が消えていた件の回帰試験。Microsoft.ML.Tokenizers の BertTokenizer は
    /// これを [101, 2709, 1060, 1015, 102] にしてしまい、`x + 1` と `x - 1` の
    /// 区別が失われていた。
    /// </summary>
    [Fact]
    public void PlusSignSurvives()
    {
        Assert.Equal([101, 2709, 1060, 1009, 1015, 102], Create().Encode("    return x + 1", 512));
    }

    /// <summary>C++ のテンプレートや論理演算子。`&lt;` `&gt;` が落ちると差が見えなくなる。</summary>
    [Fact]
    public void AngleBracketsAndAmpersandsSurvive()
    {
        Assert.Equal(
            [101, 2358, 2094, 1024, 1024, 9207, 1026, 20014, 1028, 1004, 1060, 2015, 1027, 1037, 1004, 1004, 1038, 1025, 102],
            Create().Encode("std::vector<int>& xs = a && b;", 512));
    }

    [Fact]
    public void ParenthesesAndColonsMatchTheReference()
    {
        Assert.Equal([101, 13366, 2364, 1006, 1007, 1024, 102], Create().Encode("def main():", 512));
    }

    /// <summary>
    /// この正規化は非可逆で、日本語の濁点が落ちる（「だ」→「た」）。
    /// 望ましい挙動ではないがモデル本来のもので、参照実装と一致させるには再現が要る。
    /// </summary>
    [Fact]
    public void NormalizationStripsDakutenJustLikeTheReference()
    {
        var tokenizer = Create();
        Assert.Equal("たかは", tokenizer.Normalize("だがぱ"));
        Assert.Equal("cafe", tokenizer.Normalize("café"));
    }

    /// <summary>漢字は 1 文字ずつ独立した語になる。ひらがな・カタカナは繋がったまま。</summary>
    [Fact]
    public void ChineseCharactersAreSeparatedButKanaIsNot()
    {
        Assert.Equal(" 設  定 ", Create().Normalize("設定"));
        Assert.Equal("ファイル", Create().Normalize("ファイル"));
    }

    [Fact]
    public void WhitespaceIsCollapsedToSpacesAndLowercased()
    {
        Assert.Equal("hello  world", Create().Normalize("Hello  World"));
    }

    [Fact]
    public void EmptyInputYieldsOnlySpecialTokens()
    {
        Assert.Equal([101, 102], Create().Encode("", 512));
    }

    /// <summary>切り詰めても [SEP] で終わること。ここが崩れるとモデルの入力として壊れる。</summary>
    [Fact]
    public void TruncationStillEndsWithSeparator()
    {
        var ids = Create().Encode(string.Join(' ', Enumerable.Repeat("hello", 200)), 32);
        Assert.Equal(32, ids.Count);
        Assert.Equal(101, ids[0]);
        Assert.Equal(102, ids[^1]);
    }

    [Fact]
    public void PunctuationClassificationCoversAsciiSymbols()
    {
        foreach (var c in "+<>&=;:(){}[]!@#$%^*-/\\|~`'\"?,.")
        {
            Assert.True(WordPieceTokenizer.IsPunctuation(c), $"{c} が記号として扱われていない");
        }
        Assert.False(WordPieceTokenizer.IsPunctuation('a'));
        Assert.False(WordPieceTokenizer.IsPunctuation('0'));
    }
}
