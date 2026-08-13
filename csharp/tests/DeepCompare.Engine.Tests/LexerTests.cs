using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class LexerTests
{
    private static readonly Language CSharp = Lexer.ForPath("a.cs")!;
    private static readonly Language Python = Lexer.ForPath("a.py")!;

    private static List<string> Text(string line, Language? language)
        => [.. Lexer.Tokenize(line, language).Select(t => line.Substring(t.Start, t.Length))];

    private static List<TokenKind> Kinds(string line, Language? language)
        => [.. Lexer.Tokenize(line, language).Select(t => t.Kind)];

    /// <summary>分解したものを繋ぎ直すと元の行に戻ること。抜けや重複があると表示が壊れる。</summary>
    [Theory]
    [InlineData("var x = 1; // ほげ")]
    [InlineData("if (a != null) { b.c(\"文字列\"); }")]
    [InlineData("    // 全部コメント")]
    [InlineData("\"unterminated")]
    [InlineData("")]
    [InlineData("日本語だけの行")]
    public void TokensCoverTheWholeLineWithoutGaps(string line)
    {
        Assert.Equal(line, string.Concat(Text(line, CSharp)));
    }

    [Fact]
    public void UnknownLanguagesProduceOneSpanForTheWholeLine()
    {
        var tokens = Lexer.Tokenize("anything at all", null);

        var token = Assert.Single(tokens);
        Assert.Equal(TokenKind.Text, token.Kind);
        Assert.Equal(15, token.Length);
    }

    [Fact]
    public void KeywordsAreSeparatedFromIdentifiers()
    {
        var kinds = Kinds("return value", CSharp);

        Assert.Equal([TokenKind.Keyword, TokenKind.Whitespace, TokenKind.Identifier], kinds);
    }

    [Fact]
    public void LineCommentsRunToTheEnd()
    {
        var tokens = Lexer.Tokenize("x = 1; // これは \"文字列\" ではない", CSharp);

        var comment = Assert.Single(tokens, t => t.Kind == TokenKind.Comment);
        Assert.Contains("\"文字列\"", "x = 1; // これは \"文字列\" ではない"[comment.Start..]);
    }

    /// <summary>逃がし文字の次の引用符で閉じないこと。</summary>
    [Fact]
    public void EscapedQuotesDoNotCloseTheString()
    {
        var line = "var s = \"a\\\"b\"; var t = 1;";
        var strings = Lexer.Tokenize(line, CSharp).Where(t => t.Kind == TokenKind.String).ToList();

        var single = Assert.Single(strings);
        Assert.Equal("\"a\\\"b\"", line.Substring(single.Start, single.Length));
    }

    [Fact]
    public void UnterminatedStringsRunToTheEndOfTheLine()
    {
        var line = "var s = \"open";
        var token = Assert.Single(Lexer.Tokenize(line, CSharp), t => t.Kind == TokenKind.String);

        Assert.Equal("\"open", line.Substring(token.Start, token.Length));
    }

    [Fact]
    public void BlockCommentsClosingOnTheSameLineDoNotLeakState()
    {
        var state = LexState.Start;
        Lexer.Tokenize("a /* c */ b", CSharp, ref state);

        Assert.False(state.InBlockComment);
    }

    /// <summary>閉じないブロックコメントは次の行へ引き継ぐこと。</summary>
    [Fact]
    public void BlockCommentStateCarriesToTheNextLine()
    {
        var state = LexState.Start;
        Lexer.Tokenize("a /* start", CSharp, ref state);
        Assert.True(state.InBlockComment);

        var middle = Lexer.Tokenize("まだコメント", CSharp, ref state);
        Assert.Equal(TokenKind.Comment, Assert.Single(middle).Kind);
        Assert.True(state.InBlockComment);

        Lexer.Tokenize("end */ code", CSharp, ref state);
        Assert.False(state.InBlockComment);
    }

    /// <summary>
    /// 開始と終了が同じ綴りの場合（Python の三重引用符）に、自分自身で閉じたと
    /// 誤らないこと。
    /// </summary>
    [Fact]
    public void BlockCommentsWithTheSameOpenAndCloseNeedTwoOccurrences()
    {
        var state = LexState.Start;
        Lexer.Tokenize("\"\"\"始まり", Python, ref state);

        Assert.True(state.InBlockComment);
    }

    [Fact]
    public void NumbersAreOneToken()
    {
        var tokens = Lexer.Tokenize("x = 3.14;", CSharp);

        var number = Assert.Single(tokens, t => t.Kind == TokenKind.Number);
        Assert.Equal(4, number.Length);
    }

    [Theory]
    [InlineData("a.cs", true)]
    [InlineData("a.py", true)]
    [InlineData("a.yaml", true)]
    [InlineData("a.axaml", true)]
    [InlineData("a.unknownext", false)]
    [InlineData("noextension", false)]
    public void LanguagesAreChosenByExtension(string path, bool expected)
    {
        Assert.Equal(expected, Lexer.ForPath(path) is not null);
    }

    // ---- 行内差分がトークン単位になること ----

    /// <summary>
    /// 綴りの似た語の書き換えは、文字単位だと共通部分を拾って断片的に光る。
    /// `Update` → `Upgrade` は `Up` と `ade` が共通なので、左は 2 つ以上に割れる。
    /// トークン単位なら語まるごと 1 つになる。
    /// </summary>
    [Fact]
    public void TokenBasedInlineDiffHighlightsWholeWords()
    {
        const string left = "data.Update(15);";
        const string right = "data.Upgrade(15);";

        var (byChar, _) = InlineDiff.Compute(left, right);
        var (byToken, _) = InlineDiff.Compute(left, right, CSharp);

        var changedByChar = byChar.Count(s => s.Kind == SpanKind.Changed);
        var changedByToken = byToken.Count(s => s.Kind == SpanKind.Changed);

        Assert.True(changedByChar > changedByToken,
            $"文字単位 {changedByChar} 個 / トークン単位 {changedByToken} 個");

        // 変わったのは語まるごと 1 つ。
        var span = Assert.Single(byToken, s => s.Kind == SpanKind.Changed);
        Assert.Equal("Update", left.Substring(span.Start, span.Length));
    }

    [Fact]
    public void TokenBasedInlineDiffStillCoversTheWholeLine()
    {
        const string left = "if (a == 1) { return x; }";
        const string right = "if (b == 2) { return y; }";

        var (spans, _) = InlineDiff.Compute(left, right, CSharp);

        Assert.Equal(left.Length, spans.Sum(s => s.Length));
        Assert.Equal(0, spans[0].Start);
    }
}

/// <summary>
/// 言語の判定。**拡張子だけでは足りない。**
/// Makefile や Dockerfile は実際に比べる機会が多いのに、名前で判定しないと
/// 色分けが効かない。
/// </summary>
public sealed class LexerLanguageTests
{
    [Theory]
    [InlineData("a.lua", "Lua")]
    [InlineData("a.ps1", "PowerShell")]
    [InlineData("a.hs", "Haskell")]
    [InlineData("a.ex", "Elixir")]
    [InlineData("a.tf", "HCL")]
    [InlineData("a.md", "Markdown")]
    [InlineData("a.dart", "Dart")]
    [InlineData("a.proto", "Protocol Buffers")]
    public void 拡張子から言語を決める(string path, string expected)
    {
        Assert.Equal(expected, Lexer.ForPath(path)?.Name);
    }

    [Theory]
    [InlineData("Makefile", "Makefile")]
    [InlineData("makefile", "Makefile")]
    [InlineData("Dockerfile", "Dockerfile")]
    [InlineData("CMakeLists.txt", "CMake")]
    [InlineData(".gitignore", "無視の指定")]
    [InlineData(".env", "環境変数")]
    public void 拡張子が無いものは名前で決める(string name, string expected)
    {
        Assert.Equal(expected, Lexer.ForPath(name)?.Name);
    }

    [Fact]
    public void 名前の判定は道の途中でも効く()
    {
        Assert.Equal("Dockerfile", Lexer.ForPath("/src/docker/Dockerfile")?.Name);
    }

    [Fact]
    public void 知らないものは色分けしない()
    {
        Assert.Null(Lexer.ForPath("a.unknownext"));
        Assert.Null(Lexer.ForPath("readme"));
    }

    [Fact]
    public void Luaの行注釈は二重ハイフン()
    {
        var language = Lexer.ForPath("a.lua")!;
        var state = LexState.Start;
        var tokens = Lexer.Tokenize("x = 1 -- めも", language, ref state);

        Assert.Contains(tokens, t => t.Kind == TokenKind.Comment);
    }

    [Fact]
    public void Dockerfileの命令を語として拾う()
    {
        var language = Lexer.ForPath("Dockerfile")!;
        var state = LexState.Start;
        var tokens = Lexer.Tokenize("FROM alpine:3", language, ref state);

        Assert.Contains(tokens, t => t.Kind == TokenKind.Keyword);
    }
}
