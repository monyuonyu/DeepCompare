using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 「重要でない差分」の定義。
///
/// ここは比較の入口に効くので、間違えると「無視したはずの違いが差分として残る」または
/// もっと悪く「無視するつもりのなかった違いまで消える」。後者は差分ツールとして致命的
/// なので、無視の範囲が広がっていないことを個別に確かめる。
/// </summary>
public sealed class ImportanceTests
{
    private static DecodedText Text(params string[] lines)
        => TextDecoder.Decode(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines)));

    private static Comparison Compare(DecodedText left, DecodedText right, Importance importance)
        => DiffComparer.Compare(left, right, embedder: null, new CompareOptions(Importance: importance));

    [Fact]
    public void DefaultIgnoresNothing()
    {
        var importance = Importance.Default;
        Assert.True(importance.IgnoresNothing);
        Assert.Equal("  a  b  ", importance.Normalize("  a  b  "));
    }

    [Theory]
    [InlineData(WhitespaceMode.IgnoreTrailing, "  a  b", "  a  b  ")]
    [InlineData(WhitespaceMode.IgnoreLeadingTrailing, "a  b", "  a  b  ")]
    [InlineData(WhitespaceMode.CollapseRuns, "a b", "  a  b  ")]
    [InlineData(WhitespaceMode.IgnoreAll, "ab", "  a  b  ")]
    public void WhitespaceModesNormalizeAsDocumented(
        WhitespaceMode mode, string expected, string input)
    {
        Assert.Equal(expected, new Importance(mode).Normalize(input));
    }

    /// <summary>
    /// 字下げだけを変えた行が一致として畳まれ、かつ「元は違う」ことも残ること。
    /// 一致に畳むだけなら情報が消えるので、印が付いているところまでを確かめる。
    /// </summary>
    [Fact]
    public void IndentationOnlyChangeBecomesUnimportant()
    {
        var result = Compare(
            Text("class A {", "  int x;", "}"),
            Text("class A {", "        int x;", "}"),
            new Importance(WhitespaceMode.IgnoreLeadingTrailing));

        Assert.Equal(3, result.Rows.Count);
        Assert.All(result.Rows, row => Assert.True(row.IsUnchanged));

        var changed = result.Rows[1];
        Assert.True(changed.HasUnimportantDifferences);
        Assert.Equal(1, result.Stats.UnimportantRows);
        // 一致扱いなので、埋め込みに回す必要も無い。
        Assert.Equal(0, result.Stats.EmbeddedLines);
    }

    [Fact]
    public void CaseOnlyChangeBecomesUnimportantWhenIgnoringCase()
    {
        var result = Compare(Text("Hello"), Text("HELLO"), new Importance(IgnoreCase: true));

        Assert.Single(result.Rows);
        Assert.True(result.Rows[0].IsUnchanged);
        Assert.True(result.Rows[0].HasUnimportantDifferences);
    }

    [Fact]
    public void CaseOnlyChangeStaysADifferenceByDefault()
    {
        var result = Compare(Text("Hello"), Text("HELLO"), Importance.Default);

        Assert.All(result.Rows, row => Assert.False(row.IsUnchanged));
    }

    /// <summary>
    /// 正規表現で落とすのは「一致した部分」だけで、行全体ではないこと。
    /// ここを取り違えると、日付が違う行はすべて一致になってしまう。
    /// </summary>
    [Fact]
    public void IgnoredPatternRemovesOnlyTheMatchedPart()
    {
        var importance = new Importance(IgnoredPatterns: [@"\d{4}-\d{2}-\d{2}"]);

        var same = Compare(
            Text("// 更新 2024-01-01 by shota"),
            Text("// 更新 2026-08-12 by shota"),
            importance);
        Assert.True(same.Rows[0].IsUnchanged);
        Assert.True(same.Rows[0].HasUnimportantDifferences);

        // 日付以外が違えば、ちゃんと差分として残る。
        var differs = Compare(
            Text("// 更新 2024-01-01 by shota"),
            Text("// 更新 2026-08-12 by taro"),
            importance);
        Assert.All(differs.Rows, row => Assert.False(row.IsUnchanged));
    }

    /// <summary>
    /// 複数の式が互いを壊さないこと。順に適用する実装だと、先に \d+ が #a1b2c3 の
    /// 数字を削り、後の #[0-9a-f]{6} が一致しなくなって色指定が残る。
    /// </summary>
    [Fact]
    public void SeveralIgnoredPatternsDoNotInterfere()
    {
        var importance = new Importance(IgnoredPatterns: [@"\d+", @"#[0-9a-f]{6}"]);
        Assert.Equal("build  color ", importance.Normalize("build 123 color #a1b2c3"));
    }

    /// <summary>指定した順序で結果が変わらないこと。</summary>
    [Fact]
    public void IgnoredPatternOrderDoesNotMatter()
    {
        const string line = "build 123 color #a1b2c3";
        Assert.Equal(
            new Importance(IgnoredPatterns: [@"\d+", @"#[0-9a-f]{6}"]).Normalize(line),
            new Importance(IgnoredPatterns: [@"#[0-9a-f]{6}", @"\d+"]).Normalize(line));
    }

    /// <summary>最上位に | を含む式を渡しても、結合の優先順位が壊れないこと。</summary>
    [Fact]
    public void PatternsContainingAlternationAreGrouped()
    {
        var importance = new Importance(IgnoredPatterns: ["cat|dog", @"\d+"]);
        Assert.Equal("a  b ", importance.Normalize("a cat b 12"));
    }

    [Fact]
    public void InvalidPatternFailsWithTheOffendingPattern()
    {
        var error = Assert.Throws<ArgumentException>(
            () => new Importance(IgnoredPatterns: ["("]));
        Assert.Contains("(", error.Message);
    }

    /// <summary>
    /// 無視の指定があっても、本当に違う行は差分のまま残ること。
    /// 「全部一致になってしまう」壊れ方を検出する。
    /// </summary>
    [Fact]
    public void RealChangesSurviveEveryIgnoreSetting()
    {
        var importance = new Importance(
            WhitespaceMode.IgnoreAll, IgnoreCase: true, IgnoredPatterns: [@"\d+"]);

        var result = Compare(Text("var total = 1;"), Text("var subtotal = 2;"), importance);

        Assert.Contains(result.Rows, row => !row.IsUnchanged);
    }

    /// <summary>
    /// 正規表現を落とした跡に残る空白が、空白の畳み込みまで届くこと。
    /// 順序を逆にすると「無視したはずの箇所が空白の差として残る」。
    /// </summary>
    [Fact]
    public void PatternsAreRemovedBeforeWhitespaceIsCollapsed()
    {
        var importance = new Importance(
            WhitespaceMode.CollapseRuns, IgnoredPatterns: [@"v\d+"]);

        Assert.Equal(
            importance.Normalize("release v1 build"),
            importance.Normalize("release v22 build"));
    }

    [Fact]
    public void NormalizeAllReturnsTheSameInstanceWhenNothingIsIgnored()
    {
        var lines = new[] { "a", "b" };
        Assert.Same(lines, Importance.Default.NormalizeAll(lines));
    }
}
