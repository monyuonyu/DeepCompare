using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

public sealed class TextSearchTests
{
    private static readonly string[] Lines =
    [
        "var total = 1;",
        "var subtotal = total + 2;",
        "// TOTAL",
        string.Empty,
    ];

    [Fact]
    public void FindsEveryOccurrenceIncludingSeveralOnOneLine()
    {
        var hits = TextSearch.Find(Lines, new SearchQuery("total"));

        // 1 行目、2 行目に 2 箇所（subtotal と total）、3 行目（大小文字を区別しない）。
        Assert.Equal(4, hits.Count);
        Assert.Equal(0, hits[0].Line);
        Assert.Equal(1, hits[1].Line);
        Assert.Equal(1, hits[2].Line);
        Assert.Equal(2, hits[3].Line);
    }

    [Fact]
    public void MatchCaseRestrictsToExactCasing()
    {
        var hits = TextSearch.Find(Lines, new SearchQuery("TOTAL", MatchCase: true));

        var hit = Assert.Single(hits);
        Assert.Equal(2, hit.Line);
    }

    /// <summary>単語単位なら subtotal の中の total は拾わない。</summary>
    [Fact]
    public void WholeWordSkipsMatchesInsideLongerWords()
    {
        var hits = TextSearch.Find(Lines, new SearchQuery("total", WholeWord: true));

        Assert.Equal(3, hits.Count);
        Assert.DoesNotContain(hits, h => h.Line == 1 && h.Start == 4);
    }

    /// <summary>
    /// 記号で始まる語に単語境界を付けると一致しなくなる。境界を付けるかは
    /// 語の端が語構成文字かどうかで決める。
    /// </summary>
    [Fact]
    public void WholeWordStillFindsSymbolBoundedText()
    {
        var hits = TextSearch.Find(["a = 1;", "b = 2;"], new SearchQuery("= 1;", WholeWord: true));

        Assert.Single(hits);
    }

    [Fact]
    public void RegexIsUsedOnlyWhenAsked()
    {
        Assert.Empty(TextSearch.Find(["a.c"], new SearchQuery("a.c", UseRegex: false) with { Pattern = "axc" }));
        Assert.Single(TextSearch.Find(["axc"], new SearchQuery("a.c", UseRegex: true)));
        // 正規表現でなければ . は文字そのもの。
        Assert.Empty(TextSearch.Find(["axc"], new SearchQuery("a.c")));
    }

    [Fact]
    public void InvalidRegexFailsWithThePattern()
    {
        var error = Assert.Throws<ArgumentException>(
            () => TextSearch.Find(["x"], new SearchQuery("(", UseRegex: true)));
        Assert.Contains("(", error.Message);
    }

    [Fact]
    public void EmptyPatternFindsNothing()
    {
        Assert.Empty(TextSearch.Find(Lines, new SearchQuery(string.Empty)));
    }

    /// <summary>幅 0 の一致は捨てる。拾うと同じ位置で止まって進めなくなる。</summary>
    [Fact]
    public void ZeroWidthMatchesAreIgnored()
    {
        Assert.Empty(TextSearch.Find(["abc"], new SearchQuery("^", UseRegex: true)));
    }

    [Fact]
    public void NextAndPreviousWrapAround()
    {
        var hits = TextSearch.Find(Lines, new SearchQuery("total", WholeWord: true));

        Assert.Equal(hits[0], TextSearch.Next(hits, null));
        Assert.Equal(hits[1], TextSearch.Next(hits, hits[0]));
        // 末尾の次は先頭へ戻る。
        Assert.Equal(hits[0], TextSearch.Next(hits, hits[^1]));

        Assert.Equal(hits[^1], TextSearch.Previous(hits, null));
        Assert.Equal(hits[0], TextSearch.Previous(hits, hits[1]));
        // 先頭の前は末尾へ回る。
        Assert.Equal(hits[^1], TextSearch.Previous(hits, hits[0]));
    }

    [Fact]
    public void NextAndPreviousOnNoHitsReturnNull()
    {
        Assert.Null(TextSearch.Next([], null));
        Assert.Null(TextSearch.Previous([], null));
    }

    [Fact]
    public void ReplaceAllReportsHowManyItChanged()
    {
        var result = TextSearch.ReplaceAll(
            Lines, new SearchQuery("total", WholeWord: true, MatchCase: true), "sum", out var count);

        Assert.Equal(2, count);
        Assert.Equal("var sum = 1;", result[0]);
        Assert.Equal("var subtotal = sum + 2;", result[1]);
        // 大小文字を区別しているので、コメントの TOTAL は残る。
        Assert.Equal("// TOTAL", result[2]);
    }

    /// <summary>
    /// 正規表現でないときに $1 のような文字列を置換後に入れても、そのまま入ること。
    /// 解釈してしまうと、意図せず空文字になる。
    /// </summary>
    [Fact]
    public void PlainReplacementDoesNotInterpretDollarGroups()
    {
        var result = TextSearch.ReplaceAll(["a"], new SearchQuery("a"), "$1", out var count);

        Assert.Equal(1, count);
        Assert.Equal("$1", result[0]);
    }

    [Fact]
    public void RegexReplacementCanUseGroups()
    {
        var result = TextSearch.ReplaceAll(
            ["name=value"], new SearchQuery(@"(\w+)=(\w+)", UseRegex: true), "$2=$1", out var count);

        Assert.Equal(1, count);
        Assert.Equal("value=name", result[0]);
    }
}
