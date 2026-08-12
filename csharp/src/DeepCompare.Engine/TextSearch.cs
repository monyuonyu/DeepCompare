using System.Text.RegularExpressions;

namespace DeepCompare.Engine;

/// <summary>検索で一致した 1 箇所。行と、その行の中の範囲。</summary>
public readonly record struct SearchHit(int Line, int Start, int Length);

/// <summary>検索の条件。</summary>
public sealed record SearchQuery(
    string Pattern,
    bool UseRegex = false,
    bool MatchCase = false,
    bool WholeWord = false)
{
    public bool IsEmpty => Pattern.Length == 0;
}

/// <summary>
/// 本文の検索。
///
/// 行の並びに対して素直に当てるだけだが、正規表現・大小文字・単語単位の組み合わせを
/// ここに閉じ込めてある。呼ぶ側（画面と CLI）で別々に組むと、片方だけ単語境界の扱いが
/// 違う、といったずれが出る。
/// </summary>
public static class TextSearch
{
    /// <summary>
    /// 一致箇所を先頭から順に返す。1 行に複数あればすべて返す。
    /// </summary>
    public static List<SearchHit> Find(IReadOnlyList<string> lines, SearchQuery query)
    {
        var hits = new List<SearchHit>();
        if (query.IsEmpty)
        {
            return hits;
        }

        var regex = Build(query);
        for (var line = 0; line < lines.Count; line++)
        {
            foreach (Match match in regex.Matches(lines[line]))
            {
                // 幅 0 の一致（`^` や `\b` だけの式）は無限に進まないよう捨てる。
                if (match.Length == 0)
                {
                    continue;
                }
                hits.Add(new SearchHit(line, match.Index, match.Length));
            }
        }
        return hits;
    }

    /// <summary>
    /// <paramref name="from"/> より後の最初の一致。無ければ先頭へ回り込む。
    /// 見つからなければ null。
    /// </summary>
    public static SearchHit? Next(IReadOnlyList<SearchHit> hits, SearchHit? from)
    {
        if (hits.Count == 0)
        {
            return null;
        }
        if (from is not { } current)
        {
            return hits[0];
        }
        foreach (var hit in hits)
        {
            if (hit.Line > current.Line || (hit.Line == current.Line && hit.Start > current.Start))
            {
                return hit;
            }
        }
        return hits[0];
    }

    /// <summary><paramref name="from"/> より前の最後の一致。無ければ末尾へ回り込む。</summary>
    public static SearchHit? Previous(IReadOnlyList<SearchHit> hits, SearchHit? from)
    {
        if (hits.Count == 0)
        {
            return null;
        }
        if (from is not { } current)
        {
            return hits[^1];
        }
        for (var i = hits.Count - 1; i >= 0; i--)
        {
            var hit = hits[i];
            if (hit.Line < current.Line || (hit.Line == current.Line && hit.Start < current.Start))
            {
                return hit;
            }
        }
        return hits[^1];
    }

    /// <summary>一致箇所を置き換えた行の並びを返す。元の並びは変更しない。</summary>
    public static List<string> ReplaceAll(
        IReadOnlyList<string> lines, SearchQuery query, string replacement, out int count)
    {
        count = 0;
        var result = new List<string>(lines.Count);
        if (query.IsEmpty)
        {
            result.AddRange(lines);
            return result;
        }

        // out 引数はラムダの中で触れないので、いったん局所変数で数える。
        var replaced = 0;
        var regex = Build(query);
        foreach (var line in lines)
        {
            result.Add(regex.Replace(line, match =>
            {
                if (match.Length == 0)
                {
                    return match.Value;
                }
                replaced++;
                // 正規表現でないときは $1 などを置換文字列として解釈しない。
                // 単なる文字列置換のつもりで $ を含む文字を入れたときに壊れる。
                return query.UseRegex ? match.Result(replacement) : replacement;
            }));
        }
        count = replaced;
        return result;
    }

    private static Regex Build(SearchQuery query)
    {
        var body = query.UseRegex ? query.Pattern : Regex.Escape(query.Pattern);
        if (query.WholeWord)
        {
            // \b は語の途中では効かないので、記号で始まる／終わる語のために
            // 境界が使える場合だけ付ける。
            var prefix = StartsWithWordCharacter(query) ? @"\b" : string.Empty;
            var suffix = EndsWithWordCharacter(query) ? @"\b" : string.Empty;
            body = $"{prefix}(?:{body}){suffix}";
        }

        var options = RegexOptions.CultureInvariant;
        if (!query.MatchCase)
        {
            options |= RegexOptions.IgnoreCase;
        }

        try
        {
            return new Regex(body, options);
        }
        catch (ArgumentException error)
        {
            throw new ArgumentException(
                $"検索の正規表現が不正: {query.Pattern}（{error.Message}）", nameof(query));
        }
    }

    // 正規表現のときは端が何になるか分からないので、常に境界を付ける。
    private static bool StartsWithWordCharacter(SearchQuery query)
        => query.UseRegex || (query.Pattern.Length > 0 && IsWord(query.Pattern[0]));

    private static bool EndsWithWordCharacter(SearchQuery query)
        => query.UseRegex || (query.Pattern.Length > 0 && IsWord(query.Pattern[^1]));

    private static bool IsWord(char c) => char.IsLetterOrDigit(c) || c == '_';
}
