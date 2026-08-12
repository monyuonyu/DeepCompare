using System.Text;
using System.Text.RegularExpressions;

namespace DeepCompare.Engine;

/// <summary>空白の扱い。</summary>
public enum WhitespaceMode
{
    /// <summary>そのまま比較する。</summary>
    Respect,

    /// <summary>行末の空白を無視する。</summary>
    IgnoreTrailing,

    /// <summary>行頭と行末の空白を無視する。字下げの深さだけが違う行が一致になる。</summary>
    IgnoreLeadingTrailing,

    /// <summary>連続する空白を 1 つとみなし、行頭行末は落とす。整形の揺れを吸収する。</summary>
    CollapseRuns,

    /// <summary>空白をすべて無視する。</summary>
    IgnoreAll,
}

/// <summary>
/// 「重要でない差分」の定義。
///
/// 比較の中核に効く。ここで指定した違いは、行の対応付けの段階から無かったものとして
/// 扱われる。空白だけが違う行は一致した行として畳まれ、意味的な対応付けにも回らない。
/// 結果として表示が読みやすくなるだけでなく、埋め込みに通す行が減って速くもなる。
///
/// 表示では消さずに残す。<see cref="SpanKind.Unimportant"/> として印を付けるので、
/// 「一致しているが厳密には違う」ことは画面から分かる。
///
/// 既定は何も無視しない。<see cref="IgnoresNothing"/> が真のとき正規化は素通しになり、
/// 従来と完全に同じ動作になる。
/// </summary>
public sealed record Importance(
    WhitespaceMode Whitespace = WhitespaceMode.Respect,
    bool IgnoreCase = false,

    /// <summary>
    /// ここに一致した部分は比較から除く。タイムスタンプ、著作権年、ビルド番号のように
    /// 「毎回変わるが意味は無い」箇所を落とすために使う。
    /// </summary>
    IReadOnlyList<string>? IgnoredPatterns = null)
{
    public static readonly Importance Default = new();

    private readonly Regex? _ignored = Compile(IgnoredPatterns);

    /// <summary>何も無視しない。正規化を丸ごと省ける。</summary>
    public bool IgnoresNothing =>
        Whitespace == WhitespaceMode.Respect && !IgnoreCase && _ignored is null;

    /// <summary>
    /// 複数の正規表現を 1 本にまとめる。順に適用してはいけない。
    ///
    /// 例えば <c>\d+</c> と <c>#[0-9a-f]{6}</c> を順に当てると、先に数字が消えるせいで
    /// <c>#a1b2c3</c> が <c>#abc</c> になり、後者が一致しなくなる。**指定した順序で
    /// 結果が変わる**のは使う側から見て予測できない。1 本の選択にまとめて 1 度だけ
    /// 走らせれば、各位置でどれか 1 つが丸ごと一致するので、この相互作用が起きない。
    ///
    /// 各要素を <c>(?:...)</c> で囲むのは、利用者の式が最上位に <c>|</c> を含んでいても
    /// 結合の優先順位が壊れないようにするため。
    /// </summary>
    private static Regex? Compile(IReadOnlyList<string>? patterns)
    {
        if (patterns is null || patterns.Count == 0)
        {
            return null;
        }

        // 1 本にまとめる前に個別に検証する。まとめてから失敗すると、どの式が悪いのか
        // 利用者に示せない。
        foreach (var pattern in patterns)
        {
            try
            {
                _ = new Regex(pattern, RegexOptions.CultureInvariant);
            }
            catch (ArgumentException error)
            {
                throw new ArgumentException(
                    $"無視する正規表現が不正: {pattern}（{error.Message}）", nameof(patterns));
            }
        }

        // NativeAOT では RegexOptions.Compiled は解釈実行に落ちる。指定しても速く
        // ならず、警告だけが増えるので付けない。
        return new Regex(
            string.Join('|', patterns.Select(p => $"(?:{p})")), RegexOptions.CultureInvariant);
    }

    /// <summary>比較に使う形へ直す。返り値は表示には使わない。</summary>
    public string Normalize(string line)
    {
        if (IgnoresNothing)
        {
            return line;
        }

        var text = line;

        // 正規表現を先に落とす。空白の畳み込みより前にやらないと、落とした跡に
        // 残った空白が畳まれず、無視したはずの箇所が差分として残る。
        if (_ignored is not null)
        {
            text = _ignored.Replace(text, string.Empty);
        }

        if (IgnoreCase)
        {
            text = text.ToLowerInvariant();
        }

        return Whitespace switch
        {
            WhitespaceMode.IgnoreTrailing => text.TrimEnd(),
            WhitespaceMode.IgnoreLeadingTrailing => text.Trim(),
            WhitespaceMode.CollapseRuns => Collapse(text),
            WhitespaceMode.IgnoreAll => Remove(text),
            _ => text,
        };
    }

    /// <summary>行の並びをまとめて正規化する。何も無視しないなら元の列をそのまま返す。</summary>
    public IReadOnlyList<string> NormalizeAll(IReadOnlyList<string> lines)
    {
        if (IgnoresNothing)
        {
            return lines;
        }

        var result = new string[lines.Count];
        for (var i = 0; i < lines.Count; i++)
        {
            result[i] = Normalize(lines[i]);
        }
        return result;
    }

    private static string Collapse(string text)
    {
        var builder = new StringBuilder(text.Length);
        var pendingSpace = false;
        foreach (var c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                // 行頭の空白は出力しない。builder.Length が 0 のうちは保留も立てない。
                pendingSpace = builder.Length > 0;
                continue;
            }
            if (pendingSpace)
            {
                builder.Append(' ');
                pendingSpace = false;
            }
            builder.Append(c);
        }
        return builder.ToString();
    }

    private static string Remove(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            if (!char.IsWhiteSpace(c))
            {
                builder.Append(c);
            }
        }
        return builder.ToString();
    }
}
