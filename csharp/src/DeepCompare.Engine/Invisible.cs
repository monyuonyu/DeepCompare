using System.Globalization;
using System.Text;

namespace DeepCompare.Engine;

/// <summary>見つかったものの種類。</summary>
public enum InvisibleKind
{
    /// <summary>幅を持たない文字（ゼロ幅空白、結合子、方向指定など）。</summary>
    ZeroWidth,

    /// <summary>全角空白（U+3000）。日本語入力で紛れ込む。</summary>
    IdeographicSpace,

    /// <summary>普通の空白に見えるが別の文字（ノーブレークスペースなど）。</summary>
    LookalikeSpace,

    /// <summary>正規化されていない文字（NFC でない）。</summary>
    NotNormalized,

    /// <summary>行末の空白。</summary>
    TrailingWhitespace,

    /// <summary>同じ行でタブと空白が混ざっている。</summary>
    MixedIndent,

    /// <summary>ファイル全体で改行コードが混ざっている。</summary>
    MixedLineEnding,

    /// <summary>BOM が付いている。</summary>
    ByteOrderMark,

    /// <summary>最終行が改行で終わっていない。</summary>
    NoFinalNewline,
}

/// <summary>
/// 見つけたもの 1 件。
///
/// <paramref name="Line"/> と <paramref name="Column"/> は 1 始まり。
/// ファイル全体に関わるもの（改行の混在など）では 0。
/// </summary>
public sealed record InvisibleFinding(
    InvisibleKind Kind,
    int Line,
    int Column,
    string Detail)
{
    public string Describe()
        => Line > 0
            ? $"{Line}:{Column} {Label(Kind)} — {Detail}"
            : $"（全体） {Label(Kind)} — {Detail}";

    public static string Label(InvisibleKind kind) => kind switch
    {
        InvisibleKind.ZeroWidth => "幅の無い文字",
        InvisibleKind.IdeographicSpace => "全角空白",
        InvisibleKind.LookalikeSpace => "空白に似た別の文字",
        InvisibleKind.NotNormalized => "正規化されていない",
        InvisibleKind.TrailingWhitespace => "行末の空白",
        InvisibleKind.MixedIndent => "字下げにタブと空白が混在",
        InvisibleKind.MixedLineEnding => "改行コードの混在",
        InvisibleKind.ByteOrderMark => "BOM",
        InvisibleKind.NoFinalNewline => "最終行に改行が無い",
        _ => kind.ToString(),
    };
}

/// <summary>
/// 「同じに見えるのに一致しない」を見つける。
///
/// **これは差分ツールが解くべき問題の本丸**で、しかしどの道具も弱い。
/// 目で見て気づけない差は、目で見て直せない。
///
/// 無視する側（<see cref="Importance"/>）とは向きが逆で、こちらは**目立たせる**。
/// 「なぜか一致しない」と悩んでいる人に、理由を示すのが役目。
/// </summary>
public static class InvisibleScanner
{
    /// <summary>
    /// 幅を持たない、あるいは持たないに等しい文字。
    ///
    /// 絵文字の異体字選択子（U+FE0F など）や、書字方向の指定（U+202A〜）まで含む。
    /// これらは表示に出ないまま比較結果を変える。
    /// </summary>
    private static bool IsZeroWidth(char c) => c switch
    {
        '\u200b' => true,   // ゼロ幅空白
        '\u200c' => true,   // ゼロ幅非結合子
        '\u200d' => true,   // ゼロ幅結合子
        '\u2060' => true,   // 単語結合子
        '\ufeff' => true,   // 行の途中に来た BOM（ゼロ幅ノーブレークスペース）
        '\u00ad' => true,   // 分音ハイフン
        _ => c is >= '\u202a' and <= '\u202e'   // 書字方向の上書き
             || c is >= '\u2066' and <= '\u2069'
             || c is >= '\ufe00' and <= '\ufe0f',   // 異体字選択子
    };

    /// <summary>
    /// 空白に見えるが <c>' '</c> ではない文字。
    ///
    /// 貼り付けで紛れ込む。とくにノーブレークスペースは、
    /// 見た目が完全に同じで幅も同じなので、目では絶対に気づけない。
    /// </summary>
    private static bool IsLookalikeSpace(char c) => c switch
    {
        '\u00a0' => true,   // ノーブレークスペース。**幅も見た目も普通の空白と同じ**
        '\u202f' => true,   // 狭いノーブレークスペース
        '\u205f' => true,   // 数式用の中程度の空白
        '\u3000' => false,  // 全角空白は別に扱う（日本語では正当な場合がある）
        _ => c is >= '\u2000' and <= '\u200a',   // 各種の幅の空白（U+2007 の数字幅を含む）
    };

    public static IReadOnlyList<InvisibleFinding> Scan(DecodedText text)
    {
        var findings = new List<InvisibleFinding>();

        // ファイル全体に関わるもの。既に TextDecoder が判定しているので、ここでは拾うだけ。
        if (text.LineEnding == LineEnding.Mixed)
        {
            findings.Add(new InvisibleFinding(
                InvisibleKind.MixedLineEnding, 0, 0,
                "同じファイルの中で CRLF と LF が混ざっています"));
        }
        if (text.Encoding == TextEncoding.Utf8Bom)
        {
            findings.Add(new InvisibleFinding(
                InvisibleKind.ByteOrderMark, 0, 0,
                "先頭に BOM があります。付いていない相手とは先頭行が一致しません"));
        }
        if (!text.EndsWithNewline && text.Lines.Count > 0)
        {
            findings.Add(new InvisibleFinding(
                InvisibleKind.NoFinalNewline, text.Lines.Count, 0,
                "最終行が改行で終わっていません"));
        }

        for (var i = 0; i < text.Lines.Count; i++)
        {
            ScanLine(text.Lines[i], i + 1, findings);
        }

        return findings;
    }

    private static void ScanLine(string line, int lineNumber, List<InvisibleFinding> findings)
    {
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (IsZeroWidth(c))
            {
                findings.Add(new InvisibleFinding(
                    InvisibleKind.ZeroWidth, lineNumber, i + 1,
                    $"U+{(int)c:X4}（{CharUnicodeInfo.GetUnicodeCategory(c)}）"));
            }
            else if (c == '\u3000')
            {
                findings.Add(new InvisibleFinding(
                    InvisibleKind.IdeographicSpace, lineNumber, i + 1,
                    "U+3000。半角の空白とは一致しません"));
            }
            else if (IsLookalikeSpace(c))
            {
                findings.Add(new InvisibleFinding(
                    InvisibleKind.LookalikeSpace, lineNumber, i + 1,
                    $"U+{(int)c:X4}。見た目は空白ですが別の文字です"));
            }
        }

        // 行末の空白。差分では見えないが、比較では確実に効く。
        if (line.Length > 0 && char.IsWhiteSpace(line[^1]))
        {
            var start = line.Length;
            while (start > 0 && char.IsWhiteSpace(line[start - 1]))
            {
                start--;
            }
            findings.Add(new InvisibleFinding(
                InvisibleKind.TrailingWhitespace, lineNumber, start + 1,
                $"{line.Length - start} 文字"));
        }

        // 字下げにタブと空白が混ざっている。表示幅が環境で変わるので、
        // 見た目が揃っていても中身は違う。
        var indent = 0;
        while (indent < line.Length && (line[indent] == ' ' || line[indent] == '\t'))
        {
            indent++;
        }
        var head = line.AsSpan(0, indent);
        if (head.Contains('\t') && head.Contains(' '))
        {
            findings.Add(new InvisibleFinding(
                InvisibleKind.MixedIndent, lineNumber, 1,
                "タブ幅の設定で見た目が変わります"));
        }

        // 正規化されていない文字。**NFC でない**行を挙げる。
        // 濁点付きのかなが 2 文字で表されている場合などがここに出る。
        if (!line.IsNormalized(NormalizationForm.FormC))
        {
            findings.Add(new InvisibleFinding(
                InvisibleKind.NotNormalized, lineNumber, 1,
                "NFC ではありません。同じに見える NFC の文字列とは一致しません"));
        }
    }

    /// <summary>一覧を人が読む形に整える。</summary>
    public static string Format(IReadOnlyList<InvisibleFinding> findings)
    {
        if (findings.Count == 0)
        {
            return "見えない差分は見つかりませんでした。" + Environment.NewLine;
        }

        var builder = new StringBuilder();
        foreach (var finding in findings)
        {
            builder.AppendLine(finding.Describe());
        }
        builder.AppendLine();
        foreach (var group in findings.GroupBy(f => f.Kind).OrderByDescending(g => g.Count()))
        {
            builder.AppendLine($"{InvisibleFinding.Label(group.Key)}: {group.Count()} 件");
        }
        return builder.ToString();
    }

    /// <summary>
    /// 行を見える形に直す。画面や報告に出すときに使う。
    ///
    /// 消さずに**印に置き換える**。消してしまうと、そこに何かあったことまで
    /// 消えてしまい、元の問題（見えない）が別の形で残る。
    /// </summary>
    public static string Reveal(string line)
    {
        var builder = new StringBuilder(line.Length + 8);
        foreach (var c in line)
        {
            if (IsZeroWidth(c))
            {
                builder.Append(CultureInfo.InvariantCulture, $"<U+{(int)c:X4}>");
            }
            else if (c == '\u3000')
            {
                builder.Append('\u2423');   // 全角空白（␣）
            }
            else if (IsLookalikeSpace(c))
            {
                builder.Append('\u00b7');   // ·
            }
            else if (c == '\t')
            {
                builder.Append('\u2192');   // →
            }
            else
            {
                builder.Append(c);
            }
        }
        return builder.ToString();
    }
}

/// <summary>
/// ファイル名の突き合わせ方。
///
/// **フォルダー比較で実際に困る所。** macOS が作る日本語のファイル名は NFD
/// （「が」が「か」＋濁点の 2 文字）、Windows と Linux は NFC。同じ名前のはずの
/// ファイルが「片方にしか無い」と出る。
///
/// 大小文字も同じ問題を持つ。Windows は区別せず Linux は区別するので、
/// 往復すると <c>README.md</c> と <c>readme.md</c> が別物として増える。
/// </summary>
public sealed record NameMatching(
    /// <summary>NFC に揃えてから突き合わせる。macOS を経由したファイルで要る。</summary>
    bool NormalizeUnicode = false,

    /// <summary>大小文字を区別しない。Windows と Linux をまたぐときに要る。</summary>
    bool IgnoreCase = false)
{
    public static readonly NameMatching Exact = new();

    /// <summary>両方の癖を吸収する設定。異なる OS の間で比べるとき用。</summary>
    public static readonly NameMatching Lenient = new(NormalizeUnicode: true, IgnoreCase: true);

    public bool IsExact => !NormalizeUnicode && !IgnoreCase;

    /// <summary>突き合わせに使う形。表示には使わない。</summary>
    public string Key(string name)
    {
        if (IsExact)
        {
            return name;
        }
        var key = name;
        if (NormalizeUnicode && !key.IsNormalized(NormalizationForm.FormC))
        {
            key = key.Normalize(NormalizationForm.FormC);
        }
        return IgnoreCase ? key.ToLowerInvariant() : key;
    }

    /// <summary>同じ名前とみなすか。</summary>
    public bool Same(string left, string right)
        => string.Equals(Key(left), Key(right), StringComparison.Ordinal);

    /// <summary>
    /// 「同じに見えるのに一致しない」名前の組を挙げる。
    ///
    /// 揃えて比べれば同じだが、そのままでは別物になる組を返す。
    /// フォルダー比較で「片方にしか無い」が並んだときの原因を示すために使う。
    /// </summary>
    public static List<(string Left, string Right, string Reason)> FindNearMisses(
        IEnumerable<string> left, IEnumerable<string> right)
    {
        var result = new List<(string, string, string)>();
        var rightNames = right.ToList();
        var exact = new HashSet<string>(rightNames, StringComparer.Ordinal);

        foreach (var name in left)
        {
            if (exact.Contains(name))
            {
                continue;
            }
            foreach (var other in rightNames)
            {
                if (string.Equals(name, other, StringComparison.Ordinal))
                {
                    continue;
                }

                var normalizedSame = string.Equals(
                    name.Normalize(NormalizationForm.FormC),
                    other.Normalize(NormalizationForm.FormC),
                    StringComparison.Ordinal);
                if (normalizedSame)
                {
                    result.Add((name, other, "Unicode 正規化の違い（NFC と NFD）"));
                    break;
                }

                if (string.Equals(name, other, StringComparison.OrdinalIgnoreCase))
                {
                    result.Add((name, other, "大小文字だけの違い"));
                    break;
                }
            }
        }
        return result;
    }
}
