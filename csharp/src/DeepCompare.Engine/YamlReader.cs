using System.Globalization;

namespace DeepCompare.Engine;

/// <summary>
/// YAML を木へ。
///
/// **文法の一部だけを扱う。** 対応するのは、字下げによる入れ子、
/// マッピング（<c>鍵: 値</c>）、並び（<c>- 値</c>）、行注釈、引用符付きの
/// 文字列、複数行の畳み込み（<c>|</c> と <c>&gt;</c>）、複数文書の区切り。
///
/// **対応しないもの**: 錨と参照（<c>&amp;a</c> / <c>*a</c>）、複雑な鍵、
/// タグ、流れ形式の入れ子の一部。これらを完全に扱うには本格的な実装が要り、
/// **外部の部品を足すことになる**。設定ファイルの比較という目的に対しては
/// 割に合わない。読めない書き方に当たったら、そう言って止まる方が誠実。
///
/// 完全な YAML が要るようになったら、そのときに部品を入れる判断をする。
/// </summary>
public static class YamlReader
{
    private sealed record Line(int Indent, string Text, int Number);

    public static JsonNode Parse(string text)
    {
        var lines = new List<Line>();
        var raw = text.Split('\n');

        for (var i = 0; i < raw.Length; i++)
        {
            var line = raw[i].TrimEnd('\r');

            // 複数文書は最初の 1 つだけを見る。2 つ目以降は別の木なので、
            // 1 本の比較には収まらない。
            if (line.StartsWith("---", StringComparison.Ordinal) && lines.Count > 0)
            {
                break;
            }
            if (line.StartsWith("---", StringComparison.Ordinal)
                || line.StartsWith("...", StringComparison.Ordinal))
            {
                continue;
            }

            var indent = 0;
            while (indent < line.Length && line[indent] == ' ')
            {
                indent++;
            }
            if (indent < line.Length && line[indent] == '\t')
            {
                throw new StructuredParseException(
                    $"YAML として読めません（{i + 1} 行目）: 字下げにタブは使えません");
            }

            var body = line[indent..];
            if (body.Length == 0 || body[0] == '#')
            {
                continue;
            }

            // 錨と参照は扱わない。**黙って無視すると、別物を同じと言うことになる。**
            if (body.StartsWith('&') || body.StartsWith('*')
                || body.Contains(": &", StringComparison.Ordinal)
                || body.Contains(": *", StringComparison.Ordinal))
            {
                throw new StructuredParseException(
                    $"YAML の錨と参照には対応していません（{i + 1} 行目）。"
                    + "テキストとして比べてください。");
            }

            lines.Add(new Line(indent, body, i + 1));
        }

        var at = 0;
        return lines.Count == 0
            ? JsonNode.Null
            : ParseBlock(lines, ref at, lines[0].Indent);
    }

    private static JsonNode ParseBlock(List<Line> lines, ref int at, int indent)
    {
        if (at >= lines.Count)
        {
            return JsonNode.Null;
        }

        return lines[at].Text.StartsWith("- ", StringComparison.Ordinal)
            || lines[at].Text == "-"
                ? ParseSequence(lines, ref at, indent)
                : ParseMapping(lines, ref at, indent);
    }

    private static JsonNode ParseSequence(List<Line> lines, ref int at, int indent)
    {
        var items = new List<JsonNode>();

        while (at < lines.Count && lines[at].Indent == indent
               && (lines[at].Text.StartsWith("- ", StringComparison.Ordinal) || lines[at].Text == "-"))
        {
            var rest = lines[at].Text == "-" ? string.Empty : lines[at].Text[2..].Trim();
            at++;

            if (rest.Length == 0)
            {
                // 中身は次の行から。
                items.Add(at < lines.Count && lines[at].Indent > indent
                    ? ParseBlock(lines, ref at, lines[at].Indent)
                    : JsonNode.Null);
                continue;
            }

            // 「- 鍵: 値」は、その要素が写像であることを意味する。
            var separator = FindColon(rest);
            if (separator >= 0)
            {
                var members = new List<KeyValuePair<string, JsonNode>>
                {
                    Pair(rest, separator, lines, ref at, indent + 2),
                };
                // 同じ要素の続き（字下げが深い行）を集める。
                while (at < lines.Count && lines[at].Indent > indent
                       && !lines[at].Text.StartsWith("- ", StringComparison.Ordinal))
                {
                    var inner = lines[at].Text;
                    var innerSeparator = FindColon(inner);
                    if (innerSeparator < 0)
                    {
                        break;
                    }
                    var childIndent = lines[at].Indent;
                    at++;
                    members.Add(Pair(inner, innerSeparator, lines, ref at, childIndent));
                }
                items.Add(new JsonNode { Kind = JsonKind.Object, Members = members });
                continue;
            }

            items.Add(Scalar(rest));
        }

        return new JsonNode { Kind = JsonKind.Array, Items = items };
    }

    private static JsonNode ParseMapping(List<Line> lines, ref int at, int indent)
    {
        var members = new List<KeyValuePair<string, JsonNode>>();

        while (at < lines.Count && lines[at].Indent == indent)
        {
            var text = lines[at].Text;
            var separator = FindColon(text);
            if (separator < 0)
            {
                throw new StructuredParseException(
                    $"YAML として読めません（{lines[at].Number} 行目）: 鍵と値に分けられません");
            }
            at++;
            members.Add(Pair(text, separator, lines, ref at, indent));
        }

        return new JsonNode { Kind = JsonKind.Object, Members = members };
    }

    private static KeyValuePair<string, JsonNode> Pair(
        string text, int separator, List<Line> lines, ref int at, int indent)
    {
        var key = text[..separator].Trim().Trim('"', '\'');
        var value = text[(separator + 1)..].Trim();

        if (value.Length == 0)
        {
            // 中身は次の行から。字下げが深ければそこが中身。
            if (at < lines.Count && lines[at].Indent > indent)
            {
                return new KeyValuePair<string, JsonNode>(
                    key, ParseBlock(lines, ref at, lines[at].Indent));
            }
            // 同じ深さで「- 」が続くなら、それも中身（YAML では許される書き方）。
            if (at < lines.Count && lines[at].Indent == indent
                && lines[at].Text.StartsWith("- ", StringComparison.Ordinal))
            {
                return new KeyValuePair<string, JsonNode>(
                    key, ParseSequence(lines, ref at, indent));
            }
            return new KeyValuePair<string, JsonNode>(key, JsonNode.Null);
        }

        // 複数行の畳み込み。中身は字下げの深い行を集める。
        if (value is "|" or ">" or "|-" or ">-" or "|+" or ">+")
        {
            var parts = new List<string>();
            while (at < lines.Count && lines[at].Indent > indent)
            {
                parts.Add(lines[at].Text);
                at++;
            }
            // `>` は行を空白で継ぐ、`|` は改行を保つ。比較では改行の有無が
            // 効くので、区別して持つ。
            var joined = value[0] == '>' ? string.Join(' ', parts) : string.Join('\n', parts);
            return new KeyValuePair<string, JsonNode>(
                key, new JsonNode { Kind = JsonKind.String, Value = joined });
        }

        return new KeyValuePair<string, JsonNode>(key, Scalar(value));
    }

    /// <summary>鍵と値を分ける <c>:</c> を探す。引用符の中と、URL の <c>://</c> は避ける。</summary>
    private static int FindColon(string text)
    {
        var quote = '\0';
        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            if (quote != '\0')
            {
                if (c == quote)
                {
                    quote = '\0';
                }
                continue;
            }
            if (c is '"' or '\'')
            {
                quote = c;
            }
            else if (c == '#' && i > 0 && text[i - 1] == ' ')
            {
                return -1;   // ここから先は注釈
            }
            else if (c == ':')
            {
                // 「鍵: 値」の形だけを鍵の区切りとみなす。`http://` の `:` を
                // 拾うと、URL を値に持つ行が全部壊れる。
                if (i + 1 >= text.Length || text[i + 1] == ' ')
                {
                    return i;
                }
            }
        }
        return -1;
    }

    private static JsonNode Scalar(string raw)
    {
        var value = StripComment(raw).Trim();

        if (value.Length >= 2 && (value[0] is '"' or '\'') && value[^1] == value[0])
        {
            return new JsonNode { Kind = JsonKind.String, Value = value[1..^1] };
        }

        // 流れ形式の並びと写像。設定ファイルでよく出るので拾う。
        if (value.StartsWith('[') && value.EndsWith(']'))
        {
            return new JsonNode
            {
                Kind = JsonKind.Array,
                Items = [.. Split(value[1..^1]).Select(Scalar)],
            };
        }
        if (value.StartsWith('{') && value.EndsWith('}'))
        {
            var members = new List<KeyValuePair<string, JsonNode>>();
            foreach (var part in Split(value[1..^1]))
            {
                var separator = FindColon(part);
                if (separator >= 0)
                {
                    members.Add(new KeyValuePair<string, JsonNode>(
                        part[..separator].Trim().Trim('"', '\''),
                        Scalar(part[(separator + 1)..])));
                }
            }
            return new JsonNode { Kind = JsonKind.Object, Members = members };
        }

        if (value.Length == 0 || value is "null" or "~" or "Null" or "NULL")
        {
            return JsonNode.Null;
        }
        // YAML では yes/no/on/off も真偽として扱われる。**ここを外すと、
        // `yes` と `true` を別物として出してしまう。**
        if (value is "true" or "True" or "TRUE" or "yes" or "Yes" or "on" or "On")
        {
            return new JsonNode { Kind = JsonKind.Bool, Value = "true" };
        }
        if (value is "false" or "False" or "FALSE" or "no" or "No" or "off" or "Off")
        {
            return new JsonNode { Kind = JsonKind.Bool, Value = "false" };
        }
        if (decimal.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
        {
            return new JsonNode { Kind = JsonKind.Number, Value = value };
        }
        return new JsonNode { Kind = JsonKind.String, Value = value };
    }

    private static string StripComment(string value)
    {
        var quote = '\0';
        for (var i = 0; i < value.Length; i++)
        {
            var c = value[i];
            if (quote != '\0')
            {
                if (c == quote)
                {
                    quote = '\0';
                }
                continue;
            }
            if (c is '"' or '\'')
            {
                quote = c;
            }
            else if (c == '#' && (i == 0 || value[i - 1] == ' '))
            {
                return value[..i];
            }
        }
        return value;
    }

    private static List<string> Split(string text)
    {
        var parts = new List<string>();
        var builder = new System.Text.StringBuilder();
        var depth = 0;
        var quote = '\0';

        foreach (var c in text)
        {
            if (quote != '\0')
            {
                builder.Append(c);
                if (c == quote)
                {
                    quote = '\0';
                }
                continue;
            }
            switch (c)
            {
                case '"' or '\'':
                    quote = c;
                    builder.Append(c);
                    break;
                case '[' or '{':
                    depth++;
                    builder.Append(c);
                    break;
                case ']' or '}':
                    depth--;
                    builder.Append(c);
                    break;
                case ',' when depth == 0:
                    parts.Add(builder.ToString().Trim());
                    builder.Clear();
                    break;
                default:
                    builder.Append(c);
                    break;
            }
        }

        var last = builder.ToString().Trim();
        if (last.Length > 0)
        {
            parts.Add(last);
        }
        return parts;
    }
}
