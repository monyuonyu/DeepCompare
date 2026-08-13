using System.Globalization;
using System.Text;
using System.Xml;
using System.Xml.Linq;

namespace DeepCompare.Engine;

/// <summary>読める形式。</summary>
public enum StructuredFormat
{
    Json,
    Xml,
    Toml,
    Yaml,
}

/// <summary>
/// 形式ごとの読み取り。**どれも同じ木（<see cref="JsonNode"/>）に落とす。**
///
/// 比較の側は木しか知らないので、読み取りを足せば比較・画面・CLI が
/// そのまま使える。形式ごとに比較を書き分けると、同じ間違いを何度も繰り返す。
/// </summary>
public static class StructuredReaders
{
    /// <summary>拡張子から形式を決める。分からなければ JSON として読む。</summary>
    public static StructuredFormat ForPath(string path)
        => Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".xml" or ".xaml" or ".axaml" or ".csproj" or ".props" or ".targets"
                or ".svg" or ".plist" or ".resx" => StructuredFormat.Xml,
            ".toml" => StructuredFormat.Toml,
            ".yaml" or ".yml" => StructuredFormat.Yaml,
            _ => StructuredFormat.Json,
        };

    public static JsonNode Parse(string text, StructuredFormat format) => format switch
    {
        StructuredFormat.Xml => XmlReaderAdapter.Parse(text),
        StructuredFormat.Toml => TomlReader.Parse(text),
        StructuredFormat.Yaml => YamlReader.Parse(text),
        _ => JsonReader.Parse(text),
    };

    public static JsonNode ParseFile(string path)
        => Parse(File.ReadAllText(path), ForPath(path));
}

/// <summary>
/// XML を木へ。
///
/// **属性と子要素を混ぜない。** 属性は <c>@name</c>、本文は <c>#text</c> に置く。
/// 混ぜると、同じ名前の属性と要素があるときに片方が消える。
///
/// 同じ名前の要素が複数あれば配列にする。1 つしか無い要素を配列にするか
/// どうかは XML だけでは決められないので、**数で決める**（1 つなら値、
/// 2 つ以上なら配列）。差分としては、増減が「型の変化」として出る。
/// </summary>
public static class XmlReaderAdapter
{
    public static JsonNode Parse(string text)
    {
        XDocument document;
        try
        {
            document = XDocument.Parse(text, LoadOptions.None);
        }
        catch (XmlException e)
        {
            throw new StructuredParseException(
                $"XML として読めません（{e.LineNumber} 行目 {e.LinePosition} 桁目）: {e.Message}");
        }

        if (document.Root is null)
        {
            return JsonNode.Null;
        }

        // 根の名前も木に残す。名前が変わったことを差分として見せたい。
        return new JsonNode
        {
            Kind = JsonKind.Object,
            Members = [new KeyValuePair<string, JsonNode>(document.Root.Name.LocalName,
                Convert(document.Root))],
        };
    }

    private static JsonNode Convert(XElement element)
    {
        var members = new List<KeyValuePair<string, JsonNode>>();

        foreach (var attribute in element.Attributes())
        {
            // 名前空間の宣言は中身ではないので落とす。差分に出しても直しようがない。
            if (attribute.IsNamespaceDeclaration)
            {
                continue;
            }
            members.Add(new KeyValuePair<string, JsonNode>(
                "@" + attribute.Name.LocalName, Scalar(attribute.Value)));
        }

        var children = element.Elements().ToList();
        if (children.Count == 0)
        {
            var value = element.Value;
            if (members.Count == 0)
            {
                // 属性も子も無いなら、ただの値。
                return Scalar(value);
            }
            if (value.Trim().Length > 0)
            {
                members.Add(new KeyValuePair<string, JsonNode>("#text", Scalar(value)));
            }
            return new JsonNode { Kind = JsonKind.Object, Members = members };
        }

        // 同じ名前の子をまとめる。順序は最初に出てきた順を保つ。
        foreach (var group in children.GroupBy(c => c.Name.LocalName))
        {
            var items = group.Select(Convert).ToList();
            members.Add(new KeyValuePair<string, JsonNode>(
                group.Key,
                items.Count == 1
                    ? items[0]
                    : new JsonNode { Kind = JsonKind.Array, Items = items }));
        }

        return new JsonNode { Kind = JsonKind.Object, Members = members };
    }

    /// <summary>
    /// 値の種類を推し量る。
    ///
    /// **XML はすべて文字列**だが、そのまま扱うと <c>1</c> と <c>1.0</c> の
    /// 違いが見えない。数と真偽に見えるものはその種類にして、比較の側の
    /// 「型の変化」や「数として同じ」の判定を効かせる。
    /// </summary>
    private static JsonNode Scalar(string raw)
    {
        var value = raw.Trim();
        if (value.Length == 0)
        {
            return new JsonNode { Kind = JsonKind.String, Value = string.Empty };
        }
        if (string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "false", StringComparison.OrdinalIgnoreCase))
        {
            return new JsonNode { Kind = JsonKind.Bool, Value = value.ToLowerInvariant() };
        }
        if (decimal.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
        {
            return new JsonNode { Kind = JsonKind.Number, Value = value };
        }
        return new JsonNode { Kind = JsonKind.String, Value = value };
    }
}

/// <summary>
/// TOML を木へ。
///
/// **文法の小さい部分だけを扱う。** 表（<c>[a.b]</c>）、表の配列
/// （<c>[[a]]</c>）、鍵と値、基本的な型、行注釈。設定ファイルの比較には
/// これで足りる。複数行文字列や日時の細かい書式までは追わない。
/// </summary>
public static class TomlReader
{
    public static JsonNode Parse(string text)
    {
        var root = new Builder();
        var current = root;

        var lines = text.Split('\n');
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim().TrimEnd('\r');
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }

            if (line[0] == '[')
            {
                var isArray = line.StartsWith("[[", StringComparison.Ordinal);
                var end = line.IndexOf(isArray ? "]]" : "]", StringComparison.Ordinal);
                if (end < 0)
                {
                    throw new StructuredParseException(
                        $"TOML として読めません（{i + 1} 行目）: 表の見出しが閉じていません");
                }
                var name = line[(isArray ? 2 : 1)..end].Trim();
                current = root.Table(SplitKey(name), isArray);
                continue;
            }

            var separator = FindSeparator(line);
            if (separator < 0)
            {
                throw new StructuredParseException(
                    $"TOML として読めません（{i + 1} 行目）: 鍵と値に分けられません");
            }

            var key = line[..separator].Trim();
            var value = StripComment(line[(separator + 1)..].Trim());
            current.Set(SplitKey(key), ParseValue(value));
        }

        return root.Build();
    }

    /// <summary>鍵と値を分ける <c>=</c> を探す。引用符の中は見ない。</summary>
    private static int FindSeparator(string line)
    {
        var quote = '\0';
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
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
            else if (c == '=')
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>行末の注釈を落とす。引用符の中の <c>#</c> は残す。</summary>
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
            else if (c == '#')
            {
                return value[..i].TrimEnd();
            }
        }
        return value;
    }

    private static List<string> SplitKey(string key)
    {
        var parts = new List<string>();
        var builder = new StringBuilder();
        var quote = '\0';

        foreach (var c in key)
        {
            if (quote != '\0')
            {
                if (c == quote)
                {
                    quote = '\0';
                }
                else
                {
                    builder.Append(c);
                }
                continue;
            }
            if (c is '"' or '\'')
            {
                quote = c;
            }
            else if (c == '.')
            {
                parts.Add(builder.ToString().Trim());
                builder.Clear();
            }
            else
            {
                builder.Append(c);
            }
        }
        parts.Add(builder.ToString().Trim());
        return parts;
    }

    private static JsonNode ParseValue(string value)
    {
        if (value.Length == 0)
        {
            return new JsonNode { Kind = JsonKind.String, Value = string.Empty };
        }

        if (value[0] is '"' or '\'')
        {
            var quote = value[0];
            var end = value.LastIndexOf(quote);
            return new JsonNode
            {
                Kind = JsonKind.String,
                Value = end > 0 ? value[1..end] : value[1..],
            };
        }

        if (value[0] == '[')
        {
            var inner = value.TrimEnd();
            inner = inner.EndsWith(']') ? inner[1..^1] : inner[1..];
            return new JsonNode
            {
                Kind = JsonKind.Array,
                Items = [.. SplitList(inner).Select(ParseValue)],
            };
        }

        if (value[0] == '{')
        {
            var inner = value.TrimEnd();
            inner = inner.EndsWith('}') ? inner[1..^1] : inner[1..];
            var members = new List<KeyValuePair<string, JsonNode>>();
            foreach (var part in SplitList(inner))
            {
                var separator = FindSeparator(part);
                if (separator < 0)
                {
                    continue;
                }
                members.Add(new KeyValuePair<string, JsonNode>(
                    part[..separator].Trim().Trim('"', '\''),
                    ParseValue(part[(separator + 1)..].Trim())));
            }
            return new JsonNode { Kind = JsonKind.Object, Members = members };
        }

        if (value is "true" or "false")
        {
            return new JsonNode { Kind = JsonKind.Bool, Value = value };
        }

        if (decimal.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
            || (value.StartsWith("0x", StringComparison.Ordinal) && value.Length > 2))
        {
            return new JsonNode { Kind = JsonKind.Number, Value = value };
        }

        // 日時などはそのまま文字列。書式の違いを差分として出せば足りる。
        return new JsonNode { Kind = JsonKind.String, Value = value };
    }

    /// <summary>括弧の入れ子と引用符を見ながら、要素を切る。</summary>
    private static List<string> SplitList(string text)
    {
        var parts = new List<string>();
        var builder = new StringBuilder();
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

    /// <summary>組み立て中の木。表の入れ子を辿れるようにする。</summary>
    private sealed class Builder
    {
        private readonly List<KeyValuePair<string, object>> _members = [];

        public Builder Table(List<string> path, bool asArray)
        {
            var node = this;
            for (var i = 0; i < path.Count; i++)
            {
                var last = i == path.Count - 1;
                node = node.Child(path[i], last && asArray);
            }
            return node;
        }

        private Builder Child(string name, bool asArray)
        {
            var index = _members.FindIndex(m => m.Key == name);
            if (index < 0)
            {
                var created = new Builder();
                _members.Add(new KeyValuePair<string, object>(
                    name, asArray ? new List<Builder> { created } : created));
                return created;
            }

            var existing = _members[index].Value;
            if (asArray && existing is List<Builder> list)
            {
                var added = new Builder();
                list.Add(added);
                return added;
            }
            if (existing is Builder builder)
            {
                return builder;
            }
            if (existing is List<Builder> single)
            {
                return single[^1];
            }

            var replacement = new Builder();
            _members[index] = new KeyValuePair<string, object>(name, replacement);
            return replacement;
        }

        public void Set(List<string> path, JsonNode value)
        {
            var node = this;
            for (var i = 0; i < path.Count - 1; i++)
            {
                node = node.Child(path[i], asArray: false);
            }
            node._members.Add(new KeyValuePair<string, object>(path[^1], value));
        }

        public JsonNode Build()
        {
            var members = new List<KeyValuePair<string, JsonNode>>();
            foreach (var (key, value) in _members)
            {
                members.Add(new KeyValuePair<string, JsonNode>(key, value switch
                {
                    JsonNode node => node,
                    Builder builder => builder.Build(),
                    List<Builder> list => new JsonNode
                    {
                        Kind = JsonKind.Array,
                        Items = [.. list.Select(b => b.Build())],
                    },
                    _ => JsonNode.Null,
                }));
            }
            return new JsonNode { Kind = JsonKind.Object, Members = members };
        }
    }
}
