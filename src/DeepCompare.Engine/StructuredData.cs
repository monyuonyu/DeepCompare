using System.Text;
using System.Text.Json;

namespace DeepCompare.Engine;

/// <summary>
/// 木の節点の種類。
///
/// 数と真偽と文字列を分けて持つのは、**型の変化を差分として出すため**。
/// すべて文字列にしてしまうと <c>"1"</c> と <c>1</c> の違いが消える。
/// 設定ファイルではこれが動作の違いになる（真偽値のつもりが文字列、など）。
/// </summary>
public enum JsonKind
{
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

/// <summary>
/// 構造化データの木。
///
/// JSON をそのまま持つのではなく、**比較のための形**に直してある。
/// - 物体（オブジェクト）の要素は名前で引ける辞書。書かれた順は保つが、比較では使わない
/// - 数は文字列のまま持つ。<c>1.0</c> と <c>1</c> を同じとするか別とするかを後で選べる
///   ようにするため。倍精度に丸めると選べなくなる
/// </summary>
public sealed record JsonNode
{
    public JsonKind Kind { get; init; }

    /// <summary>葉の値。物体と配列では空。</summary>
    public string Value { get; init; } = string.Empty;

    /// <summary>物体の要素。書かれた順に並ぶ。</summary>
    public IReadOnlyList<KeyValuePair<string, JsonNode>> Members { get; init; } = [];

    /// <summary>配列の要素。</summary>
    public IReadOnlyList<JsonNode> Items { get; init; } = [];

    public static readonly JsonNode Null = new() { Kind = JsonKind.Null };

    public JsonNode? Member(string name)
    {
        foreach (var (key, node) in Members)
        {
            if (string.Equals(key, name, StringComparison.Ordinal))
            {
                return node;
            }
        }
        return null;
    }

    /// <summary>
    /// 表示用の短い文字列。差分の一覧に出すので、長い物体は畳む。
    /// </summary>
    public string Display() => Kind switch
    {
        JsonKind.Null => "null",
        JsonKind.Bool or JsonKind.Number => Value,
        JsonKind.String => "\"" + Value + "\"",
        JsonKind.Array => "[" + Items.Count + " 要素]",
        JsonKind.Object => "{" + Members.Count + " 項目}",
        _ => string.Empty,
    };

    /// <summary>種類の名前。型の変化を伝えるときに使う。</summary>
    public string KindName() => Kind switch
    {
        JsonKind.Null => "null",
        JsonKind.Bool => "真偽値",
        JsonKind.Number => "数値",
        JsonKind.String => "文字列",
        JsonKind.Array => "配列",
        JsonKind.Object => "物体",
        _ => "?",
    };
}

/// <summary>読み取りに失敗したときに投げる。行と桁を添えて、どこが悪いか示す。</summary>
public sealed class StructuredParseException(string message) : Exception(message);

/// <summary>
/// JSON の読み取り。
///
/// System.Text.Json の <see cref="JsonDocument"/> を使う。自前で書く理由が無いうえ、
/// AOT でも反射を伴わない（<see cref="JsonDocument"/> は型に依存しない）。
///
/// **末尾カンマと注釈を許す。** 設定ファイルには実際に混ざっており、
/// 「読めない」で止まると比較の役に立たない。
/// </summary>
public static class JsonReader
{
    private static readonly JsonDocumentOptions Options = new()
    {
        AllowTrailingCommas = true,
        CommentHandling = JsonCommentHandling.Skip,
        MaxDepth = 256,
    };

    public static JsonNode Parse(string text)
    {
        try
        {
            using var document = JsonDocument.Parse(text, Options);
            return Convert(document.RootElement);
        }
        catch (JsonException e)
        {
            // 行と桁は例外が持っている。そのまま伝える方が直しやすい。
            throw new StructuredParseException(
                $"JSON として読めません（{e.LineNumber + 1} 行目 {e.BytePositionInLine + 1} 桁目）: {e.Message}");
        }
    }

    private static JsonNode Convert(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => new JsonNode
        {
            Kind = JsonKind.Object,
            Members = [.. element.EnumerateObject()
                .Select(p => new KeyValuePair<string, JsonNode>(p.Name, Convert(p.Value)))],
        },
        JsonValueKind.Array => new JsonNode
        {
            Kind = JsonKind.Array,
            Items = [.. element.EnumerateArray().Select(Convert)],
        },
        // GetRawText なら書かれたままの数字が取れる。1.0 と 1 を区別できる。
        JsonValueKind.Number => new JsonNode { Kind = JsonKind.Number, Value = element.GetRawText() },
        JsonValueKind.String => new JsonNode { Kind = JsonKind.String, Value = element.GetString() ?? string.Empty },
        JsonValueKind.True => new JsonNode { Kind = JsonKind.Bool, Value = "true" },
        JsonValueKind.False => new JsonNode { Kind = JsonKind.Bool, Value = "false" },
        _ => JsonNode.Null,
    };
}

/// <summary>差分の種類。</summary>
public enum StructuralChangeKind
{
    /// <summary>右にだけある。</summary>
    Added,

    /// <summary>左にだけある。</summary>
    Removed,

    /// <summary>両方にあり、値が違う。</summary>
    Changed,

    /// <summary>両方にあり、値は等しいが種類が違う（"1" と 1 など）。</summary>
    TypeChanged,

    /// <summary>両方にあり、配列の中の位置だけが違う。</summary>
    Moved,
}

/// <summary>
/// 1 個の差分。
///
/// <paramref name="Path"/> は <c>spec.containers[0].image</c> の形。
/// 行番号ではなく構造上の位置で示す。整形が違えば行番号は当てにならない。
/// </summary>
public sealed record StructuralChange(
    StructuralChangeKind Kind,
    string Path,
    JsonNode? Left,
    JsonNode? Right)
{
    /// <summary>人が読む一行。</summary>
    public string Describe() => Kind switch
    {
        StructuralChangeKind.Added => $"+ {Path} = {Right?.Display()}",
        StructuralChangeKind.Removed => $"- {Path} = {Left?.Display()}",
        StructuralChangeKind.Changed => $"~ {Path}: {Left?.Display()} → {Right?.Display()}",
        StructuralChangeKind.TypeChanged =>
            $"! {Path}: {Left?.KindName()} → {Right?.KindName()}（{Left?.Display()} → {Right?.Display()}）",
        StructuralChangeKind.Moved => $"→ {Path}: 位置が変わった",
        _ => Path,
    };
}

/// <summary>
/// 構造比較の設定。
///
/// 既定はすべて「厳しい方」ではなく「実用の方」に寄せてある。整形の違いを差分に
/// 出しても誰も得をしない。
/// </summary>
public sealed record StructuredCompareOptions
{
    /// <summary>
    /// 配列の要素を対応付けるキー。<c>["id", "name"]</c> のように与えると、
    /// 要素がその名前を持つ物体のとき、値で組にする。順序だけの違いは差分にしない。
    ///
    /// 空なら位置で組にする（従来どおり）。
    /// </summary>
    public IReadOnlyList<string> ArrayKeys { get; init; } = ["id", "name", "key", "path"];

    /// <summary>
    /// 配列の順序の違いを差分として報告するか。
    ///
    /// 既定は報告する（<see cref="StructuralChangeKind.Moved"/>）。順序に意味がある
    /// 配列（処理の手順など）で、黙って無視されると困る。
    /// </summary>
    public bool ReportMoves { get; init; } = true;

    /// <summary>
    /// <c>1.0</c> と <c>1</c> を同じとみなすか。既定は同じ。
    /// 書式の違いでしかなく、値としては等しい。
    /// </summary>
    public bool NumbersByValue { get; init; } = true;

    /// <summary>比べない位置。<c>metadata.generated_at</c> のような、毎回変わるもの。</summary>
    public IReadOnlyList<string> IgnoredPaths { get; init; } = [];
}

/// <summary>
/// 構造化データの比較。
///
/// **なぜ要るか。** JSON をテキストとして比べると、キーの順序が変わっただけ、
/// 整形が変わっただけで全行が差分になる。本当に変わった 1 個の値が埋もれる。
/// ここでは木として比べ、位置（パス）と変化の種類だけを出す。
///
/// 配列の扱いが要点。<c>[{id:1},{id:2}]</c> と <c>[{id:2},{id:1}]</c> を
/// 「全部違う」と言うのは正しくない。キーで組にしてから中身を比べる。
/// </summary>
public static class StructuredCompare
{
    public static IReadOnlyList<StructuralChange> Compare(
        JsonNode left, JsonNode right, StructuredCompareOptions? options = null)
    {
        options ??= new StructuredCompareOptions();
        var changes = new List<StructuralChange>();
        Walk(left, right, "$", changes, options);
        return changes;
    }

    /// <summary>読み取りから比較まで一息に。</summary>
    public static IReadOnlyList<StructuralChange> CompareJson(
        string leftText, string rightText, StructuredCompareOptions? options = null)
        => Compare(JsonReader.Parse(leftText), JsonReader.Parse(rightText), options);

    private static void Walk(
        JsonNode left,
        JsonNode right,
        string path,
        List<StructuralChange> changes,
        StructuredCompareOptions options)
    {
        if (IsIgnored(path, options))
        {
            return;
        }

        if (left.Kind != right.Kind)
        {
            // 種類が違う。値まで等しく見えるなら（"1" と 1）、それが伝わる形で出す。
            var kind = SameText(left, right)
                ? StructuralChangeKind.TypeChanged
                : StructuralChangeKind.Changed;
            changes.Add(new StructuralChange(kind, path, left, right));
            return;
        }

        switch (left.Kind)
        {
            case JsonKind.Object:
                WalkObject(left, right, path, changes, options);
                break;
            case JsonKind.Array:
                WalkArray(left, right, path, changes, options);
                break;
            default:
                if (!SameLeaf(left, right, options))
                {
                    changes.Add(new StructuralChange(StructuralChangeKind.Changed, path, left, right));
                }
                break;
        }
    }

    private static void WalkObject(
        JsonNode left,
        JsonNode right,
        string path,
        List<StructuralChange> changes,
        StructuredCompareOptions options)
    {
        // 左に書かれた順で回る。順序は比較に使わないが、出力の順序は安定させたい。
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (key, leftValue) in left.Members)
        {
            if (!seen.Add(key))
            {
                // 同じ名前が 2 回。JSON としては不正だが読めてしまう場合がある。
                // 最初の 1 個だけを見る（JsonDocument も後勝ちにはしない）。
                continue;
            }
            var child = Join(path, key);
            var rightValue = right.Member(key);
            if (rightValue is null)
            {
                if (!IsIgnored(child, options))
                {
                    changes.Add(new StructuralChange(StructuralChangeKind.Removed, child, leftValue, null));
                }
            }
            else
            {
                Walk(leftValue, rightValue, child, changes, options);
            }
        }

        foreach (var (key, rightValue) in right.Members)
        {
            if (left.Member(key) is null && seen.Add(key))
            {
                var child = Join(path, key);
                if (!IsIgnored(child, options))
                {
                    changes.Add(new StructuralChange(StructuralChangeKind.Added, child, null, rightValue));
                }
            }
        }
    }

    private static void WalkArray(
        JsonNode left,
        JsonNode right,
        string path,
        List<StructuralChange> changes,
        StructuredCompareOptions options)
    {
        // 要素を対応付けられるキーがあるか調べる。両側で同じ名前が使えて、
        // かつ**その側の中で値が重複していない**ことを求める。重複していると
        // どれと組にするか決められず、位置で組にした方がまだ正しい。
        var key = ChooseKey(left, right, options);
        if (key is null)
        {
            WalkByPosition(left, right, path, changes, options);
            return;
        }

        var leftByKey = Index(left, key);
        var rightByKey = Index(right, key);

        for (var i = 0; i < left.Items.Count; i++)
        {
            var id = KeyOf(left.Items[i], key)!;
            var child = $"{path}[{key}={id}]";
            if (rightByKey.TryGetValue(id, out var pair))
            {
                Walk(left.Items[i], pair.Node, child, changes, options);
                if (options.ReportMoves && pair.Index != i && !IsIgnored(child, options))
                {
                    changes.Add(new StructuralChange(
                        StructuralChangeKind.Moved, child, left.Items[i], pair.Node));
                }
            }
            else if (!IsIgnored(child, options))
            {
                changes.Add(new StructuralChange(
                    StructuralChangeKind.Removed, child, left.Items[i], null));
            }
        }

        for (var i = 0; i < right.Items.Count; i++)
        {
            var id = KeyOf(right.Items[i], key)!;
            if (!leftByKey.ContainsKey(id))
            {
                var child = $"{path}[{key}={id}]";
                if (!IsIgnored(child, options))
                {
                    changes.Add(new StructuralChange(
                        StructuralChangeKind.Added, child, null, right.Items[i]));
                }
            }
        }
    }

    private static void WalkByPosition(
        JsonNode left,
        JsonNode right,
        string path,
        List<StructuralChange> changes,
        StructuredCompareOptions options)
    {
        var shared = Math.Min(left.Items.Count, right.Items.Count);
        for (var i = 0; i < shared; i++)
        {
            Walk(left.Items[i], right.Items[i], $"{path}[{i}]", changes, options);
        }
        for (var i = shared; i < left.Items.Count; i++)
        {
            var child = $"{path}[{i}]";
            if (!IsIgnored(child, options))
            {
                changes.Add(new StructuralChange(
                    StructuralChangeKind.Removed, child, left.Items[i], null));
            }
        }
        for (var i = shared; i < right.Items.Count; i++)
        {
            var child = $"{path}[{i}]";
            if (!IsIgnored(child, options))
            {
                changes.Add(new StructuralChange(
                    StructuralChangeKind.Added, child, null, right.Items[i]));
            }
        }
    }

    /// <summary>
    /// 配列の対応付けに使える名前を選ぶ。
    ///
    /// 条件は 3 つ。**すべての要素が物体でその名前を持つ**、**葉の値である**、
    /// **その側の中で値が重複していない**。一つでも欠けたら位置で組にする方が安全。
    /// </summary>
    private static string? ChooseKey(JsonNode left, JsonNode right, StructuredCompareOptions options)
    {
        if (left.Items.Count == 0 || right.Items.Count == 0)
        {
            return null;
        }

        foreach (var candidate in options.ArrayKeys)
        {
            if (Usable(left, candidate) && Usable(right, candidate))
            {
                return candidate;
            }
        }
        return null;
    }

    private static bool Usable(JsonNode array, string key)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in array.Items)
        {
            var value = KeyOf(item, key);
            if (value is null || !seen.Add(value))
            {
                return false;
            }
        }
        return true;
    }

    private static string? KeyOf(JsonNode item, string key)
    {
        if (item.Kind != JsonKind.Object)
        {
            return null;
        }
        var value = item.Member(key);
        return value is null || value.Kind is JsonKind.Object or JsonKind.Array
            ? null
            : value.Value;
    }

    private static Dictionary<string, (int Index, JsonNode Node)> Index(JsonNode array, string key)
    {
        var map = new Dictionary<string, (int, JsonNode)>(StringComparer.Ordinal);
        for (var i = 0; i < array.Items.Count; i++)
        {
            if (KeyOf(array.Items[i], key) is { } id)
            {
                map[id] = (i, array.Items[i]);
            }
        }
        return map;
    }

    private static bool SameLeaf(JsonNode left, JsonNode right, StructuredCompareOptions options)
    {
        if (left.Kind == JsonKind.Number && options.NumbersByValue)
        {
            // 1.0 と 1、1e3 と 1000 を同じとみなす。
            //
            // **倍精度は使わない。** 123456789012345678901 と ...902 が同じ値に
            // 丸められ、差を見逃す。JSON の数値には識別子や金額が入っていることが
            // あり、これは実害になる。10 進数なら 28 桁まで正確に区別できる。
            // それでも読めない（桁が多すぎる、指数が大きすぎる）ときは文字列で比べる。
            if (decimal.TryParse(left.Value, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var a)
                && decimal.TryParse(right.Value, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out var b))
            {
                return a == b;
            }
        }
        return string.Equals(left.Value, right.Value, StringComparison.Ordinal);
    }

    /// <summary>種類は違うが、書かれた中身は同じか。<c>"1"</c> と <c>1</c> を拾う。</summary>
    private static bool SameText(JsonNode left, JsonNode right)
        => left.Kind is not (JsonKind.Object or JsonKind.Array)
           && right.Kind is not (JsonKind.Object or JsonKind.Array)
           && string.Equals(left.Value, right.Value, StringComparison.Ordinal);

    private static bool IsIgnored(string path, StructuredCompareOptions options)
    {
        foreach (var ignored in options.IgnoredPaths)
        {
            if (PathMatches(path, ignored))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// 無視する位置の照合。<c>*</c> は 1 段ぶんの任意に当たる。
    /// <c>$.cells[*].execution_count</c> のように書ける。
    /// </summary>
    private static bool PathMatches(string path, string pattern)
    {
        if (string.Equals(path, pattern, StringComparison.Ordinal))
        {
            return true;
        }
        if (!pattern.Contains('*'))
        {
            // 前方一致も許す。親を無視すれば子も無視される。
            return path.StartsWith(pattern + ".", StringComparison.Ordinal)
                   || path.StartsWith(pattern + "[", StringComparison.Ordinal);
        }

        var parts = pattern.Split('*');
        var position = 0;
        for (var i = 0; i < parts.Length; i++)
        {
            if (parts[i].Length == 0)
            {
                continue;
            }
            var found = path.IndexOf(parts[i], position, StringComparison.Ordinal);
            if (found < 0 || (i == 0 && found != 0))
            {
                return false;
            }
            position = found + parts[i].Length;
        }
        return parts[^1].Length == 0 || path.EndsWith(parts[^1], StringComparison.Ordinal);
    }

    private static string Join(string path, string key)
    {
        // 識別子として素直な名前だけを点で継ぐ。空白や記号を含むものは括弧書きにする。
        var plain = key.Length > 0
            && (char.IsLetter(key[0]) || key[0] == '_')
            && key.All(c => char.IsLetterOrDigit(c) || c == '_' || c == '-');
        return plain ? $"{path}.{key}" : $"{path}[\"{Escape(key)}\"]";
    }

    private static string Escape(string key)
        => key.Replace("\\", "\\\\", StringComparison.Ordinal)
              .Replace("\"", "\\\"", StringComparison.Ordinal);

    /// <summary>差分の一覧を人が読む形に整える。CLI の出力に使う。</summary>
    public static string Format(IReadOnlyList<StructuralChange> changes)
    {
        if (changes.Count == 0)
        {
            return "構造としては同じです。" + Environment.NewLine;
        }
        var builder = new StringBuilder();
        foreach (var change in changes)
        {
            builder.AppendLine(change.Describe());
        }
        builder.AppendLine();
        builder.AppendLine(Summarize(changes));
        return builder.ToString();
    }

    public static string Summarize(IReadOnlyList<StructuralChange> changes)
    {
        var added = changes.Count(c => c.Kind == StructuralChangeKind.Added);
        var removed = changes.Count(c => c.Kind == StructuralChangeKind.Removed);
        var changed = changes.Count(c => c.Kind == StructuralChangeKind.Changed);
        var typed = changes.Count(c => c.Kind == StructuralChangeKind.TypeChanged);
        var moved = changes.Count(c => c.Kind == StructuralChangeKind.Moved);

        var parts = new List<string>();
        if (added > 0) { parts.Add($"追加 {added}"); }
        if (removed > 0) { parts.Add($"削除 {removed}"); }
        if (changed > 0) { parts.Add($"変更 {changed}"); }
        if (typed > 0) { parts.Add($"型の変化 {typed}"); }
        if (moved > 0) { parts.Add($"移動 {moved}"); }
        return string.Join("、", parts);
    }
}
