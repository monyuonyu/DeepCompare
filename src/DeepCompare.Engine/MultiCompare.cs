using System.Text;

namespace DeepCompare.Engine;

/// <summary>N 個を並べたときの、1 つの位置の状態。</summary>
public enum MultiStatus
{
    /// <summary>全部同じ。</summary>
    Same,

    /// <summary>1 つだけ違う。**設定の間違いはたいていこの形で現れる。**</summary>
    OneDiffers,

    /// <summary>全部違う。環境ごとに違って当然の値（接続先など）。</summary>
    AllDiffer,

    /// <summary>一部にしか無い。</summary>
    Missing,
}

/// <summary>1 つの位置と、そこに並ぶ値。</summary>
public sealed record MultiRow(
    string Path,
    /// <summary>それぞれの値。無ければ null。</summary>
    IReadOnlyList<JsonNode?> Values,
    MultiStatus Status)
{
    /// <summary>ほかと違うものの位置。<see cref="MultiStatus.OneDiffers"/> のときだけ意味を持つ。</summary>
    public int Odd { get; init; } = -1;

    public string Display(int index)
        => Values[index] is { } node ? node.Display() : "（無し）";
}

public sealed record MultiComparison(
    IReadOnlyList<string> Names,
    IReadOnlyList<MultiRow> Rows)
{
    public int Differences => Rows.Count(r => r.Status != MultiStatus.Same);
}

/// <summary>
/// 3 つ以上を並べて比べる。
///
/// **dev / staging / prod の設定を突き合わせる**のが主な用途。2 つずつ
/// 比べると、3 つの関係（どれが仲間外れか）が見えない。実際に知りたいのは
/// 「本番だけ値が違う」「検証にだけ項目が無い」といった形。
///
/// 構造比較と同じ木を使う。**位置（パス）で揃える**ので、鍵の順序や整形の
/// 違いは無視される。
/// </summary>
public static class MultiCompare
{
    public static MultiComparison Compare(
        IReadOnlyList<string> names,
        IReadOnlyList<JsonNode> sources,
        StructuredCompareOptions? options = null)
    {
        if (sources.Count < 2)
        {
            throw new ArgumentException("2 つ以上が要ります。", nameof(sources));
        }

        options ??= new StructuredCompareOptions();

        // すべての位置を集める。**どれか 1 つにしか無い位置も拾う。**
        // 片方を基準にすると、基準に無い項目が最初から見えなくなる。
        var paths = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var source in sources)
        {
            foreach (var path in Leaves(source, "$"))
            {
                if (seen.Add(path))
                {
                    paths.Add(path);
                }
            }
        }

        var rows = new List<MultiRow>();
        foreach (var path in paths)
        {
            var values = sources.Select(s => Find(s, path)).ToList();
            rows.Add(Classify(path, values, options));
        }

        return new MultiComparison(names, rows);
    }

    private static MultiRow Classify(
        string path, List<JsonNode?> values, StructuredCompareOptions options)
    {
        var present = values.Where(v => v is not null).ToList();
        if (present.Count != values.Count)
        {
            return new MultiRow(path, values, MultiStatus.Missing);
        }

        // 値ごとに仲間を数える。
        var groups = new List<(JsonNode Sample, List<int> Members)>();
        for (var i = 0; i < values.Count; i++)
        {
            var value = values[i]!;
            var group = groups.FirstOrDefault(g => SameValue(g.Sample, value, options));
            if (group.Members is null)
            {
                groups.Add((value, [i]));
            }
            else
            {
                group.Members.Add(i);
            }
        }

        if (groups.Count == 1)
        {
            return new MultiRow(path, values, MultiStatus.Same);
        }

        // **1 つだけ違う形を特に拾う。** 設定の間違いはたいていこれ。
        if (groups.Count == 2)
        {
            var odd = groups.FirstOrDefault(g => g.Members.Count == 1);
            if (odd.Members is not null && values.Count > 2)
            {
                return new MultiRow(path, values, MultiStatus.OneDiffers)
                {
                    Odd = odd.Members[0],
                };
            }
        }

        return new MultiRow(path, values, MultiStatus.AllDiffer);
    }

    private static bool SameValue(JsonNode a, JsonNode b, StructuredCompareOptions options)
    {
        if (a.Kind != b.Kind)
        {
            return false;
        }
        if (a.Kind is JsonKind.Object or JsonKind.Array)
        {
            return true;   // 葉だけを比べるので、ここへは来ない
        }
        if (a.Kind == JsonKind.Number && options.NumbersByValue
            && decimal.TryParse(a.Value, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var x)
            && decimal.TryParse(b.Value, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var y))
        {
            return x == y;
        }
        return string.Equals(a.Value, b.Value, StringComparison.Ordinal);
    }

    /// <summary>葉の位置を集める。物体と配列そのものは出さない。</summary>
    private static IEnumerable<string> Leaves(JsonNode node, string path)
    {
        switch (node.Kind)
        {
            case JsonKind.Object:
                foreach (var (key, value) in node.Members)
                {
                    foreach (var leaf in Leaves(value, Join(path, key)))
                    {
                        yield return leaf;
                    }
                }
                break;

            case JsonKind.Array:
                for (var i = 0; i < node.Items.Count; i++)
                {
                    foreach (var leaf in Leaves(node.Items[i], $"{path}[{i}]"))
                    {
                        yield return leaf;
                    }
                }
                break;

            default:
                yield return path;
                break;
        }
    }

    private static string Join(string path, string key)
    {
        var plain = key.Length > 0
            && (char.IsLetter(key[0]) || key[0] == '_' || key[0] == '@')
            && key.All(c => char.IsLetterOrDigit(c) || c is '_' or '-' or '@' or '.');
        return plain ? $"{path}.{key}" : $"{path}[\"{key}\"]";
    }

    /// <summary>位置をたどって値を取る。無ければ null。</summary>
    private static JsonNode? Find(JsonNode node, string path)
    {
        var current = node;
        foreach (var step in Steps(path))
        {
            if (current is null)
            {
                return null;
            }
            current = step.Index >= 0
                ? (current.Kind == JsonKind.Array && step.Index < current.Items.Count
                    ? current.Items[step.Index]
                    : null)
                : current.Member(step.Name);
        }
        return current;
    }

    private static IEnumerable<(string Name, int Index)> Steps(string path)
    {
        var i = 0;
        // 先頭の `$` を飛ばす。
        if (path.StartsWith('$'))
        {
            i = 1;
        }

        while (i < path.Length)
        {
            if (path[i] == '.')
            {
                var start = ++i;
                while (i < path.Length && path[i] != '.' && path[i] != '[')
                {
                    i++;
                }
                yield return (path[start..i], -1);
            }
            else if (path[i] == '[')
            {
                i++;
                if (i < path.Length && path[i] == '"')
                {
                    var start = ++i;
                    while (i < path.Length && path[i] != '"')
                    {
                        i++;
                    }
                    yield return (path[start..i], -1);
                    i += 2;   // 閉じる引用符と ]
                }
                else
                {
                    var start = i;
                    while (i < path.Length && path[i] != ']')
                    {
                        i++;
                    }
                    yield return (string.Empty,
                        int.TryParse(path[start..i], out var index) ? index : -1);
                    i++;
                }
            }
            else
            {
                i++;
            }
        }
    }

    /// <summary>人が読む形に整える。</summary>
    public static string Format(MultiComparison comparison, bool differencesOnly = true)
    {
        var text = new StringBuilder();
        text.AppendLine(string.Join("  |  ", comparison.Names));
        text.AppendLine("legend = 全部同じ / ! 1 つだけ違う / ~ 全部違う / ? 一部に無い");
        text.AppendLine("---");

        foreach (var row in comparison.Rows)
        {
            if (differencesOnly && row.Status == MultiStatus.Same)
            {
                continue;
            }

            var mark = row.Status switch
            {
                MultiStatus.Same => '=',
                MultiStatus.OneDiffers => '!',
                MultiStatus.AllDiffer => '~',
                _ => '?',
            };

            var values = new List<string>();
            for (var i = 0; i < row.Values.Count; i++)
            {
                // 仲間外れに印を付ける。**そこが見たいものなので、目に入る形にする。**
                values.Add(i == row.Odd ? $"[{row.Display(i)}]" : row.Display(i));
            }
            text.AppendLine($"{mark} {row.Path}: {string.Join("  |  ", values)}");
        }

        text.AppendLine("---");
        text.AppendLine(comparison.Differences == 0
            ? "全部同じです。"
            : $"{comparison.Differences} 箇所が違います"
              + $"（1 つだけ違う {comparison.Rows.Count(r => r.Status == MultiStatus.OneDiffers)}"
              + $" / 一部に無い {comparison.Rows.Count(r => r.Status == MultiStatus.Missing)}）");
        return text.ToString();
    }
}
