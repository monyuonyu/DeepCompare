using System.Text;
using System.Text.RegularExpressions;

namespace DeepCompare.Engine;

/// <summary>依存 1 つの変化。</summary>
public enum DependencyChangeKind
{
    Added,
    Removed,
    Upgraded,
    Downgraded,
    Changed,
}

/// <summary>依存 1 つの変化。</summary>
public sealed record DependencyChange(
    string Name,
    DependencyChangeKind Kind,
    string? From,
    string? To)
{
    public string Describe() => Kind switch
    {
        DependencyChangeKind.Added => $"+ {Name} {To}",
        DependencyChangeKind.Removed => $"- {Name} {From}",
        DependencyChangeKind.Upgraded => $"↑ {Name} {From} → {To}",
        DependencyChangeKind.Downgraded => $"↓ {Name} {From} → {To}",
        _ => $"~ {Name} {From} → {To}",
    };
}

/// <summary>
/// 依存の一覧（ロックファイル）の変化をまとめる。
///
/// **数千行動いても、意味のある変化は数行。** package-lock.json や Cargo.lock は
/// 1 つ依存を上げるだけで整合性ハッシュや推移的な依存が大量に書き換わる。
/// 差分をそのまま見ても「何が起きたか」は読み取れない。
///
/// 構造比較（<see cref="StructuredCompare"/>）の上に載せる。**新しい解析器を
/// 書かない。** ロックファイルはどれも JSON か TOML か YAML なので、木にして
/// から「名前と版」の対を拾えば足りる。
/// </summary>
public static class DependencySummary
{
    /// <summary>
    /// 依存の一覧が入っていそうな場所。
    ///
    /// **形式ごとに置き場所が違う**ので、当たりを付けて探す。見つからなければ
    /// 木全体から「version を持つ物体」を拾う。
    /// </summary>
    private static readonly string[][] KnownRoots =
    [
        ["packages"],           // package-lock.json (v2/v3)
        ["dependencies"],       // package-lock.json (v1)、その他
        ["package"],            // Cargo.lock
        ["importers"],          // pnpm-lock.yaml
        ["default"],            // Pipfile.lock
        ["develop"],
    ];

    /// <summary>その名前がロックファイルらしいか。</summary>
    public static bool LooksLikeLockFile(string path)
    {
        var name = Path.GetFileName(path).ToLowerInvariant();
        return name is "package-lock.json" or "npm-shrinkwrap.json" or "cargo.lock"
            or "pnpm-lock.yaml" or "yarn.lock" or "poetry.lock" or "pipfile.lock"
            or "composer.lock" or "gemfile.lock" or "go.sum" or "packages.lock.json";
    }

    public static IReadOnlyList<DependencyChange> Compare(JsonNode left, JsonNode right)
    {
        var before = Collect(left);
        var after = Collect(right);
        var changes = new List<DependencyChange>();

        foreach (var (name, from) in before)
        {
            if (!after.TryGetValue(name, out var to))
            {
                changes.Add(new DependencyChange(name, DependencyChangeKind.Removed, from, null));
            }
            else if (!string.Equals(from, to, StringComparison.Ordinal))
            {
                changes.Add(new DependencyChange(name, Direction(from, to), from, to));
            }
        }

        foreach (var (name, to) in after)
        {
            if (!before.ContainsKey(name))
            {
                changes.Add(new DependencyChange(name, DependencyChangeKind.Added, null, to));
            }
        }

        return [.. changes.OrderBy(c => c.Name, StringComparer.OrdinalIgnoreCase)];
    }

    /// <summary>
    /// 木から「名前 → 版」を拾う。
    ///
    /// **入れ子の深さを決め打たない。** 形式ごとに置き場所が違ううえ、
    /// 同じ形式でも版によって変わる（package-lock は v1 と v2 で別物）。
    /// 「version を持つ物体」を探す方が、形式が増えても壊れない。
    /// </summary>
    internal static Dictionary<string, string> Collect(JsonNode node)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var root in KnownRoots)
        {
            var target = node;
            foreach (var step in root)
            {
                target = target?.Member(step);
            }
            if (target is not null)
            {
                Walk(target, result, string.Empty, 0);
            }
        }

        // 当たりが外れたら木全体を見る。
        if (result.Count == 0)
        {
            Walk(node, result, string.Empty, 0);
        }
        return result;
    }

    private const int MaximumDepth = 6;

    private static void Walk(JsonNode node, Dictionary<string, string> into, string name, int depth)
    {
        if (depth > MaximumDepth)
        {
            return;
        }

        switch (node.Kind)
        {
            case JsonKind.Object:
            {
                // その物体自身が「版を持つ依存」なら、そこで拾う。
                var version = node.Member("version") ?? node.Member("Version");
                if (version is not null && version.Kind is not (JsonKind.Object or JsonKind.Array))
                {
                    var key = node.Member("name")?.Value is { Length: > 0 } explicitName
                        ? explicitName
                        : Clean(name);
                    if (key.Length > 0)
                    {
                        into[key] = version.Value;
                        // 版を持つものの中は見ない。推移的な依存まで拾うと、
                        // **上げた 1 つが数十件に膨れて要約にならない。**
                        return;
                    }
                }

                foreach (var (key, value) in node.Members)
                {
                    Walk(value, into, key, depth + 1);
                }
                break;
            }

            case JsonKind.Array:
                foreach (var item in node.Items)
                {
                    Walk(item, into, name, depth + 1);
                }
                break;

            case JsonKind.String when name.Length > 0 && LooksLikeVersion(node.Value):
                // 「名前: 版」の素朴な形（package.json の dependencies など）。
                into[Clean(name)] = node.Value;
                break;
        }
    }

    /// <summary>置き場所の飾りを落とす。<c>node_modules/foo</c> → <c>foo</c>。</summary>
    private static string Clean(string name)
    {
        var index = name.LastIndexOf("node_modules/", StringComparison.Ordinal);
        return index >= 0 ? name[(index + "node_modules/".Length)..] : name;
    }

    private static readonly Regex VersionShape = new(@"^[\^~>=<]*\s*v?\d+(\.\d+)*");

    private static bool LooksLikeVersion(string value)
        => value.Length > 0 && value.Length < 40 && VersionShape.IsMatch(value);

    /// <summary>
    /// 上がったか下がったか。
    ///
    /// **番号ごとに数として比べる。** 文字列のままだと "1.10.0" &lt; "1.9.0" に
    /// なり、上げたのを下げたと言うことになる。
    /// </summary>
    internal static DependencyChangeKind Direction(string from, string to)
    {
        var a = Numbers(from);
        var b = Numbers(to);
        if (a.Count == 0 || b.Count == 0)
        {
            return DependencyChangeKind.Changed;
        }

        for (var i = 0; i < Math.Max(a.Count, b.Count); i++)
        {
            var x = i < a.Count ? a[i] : 0;
            var y = i < b.Count ? b[i] : 0;
            if (x != y)
            {
                return y > x ? DependencyChangeKind.Upgraded : DependencyChangeKind.Downgraded;
            }
        }

        // 数は同じで文字列が違う（前置きや接尾辞の違い）。
        return DependencyChangeKind.Changed;
    }

    private static List<long> Numbers(string version)
    {
        var result = new List<long>();
        foreach (Match match in Regex.Matches(version, @"\d+"))
        {
            if (long.TryParse(match.Value, out var value))
            {
                result.Add(value);
            }
            // 前置きの区切り（-beta など）が来たら、そこで止める。
            // 1.0.0-beta.2 の 2 まで数に混ぜると、比べ方が狂う。
            if (result.Count >= 4)
            {
                break;
            }
        }
        return result;
    }

    public static string Format(IReadOnlyList<DependencyChange> changes)
    {
        if (changes.Count == 0)
        {
            return "依存の変化はありません。" + Environment.NewLine;
        }

        var text = new StringBuilder();
        foreach (var change in changes)
        {
            text.AppendLine(change.Describe());
        }
        text.AppendLine();

        var parts = new List<string>();
        void Count(DependencyChangeKind kind, string label)
        {
            var n = changes.Count(c => c.Kind == kind);
            if (n > 0)
            {
                parts.Add($"{label} {n}");
            }
        }
        Count(DependencyChangeKind.Added, "追加");
        Count(DependencyChangeKind.Removed, "削除");
        Count(DependencyChangeKind.Upgraded, "上げた");
        Count(DependencyChangeKind.Downgraded, "**下げた**");
        Count(DependencyChangeKind.Changed, "その他");
        text.AppendLine(string.Join("、", parts));
        return text.ToString();
    }
}
