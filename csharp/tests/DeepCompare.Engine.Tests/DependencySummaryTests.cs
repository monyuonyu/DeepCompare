using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// ロックファイルの要約。
///
/// **数千行動いても、意味のある変化は数行。** そこを取り出せるかを見る。
/// </summary>
public sealed class DependencySummaryTests
{
    private static IReadOnlyList<DependencyChange> Compare(string left, string right)
        => DependencySummary.Compare(JsonReader.Parse(left), JsonReader.Parse(right));

    [Theory]
    [InlineData("package-lock.json", true)]
    [InlineData("Cargo.lock", true)]
    [InlineData("pnpm-lock.yaml", true)]
    [InlineData("poetry.lock", true)]
    [InlineData("go.sum", true)]
    [InlineData("package.json", false)]
    [InlineData("main.cs", false)]
    public void ロックファイルらしい名前を見分ける(string name, bool expected)
    {
        Assert.Equal(expected, DependencySummary.LooksLikeLockFile(name));
    }

    [Fact]
    public void 上げた依存だけを取り出す()
    {
        var left = """
            {"packages": {
              "node_modules/a": {"version": "1.0.0", "resolved": "https://x/a-1.0.0.tgz", "integrity": "sha512-AAA"},
              "node_modules/b": {"version": "2.0.0", "resolved": "https://x/b-2.0.0.tgz", "integrity": "sha512-BBB"}
            }}
            """;
        var right = """
            {"packages": {
              "node_modules/a": {"version": "1.2.0", "resolved": "https://x/a-1.2.0.tgz", "integrity": "sha512-CCC"},
              "node_modules/b": {"version": "2.0.0", "resolved": "https://x/b-2.0.0.tgz", "integrity": "sha512-BBB"}
            }}
            """;

        // ハッシュや取得先も動いているが、**要約には出さない。**
        var change = Assert.Single(Compare(left, right));
        Assert.Equal("a", change.Name);
        Assert.Equal(DependencyChangeKind.Upgraded, change.Kind);
        Assert.Equal("1.0.0", change.From);
        Assert.Equal("1.2.0", change.To);
    }

    [Fact]
    public void 下げた依存を上げたと言わない()
    {
        // **文字列で比べると "1.10.0" < "1.9.0" になる。**
        // 番号ごとに数として比べないと、上げたのを下げたと言うことになる。
        Assert.Equal(DependencyChangeKind.Upgraded, DependencySummary.Direction("1.9.0", "1.10.0"));
        Assert.Equal(DependencyChangeKind.Downgraded, DependencySummary.Direction("1.10.0", "1.9.0"));
    }

    [Fact]
    public void 追加と削除を出す()
    {
        var left = """{"packages": {"node_modules/a": {"version": "1.0.0"}}}""";
        var right = """{"packages": {"node_modules/b": {"version": "2.0.0"}}}""";

        var changes = Compare(left, right);

        Assert.Equal(2, changes.Count);
        Assert.Contains(changes, c => c.Name == "a" && c.Kind == DependencyChangeKind.Removed);
        Assert.Contains(changes, c => c.Name == "b" && c.Kind == DependencyChangeKind.Added);
    }

    [Fact]
    public void 置き場所の飾りを落とす()
    {
        var left = """{"packages": {"node_modules/@scope/pkg": {"version": "1.0.0"}}}""";
        var right = """{"packages": {"node_modules/@scope/pkg": {"version": "1.1.0"}}}""";

        Assert.Equal("@scope/pkg", Assert.Single(Compare(left, right)).Name);
    }

    [Fact]
    public void 推移的な依存の中までは見ない()
    {
        // **上げた 1 つが数十件に膨れると要約にならない。**
        var left = """
            {"packages": {"node_modules/a": {"version": "1.0.0",
              "dependencies": {"inner": {"version": "0.1.0"}}}}}
            """;
        var right = """
            {"packages": {"node_modules/a": {"version": "1.1.0",
              "dependencies": {"inner": {"version": "0.2.0"}}}}}
            """;

        var change = Assert.Single(Compare(left, right));
        Assert.Equal("a", change.Name);
    }

    [Fact]
    public void 版を持つ形が違っても拾う()
    {
        // Cargo.lock は [[package]] の配列。名前は中の name に入る。
        var left = """{"package": [{"name": "serde", "version": "1.0.100"}]}""";
        var right = """{"package": [{"name": "serde", "version": "1.0.200"}]}""";

        var change = Assert.Single(Compare(left, right));
        Assert.Equal("serde", change.Name);
        Assert.Equal(DependencyChangeKind.Upgraded, change.Kind);
    }

    [Fact]
    public void 素朴な名前と版の対も拾う()
    {
        var left = """{"dependencies": {"react": "18.2.0", "lodash": "^4.17.20"}}""";
        var right = """{"dependencies": {"react": "18.3.0", "lodash": "^4.17.20"}}""";

        var change = Assert.Single(Compare(left, right));
        Assert.Equal("react", change.Name);
    }

    [Fact]
    public void 変化が無ければそう言う()
    {
        var text = """{"packages": {"node_modules/a": {"version": "1.0.0"}}}""";

        Assert.Empty(Compare(text, text));
        Assert.Contains("変化はありません", DependencySummary.Format(Compare(text, text)));
    }

    [Fact]
    public void 下げたことを強めに出す()
    {
        // 依存を下げるのは、たいてい意図しない事故。目に入る形にする。
        var left = """{"packages": {"node_modules/a": {"version": "2.0.0"}}}""";
        var right = """{"packages": {"node_modules/a": {"version": "1.0.0"}}}""";

        Assert.Contains("下げた", DependencySummary.Format(Compare(left, right)));
    }

    [Fact]
    public void YAMLのロックファイルも扱える()
    {
        var left = YamlReader.Parse("""
            importers:
              app:
                version: 1.0.0
            """);
        var right = YamlReader.Parse("""
            importers:
              app:
                version: 1.1.0
            """);

        Assert.Single(DependencySummary.Compare(left, right));
    }
}
