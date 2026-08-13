using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// XML / TOML / YAML の読み取り。
///
/// **どれも同じ木に落とす。** 比較の側は木しか知らないので、ここが正しければ
/// 比較・画面・CLI がそのまま使える。
/// </summary>
public sealed class StructuredReadersTests
{
    private static JsonNode Xml(string text) => XmlReaderAdapter.Parse(text);
    private static JsonNode Toml(string text) => TomlReader.Parse(text);
    private static JsonNode Yaml(string text) => YamlReader.Parse(text);

    // --- 形式の判定 ---

    [Theory]
    [InlineData("a.xml", StructuredFormat.Xml)]
    [InlineData("a.csproj", StructuredFormat.Xml)]
    [InlineData("a.axaml", StructuredFormat.Xml)]
    [InlineData("a.toml", StructuredFormat.Toml)]
    [InlineData("a.yaml", StructuredFormat.Yaml)]
    [InlineData("a.yml", StructuredFormat.Yaml)]
    [InlineData("a.json", StructuredFormat.Json)]
    [InlineData("a.unknown", StructuredFormat.Json)]
    public void 拡張子から形式を決める(string path, StructuredFormat expected)
    {
        Assert.Equal(expected, StructuredReaders.ForPath(path));
    }

    // --- XML ---

    [Fact]
    public void XMLの属性と子要素を分けて持つ()
    {
        // **混ぜない。** 同じ名前の属性と要素があるとき、混ぜると片方が消える。
        var node = Xml("""<root id="1"><id>2</id></root>""");
        var root = node.Member("root")!;

        Assert.Equal("1", root.Member("@id")!.Value);
        Assert.Equal("2", root.Member("id")!.Value);
    }

    [Fact]
    public void 同じ名前の要素が複数あれば配列にする()
    {
        var root = Xml("<r><item>a</item><item>b</item></r>").Member("r")!;

        var items = root.Member("item")!;
        Assert.Equal(JsonKind.Array, items.Kind);
        Assert.Equal(2, items.Items.Count);
    }

    [Fact]
    public void 一つしか無い要素は値のまま()
    {
        var root = Xml("<r><item>a</item></r>").Member("r")!;

        Assert.Equal(JsonKind.String, root.Member("item")!.Kind);
    }

    [Fact]
    public void 数と真偽に見えるものはその種類にする()
    {
        // XML はすべて文字列だが、そのままだと 1 と 1.0 の違いが見えない。
        var root = Xml("""<r n="1" b="true" s="x" />""").Member("r")!;

        Assert.Equal(JsonKind.Number, root.Member("@n")!.Kind);
        Assert.Equal(JsonKind.Bool, root.Member("@b")!.Kind);
        Assert.Equal(JsonKind.String, root.Member("@s")!.Kind);
    }

    [Fact]
    public void 名前空間の宣言は落とす()
    {
        // 中身ではないので、差分に出しても直しようがない。
        var root = Xml("""<r xmlns="http://example.com" a="1" />""").Member("r")!;

        Assert.Single(root.Members);
        Assert.Equal("@a", root.Members[0].Key);
    }

    [Fact]
    public void 属性と本文が両方あれば本文を別に持つ()
    {
        var root = Xml("""<r a="1">text</r>""").Member("r")!;

        Assert.Equal("text", root.Member("#text")!.Value);
    }

    [Fact]
    public void 読めないXMLは場所を添えて知らせる()
    {
        var e = Assert.Throws<StructuredParseException>(() => Xml("<r><unclosed></r>"));

        Assert.Contains("行目", e.Message);
    }

    [Fact]
    public void XMLの属性の順序は差分にしない()
    {
        // 構造比較の側が順序を無視するので、読み取りは順序を保つだけでよい。
        var left = Xml("""<r a="1" b="2" />""");
        var right = Xml("""<r b="2" a="1" />""");

        Assert.Empty(StructuredCompare.Compare(left, right));
    }

    // --- TOML ---

    [Fact]
    public void TOMLの鍵と値を読む()
    {
        var node = Toml("""
            name = "app"
            port = 8080
            debug = true
            """);

        Assert.Equal("app", node.Member("name")!.Value);
        Assert.Equal(JsonKind.Number, node.Member("port")!.Kind);
        Assert.Equal(JsonKind.Bool, node.Member("debug")!.Kind);
    }

    [Fact]
    public void TOMLの表を入れ子にする()
    {
        var node = Toml("""
            [server]
            host = "localhost"

            [server.tls]
            enabled = true
            """);

        var tls = node.Member("server")!.Member("tls")!;
        Assert.Equal("true", tls.Member("enabled")!.Value);
    }

    [Fact]
    public void TOMLの表の配列を読む()
    {
        var node = Toml("""
            [[deps]]
            name = "a"

            [[deps]]
            name = "b"
            """);

        var deps = node.Member("deps")!;
        Assert.Equal(JsonKind.Array, deps.Kind);
        Assert.Equal(2, deps.Items.Count);
        Assert.Equal("b", deps.Items[1].Member("name")!.Value);
    }

    [Fact]
    public void TOMLの並びを読む()
    {
        var items = Toml("""ports = [80, 443, 8080]""").Member("ports")!;

        Assert.Equal(3, items.Items.Count);
        Assert.Equal("443", items.Items[1].Value);
    }

    [Fact]
    public void TOMLの行注釈を落とす()
    {
        Assert.Equal("app", Toml("""
            # これは注釈
            name = "app"   # 行末の注釈
            """).Member("name")!.Value);
    }

    [Fact]
    public void 引用符の中の記号を注釈と間違えない()
    {
        Assert.Equal("a#b", Toml("""key = "a#b" """).Member("key")!.Value);
    }

    [Fact]
    public void TOMLの点付きの鍵を入れ子にする()
    {
        var node = Toml("""a.b.c = 1""");

        Assert.Equal("1", node.Member("a")!.Member("b")!.Member("c")!.Value);
    }

    // --- YAML ---

    [Fact]
    public void YAMLの写像を読む()
    {
        var node = Yaml("""
            name: app
            port: 8080
            debug: true
            """);

        Assert.Equal("app", node.Member("name")!.Value);
        Assert.Equal(JsonKind.Number, node.Member("port")!.Kind);
        Assert.Equal(JsonKind.Bool, node.Member("debug")!.Kind);
    }

    [Fact]
    public void YAMLの入れ子を字下げで読む()
    {
        var node = Yaml("""
            server:
              host: localhost
              tls:
                enabled: true
            """);

        Assert.Equal("true", node.Member("server")!.Member("tls")!.Member("enabled")!.Value);
    }

    [Fact]
    public void YAMLの並びを読む()
    {
        var items = Yaml("""
            ports:
              - 80
              - 443
            """).Member("ports")!;

        Assert.Equal(JsonKind.Array, items.Kind);
        Assert.Equal(2, items.Items.Count);
    }

    [Fact]
    public void 並びの中の写像を読む()
    {
        var items = Yaml("""
            items:
              - name: a
                value: 1
              - name: b
                value: 2
            """).Member("items")!;

        Assert.Equal(2, items.Items.Count);
        Assert.Equal("b", items.Items[1].Member("name")!.Value);
        Assert.Equal("2", items.Items[1].Member("value")!.Value);
    }

    [Fact]
    public void 字下げしない並びも読む()
    {
        // YAML では鍵と同じ深さに「- 」を置く書き方も許される。
        var items = Yaml("""
            ports:
            - 80
            - 443
            """).Member("ports")!;

        Assert.Equal(2, items.Items.Count);
    }

    [Theory]
    [InlineData("yes", "true")]
    [InlineData("no", "false")]
    [InlineData("on", "true")]
    [InlineData("off", "false")]
    [InlineData("True", "true")]
    public void YAMLの真偽の書き方を揃える(string written, string expected)
    {
        // **ここを外すと `yes` と `true` を別物として出す。**
        var node = Yaml($"flag: {written}");

        Assert.Equal(JsonKind.Bool, node.Member("flag")!.Kind);
        Assert.Equal(expected, node.Member("flag")!.Value);
    }

    [Fact]
    public void URLのコロンを鍵の区切りと間違えない()
    {
        // `http://` の `:` を拾うと、URL を値に持つ行が全部壊れる。
        Assert.Equal("https://example.com/a",
            Yaml("url: https://example.com/a").Member("url")!.Value);
    }

    [Fact]
    public void YAMLの注釈を落とす()
    {
        Assert.Equal("app", Yaml("""
            # 先頭の注釈
            name: app   # 行末の注釈
            """).Member("name")!.Value);
    }

    [Fact]
    public void 複数行の文字列を読む()
    {
        var folded = Yaml("""
            text: |
              1 行目
              2 行目
            """).Member("text")!;

        Assert.Contains("1 行目", folded.Value);
        Assert.Contains("\n", folded.Value);
    }

    [Fact]
    public void 畳み込みは行を空白で継ぐ()
    {
        var folded = Yaml("""
            text: >
              1 行目
              2 行目
            """).Member("text")!;

        Assert.DoesNotContain("\n", folded.Value);
    }

    [Fact]
    public void 錨と参照には対応していないと言って止まる()
    {
        // **黙って無視すると、別物を同じと言うことになる。**
        // 読めないなら読めないと言う方が誠実。
        var e = Assert.Throws<StructuredParseException>(() => Yaml("""
            base: &anchor
              a: 1
            derived: *anchor
            """));

        Assert.Contains("錨と参照", e.Message);
    }

    [Fact]
    public void 字下げにタブを使っていたら知らせる()
    {
        var e = Assert.Throws<StructuredParseException>(() => Yaml("a:\n\tb: 1"));

        Assert.Contains("タブ", e.Message);
    }

    [Fact]
    public void 流れ形式の並びを読む()
    {
        var items = Yaml("ports: [80, 443]").Member("ports")!;

        Assert.Equal(2, items.Items.Count);
    }

    // --- 形式をまたいだ比較 ---

    [Fact]
    public void 鍵の順序が違うだけのYAMLは差分にしない()
    {
        var left = Yaml("a: 1\nb: 2");
        var right = Yaml("b: 2\na: 1");

        Assert.Empty(StructuredCompare.Compare(left, right));
    }

    [Fact]
    public void 設定の一箇所だけが変わったことを示せる()
    {
        var left = Yaml("""
            server:
              host: localhost
              port: 8080
            debug: false
            """);
        var right = Yaml("""
            debug: false
            server:
              port: 9090
              host: localhost
            """);

        var change = Assert.Single(StructuredCompare.Compare(left, right));
        Assert.Equal("$.server.port", change.Path);
    }
}
